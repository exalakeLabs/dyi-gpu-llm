import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as _tf_version

from project_config import ADAPTER_DIR, BASE_MODEL, MAX_NEW_TOKENS

# transformers ≥ 4.51 renamed the from_pretrained dtype kwarg from
# `torch_dtype` to `dtype`; older builds silently ignore `dtype`.
_tf_ver = tuple(int(x) for x in _tf_version.split(".")[:2])
_DTYPE_KWARG = "dtype" if _tf_ver >= (4, 51) else "torch_dtype"


def is_rocm() -> bool:
    """Return True when running on AMD ROCm (HIP), False for NVIDIA CUDA or CPU."""
    return torch.cuda.is_available() and torch.version.hip is not None


def _rocm_supports_bf16() -> bool:
    """
    Return True for AMD architectures with native bfloat16 support.
    CDNA2 (gfx90a / MI200) and CDNA3 (gfx940-942 / MI300) have full bf16
    ALUs. RDNA3 (gfx1100-1102) also supports bf16 natively.
    """
    if not is_rocm():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return any(
            arch.startswith(p)
            for p in ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101", "gfx1102")
        )
    except Exception:
        return False


def _model_dtype() -> torch.dtype:
    """Select the best dtype: bf16 on capable AMD hardware, fp16 elsewhere."""
    if _rocm_supports_bf16():
        return torch.bfloat16
    return torch.float16


def patch_rocm_isin() -> None:
    """
    Work around a ROCm bug where torch.isin raises on GPU tensors.
    Only applied on ROCm; real CUDA handles isin on-device correctly.
    """
    if not is_rocm():
        return
    if getattr(torch.isin, "_llama_local_patched", False):
        return

    original_isin = torch.isin

    def safe_isin(elements, test_elements, *args, **kwargs):
        elems_dev = getattr(elements, "device", None)
        test_dev = getattr(test_elements, "device", None)
        elems_on_gpu = elems_dev is not None and elems_dev.type == "cuda"
        test_on_gpu = test_dev is not None and test_dev.type == "cuda"

        if elems_on_gpu or test_on_gpu:
            cpu_elems = elements.cpu() if elems_on_gpu else elements
            cpu_test = test_elements.cpu() if test_on_gpu else test_elements
            result = original_isin(cpu_elems, cpu_test, *args, **kwargs)
            # Output lives where elements lives; only move if elements was on GPU.
            return result.to(elems_dev) if elems_on_gpu else result

        return original_isin(elements, test_elements, *args, **kwargs)

    safe_isin._llama_local_patched = True
    torch.isin = safe_isin


def load_tokenizer(base_model: str = BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_model: str = BASE_MODEL, **kwargs):
    # device_map={"": 0} targets device 0 (works for ROCm/HIP and CUDA alike).
    # Omitted on CPU-only builds: transformers raises when no accelerator is present.
    model_kwargs: dict = {
        _DTYPE_KWARG: _model_dtype(),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model_kwargs.update(kwargs)
    return AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)


def load_generation_model(
    base_model: str = BASE_MODEL,
    adapter_path=ADAPTER_DIR,
    use_adapter=True,
    **model_kwargs,
):
    patch_rocm_isin()
    tokenizer = load_tokenizer(base_model)
    model = load_base_model(base_model, **model_kwargs)

    if use_adapter:
        adapter_path = pathlib.Path(adapter_path)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No adapter_config.json found at '{adapter_path}'. "
                "Run training first, or pass use_adapter=False / --no-adapter to load the base model only."
            )
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            autocast_adapter_dtype=False,
        )

    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


def chat_inputs(tokenizer, model, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(model.device) for k, v in inputs.items()}


def generate_text(tokenizer, model, messages, **generate_kwargs) -> str:
    inputs = chat_inputs(tokenizer, model, messages)
    defaults = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.15,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    defaults.update(generate_kwargs)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **defaults)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
