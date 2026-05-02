import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from project_config import ADAPTER_DIR, BASE_MODEL


def patch_rocm_isin() -> None:
    if getattr(torch.isin, "_llama_local_patched", False):
        return

    original_isin = torch.isin

    def safe_isin(elements, test_elements, *args, **kwargs):
        elems_dev = getattr(elements, "device", None)
        test_dev = getattr(test_elements, "device", None)
        elems_cuda = elems_dev is not None and elems_dev.type == "cuda"
        test_cuda = test_dev is not None and test_dev.type == "cuda"

        if elems_cuda or test_cuda:
            out = original_isin(elements.cpu(), test_elements.cpu(), *args, **kwargs)
            if elems_cuda:
                return out.to(elems_dev)
            if test_cuda:
                return out.to(test_dev)
        return original_isin(elements, test_elements, *args, **kwargs)

    safe_isin._llama_local_patched = True
    torch.isin = safe_isin


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(**kwargs):
    # device_map={"": 0} targets CUDA device 0. On CPU-only builds, transformers
    # still expands that map and touches cuda:0 during load, which raises
    # "Cannot access accelerator device when none is available."
    model_kwargs = {
        "dtype": torch.float16,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model_kwargs.update(kwargs)
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        **model_kwargs,
    )


def load_generation_model(adapter_path=ADAPTER_DIR, use_adapter=True):
    patch_rocm_isin()
    tokenizer = load_tokenizer()
    model = load_base_model()

    if use_adapter:
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
        "max_new_tokens": 180,
        "do_sample": False,
        "repetition_penalty": 1.15,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    defaults.update(generate_kwargs)

    with torch.no_grad():
        outputs = model.generate(**inputs, **defaults)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
