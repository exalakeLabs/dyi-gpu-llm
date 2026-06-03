#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as TRANSFORMERS_VERSION

from runtime_env import env_int, env_str

DEFAULT_ATTENTION = env_str("DEFAULT_ATTENTION", "auto")
DEFAULT_BACKEND = env_str("DEFAULT_BACKEND", "rocm:0")
DEFAULT_DTYPE = env_str("DEFAULT_DTYPE", "auto")
DEFAULT_MODEL_PATH = env_str("DEFAULT_MODEL_PATH")
DEFAULT_SYSTEM_PROMPT = env_str("SYSTEM_PROMPT")
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _transformers_dtype_kwarg() -> str:
    match = re.match(r"^(\d+)\.(\d+)", TRANSFORMERS_VERSION)
    if not match:
        return "dtype"

    major, minor = (int(part) for part in match.groups())
    return "dtype" if (major, minor) >= (4, 51) else "torch_dtype"


def _checkpoint_step(path: Path) -> int:
    match = re.match(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(model_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in model_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "config.json").exists()
    ]
    return max(checkpoints, key=_checkpoint_step) if checkpoints else None


def _has_tokenizer_files(path: Path) -> bool:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "spiece.model",
    )
    return any((path / name).exists() for name in tokenizer_files)


def _resolve_model_path(model_path: str, checkpoint: str) -> Path:
    base = Path(model_path).expanduser()

    print(f"Model path: {base}")

    if checkpoint == "root":
        return base

    latest = _latest_checkpoint(base)
    if checkpoint == "latest":
        if latest is None:
            raise FileNotFoundError(f"No checkpoint-* directory found under {base}")
        return latest

    if (base / "config.json").exists():
        return base
    if latest is not None:
        return latest

    raise FileNotFoundError(
        f"No model config found at {base} and no checkpoint-* directory exists below it."
    )


def _resolve_tokenizer_path(model_path: Path, tokenizer_path: str | None) -> Path:
    if tokenizer_path is not None:
        return Path(tokenizer_path).expanduser()

    if _has_tokenizer_files(model_path):
        return model_path

    parent = model_path.parent
    if _checkpoint_step(model_path) >= 0 and _has_tokenizer_files(parent):
        return parent

    return model_path


def _parse_backend(backend: str) -> tuple[str, int | None]:
    parts = backend.split(":", 1)
    name = parts[0].lower()

    if name == "cpu":
        return name, None
    if name in {"cuda", "rocm", "hip"}:
        return name, int(parts[1]) if len(parts) > 1 else 0

    raise argparse.ArgumentTypeError(
        f"Unknown backend {backend!r}. Use rocm, rocm:<id>, cuda, cuda:<id>, or cpu."
    )


def _resolve_dtype(dtype_name: str, backend_name: str) -> torch.dtype:
    if dtype_name != "auto":
        return _DTYPES[dtype_name]

    if backend_name == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _configure_accelerator(backend_name: str, device_id: int, tf32: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{backend_name.upper()} backend requested, but torch.cuda.is_available() is false."
        )
    if backend_name in {"rocm", "hip"} and torch.version.hip is None:
        raise RuntimeError(
            f"{backend_name.upper()} backend requested, but this PyTorch build is not ROCm/HIP."
        )

    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

    props = torch.cuda.get_device_properties(device_id)
    total_gb = props.total_memory / (1024**3)
    backend_label = "ROCm" if torch.version.hip else "CUDA"
    print(f"{backend_label} device {device_id}: {props.name} ({total_gb:.1f} GiB)")
    print(f"TF32 enabled: {tf32}")


def _device_map(backend_name: str, device_id: int | None) -> dict:
    if backend_name == "cpu":
        return {"": "cpu"}
    return {"": device_id if device_id is not None else 0}


def _load_tokenizer(tokenizer_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_path: Path,
    dtype: torch.dtype,
    device_map: dict,
    attention: str,
    low_cpu_mem_usage: bool,
):
    kwargs = {
        _transformers_dtype_kwarg(): dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }

    if attention == "default":
        print("Attention: default")
        return AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)

    if attention != "auto":
        print(f"Attention: {attention}")
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            attn_implementation=attention,
            **kwargs,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            attn_implementation="flash_attention_2",
            **kwargs,
        )
        print("Attention: flash_attention_2")
        return model
    except (ImportError, TypeError, ValueError) as exc:
        print(f"Flash Attention 2 unavailable; trying SDPA. ({exc})")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            attn_implementation="sdpa",
            **kwargs,
        )
        print("Attention: sdpa")
        return model
    except (ImportError, TypeError, ValueError) as exc:
        print(f"SDPA unavailable; using default attention. ({exc})")

    print("Attention: default")
    return AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)


def _input_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _format_messages(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = _build_messages(system_prompt, user_prompt)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def _generate_response(tokenizer, model, prompt: str, args) -> str:
    text = _format_messages(tokenizer, args.system_prompt, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(_input_device(model))
    input_tokens = inputs["input_ids"].shape[-1]

    do_sample = args.temperature > 0
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    response = outputs[0][input_tokens:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with the continued-pretrained model on an RTX-class CUDA GPU."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Model/checkpoint directory. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("auto", "root", "latest"),
        default="auto",
        help="Load the root model dir, latest checkpoint, or root if available else latest.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer directory. Defaults to the model path or checkpoint parent.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="Inference backend: rocm:<id>, cuda:<id>, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=DEFAULT_DTYPE,
        help="Inference dtype. Auto prefers bf16 on capable NVIDIA GPUs, else fp16.",
    )
    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "default"),
        default=DEFAULT_ATTENTION,
        help="Attention backend. Auto tries Flash Attention 2, then SDPA, then default.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmul/convolution on NVIDIA GPUs.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Optionally torch.compile the loaded model. Startup is slower.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 disables sampling for deterministic output.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Tokenizer path: {args.tokenizer_path}")
    print(f"Backend: {args.backend}")

    backend_name, device_id = _parse_backend(args.backend)
    dtype = _resolve_dtype(args.dtype, backend_name)
    if backend_name in {"cuda", "rocm", "hip"}:
        _configure_accelerator(backend_name, device_id or 0, args.tf32)

    model_path = _resolve_model_path(args.model_path, args.checkpoint)
    tokenizer_path = _resolve_tokenizer_path(model_path, args.tokenizer_path)
    device_map = _device_map(backend_name, device_id)

    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Backend: {args.backend}")
    print(f"Device map: {device_map}")
    print(f"Dtype: {dtype}")

    tokenizer = _load_tokenizer(tokenizer_path)
    model = _load_model(
        model_path=model_path,
        dtype=dtype,
        device_map=device_map,
        attention=args.attention,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    model.eval()

    if args.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    while True:
        prompt = input("\nPrompt> ").strip()
        if not prompt or prompt.lower() in {"exit", "quit"}:
            break

        print("\n" + _generate_response(tokenizer, model, prompt, args))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
