#!/usr/bin/env python

import argparse

import torch

from model_runtime import _DTYPE_KWARG, generate_text, load_generation_model

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _backend_kwargs(backend: str, dtype: str | None) -> dict:
    """
    Translate --backend / --dtype into from_pretrained keyword overrides.

    Backend format: cuda[:<id>] | rocm[:<id>] | hip[:<id>] | cpu
      cuda / rocm / hip   → GPU device 0  (ROCm exposes itself as "cuda" in PyTorch)
      cuda:1              → GPU device 1
      cpu                 → run on CPU, default dtype float32
    """
    kwargs: dict = {}

    parts = backend.split(":", 1)
    name = parts[0].lower()

    if name == "cpu":
        kwargs["device_map"] = {"": "cpu"}
        if dtype is None:
            # fp16/bf16 on CPU is slow; default to fp32 unless the user asked
            kwargs[_DTYPE_KWARG] = torch.float32
    elif name in ("cuda", "rocm", "hip"):
        device_id = int(parts[1]) if len(parts) > 1 else 0
        kwargs["device_map"] = {"": device_id}
    else:
        raise argparse.ArgumentTypeError(
            f"Unknown backend {backend!r}. Use cuda, rocm, hip, or cpu (optionally with :N for device index)."
        )

    if dtype is not None:
        kwargs[_DTYPE_KWARG] = _DTYPES[dtype]

    return kwargs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned model")
    parser.add_argument(
        "--backend",
        default="cuda",
        metavar="BACKEND",
        help=(
            "Compute backend: cuda, rocm/hip (AMD), or cpu. "
            "Append :<id> to select a specific GPU (e.g. cuda:1). "
            "Default: cuda"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=list(_DTYPES),
        default=None,
        help=(
            "Model dtype: fp16, bf16, or fp32. "
            "Default: bf16 on capable AMD GPUs, fp16 on CUDA, fp32 on cpu."
        ),
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Load the base model without the PEFT adapter.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    model_kwargs = _backend_kwargs(args.backend, args.dtype)

    tokenizer, model = load_generation_model(
        use_adapter=not args.no_adapter,
        **model_kwargs,
    )

    device_label = next(iter(model_kwargs.get("device_map", {}).values()), "auto")
    dtype_label = next(
        (k for k, v in _DTYPES.items() if v == model_kwargs.get("torch_dtype")), "auto"
    )
    print(f"[backend={args.backend}  device={device_label}  dtype={dtype_label}]")

    system_prompt = "You are a concise assistant."

    while True:
        prompt = input("\nPrompt> ").strip()
        if not prompt or prompt.lower() in {"exit", "quit"}:
            break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        print("\n" + generate_text(tokenizer, model, messages))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
