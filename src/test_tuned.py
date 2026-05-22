#!/usr/bin/env python3

import argparse
from model_runtime import load_generation_model
from runtime_env import env_int, env_path, env_str

ADAPTER_DIR = env_path("ADAPTER_DIR", "output/lora/final")
BASE_MODEL = env_str("BASE_MODEL")
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        default=str(ADAPTER_DIR),
        help="LoRA adapter path",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Load base model only",
    )
    parser.add_argument(
        "--prompt",
        default="Explain RAG indexing in plain English.",
        help="Prompt to test",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    args = parser.parse_args()

    tokenizer, model = load_generation_model(
        base_model=args.base_model,
        adapter_path=args.adapter,
        use_adapter=not args.no_adapter,
    )

    messages = [
        {"role": "user", "content": args.prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print(response)


if __name__ == "__main__":
    raise SystemExit(main())
