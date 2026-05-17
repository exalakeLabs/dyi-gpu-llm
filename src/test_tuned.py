#!/usr/bin/env python3

import argparse
from model_runtime import load_generation_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        default="/home/alex2/llrun/output/lora/final",
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
        default=256,
    )

    args = parser.parse_args()

    tokenizer, model = load_generation_model(
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
