#!/usr/bin/env python

from model_runtime import generate_text, load_generation_model


def main() -> int:
    tokenizer, model = load_generation_model()
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
