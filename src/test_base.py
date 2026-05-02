from model_runtime import generate_text, load_generation_model


def main() -> int:
    tokenizer, model = load_generation_model(use_adapter=False)
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarize the employee handbook communication guidelines."},
    ]

    print("\n--- BASE MODEL OUTPUT ---\n")
    print(generate_text(tokenizer, model, messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
