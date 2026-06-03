#!/usr/bin/env python3

import argparse
from model_runtime import generate_text, load_generation_model
from rag_model_config import validate_generator_model
from runtime_env import env_int, env_str

BASE_MODEL = env_str("BASE_MODEL")
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)
TEST_BASE_MODEL = env_str("TEST_BASE_MODEL", BASE_MODEL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        "--generator-model",
        dest="generator_model",
        default=TEST_BASE_MODEL,
        help="Transformers model name or path",
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

    validate_generator_model(args.generator_model)
    tokenizer, model = load_generation_model(
        base_model=args.generator_model,
        use_adapter=False,
    )

    response = generate_text(
        tokenizer,
        model,
        [{"role": "user", "content": args.prompt}],
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    print(response)


if __name__ == "__main__":
    raise SystemExit(main())
