from __future__ import annotations

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def is_gpt_oss_model(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return "gpt-oss" in normalized


def validate_embedding_model(model_name: str) -> None:
    if is_gpt_oss_model(model_name):
        raise SystemExit(
            "\n".join(
                [
                    f"Invalid embedding model: {model_name}",
                    "gpt-oss is a generator model, not a sentence embedding model.",
                    f"Set EMBED_MODEL={DEFAULT_EMBED_MODEL} and rebuild the RAG index.",
                ]
            )
        )


def validate_reranker_model(model_name: str) -> None:
    if is_gpt_oss_model(model_name):
        raise SystemExit(
            "\n".join(
                [
                    f"Invalid reranker model: {model_name}",
                    "gpt-oss is a generator model, not a cross-encoder reranker.",
                    f"Set RERANKER_MODEL={DEFAULT_RERANKER_MODEL}.",
                ]
            )
        )


def validate_generator_model(model_name: str) -> None:
    normalized = model_name.strip().lower()
    if normalized.startswith("gpt-oss:"):
        raise SystemExit(
            "\n".join(
                [
                    f"Invalid generator model: {model_name}",
                    "Use a Hugging Face Transformers model id for local generation.",
                    f"Set GENERATOR_MODEL={DEFAULT_GENERATOR_MODEL}.",
                ]
            )
        )
