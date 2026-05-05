def tokenize_dataset(dataset, tokenizer):
    def format_messages(messages):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    if "text" in dataset.column_names:
        def tokenize_fn(batch):
            texts = batch["text"]
            return tokenizer(
                texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )

    elif "messages" in dataset.column_names:
        def tokenize_fn(batch):
            texts = [format_messages(msgs) for msgs in batch["messages"]]
            return tokenizer(
                texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )

    else:
        raise ValueError(
            f"Expected either 'text' or 'messages' column in {TRAIN_FILE}, "
            f"but found: {dataset.column_names}"
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing train dataset",
    )
    return tokenized
