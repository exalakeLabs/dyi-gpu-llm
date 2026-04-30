#!/usr/bin/env python

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from model_runtime import load_base_model, load_tokenizer
from project_config import ADAPTER_DIR, LORA_DIR, TRAIN_FILE


class RadeonSafeSFTTrainer(SFTTrainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        return batch_samples, None


def print_device_info(model) -> None:
    print("Model first param device:", next(model.parameters()).device)
    print("HIP version:", torch.version.hip)
    print("CUDA version:", torch.version.cuda)
    print("GPU available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def prepare_lora_model(model):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, dataset):
    args = SFTConfig(
        output_dir=str(LORA_DIR),
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=50,
        max_length=192,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        average_tokens_across_devices=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    return RadeonSafeSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def main() -> int:
    LORA_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_base_model(attn_implementation="eager")
    print_device_info(model)

    dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    model = prepare_lora_model(model)
    trainer = build_trainer(model, tokenizer, dataset)
    trainer.train()

    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
