import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "data/train.jsonl"
OUTPUT_DIR = "output/lora"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
    attn_implementation="eager",  # safer on your ROCm setup
)

print("Model first param device:", next(model.parameters()).device)
print("HIP version:", torch.version.hip)
print("CUDA version:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model.gradient_checkpointing_enable()
model.config.use_cache = False

dataset = load_dataset(
    "json",
    data_files=DATA_FILE,
    split="train",
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Keep adapter creation stable on ROCm
model = get_peft_model(
    model,
    peft_config,
    autocast_adapter_dtype=False,
)

# Critical fix: trainable params must not remain fp16 when using AMP/fp16=True
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

model.print_trainable_parameters()

class RadeonSafeSFTTrainer(SFTTrainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        return batch_samples, None

args = SFTConfig(
    output_dir=OUTPUT_DIR,
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

trainer = RadeonSafeSFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

final_dir = os.path.join(OUTPUT_DIR, "final")
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"Saved LoRA adapter to {final_dir}")