import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "output/lora/final"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    autocast_adapter_dtype=False,
)

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Summarize the employee handbook communication guidelines."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n--- MODEL OUTPUT ---\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))