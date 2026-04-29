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

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

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
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n--- MODEL OUTPUT ---\n")
generated = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True).strip())
