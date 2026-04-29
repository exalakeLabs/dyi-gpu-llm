import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "output/lora/final"

# ROCm workaround: some tiny-tensor torch.isin GPU calls fail on gfx1102
_orig_isin = torch.isin
def _safe_isin(elements, test_elements, *args, **kwargs):
    elems_dev = getattr(elements, "device", None)
    test_dev = getattr(test_elements, "device", None)

    if (elems_dev is not None and elems_dev.type == "cuda") or (
        test_dev is not None and test_dev.type == "cuda"
    ):
        out = _orig_isin(elements.cpu(), test_elements.cpu(), *args, **kwargs)
        if elems_dev is not None and elems_dev.type == "cuda":
            return out.to(elems_dev)
        if test_dev is not None and test_dev.type == "cuda":
            return out.to(test_dev)
    return _orig_isin(elements, test_elements, *args, **kwargs)

torch.isin = _safe_isin

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

# Explicit IDs to reduce generate() surprises
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Summarize the employee handbook communication guidelines."},
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
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

print("\n--- MODEL OUTPUT ---\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))