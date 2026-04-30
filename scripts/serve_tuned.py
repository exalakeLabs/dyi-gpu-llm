import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]


def env_dir(var: str, default_rel: str) -> Path:
    v = os.environ.get(var, "").strip()
    p = Path(v).expanduser() if v else (REPO_ROOT / default_rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


OUTPUT_DIR = env_dir("LLAMA_OUTPUT_DIR", "output")
ADAPTER_PATH = str(OUTPUT_DIR / "lora" / "final")

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

class ChatRequest(BaseModel):
    prompt: str
    system: str = "You are a concise technical assistant."
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/chat")
def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": req.system},
        {"role": "user", "content": req.prompt},
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
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # crude but useful: return only after the user's prompt when possible
    answer = decoded.split(req.prompt)[-1].strip()

    return {"response": answer}
