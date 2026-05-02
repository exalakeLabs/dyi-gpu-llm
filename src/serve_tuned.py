import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model_runtime import chat_inputs, load_generation_model

app = FastAPI()

tokenizer, model = load_generation_model()


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

    inputs = chat_inputs(tokenizer, model, messages)

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
    answer = decoded.split(req.prompt)[-1].strip()
    return {"response": answer}
