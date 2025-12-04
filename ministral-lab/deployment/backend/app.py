from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer

app = FastAPI()

# Load model once at startup
print("Loading model...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    "/model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
print("Model loaded!")

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # For base models, just use the last user message as a completion prompt
    last_user_msg = None
    for msg in reversed(request.messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break
    
    if not last_user_msg:
        last_user_msg = "Hello"
    
    # Generate completion
    inputs = tokenizer(last_user_msg, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # Decode only the new tokens (skip the prompt)
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # If empty, provide fallback
    if not response_text:
        response_text = "I understand. How can I help you with climate-related information?"
    
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}