from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import json

app = FastAPI()

print("Loading model...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    "/model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("Model loaded!")

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

def generate_stream(prompt: str, temperature: float, max_tokens: int):
    """Generator for streaming tokens"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        repetition_penalty=1.15,
        streamer=streamer
    )
    
    # Run generation in background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens as they're generated
    for text in streamer:
        if text:
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "model": "ministral",
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send final chunk
    final_chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "model": "ministral",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Extract last user message
    last_msg = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "Hello")
    
    # Add primer for base models
    prompt = f"{last_msg}\n\nAnswer:"
    
    if request.stream:
        return StreamingResponse(
            generate_stream(prompt, request.temperature, request.max_tokens),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming path
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.15
            )
        
        generated = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        if not response:
            response = "I'm here to help with climate-related questions."
        
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }]
        }

@app.get("/health")
async def health():
    return {"status": "healthy"}