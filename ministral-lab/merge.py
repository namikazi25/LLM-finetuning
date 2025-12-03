import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

# Config
BASE_MODEL = "mistralai/Ministral-3-8B-Base-2512"
ADAPTER_DIR = "ministral-climate-v1"
OUTPUT_DIR = "ministral-climate-merged"

print(f"Loading base model: {BASE_MODEL}")
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapter: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("Merging weights (this may take a minute)...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Success! Merged model is ready.")
