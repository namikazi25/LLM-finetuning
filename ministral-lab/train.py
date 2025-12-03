import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator, # <--- THE FIX
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets, Dataset
from itertools import chain

# 1. Config
MODEL_ID = "mistralai/Ministral-3-8B-Base-2512"
OUTPUT_DIR = "ministral-climate-v1"
BLOCK_SIZE = 4096 

# 2. Load Tokenizer & Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

print(f"Loading {MODEL_ID}...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# 3. Apply LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. Prepare Data (Manual RAM Packing)
print("Loading Datasets...")
ds_policy = load_dataset("ClimatePolicyRadar/all-document-text-data", split="train", streaming=True)
ds_ipcc = load_dataset("Ekimetrics/climateqa-ipcc-ipbes-reports-1.0", split="train", streaming=True)

def gen_text():
    # Take a robust sample
    for ex in chain(ds_policy.take(5000), ds_ipcc.take(1000)):
        yield ex.get("text") or ex.get("content") or ""

print("Tokenizing and Packing in Memory...")
all_input_ids = []

# Tokenize everything into one flat stream
for text in gen_text():
    ids = tokenizer.encode(text, add_special_tokens=False) 
    all_input_ids.extend(ids)
    all_input_ids.append(tokenizer.eos_token_id)

# Truncate to exact multiple of BLOCK_SIZE
total_tokens = len(all_input_ids)
truncated_length = (total_tokens // BLOCK_SIZE) * BLOCK_SIZE
all_input_ids = all_input_ids[:truncated_length]

print(f"Total tokens: {total_tokens}. Packed into {truncated_length // BLOCK_SIZE} blocks.")

# Slice into chunks
chunks = [all_input_ids[i : i + BLOCK_SIZE] for i in range(0, truncated_length, BLOCK_SIZE)]

# Create final clean dataset
lm_dataset = Dataset.from_dict({
    "input_ids": chunks,
    "labels": chunks, 
    "attention_mask": [[1] * BLOCK_SIZE for _ in chunks]
})

# 5. Trainer
trainer = Trainer(
    model=model,
    train_dataset=lm_dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100, 
        learning_rate=2e-5,
        fp16=False,
        bf16=True, 
        logging_steps=1,
        save_strategy="no", 
        optim="paged_adamw_8bit",
        report_to="none",
    ),
    data_collator=default_data_collator, # <--- USES SIMPLE STACKING (NO PADDING LOGIC)
)

print("Starting Training...")
trainer.train()

# 6. Save
print("Saving Adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")