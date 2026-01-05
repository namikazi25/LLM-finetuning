"""
Ministral Climate LLM - Full Fine-Tuning Script
Dataset: ~80M tokens (ClimatePolicyRadar + IPCC/IPBES)
Expected time: ~8 hours on A6000 (1 epoch)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from itertools import chain
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit"  # Or your base model
OUTPUT_DIR = "ministral-climate-v2"
BLOCK_SIZE = 4096

# Training params - tuned for ~80M tokens on A6000
NUM_EPOCHS = 1          # ~8 hours for 1 epoch
BATCH_SIZE = 4          # Per device
GRAD_ACCUM = 4          # Effective batch = 16
LEARNING_RATE = 2e-5    # Good for CPT

# Set HF token if needed
# os.environ["HF_TOKEN"] = "your_token_here"

# =============================================================================
# MODEL SETUP
# =============================================================================

print("=" * 60)
print("MINISTRAL CLIMATE LLM - FULL TRAINING")
print("=" * 60)

# QLoRA config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"\n[1/6] Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # Speed boost
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# LoRA CONFIGURATION
# =============================================================================

print("\n[2/6] Applying LoRA adapters")

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =============================================================================
# DATA LOADING - Full Dataset (~80M tokens)
# =============================================================================

print("\n[3/6] Loading datasets (streaming mode)")

# Load both datasets as streaming to manage memory
ds_policy = load_dataset(
    "ClimatePolicyRadar/all-document-text-data",
    split="train",
    streaming=True
)

ds_ipcc = load_dataset(
    "Ekimetrics/climateqa-ipcc-ipbes-reports-1.0",
    split="train",
    streaming=True
)

# =============================================================================
# TOKENIZATION WITH PACKING
# =============================================================================

print("\n[4/6] Tokenizing and packing sequences")
print("      This may take 10-15 minutes for 80M tokens...")

def generate_texts():
    """Generator that yields text from both datasets."""
    # ClimatePolicyRadar - uses 'text' field
    for example in ds_policy:
        text = example.get("text_block", {})
        if isinstance(text, dict):
            text = text.get("text", "")
        elif not isinstance(text, str):
            text = str(text) if text else ""
        if text.strip():
            yield text
    
    # IPCC/IPBES - uses 'content' field
    for example in ds_ipcc:
        text = example.get("content", "")
        if text and text.strip():
            yield text

# Tokenize and pack into BLOCK_SIZE chunks
all_input_ids = []
token_count = 0

print("      Tokenizing...")
for i, text in enumerate(generate_texts()):
    ids = tokenizer.encode(text, add_special_tokens=False)
    all_input_ids.extend(ids)
    all_input_ids.append(tokenizer.eos_token_id)
    token_count += len(ids) + 1
    
    # Progress update every 100k texts
    if (i + 1) % 100000 == 0:
        print(f"      Processed {i + 1} texts, {token_count:,} tokens so far...")

print(f"\n      Total tokens: {token_count:,}")

# Truncate to exact multiple of BLOCK_SIZE
total_length = len(all_input_ids)
truncated_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
all_input_ids = all_input_ids[:truncated_length]

num_sequences = truncated_length // BLOCK_SIZE
print(f"      Packed into {num_sequences:,} sequences of {BLOCK_SIZE} tokens")

# Create chunks
print("      Creating training dataset...")
chunks = [all_input_ids[i:i + BLOCK_SIZE] for i in range(0, truncated_length, BLOCK_SIZE)]

# Build HF Dataset
train_dataset = Dataset.from_dict({
    "input_ids": chunks,
    "labels": chunks,
    "attention_mask": [[1] * BLOCK_SIZE for _ in chunks]
})

print(f"      Dataset ready: {len(train_dataset)} samples")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

print("\n[5/6] Setting up trainer")

# Estimate training time
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
estimated_hours = (steps_per_epoch * NUM_EPOCHS) / 300  # ~300 steps/hour on A6000

print(f"""
      Training Configuration:
      ─────────────────────────
      Sequences:        {len(train_dataset):,}
      Batch size:       {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}
      Steps/epoch:      {steps_per_epoch:,}
      Total epochs:     {NUM_EPOCHS}
      Learning rate:    {LEARNING_RATE}
      Estimated time:   ~{estimated_hours:.1f} hours
""")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Batch settings
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # Training duration
    num_train_epochs=NUM_EPOCHS,
    
    # Optimizer settings
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    
    # Precision
    bf16=True,
    fp16=False,
    
    # Memory optimization
    gradient_checkpointing=True,
    
    # Logging & saving
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    
    # Performance
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Misc
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)

# =============================================================================
# TRAINING
# =============================================================================

print("\n[6/6] Starting training...")
print("=" * 60)

trainer.train()

# =============================================================================
# SAVE MODEL
# =============================================================================

print("\n" + "=" * 60)
print("Training complete! Saving model...")

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"""
      ✓ Adapter saved to: {OUTPUT_DIR}/
      
      Next steps:
      1. Merge adapter: python merge.py
      2. Test model: python test_inference.py
      3. Deploy: docker-compose up
""")