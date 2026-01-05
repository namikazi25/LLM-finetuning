"""
Ministral Climate LLM - Final Optimized Script
Hardware: NVIDIA RTX A6000 (48GB)
Features: Unsloth, Smart Caching, Batch Tokenization, Robust Data Loading
"""

import os
# --- MEMORY FIX: Reduce fragmentation ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from tqdm import tqdm
import shutil

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# 🚨 MASTER SWITCH: Set True to test (1 min), False for real training (7+ hours)
DEBUG_MODE = False

MODEL_ID = "unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit"
OUTPUT_DIR = "ministral-climate-v2"
CACHE_DIR = "./cached_climate_80m"
BLOCK_SIZE = 4096 

# --- MEMORY OPTIMIZATION ---
# Reduced Batch Size to prevent OOM on Logits (16GB spike)
# Increased Grad Accum so effective batch size is still 16
BATCH_SIZE = 2          
GRAD_ACCUM = 8          
LEARNING_RATE = 1e-5      # was 2e-5
NUM_EPOCHS = 2
MAX_STEPS = 4000          # Set to -1 to use full epochs

# =============================================================================
# 2. LOAD MODEL (UNSLOTH OPTIMIZED)
# =============================================================================
print(f"Loading {MODEL_ID}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=BLOCK_SIZE,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    trust_remote_code=True,
)

# CRITICAL FIX: Unwrap tokenizer if hidden inside PixtralProcessor
if hasattr(tokenizer, "tokenizer"):
    print("✨ Unwrapping tokenizer from PixtralProcessor...")
    tokenizer = tokenizer.tokenizer

# Ensure padding token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# 3. APPLY LORA ADAPTERS
# =============================================================================
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth", 
    random_state=42,
    use_dora =True,
)

# =============================================================================
# 4. PREPARE DATA (SMART CACHING + BATCHING)
# =============================================================================
print("\n" + "="*40)
print("DATA PREPARATION")
print("="*40)

if os.path.exists(CACHE_DIR) and not DEBUG_MODE:
    print(f"✅ Found cached dataset at '{CACHE_DIR}'. Loading instantly...")
    train_dataset = Dataset.load_from_disk(CACHE_DIR)
    print(f"   Loaded {len(train_dataset):,} sequences.")

else:
    reason = "Debug Mode is On" if DEBUG_MODE else "Cache not found"
    print(f"⚠️ {reason}. Starting processing pipeline...")
    
    print("   Loading raw datasets (Streaming)...")
    ds_policy = load_dataset("ClimatePolicyRadar/all-document-text-data", split="train", streaming=True)
    ds_ipcc = load_dataset("Ekimetrics/climateqa-ipcc-ipbes-reports-1.0", split="train", streaming=True)

    def generate_texts():
        """Robust generator that handles inconsistent column names."""
        print("   -> Stream started. Reading ClimatePolicyRadar...")
        for ex in ds_policy:
            text = ex.get("text_block.text") 
            if not text: text = ex.get("text")
            if not text and isinstance(ex.get("text_block"), dict): 
                text = ex["text_block"].get("text")
            if text and len(text) > 50: yield text

        print("   -> Reading IPCC/IPBES Reports...")
        for ex in ds_ipcc:
            text = ex.get("content", "")
            if text and len(text) > 50: yield text

    print(f"   Tokenizing... (Debug Limit: {1000 if DEBUG_MODE else 'None'})")
    
    all_input_ids = []
    BATCH_SIZE_TOKENIZE = 1000
    buffer = []
    doc_count = 0
    LIMIT = 1000 if DEBUG_MODE else None

    for text in tqdm(generate_texts(), desc="   Docs", unit=" docs"):
        if LIMIT and doc_count >= LIMIT:
            print(f"   🛑 Debug limit reached ({LIMIT} docs). Stopping.")
            break
        buffer.append(text)
        doc_count += 1
        
        if len(buffer) >= BATCH_SIZE_TOKENIZE:
            batch_ids = tokenizer(buffer, add_special_tokens=False)["input_ids"]
            for ids in batch_ids:
                all_input_ids.extend(ids)
                all_input_ids.append(tokenizer.eos_token_id)
            buffer = []

    if buffer:
        batch_ids = tokenizer(buffer, add_special_tokens=False)["input_ids"]
        for ids in batch_ids:
            all_input_ids.extend(ids)
            all_input_ids.append(tokenizer.eos_token_id)

    truncated_length = (len(all_input_ids) // BLOCK_SIZE) * BLOCK_SIZE
    if truncated_length == 0 and len(all_input_ids) > 0: 
        truncated_length = len(all_input_ids)

    print(f"   Packing into blocks of {BLOCK_SIZE}...")
    chunks = [all_input_ids[i : i + BLOCK_SIZE] for i in range(0, truncated_length, BLOCK_SIZE)]

    train_dataset = Dataset.from_dict({
        "input_ids": chunks,
        "labels": chunks, 
        "attention_mask": [[1] * len(c) for c in chunks]
    })
    
    if not DEBUG_MODE:
        print(f"💾 Caching dataset to '{CACHE_DIR}'...")
        train_dataset.save_to_disk(CACHE_DIR)

print(f"Dataset ready: {len(train_dataset)} sequences.")

# =============================================================================
# 5. TRAINER SETUP (PATCHED FOR COMPATIBILITY)
# =============================================================================
print("\n" + "="*40)
print(f"STARTING TRAINING (Debug: {DEBUG_MODE})")
print("="*40)

# --- PATCH 1: Handle Missing 'push_to_hub_token' in Transformers 5.0 ---
class UnslothCompatibleArgs(TrainingArguments):
    def to_dict(self):
        # Force the dictionary to include the key Unsloth expects to pop
        d = super().to_dict()
        d["push_to_hub_token"] = None
        return d

training_args = UnslothCompatibleArgs(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    max_steps=10 if DEBUG_MODE else MAX_STEPS, 
    learning_rate=LEARNING_RATE,
    fp16=False,
    bf16=True, 
    logging_steps=1 if DEBUG_MODE else 10,
    save_strategy="steps" if DEBUG_MODE else "epoch",
    save_steps=500,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    seed=3407,
    report_to="none",
    dataloader_num_workers=4,
)

if not hasattr(training_args, "push_to_hub_token"):
    training_args.push_to_hub_token = None

# --- PATCH 2: Handle 'tokenizer' rename in TRL 0.12+ ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field=None,  # We are passing pre-tokenized IDs, so None is safer
    max_seq_length=BLOCK_SIZE,
    processing_class=tokenizer, # <--- NEW NAME (Fixes the TypeError)
    args=training_args,
)

# =============================================================================
# 6. TRAIN & SAVE
# =============================================================================
trainer.train()

print("\n" + "="*40)
print("SAVING MODEL")
print("="*40)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

if not DEBUG_MODE:
    print("Saving merged model for inference...")
    try:
        model.save_pretrained_merged(f"{OUTPUT_DIR}_merged", tokenizer, save_method="merged_16bit")
        print(f"✅ Merged model saved to {OUTPUT_DIR}_merged")
    except Exception as e:
        print(f"⚠️ Could not merge (likely OOM). Adapters are safe in {OUTPUT_DIR}")

print("Done!")