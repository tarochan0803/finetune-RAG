
import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling

print("🔥 RTX 5070向け軽量ファインチューニング")

# 設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
OUTPUT_DIR = "./rag_optimized_model"
JSONL_PATH = "/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl"

# データ準備（サンプル数制限）
dataset = load_dataset("json", data_files={"train": JSONL_PATH}, split="train")
dataset = dataset.select(range(2000))  # 2000サンプルで高速学習

def format_prompt(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip() 
    output_text = example.get("output", "").strip()
    
    if instruction:
        text = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n{output_text}<|endoftext|>"
    else:
        text = f"### 入力:\n{input_text}\n\n### 応答:\n{output_text}<|endoftext|>"
    return {"text": text}

dataset = dataset.map(format_prompt)
dataset = dataset.train_test_split(test_size=0.1)

# トークナイザー
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["instruction", "input", "output", "text"])

# 4bit量子化モデル
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 軽量LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
print(f"学習パラメータ: {model.num_parameters(only_trainable=True):,}")

# 軽量学習設定
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=1,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# データコレクター
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# トレーナー
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

print("🚀 学習開始...")
trainer.train()

print("💾 保存中...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🎉 ファインチューニング完了！")
print(f"保存先: {OUTPUT_DIR}")
