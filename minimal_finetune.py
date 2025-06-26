# minimal_finetune.py - 最小構成版（メモリ効率最重視）

import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling

print("🔥 超軽量ファインチューニング開始")

# 超軽量設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
OUTPUT_DIR = "./rag_model_light"
SAMPLE_SIZE = 1000  # 1000サンプルのみ使用

# データ準備（最小限）
print("📊 データ準備...")
dataset = load_dataset(
    "json", 
    data_files={"train": "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl"}, 
    split="train"
)
dataset = dataset.select(range(SAMPLE_SIZE))  # 1000サンプルのみ

def format_prompt(example):
    text = f"### 入力:\n{example['input']}\n### 応答:\n{example['output']}<|endoftext|>"
    return {"text": text}

dataset = dataset.map(format_prompt)
dataset = dataset.train_test_split(test_size=0.1)

# トークナイザー
print("🔤 トークナイザー...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=256)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["instruction", "input", "output", "text"])

# 4bit量子化モデル
print("🤖 モデル読み込み...")
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

# 最小LoRA
lora_config = LoraConfig(
    r=2,  # 最小
    lora_alpha=4,
    target_modules=["q_proj"],  # 1つのモジュールのみ
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
print(f"✅ 学習パラメータ: {model.num_parameters(only_trainable=True):,}")

# 学習設定（最小）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # 1エポックのみ
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=1,
    remove_unused_columns=False,
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

print("💾 保存...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🎉 完了！保存先: {OUTPUT_DIR}")
print("設定ファイルのlora_adapter_pathを更新してください。")