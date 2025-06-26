# quick_finetune.py - RTX 5070最適化版
# 12GB VRAM向け軽量ファインチューニング

import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

# RTX 5070最適化設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
JSONL_PATH = "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl"
OUTPUT_DIR = "./rag_finetuned_model"

print("🚀 RTX 5070向け軽量ファインチューニング開始")

# データ準備
def preprocess_function(examples):
    instruction = examples.get("instruction", "").strip()
    input_text = examples.get("input", "").strip()
    output_text = examples.get("output", "").strip()
    
    if instruction:
        prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
    else:
        prompt = f"### 入力:\n{input_text}\n\n### 応答:\n"
    
    return {"text": prompt + output_text + "<|endoftext|>"}

print("📊 データ読み込み中...")
raw_dataset = load_dataset("json", data_files={"train": JSONL_PATH}, split="train")
# サンプル数を制限（高速化のため）
sample_size = min(5000, len(raw_dataset))  # 5000サンプルに制限
raw_dataset = raw_dataset.select(range(sample_size))

dataset = raw_dataset.map(preprocess_function, remove_columns=raw_dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"学習データ: {len(dataset['train'])}件, 評価データ: {len(dataset['test'])}件")

# トークナイザー
print("🔤 トークナイザー準備中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,  # 短めに設定（メモリ節約）
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# モデル設定（軽量化）
print("🤖 モデル読み込み中...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # float16で軽量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 軽量LoRA設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,  # 軽量化のため小さく
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 最小限のモジュール
)

model = get_peft_model(model, lora_config)
print(f"✅ LoRA適用完了（学習可能パラメータ: {model.num_parameters(only_trainable=True):,}）")

# データコレクター
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

# 軽量学習設定
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,  # エポック数削減
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # 実効バッチサイズ=4
    learning_rate=5e-5,
    fp16=True,  # fp16で軽量化
    optim="adamw_torch",  # 標準optimizer
    dataloader_num_workers=0,  # ワーカー数削減
    
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,  # 保存数制限
    load_best_model_at_end=True,
    
    logging_steps=50,
    report_to=[],  # ログ無効化
    
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # メモリ節約
)

# トレーナー作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("🎯 学習開始...")
try:
    trainer.train()
    
    print("💾 モデル保存中...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("🎉 学習完了！")
    print(f"📁 保存先: {OUTPUT_DIR}")
    print("\n次の手順:")
    print("1. config.py の lora_adapter_path を以下に変更:")
    print(f"   self.lora_adapter_path = '{OUTPUT_DIR}'")
    print("2. RAGアプリを起動して性能確認")
    
except Exception as e:
    print(f"❌ エラー発生: {e}")
    print("メモリ不足の可能性があります。バッチサイズやシーケンス長を調整してください。")