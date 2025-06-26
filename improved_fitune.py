# improved_fitune.py - 改良版ファインチューニングスクリプト
# より効率的で高品質な学習を実現

import torch
import os
import json
import wandb
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import numpy as np

# ──────────────────────────────────────────────────────
# 0. 改良版設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"  # より軽量で日本語特化
RAW_JSONL_PATH = "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl"  # 拡張データ使用
CACHE_DIR = "cache/tokenized_enhanced"
OUTPUT_DIR = "./lora_enhanced_finetune"

# LoRA設定の最適化
LORA_R = 16  # 8→16に増加（表現力向上）
LORA_ALPHA = 32  # r*2が目安
LORA_DROPOUT = 0.1  # 過学習防止

# 学習設定の最適化
LEARNING_RATE = 1e-4  # 2e-4→1e-4に下げて安定化
EPOCHS = 5  # 3→5に増加
BATCH_SIZE = 1  # GPU VRAM制約
GRAD_ACCUMULATION = 16  # 実効バッチサイズ=16

# ──────────────────────────────────────────────────────
# 1. 改良版データ前処理
def enhanced_preprocess(example):
    """改良版データ前処理（プロンプト形式最適化）"""
    instr = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    
    # ELYZA形式に最適化
    if instr:
        prompt = f"以下は、タスクを説明する指示と、文脈を説明する入力の組み合わせです。要求を適切に満たす応答を書いてください。\n\n### 指示:\n{instr}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
    else:
        prompt = f"以下は、タスクを説明する指示です。要求を適切に満たす応答を書いてください。\n\n### 指示:\n{input_text}\n\n### 応答:\n"
    
    return {"prompt": prompt, "completion": output_text}

# ──────────────────────────────────────────────────────
# 2. データ読み込み＆前処理
print("データ読み込み中...")
raw_ds = load_dataset(
    "json",
    data_files={"train": RAW_JSONL_PATH},
    split="train",
)

print(f"元データ件数: {len(raw_ds)}")

# 前処理適用
ds = raw_ds.map(
    enhanced_preprocess,
    remove_columns=raw_ds.column_names,
    num_proc=4,
)

# 学習/評価分割（評価用を少し多めに）
ds = ds.train_test_split(test_size=0.15, seed=42)
print(f"学習データ: {len(ds['train'])}件, 評価データ: {len(ds['test'])}件")

# ──────────────────────────────────────────────────────
# 3. トークナイザー設定
print("トークナイザー読み込み中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# パディングトークン設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("pad_tokenをeos_tokenに設定")

# 改良版トークン長設定
MAX_PROMPT_LEN = 600  # プロンプト部分を長めに
MAX_TOTAL_LEN = 1200  # 全体も長めに（品質向上）

def improved_tokenize_fn(batch):
    """改良版トークナイズ関数"""
    # プロンプト部分のトークン化
    prompts = batch["prompt"]
    completions = batch["completion"]
    
    prompt_encodings = tokenizer(
        prompts,
        max_length=MAX_PROMPT_LEN,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    
    # 完全なテキスト（プロンプト+完了+EOS）
    full_texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
    
    full_encodings = tokenizer(
        full_texts,
        max_length=MAX_TOTAL_LEN,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    
    # ラベル作成（プロンプト部分をマスク）
    labels = []
    for i, full_ids in enumerate(full_encodings["input_ids"]):
        prompt_len = len(prompt_encodings["input_ids"][i])
        label_ids = full_ids.copy()
        # プロンプト部分を-100でマスク（損失計算から除外）
        label_ids[:prompt_len] = [-100] * prompt_len
        labels.append(label_ids)
    
    return {
        "input_ids": full_encodings["input_ids"],
        "attention_mask": full_encodings["attention_mask"],
        "labels": labels,
    }

# トークナイズ実行（キャッシュ利用）
print("トークナイズ実行中...")
try:
    tokenized = load_from_disk(CACHE_DIR)
    print("キャッシュからトークナイズ済みデータを読み込み")
except:
    tokenized = ds.map(
        improved_tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=["prompt", "completion"],
        num_proc=4,
    )
    tokenized.save_to_disk(CACHE_DIR)
    print("トークナイズ完了＆キャッシュ保存")

# ──────────────────────────────────────────────────────
# 4. モデル読み込み（QLoRA設定）
print("モデル読み込み中...")

# 改良版量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # float16→bfloat16で安定性向上
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# モデル読み込み
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 追加
)

# 勾配チェックポイント有効化（メモリ節約）
model.gradient_checkpointing_enable()

# 改良版LoRA設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    # より多くのレイヤーに適用
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
print(f"LoRA適用完了 (trainable parameters: {model.num_parameters(only_trainable=True):,})")

# ──────────────────────────────────────────────────────
# 5. データコレクター
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding="longest",
    label_pad_token_id=-100,
    return_tensors="pt",
)

# ──────────────────────────────────────────────────────
# 6. 改良版学習設定
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",  # 学習率スケジューラー追加
    warmup_steps=100,  # ウォームアップ追加
    bf16=True,  # fp16→bf16で安定性向上
    optim="paged_adamw_8bit",
    dataloader_num_workers=4,
    remove_unused_columns=False,
    
    # 評価・保存設定
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps", 
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # ログ設定
    logging_steps=10,
    logging_strategy="steps",
    report_to=["tensorboard"],
    
    # その他
    dataloader_pin_memory=True,
    group_by_length=True,  # 効率化
)

# ──────────────────────────────────────────────────────
# 7. 学習実行
print("学習開始...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 早期停止
)

# 学習実行
trainer.train()

# ──────────────────────────────────────────────────────
# 8. モデル保存
print("モデル保存中...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"学習完了！保存先: {OUTPUT_DIR}")
print("使用方法:")
print(f"config.py の lora_adapter_path を '{OUTPUT_DIR}' に設定してください")