# auto_optimized_finetune.py - 自動最適化版
import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

# 自動最適化設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
JSONL_PATH = "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl"
OUTPUT_DIR = "./optimized_rag_model"
CACHE_DIR = "cache/optimized"

# GPU最適化パラメータ
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
MAX_LENGTH = 800
LORA_R = 8
LEARNING_RATE = 2e-05

print("🚀 最適化ファインチューニング開始")
print(f"   バッチサイズ: {BATCH_SIZE}")
print(f"   勾配蓄積: {GRAD_ACCUMULATION}")
print(f"   最大長: {MAX_LENGTH}")
print(f"   LoRA r: {LORA_R}")
print(f"   学習率: {LEARNING_RATE}")

# データ準備
def preprocess_function(examples):
    instruction = examples.get("instruction", "").strip()
    input_text = examples.get("input", "").strip()
    output_text = examples.get("output", "").strip()
    
    if instruction:
        prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
    else:
        prompt = f"### 入力:\n{input_text}\n\n### 応答:\n"
    
    return {"prompt": prompt, "completion": output_text}

# データ読み込み
raw_dataset = load_dataset("json", data_files={"train": JSONL_PATH}, split="train")
dataset = raw_dataset.map(preprocess_function, remove_columns=raw_dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# トークナイザー
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]
    
    full_texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
    
    model_inputs = tokenizer(
        full_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )
    
    # プロンプト長を計算してラベルマスク
    prompt_inputs = tokenizer(prompts, truncation=True, padding=False)
    labels = []
    
    for i, input_ids in enumerate(model_inputs["input_ids"]):
        prompt_len = len(prompt_inputs["input_ids"][i])
        label = input_ids.copy()
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)
    
    model_inputs["labels"] = labels
    return model_inputs

# トークナイズ（キャッシュ利用）
try:
    tokenized_dataset = load_from_disk(CACHE_DIR)
    print("✅ キャッシュからデータ読み込み")
except:
    print("⏳ トークナイズ実行中...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "completion"],
        num_proc=4
    )
    tokenized_dataset.save_to_disk(CACHE_DIR)
    print("✅ トークナイズ完了")

# モデル設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.gradient_checkpointing_enable()

# LoRA設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_R * 2,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
print(f"✅ LoRA適用完了（学習可能パラメータ: {model.num_parameters(only_trainable=True):,}）")

# データコレクター
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding="longest",
    label_pad_token_id=-100,
)

# 学習設定
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    optim="paged_adamw_8bit",
    dataloader_num_workers=2,
    
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    logging_steps=10,
    report_to=["tensorboard"],
    
    remove_unused_columns=False,
    group_by_length=True,
)

# トレーナー作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🎯 学習開始...")
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
