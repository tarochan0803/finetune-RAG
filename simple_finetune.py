# simple_finetune.py - シンプル版ファインチューニング
# 互換性重視で確実に動作

import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")

print("🚀 シンプル版ファインチューニング開始")

# CPU強制
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
OUTPUT_DIR = "./rag_simple_model"
JSONL_PATH = "/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl"

print("📊 データ準備...")

# 軽量データ準備
dataset = load_dataset("json", data_files={"train": JSONL_PATH}, split="train")
sample_dataset = dataset.select(range(200))  # 200サンプルのみ

def simple_format(example):
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    text = f"質問: {input_text}\n回答: {output_text}<|endoftext|>"
    return {"text": text}

formatted_dataset = sample_dataset.map(simple_format)
train_test = formatted_dataset.train_test_split(test_size=0.1)

print(f"   学習データ: {len(train_test['train'])}件")
print(f"   評価データ: {len(train_test['test'])}件")

# トークナイザー
print("🔤 トークナイザー設定...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=128,  # 短縮
    )

tokenized = train_test.map(tokenize, batched=True, remove_columns=["instruction", "input", "output", "text"])

# モデル読み込み（CPU用）
print("🤖 モデル読み込み...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

# 最小LoRA
print("⚙️ LoRA設定...")
lora_config = LoraConfig(
    r=1,  # 最小
    lora_alpha=2,
    target_modules=["q_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
print(f"学習パラメータ: {model.num_parameters(only_trainable=True):,}")

# シンプル学習設定
print("🎯 学習設定...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-3,
    logging_steps=10,
    save_steps=100,
    remove_unused_columns=False,
    no_cuda=True,
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
    data_collator=data_collator,
)

print("🚀 学習開始...")
try:
    trainer.train()
    
    print("💾 保存中...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("🎉 ファインチューニング完了！")
    print(f"保存先: {OUTPUT_DIR}")
    
    # テスト推論
    print("\n🧪 テスト推論...")
    test_input = "質問: 株式会社三建の壁仕様について\n回答: "
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"テスト結果: {result}")
    
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()