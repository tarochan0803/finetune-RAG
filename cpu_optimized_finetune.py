# cpu_optimized_finetune.py - CPU版高速ファインチューニング
# RTX 5070のCUDA互換性問題を回避

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

print("🔥 CPU最適化ファインチューニング（RTX 5070対応）")

# CPU強制設定
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False

# 軽量設定
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
OUTPUT_DIR = "./rag_cpu_optimized_model"
JSONL_PATH = "/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl"
SAMPLE_SIZE = 500  # 超軽量（500サンプル）

print(f"📊 データ準備中...")

# データ準備
dataset = load_dataset("json", data_files={"train": JSONL_PATH}, split="train")
print(f"   元データ: {len(dataset):,}件")

# 高品質サンプル選択（エラーの少ないもの）
def quality_filter(example):
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    # 日本語を含み、適切な長さのデータを選択
    return (
        '株式会社' in input_text and
        10 <= len(input_text) <= 200 and
        5 <= len(output_text) <= 100 and
        not '[Error' in output_text
    )

filtered_dataset = dataset.filter(quality_filter)
print(f"   品質フィルタ後: {len(filtered_dataset):,}件")

# 最高品質サンプルを選択
selected_dataset = filtered_dataset.select(range(min(SAMPLE_SIZE, len(filtered_dataset))))
print(f"   選択データ: {len(selected_dataset):,}件")

def format_prompt(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    
    if instruction:
        text = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n{output_text}<|endoftext|>"
    else:
        text = f"### 入力:\n{input_text}\n\n### 応答:\n{output_text}<|endoftext|>"
    return {"text": text}

dataset_formatted = selected_dataset.map(format_prompt)
dataset_split = dataset_formatted.train_test_split(test_size=0.1)
print(f"   学習: {len(dataset_split['train'])}件, 評価: {len(dataset_split['test'])}件")

# トークナイザー
print("🔤 トークナイザー初期化...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=256,  # 短縮
    )

tokenized_dataset = dataset_split.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "input", "output", "text"]
)

# CPU版モデル読み込み
print("🤖 CPU版モデル読み込み...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # CPU用
    trust_remote_code=True,
    device_map=None,  # CPU使用
)

# 超軽量LoRA設定
print("⚙️ 超軽量LoRA設定...")
lora_config = LoraConfig(
    r=2,  # 最小
    lora_alpha=4,
    target_modules=["q_proj"],  # 1つのモジュールのみ
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
trainable_params = model.num_parameters(only_trainable=True)
total_params = model.num_parameters()

print(f"✅ LoRA適用完了")
print(f"   学習可能パラメータ: {trainable_params:,}")
print(f"   全パラメータ: {total_params:,}")
print(f"   学習可能率: {100 * trainable_params / total_params:.3f}%")

# 超軽量学習設定
print("🎯 学習設定...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # 1エポックのみ
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,  # 高めの学習率
    logging_steps=20,
    save_steps=200,
    eval_steps=200,
    evaluation_strategy="steps",
    save_total_limit=1,
    remove_unused_columns=False,
    dataloader_num_workers=0,  # CPU用
    no_cuda=True,  # CPU強制
    fp16=False,  # CPU用
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("🚀 CPU学習開始...")
print("   ※CPU学習のため時間がかかりますが、確実に動作します")

try:
    trainer.train()
    
    print("💾 モデル保存中...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("🎉 CPU版ファインチューニング完了！")
    print(f"📁 保存先: {OUTPUT_DIR}")
    print(f"📊 学習データ: {len(dataset_split['train'])}件")
    print(f"⚙️ LoRAパラメータ: {trainable_params:,}")
    
    # config更新用の情報
    print("\n📝 次の手順:")
    print("1. config_optimized.py の lora_adapter_path を以下に更新:")
    print(f"   self.lora_adapter_path = '{OUTPUT_DIR}'")
    print("2. RAGアプリを起動:")
    print("   streamlit run RAGapp_optimized.py")
    
except Exception as e:
    print(f"❌ 学習エラー: {e}")
    print("デバッグのため詳細を確認します...")
    import traceback
    traceback.print_exc()