import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

# ──────────────────────────────────────────────────────
# 0. 設定
MODEL_NAME     = "deepseek-ai/deepseek-r1-distill-qwen-7b"
RAW_JSONL_PATH = "/home/ncnadmin/my_rag_project/fine_tuning_dynamic_instruction_dataset.jsonl"
CACHE_DIR      = "cache/tokenized_qora"
OUTPUT_DIR     = "./lora_qora_finetune"

# ──────────────────────────────────────────────────────
# 1. 生データ読み込み & 前処理
raw_ds = load_dataset(
    "json",
    data_files={"train": RAW_JSONL_PATH},
    split="train",
)

def preprocess(example):
    instr = example.get("instruction", "").strip()
    if instr:
        prompt = (
            f"### Instruction:\n{instr}\n"
            f"### Input:\n{example['input']}\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Input:\n{example['input']}\n### Response:\n"
    return {"prompt": prompt, "completion": example["output"]}

ds = raw_ds.map(
    preprocess,
    remove_columns=raw_ds.column_names,
    num_proc=4,
)
ds = ds.train_test_split(test_size=0.1, seed=42)

# ──────────────────────────────────────────────────────
# 2. トークナイザー & トークナイズ (一度だけキャッシュ)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
MAX_PROMPT_LEN = 512
MAX_TOTAL_LEN  = 1024

def tokenize_fn(batch):
    # prompt 部分を token 化
    inputs = tokenizer(
        batch["prompt"],
        max_length=MAX_PROMPT_LEN,
        truncation=True,
        padding=False,
    )
    # prompt+completion+eos をまとめて token 化し、labels を作成
    full = [p + c + tokenizer.eos_token for p, c in zip(batch["prompt"], batch["completion"])]
    label_ids = tokenizer(
        full,
        max_length=MAX_TOTAL_LEN,
        truncation=True,
        padding=False,
    )["input_ids"]
    # prompt 部分を -100 でマスク
    for i, labs in enumerate(label_ids):
        plen = len(inputs["input_ids"][i])
        labs[:plen] = [-100] * plen
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": label_ids,
    }

# キャッシュがなければトークナイズ＆保存
try:
    tokenized = load_from_disk(CACHE_DIR)
except FileNotFoundError:
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=512,
        remove_columns=["prompt", "completion"],
        num_proc=4,
    )
    tokenized.save_to_disk(CACHE_DIR)

# ──────────────────────────────────────────────────────
# 3. モデル読み込み (4bit QLoRA + LoRA 設定)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# optional: Torch 2.0 compile
if torch.__version__ >= "2.0":
    model = torch.compile(model)

# ──────────────────────────────────────────────────────
# 4. DataCollator 設定 (動的パディング)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding="longest",
    label_pad_token_id=-100,
)

# ──────────────────────────────────────────────────────
# 5. トレーニング引数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    dataloader_num_workers=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    report_to="tensorboard",
)

# ──────────────────────────────────────────────────────
# 6. Trainer & 学習開始
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
