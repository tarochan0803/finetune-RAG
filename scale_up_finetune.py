# scale_up_finetune.py - 13Bモデル対応大規模ファインチューニング
# RTX 5070でも動作するよう最適化

import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import gc

class ScaledUpFineTuner:
    def __init__(self):
        # 13Bモデル設定
        self.model_options = [
            "elyza/ELYZA-japanese-Llama-2-13b-instruct",  # 第1選択
            "rinna/youri-7b-instruction",                  # 第2選択（7B高性能）
            "stabilityai/japanese-stablelm-instruct-alpha-7b-v2"  # 第3選択
        ]
        
        self.jsonl_path = "/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl"
        self.cache_dir = "cache/scaled_tokenized"
        self.output_dir = "./rag_model_13b"
        
        # RTX 5070最適化設定
        self.batch_size = 1
        self.grad_accumulation = 16  # 実効バッチサイズ=16
        self.max_length = 768
        self.lora_r = 16
        self.learning_rate = 1e-4
        
        print("🚀 13Bモデル対応ファインチューニング設定")
        
    def check_model_availability(self) -> str:
        """利用可能なモデルを確認"""
        print("📋 モデル可用性チェック中...")
        
        for model_name in self.model_options:
            try:
                print(f"   試行: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                print(f"✅ 利用可能: {model_name}")
                return model_name
            except Exception as e:
                print(f"❌ 利用不可: {model_name} - {e}")
                continue
        
        # フォールバック
        fallback_model = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        print(f"🔄 フォールバック: {fallback_model}")
        return fallback_model
    
    def optimize_for_model_size(self, model_name: str):
        """モデルサイズに応じた設定最適化"""
        if "13b" in model_name.lower():
            print("⚡ 13Bモデル向け設定")
            self.batch_size = 1
            self.grad_accumulation = 8  # メモリ節約
            self.max_length = 512       # 短縮
            self.lora_r = 8            # LoRA削減
        elif "7b" in model_name.lower():
            print("🔧 7Bモデル向け設定")
            self.batch_size = 1
            self.grad_accumulation = 12
            self.max_length = 768
            self.lora_r = 16
        
        print(f"   バッチサイズ: {self.batch_size}")
        print(f"   勾配蓄積: {self.grad_accumulation}")
        print(f"   最大長: {self.max_length}")
        print(f"   LoRA r: {self.lora_r}")
    
    def prepare_data(self):
        """データ準備"""
        print("📊 高品質データ準備中...")
        
        def preprocess_function(examples):
            instruction = examples.get("instruction", "").strip()
            input_text = examples.get("input", "").strip()
            output_text = examples.get("output", "").strip()
            
            if instruction:
                prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
            else:
                prompt = f"### 入力:\n{input_text}\n\n### 応答:\n"
            
            return {"prompt": prompt, "completion": output_text}
        
        # データロード
        raw_dataset = load_dataset("json", data_files={"train": self.jsonl_path}, split="train")
        print(f"   元データ: {len(raw_dataset):,}件")
        
        # 品質フィルタリング（長すぎる・短すぎるデータを除外）
        def quality_filter(example):
            input_len = len(example.get('input', ''))
            output_len = len(example.get('output', ''))
            return 10 <= input_len <= 500 and 5 <= output_len <= 200
        
        filtered_dataset = raw_dataset.filter(quality_filter)
        print(f"   品質フィルタ後: {len(filtered_dataset):,}件")
        
        # 前処理適用
        dataset = filtered_dataset.map(preprocess_function, remove_columns=filtered_dataset.column_names)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"   学習データ: {len(dataset['train']):,}件")
        print(f"   評価データ: {len(dataset['test']):,}件")
        
        return dataset
    
    def tokenize_data(self, dataset, tokenizer):
        """トークナイズ（キャッシュ利用）"""
        print("🔤 トークナイズ実行...")
        
        def tokenize_function(examples):
            prompts = examples["prompt"]
            completions = examples["completion"]
            
            full_texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
            
            model_inputs = tokenizer(
                full_texts,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )
            
            # プロンプト部分をマスク
            prompt_inputs = tokenizer(prompts, truncation=True, padding=False)
            labels = []
            
            for i, input_ids in enumerate(model_inputs["input_ids"]):
                prompt_len = len(prompt_inputs["input_ids"][i])
                label = input_ids.copy()
                label[:prompt_len] = [-100] * prompt_len
                labels.append(label)
            
            model_inputs["labels"] = labels
            return model_inputs
        
        # キャッシュ確認
        try:
            tokenized_dataset = load_from_disk(self.cache_dir)
            print("✅ キャッシュからデータ読み込み")
        except:
            print("⏳ トークナイズ実行中...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["prompt", "completion"],
                num_proc=4
            )
            tokenized_dataset.save_to_disk(self.cache_dir)
            print("✅ トークナイズ完了＆キャッシュ保存")
        
        return tokenized_dataset
    
    def load_model(self, model_name: str):
        """モデル読み込み（メモリ最適化）"""
        print(f"🤖 {model_name} 読み込み中...")
        
        # メモリクリア
        torch.cuda.empty_cache()
        gc.collect()
        
        # 量子化設定（メモリ効率重視）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # メモリ効率化
        )
        
        model.gradient_checkpointing_enable()
        
        # LoRA設定
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        model = get_peft_model(model, lora_config)
        
        trainable_params = model.num_parameters(only_trainable=True)
        total_params = model.num_parameters()
        print(f"✅ LoRA適用完了")
        print(f"   学習可能パラメータ: {trainable_params:,}")
        print(f"   全パラメータ: {total_params:,}")
        print(f"   学習可能率: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def run_training(self):
        """メインの学習実行"""
        print("🎯 スケールアップファインチューニング開始")
        
        # 1. モデル選択
        model_name = self.check_model_availability()
        self.optimize_for_model_size(model_name)
        
        # 2. データ準備
        dataset = self.prepare_data()
        
        # 3. トークナイザー
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4. トークナイズ
        tokenized_dataset = self.tokenize_data(dataset, tokenizer)
        
        # 5. モデル読み込み
        model = self.load_model(model_name)
        
        # 6. データコレクター
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            label_pad_token_id=-100,
        )
        
        # 7. 学習設定
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=200,
            bf16=True,
            optim="paged_adamw_8bit",
            dataloader_num_workers=2,
            
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            logging_steps=20,
            report_to=["tensorboard"],
            
            remove_unused_columns=False,
            group_by_length=True,
            dataloader_pin_memory=False,
        )
        
        # 8. トレーナー
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 9. 学習実行
        print("🚀 学習開始...")
        trainer.train()
        
        # 10. 保存
        print("💾 モデル保存中...")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        print("🎉 スケールアップ学習完了！")
        print(f"📁 保存先: {self.output_dir}")
        print(f"🤖 使用モデル: {model_name}")
        print("\n次の手順:")
        print("1. config.py の lora_adapter_path を以下に変更:")
        print(f"   self.lora_adapter_path = '{self.output_dir}'")
        print("2. vLLM最適化を実行")

if __name__ == "__main__":
    tuner = ScaledUpFineTuner()
    tuner.run_training()