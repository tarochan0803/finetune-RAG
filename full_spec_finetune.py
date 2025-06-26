# full_spec_finetune.py - フルスペック版ファインチューニング
# 最高性能・多機能・高度な設定を統合したプロフェッショナルファインチューニング

import torch
import os
import gc
import json
import logging
import wandb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, ProgressCallback,
    set_seed, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import deepspeed

class FullSpecFineTuner:
    """フルスペック版ファインチューニングクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.model = None
        self.tokenizer = None
        self.best_eval_loss = float('inf')
        
        # Weights & Biases初期化
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'rag-finetune'),
                name=self.config.get('run_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=self.config
            )
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        default_config = {
            # モデル設定
            "model_name": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
            "model_alternatives": [
                "elyza/ELYZA-japanese-Llama-2-13b-instruct",
                "rinna/youri-7b-instruction",
                "stabilityai/japanese-stablelm-instruct-alpha-7b-v2"
            ],
            
            # データ設定
            "train_data_path": "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl",
            "eval_data_path": None,  # 自動分割
            "validation_split": 0.1,
            "max_train_samples": None,  # 全データ使用
            "max_eval_samples": 1000,
            
            # トークナイザー設定
            "max_length": 1024,
            "padding_side": "right",
            "truncation_strategy": "longest_first",
            
            # モデル最適化設定
            "use_quantization": True,
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            },
            "use_gradient_checkpointing": True,
            "use_flash_attention": True,
            
            # LoRA設定
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            
            # 学習設定
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 1.0,
            
            # 評価・保存設定
            "evaluation_strategy": "steps",
            "eval_steps": 200,
            "save_strategy": "steps",
            "save_steps": 200,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            
            # ログ設定
            "logging_strategy": "steps",
            "logging_steps": 50,
            "report_to": ["tensorboard"],
            "use_wandb": False,
            "wandb_project": "rag-finetune",
            
            # 最適化設定
            "optim": "paged_adamw_8bit",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_steps": -1,
            
            # データローダー設定
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "group_by_length": True,
            "length_column_name": "length",
            
            # 高度な設定
            "use_early_stopping": True,
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.001,
            "use_deepspeed": False,
            "deepspeed_config": None,
            
            # 出力設定
            "output_dir": "./full_spec_rag_model",
            "cache_dir": "cache/full_spec",
            "run_name": f"full_spec_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # GPU最適化
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "dataloader_drop_last": False,
            "prediction_loss_only": True,
            
            # 再現性
            "seed": 42,
            "data_seed": 42,
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_logging(self):
        """ログ設定"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """ディレクトリ作成"""
        dirs = [
            self.config['output_dir'],
            self.config['cache_dir'],
            os.path.dirname(self.config['cache_dir'] + '/tokenized'),
            'logs',
            'checkpoints'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def check_environment(self) -> Dict[str, Any]:
        """実行環境チェック"""
        self.logger.info("=== 実行環境チェック ===")
        
        env_info = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
        }
        
        if env_info["cuda_available"]:
            current_gpu = torch.cuda.current_device()
            env_info.update({
                "current_gpu": current_gpu,
                "gpu_name": torch.cuda.get_device_name(current_gpu),
                "total_memory": torch.cuda.get_device_properties(current_gpu).total_memory / 1e9,
                "allocated_memory": torch.cuda.memory_allocated(current_gpu) / 1e9,
                "reserved_memory": torch.cuda.memory_reserved(current_gpu) / 1e9,
            })
            
            self.logger.info(f"✅ CUDA利用可能 - {env_info['gpu_name']}")
            self.logger.info(f"   VRAM: {env_info['total_memory']:.1f}GB")
            self.logger.info(f"   使用中: {env_info['allocated_memory']:.2f}GB")
        else:
            self.logger.error("❌ CUDAが利用できません")
            
        return env_info
    
    def auto_optimize_config(self, env_info: Dict[str, Any]):
        """環境に応じた自動最適化"""
        if not env_info["cuda_available"]:
            return
            
        vram_gb = env_info["total_memory"]
        self.logger.info(f"VRAM {vram_gb:.1f}GB用設定最適化")
        
        if vram_gb >= 24:  # RTX 4090, A100等
            self.config.update({
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 1536,
                "lora_config": dict(self.config["lora_config"], r=32, lora_alpha=64),
                "learning_rate": 1e-4,
            })
            self.logger.info("🚀 高性能GPU設定適用")
            
        elif vram_gb >= 16:  # RTX 4080, 3090等
            self.config.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 12,
                "max_length": 1024,
                "lora_config": dict(self.config["lora_config"], r=16, lora_alpha=32),
                "learning_rate": 5e-5,
            })
            self.logger.info("⚡ 中性能GPU設定適用")
            
        elif vram_gb >= 12:  # RTX 4070Ti, 3080等
            self.config.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_length": 768,
                "lora_config": dict(self.config["lora_config"], r=8, lora_alpha=16),
                "learning_rate": 2e-5,
            })
            self.logger.info("⚙️ 標準GPU設定適用")
            
        else:  # 12GB未満
            self.config.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "max_length": 512,
                "lora_config": dict(self.config["lora_config"], r=4, lora_alpha=8),
                "learning_rate": 1e-5,
            })
            self.logger.info("💧 軽量GPU設定適用")
    
    def load_and_prepare_data(self) -> Dataset:
        """データ読み込み・前処理"""
        self.logger.info("📊 データ準備開始")
        
        # データ読み込み
        if self.config["train_data_path"].endswith('.jsonl'):
            raw_dataset = load_dataset("json", data_files={"train": self.config["train_data_path"]}, split="train")
        else:
            raw_dataset = load_dataset(self.config["train_data_path"], split="train")
        
        self.logger.info(f"元データ: {len(raw_dataset):,}件")
        
        # データ品質フィルタリング
        def quality_filter(example):
            input_text = example.get('input', '') or example.get('text', '')
            output_text = example.get('output', '') or example.get('response', '')
            
            if len(input_text) < 10 or len(output_text) < 5:
                return False
            if len(input_text) > 2000 or len(output_text) > 1000:
                return False
                
            return True
        
        filtered_dataset = raw_dataset.filter(quality_filter)
        self.logger.info(f"品質フィルタ後: {len(filtered_dataset):,}件")
        
        # サンプル数制限
        if self.config["max_train_samples"] and len(filtered_dataset) > self.config["max_train_samples"]:
            filtered_dataset = filtered_dataset.select(range(self.config["max_train_samples"]))
            self.logger.info(f"サンプル制限後: {len(filtered_dataset):,}件")
        
        # 前処理関数
        def preprocess_function(examples):
            instruction = examples.get("instruction", "").strip()
            input_text = examples.get("input", "") or examples.get("text", "")
            output_text = examples.get("output", "") or examples.get("response", "")
            
            input_text = input_text.strip()
            output_text = output_text.strip()
            
            if instruction:
                prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
            else:
                prompt = f"### 入力:\n{input_text}\n\n### 応答:\n"
            
            return {"prompt": prompt, "completion": output_text}
        
        # 前処理適用
        dataset = filtered_dataset.map(preprocess_function, remove_columns=filtered_dataset.column_names)
        
        # 評価データ分割
        if self.config["eval_data_path"]:
            eval_dataset = load_dataset("json", data_files={"eval": self.config["eval_data_path"]}, split="eval")
            eval_dataset = eval_dataset.map(preprocess_function, remove_columns=eval_dataset.column_names)
            final_dataset = {"train": dataset, "eval": eval_dataset}
        else:
            split_dataset = dataset.train_test_split(
                test_size=self.config["validation_split"], 
                seed=self.config["data_seed"]
            )
            final_dataset = {"train": split_dataset["train"], "eval": split_dataset["test"]}
        
        self.logger.info(f"学習データ: {len(final_dataset['train']):,}件")
        self.logger.info(f"評価データ: {len(final_dataset['eval']):,}件")
        
        return final_dataset
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """トークナイザーセットアップ"""
        self.logger.info("🔤 トークナイザー準備")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], 
            trust_remote_code=True,
            padding_side=self.config["padding_side"]
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def tokenize_data(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        """データトークナイズ"""
        self.logger.info("⚡ トークナイズ実行")
        
        cache_path = os.path.join(self.config["cache_dir"], "tokenized")
        
        def tokenize_function(examples):
            prompts = examples["prompt"]
            completions = examples["completion"]
            
            full_texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
            
            model_inputs = tokenizer(
                full_texts,
                max_length=self.config["max_length"],
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
            
            # 長さ情報追加（group_by_length用）
            model_inputs["length"] = [len(ids) for ids in model_inputs["input_ids"]]
            
            return model_inputs
        
        # キャッシュ確認
        try:
            tokenized_dataset = load_from_disk(cache_path)
            self.logger.info("✅ キャッシュからデータ読み込み")
        except:
            self.logger.info("⏳ トークナイズ実行中...")
            tokenized_dataset = {}
            
            for split in dataset.keys():
                tokenized_dataset[split] = dataset[split].map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["prompt", "completion"],
                    num_proc=self.config["dataloader_num_workers"]
                )
            
            # キャッシュ保存
            tokenized_dataset_obj = Dataset.from_dict({
                "train": tokenized_dataset["train"],
                "eval": tokenized_dataset["eval"]
            })
            tokenized_dataset_obj.save_to_disk(cache_path)
            self.logger.info("✅ トークナイズ完了＆キャッシュ保存")
        
        return tokenized_dataset
    
    def setup_model(self, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
        """モデルセットアップ"""
        self.logger.info(f"🤖 モデル読み込み: {self.config['model_name']}")
        
        # メモリクリア
        torch.cuda.empty_cache()
        gc.collect()
        
        # 量子化設定
        bnb_config = None
        if self.config["use_quantization"]:
            qconfig = self.config["quantization_config"]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qconfig["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, qconfig["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=qconfig["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=qconfig["bnb_4bit_quant_type"],
            )
        
        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config["bf16"] else torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if self.config["use_flash_attention"] else None,
        )
        
        # 勾配チェックポイント
        if self.config["use_gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        
        # kbit学習準備
        if self.config["use_quantization"]:
            model = prepare_model_for_kbit_training(model)
        
        # LoRA設定
        lora_config = LoraConfig(**self.config["lora_config"])
        model = get_peft_model(model, lora_config)
        
        # パラメータ情報
        trainable_params = model.num_parameters(only_trainable=True)
        total_params = model.num_parameters()
        
        self.logger.info(f"✅ LoRA適用完了")
        self.logger.info(f"   学習可能パラメータ: {trainable_params:,}")
        self.logger.info(f"   全パラメータ: {total_params:,}")
        self.logger.info(f"   学習可能率: {100 * trainable_params / total_params:.3f}%")
        
        return model
    
    class CustomCallback(TrainerCallback):
        """カスタムコールバック"""
        
        def __init__(self, tuner):
            self.tuner = tuner
        
        def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
            if logs and "eval_loss" in logs:
                if logs["eval_loss"] < self.tuner.best_eval_loss:
                    self.tuner.best_eval_loss = logs["eval_loss"]
                    self.tuner.logger.info(f"🎯 新しい最良評価損失: {logs['eval_loss']:.4f}")
    
    def run_training(self):
        """メイン学習実行"""
        self.logger.info("🚀 フルスペックファインチューニング開始")
        
        # 再現性設定
        set_seed(self.config["seed"])
        
        # 環境チェック・最適化
        env_info = self.check_environment()
        self.auto_optimize_config(env_info)
        
        # データ準備
        dataset = self.load_and_prepare_data()
        
        # トークナイザー
        tokenizer = self.setup_tokenizer()
        
        # トークナイズ
        tokenized_dataset = self.tokenize_data(dataset, tokenizer)
        
        # モデル
        model = self.setup_model(tokenizer)
        
        # データコレクター
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            label_pad_token_id=-100,
        )
        
        # 学習設定
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_ratio=self.config["warmup_ratio"],
            lr_scheduler_type=self.config["lr_scheduler_type"],
            max_grad_norm=self.config["max_grad_norm"],
            
            evaluation_strategy=self.config["evaluation_strategy"],
            eval_steps=self.config["eval_steps"],
            save_strategy=self.config["save_strategy"],
            save_steps=self.config["save_steps"],
            save_total_limit=self.config["save_total_limit"],
            load_best_model_at_end=self.config["load_best_model_at_end"],
            metric_for_best_model=self.config["metric_for_best_model"],
            greater_is_better=self.config["greater_is_better"],
            
            logging_strategy=self.config["logging_strategy"],
            logging_steps=self.config["logging_steps"],
            report_to=self.config["report_to"],
            
            optim=self.config["optim"],
            adam_beta1=self.config["adam_beta1"],
            adam_beta2=self.config["adam_beta2"],
            adam_epsilon=self.config["adam_epsilon"],
            max_steps=self.config["max_steps"],
            
            dataloader_num_workers=self.config["dataloader_num_workers"],
            dataloader_pin_memory=self.config["dataloader_pin_memory"],
            group_by_length=self.config["group_by_length"],
            length_column_name=self.config["length_column_name"],
            
            bf16=self.config["bf16"],
            fp16=self.config["fp16"],
            tf32=self.config["tf32"],
            dataloader_drop_last=self.config["dataloader_drop_last"],
            prediction_loss_only=self.config["prediction_loss_only"],
            
            run_name=self.config["run_name"],
            seed=self.config["seed"],
            data_seed=self.config["data_seed"],
            
            remove_unused_columns=False,
        )
        
        # コールバック準備
        callbacks = [self.CustomCallback(self)]
        if self.config["use_early_stopping"]:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.config["early_stopping_patience"],
                early_stopping_threshold=self.config["early_stopping_threshold"]
            ))
        
        # トレーナー作成
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        
        # 学習実行
        self.logger.info("🎯 学習開始...")
        trainer.train()
        
        # 保存
        self.logger.info("💾 モデル保存中...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config["output_dir"])
        
        # 設定保存
        with open(os.path.join(self.config["output_dir"], "training_config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info("🎉 フルスペック学習完了！")
        self.logger.info(f"📁 保存先: {self.config['output_dir']}")
        self.logger.info(f"🏆 最良評価損失: {self.best_eval_loss:.4f}")
        
        return trainer

def main():
    """メイン実行関数"""
    print("🚀 フルスペック版ファインチューニング")
    print("=" * 60)
    
    # 設定ファイルのパス（オプション）
    config_path = input("設定ファイルのパス（空でデフォルト設定）: ").strip()
    if not config_path:
        config_path = None
    
    # ファインチューニング実行
    tuner = FullSpecFineTuner(config_path)
    trainer = tuner.run_training()
    
    print("\n✅ すべて完了！")
    print("\n次の手順:")
    print("1. config.py の lora_adapter_path を以下に変更:")
    print(f"   self.lora_adapter_path = '{tuner.config['output_dir']}'")
    print("2. モデル性能を評価")
    print("3. 必要に応じてvLLM最適化実行")

if __name__ == "__main__":
    main()