#!/usr/bin/env python3
# continual_learning_finetune.py - 継続学習対応フルスペック版ファインチューニング
"""
継続学習（Continual Learning）対応の高度なファインチューニングシステム
- 既存のファインチューニング済みモデルからの継続学習
- 知識の忘却を防ぐRegularization技術
- Elastic Weight Consolidation (EWC)
- 段階的なカリキュラム学習
- 複数タスクの学習履歴管理
"""

import os
import sys
import json
import logging
import shutil
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb

@dataclass
class ContinualLearningConfig:
    """継続学習設定"""
    # 基本設定
    base_model_path: str = "./full_spec_rag_model"  # 既存のファインチューニング済みモデル
    checkpoint_interval: int = 500  # チェックポイント保存間隔
    
    # EWC (Elastic Weight Consolidation) 設定
    use_ewc: bool = True
    ewc_lambda: float = 1000.0  # EWC正則化強度
    fisher_estimation_sample_size: int = 1000  # Fisher情報行列推定サンプル数
    
    # 学習率調整
    initial_lr_factor: float = 0.1  # 継続学習時の初期学習率倍率
    lr_decay_factor: float = 0.9   # 各タスクでの学習率減衰
    
    # カリキュラム学習設定
    use_curriculum: bool = True
    difficulty_threshold: float = 0.8  # 難易度閾値
    curriculum_batch_size_factor: float = 0.5  # カリキュラム学習時のバッチサイズ倍率
    
    # 知識蒸留設定
    use_knowledge_distillation: bool = True
    kd_temperature: float = 4.0     # 蒸留温度
    kd_alpha: float = 0.7          # 蒸留損失の重み
    
    # リプレイバッファ設定
    use_replay_buffer: bool = True
    replay_buffer_size: int = 10000  # リプレイバッファサイズ
    replay_sample_rate: float = 0.2  # リプレイサンプルの割合
    
    # タスク管理
    max_tasks: int = 10  # 最大タスク数
    task_memory_size: int = 5000  # タスクごとのメモリサイズ

@dataclass
class TaskInfo:
    """タスク情報"""
    task_id: str
    name: str
    description: str
    data_path: str
    model_path: str
    timestamp: str
    performance_metrics: Dict[str, float]
    fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None
    optimal_params: Optional[Dict[str, torch.Tensor]] = None

class FisherInformationMatrix:
    """Fisher情報行列計算クラス"""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, device: str = "cuda"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
    def compute_fisher_diagonal(self, sample_size: int = 1000) -> Dict[str, torch.Tensor]:
        """Fisher情報行列の対角成分を計算"""
        fisher_diagonal = {}
        
        # モデルをeval模ードに
        self.model.eval()
        
        # パラメータ初期化
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_diagonal[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch in self.dataloader:
            if sample_count >= sample_size:
                break
                
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 勾配をクリア
            self.model.zero_grad()
            
            # フォワードパス
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # バックワード
            loss.backward()
            
            # Fisher情報行列更新
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    fisher_diagonal[name] += param.grad.pow(2)
            
            sample_count += batch['input_ids'].size(0)
        
        # 平均化
        for name in fisher_diagonal:
            fisher_diagonal[name] /= sample_count
            
        return fisher_diagonal

class EWCLoss:
    """Elastic Weight Consolidation損失"""
    
    def __init__(self, fisher_diagonal: Dict[str, torch.Tensor], 
                 optimal_params: Dict[str, torch.Tensor], ewc_lambda: float = 1000.0):
        self.fisher_diagonal = fisher_diagonal
        self.optimal_params = optimal_params
        self.ewc_lambda = ewc_lambda
    
    def compute_penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC正則化項を計算"""
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_diagonal and param.requires_grad:
                penalty += (self.fisher_diagonal[name] * 
                           (param - self.optimal_params[name]).pow(2)).sum()
        
        return self.ewc_lambda * penalty

class ReplayBuffer:
    """経験リプレイバッファ"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(self, experiences: List[Dict]):
        """経験を追加"""
        for exp in experiences:
            if len(self.buffer) < self.max_size:
                self.buffer.append(exp)
            else:
                self.buffer[self.position] = exp
                self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> List[Dict]:
        """サンプリング"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        return np.random.choice(self.buffer, batch_size, replace=False).tolist()
    
    def size(self) -> int:
        return len(self.buffer)

class ContinualLearningTrainer(Trainer):
    """継続学習対応トレーナー"""
    
    def __init__(self, cl_config: ContinualLearningConfig, ewc_loss: Optional[EWCLoss] = None,
                 replay_buffer: Optional[ReplayBuffer] = None, teacher_model: Optional[nn.Module] = None, **kwargs):
        super().__init__(**kwargs)
        self.cl_config = cl_config
        self.ewc_loss = ewc_loss
        self.replay_buffer = replay_buffer
        self.teacher_model = teacher_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """拡張損失関数"""
        # 通常の損失計算
        outputs = model(**inputs)
        loss = outputs.loss
        
        # EWC正則化項
        if self.ewc_loss is not None:
            ewc_penalty = self.ewc_loss.compute_penalty(model)
            loss += ewc_penalty
        
        # 知識蒸留損失
        if self.teacher_model is not None and self.cl_config.use_knowledge_distillation:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            
            student_logits = outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # 蒸留損失計算
            kd_loss = self._compute_kd_loss(student_logits, teacher_logits, 
                                          self.cl_config.kd_temperature)
            loss = self.cl_config.kd_alpha * kd_loss + (1 - self.cl_config.kd_alpha) * loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                        temperature: float) -> torch.Tensor:
        """知識蒸留損失計算"""
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

class ContinualLearningFineTuner:
    """継続学習ファインチューナー"""
    
    def __init__(self, base_config_path: Optional[str] = None):
        self.setup_logging()
        
        # 設定読み込み
        self.cl_config = ContinualLearningConfig()
        if base_config_path:
            self.load_base_config(base_config_path)
            
        # 継続学習設定を既存設定に統合
        from full_spec_finetune import FullSpecFineTuner
        self.base_tuner = FullSpecFineTuner(base_config_path)
        
        # 状態管理
        self.task_history: List[TaskInfo] = []
        self.current_task_id = 0
        self.replay_buffer = ReplayBuffer(self.cl_config.replay_buffer_size) if self.cl_config.use_replay_buffer else None
        
        # ディレクトリ設定
        self.setup_directories()
        
        self.logger.info("🔄 継続学習ファインチューニング初期化完了")
    
    def setup_logging(self):
        """ログ設定"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'continual_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_base_config(self, config_path: str):
        """既存設定読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 継続学習設定更新
        if 'continual_learning' in config:
            cl_config_dict = config['continual_learning']
            for key, value in cl_config_dict.items():
                if hasattr(self.cl_config, key):
                    setattr(self.cl_config, key, value)
    
    def setup_directories(self):
        """ディレクトリ作成"""
        dirs = [
            "continual_learning_checkpoints",
            "continual_learning_logs", 
            "task_history",
            "fisher_matrices",
            "replay_buffers"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_previous_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """前回のモデル読み込み"""
        self.logger.info(f"📂 前回のモデル読み込み: {self.cl_config.base_model_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.cl_config.base_model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.cl_config.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.logger.info("✅ 前回のモデル読み込み完了")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"❌ モデル読み込みエラー: {e}")
            raise
    
    def compute_fisher_information(self, model: nn.Module, dataset: Dataset, 
                                 tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
        """Fisher情報行列計算"""
        self.logger.info("🧮 Fisher情報行列計算開始")
        
        # データコレクター
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            label_pad_token_id=-100,
        )
        
        # データローダー
        dataloader = DataLoader(
            dataset.select(range(min(self.cl_config.fisher_estimation_sample_size, len(dataset)))),
            batch_size=1,
            collate_fn=data_collator
        )
        
        # Fisher情報行列計算
        fisher_computer = FisherInformationMatrix(model, dataloader)
        fisher_diagonal = fisher_computer.compute_fisher_diagonal(self.cl_config.fisher_estimation_sample_size)
        
        self.logger.info("✅ Fisher情報行列計算完了")
        return fisher_diagonal
    
    def save_task_info(self, task_info: TaskInfo):
        """タスク情報保存"""
        # Fisher行列保存
        if task_info.fisher_diagonal:
            fisher_path = f"fisher_matrices/fisher_{task_info.task_id}.pkl"
            with open(fisher_path, 'wb') as f:
                pickle.dump(task_info.fisher_diagonal, f)
        
        # 最適パラメータ保存
        if task_info.optimal_params:
            params_path = f"fisher_matrices/params_{task_info.task_id}.pkl"
            with open(params_path, 'wb') as f:
                pickle.dump(task_info.optimal_params, f)
        
        # タスク履歴保存
        task_info_path = f"task_history/task_{task_info.task_id}.json"
        with open(task_info_path, 'w', encoding='utf-8') as f:
            task_data = asdict(task_info)
            # テンソルは除外
            task_data.pop('fisher_diagonal', None)
            task_data.pop('optimal_params', None)
            json.dump(task_data, f, indent=2, ensure_ascii=False)
    
    def load_task_history(self):
        """タスク履歴読み込み"""
        task_history_files = list(Path("task_history").glob("task_*.json"))
        
        for file_path in sorted(task_history_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
                task_info = TaskInfo(**task_data)
                
                # Fisher行列読み込み
                fisher_path = f"fisher_matrices/fisher_{task_info.task_id}.pkl"
                if os.path.exists(fisher_path):
                    with open(fisher_path, 'rb') as f:
                        task_info.fisher_diagonal = pickle.load(f)
                
                # 最適パラメータ読み込み
                params_path = f"fisher_matrices/params_{task_info.task_id}.pkl"
                if os.path.exists(params_path):
                    with open(params_path, 'rb') as f:
                        task_info.optimal_params = pickle.load(f)
                
                self.task_history.append(task_info)
        
        if self.task_history:
            self.current_task_id = max(int(task.task_id) for task in self.task_history) + 1
            self.logger.info(f"📚 タスク履歴読み込み完了: {len(self.task_history)}タスク")
    
    def prepare_curriculum_data(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        """カリキュラム学習用データ準備"""
        if not self.cl_config.use_curriculum:
            return dataset
        
        self.logger.info("📈 カリキュラム学習データ準備")
        
        # 難易度推定（文章長ベース）
        def estimate_difficulty(example):
            text_length = len(example.get('input', '') + example.get('output', ''))
            return min(text_length / 1000, 1.0)  # 正規化
        
        # 難易度追加
        dataset = dataset.map(lambda x: {**x, 'difficulty': estimate_difficulty(x)})
        
        # 難易度でソート
        dataset = dataset.sort('difficulty')
        
        # 段階的な難易度フィルタリング
        easy_size = int(len(dataset) * 0.3)
        medium_size = int(len(dataset) * 0.6)
        
        curriculum_stages = [
            dataset.select(range(easy_size)),                    # Easy
            dataset.select(range(easy_size, medium_size)),       # Medium  
            dataset.select(range(medium_size, len(dataset)))     # Hard
        ]
        
        self.logger.info(f"   段階1 (簡単): {len(curriculum_stages[0]):,}件")
        self.logger.info(f"   段階2 (中程度): {len(curriculum_stages[1]):,}件") 
        self.logger.info(f"   段階3 (困難): {len(curriculum_stages[2]):,}件")
        
        return curriculum_stages
    
    def add_new_task(self, task_name: str, data_path: str, description: str = "") -> str:
        """新しいタスク追加"""
        task_id = str(self.current_task_id)
        self.current_task_id += 1
        
        self.logger.info(f"📝 新タスク追加: {task_name} (ID: {task_id})")
        
        # データ読み込み・前処理
        dataset = self.base_tuner.load_and_prepare_data()
        
        # 前回のモデル読み込み
        model, tokenizer = self.load_previous_model()
        
        # Teacher model（知識蒸留用）
        teacher_model = None
        if self.cl_config.use_knowledge_distillation and self.task_history:
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
        
        # Fisher情報行列計算（前タスクのために）
        fisher_diagonal = None
        optimal_params = None
        
        if self.cl_config.use_ewc and self.task_history:
            fisher_diagonal = self.compute_fisher_information(model, dataset['train'], tokenizer)
            optimal_params = {name: param.clone().detach() 
                            for name, param in model.named_parameters() if param.requires_grad}
        
        # EWC損失準備
        ewc_loss = None
        if self.task_history and self.cl_config.use_ewc:
            # 全ての過去タスクのEWC損失を統合
            combined_fisher = {}
            combined_params = {}
            
            for prev_task in self.task_history:
                if prev_task.fisher_diagonal and prev_task.optimal_params:
                    for name in prev_task.fisher_diagonal:
                        if name not in combined_fisher:
                            combined_fisher[name] = prev_task.fisher_diagonal[name].clone()
                            combined_params[name] = prev_task.optimal_params[name].clone()
                        else:
                            combined_fisher[name] += prev_task.fisher_diagonal[name]
            
            if combined_fisher:
                ewc_loss = EWCLoss(combined_fisher, combined_params, self.cl_config.ewc_lambda)
        
        # リプレイデータ準備
        replay_data = None
        if self.replay_buffer and self.replay_buffer.size() > 0:
            replay_samples = self.replay_buffer.sample(
                int(len(dataset['train']) * self.cl_config.replay_sample_rate)
            )
            replay_data = Dataset.from_list(replay_samples)
            dataset['train'] = concatenate_datasets([dataset['train'], replay_data])
            self.logger.info(f"   リプレイデータ追加: {len(replay_samples):,}件")
        
        # カリキュラム学習設定
        training_stages = self.prepare_curriculum_data(dataset['train'], tokenizer) if self.cl_config.use_curriculum else [dataset['train']]
        
        # 各段階で学習実行
        for stage_idx, stage_dataset in enumerate(training_stages):
            self.logger.info(f"🎯 学習段階 {stage_idx + 1}/{len(training_stages)} 開始")
            
            # 学習率調整
            current_lr = (self.base_tuner.config['learning_rate'] * 
                         self.cl_config.initial_lr_factor * 
                         (self.cl_config.lr_decay_factor ** len(self.task_history)))
            
            # バッチサイズ調整
            batch_size = self.base_tuner.config['per_device_train_batch_size']
            if self.cl_config.use_curriculum and stage_idx == 0:
                batch_size = max(1, int(batch_size * self.cl_config.curriculum_batch_size_factor))
            
            # トレーナー設定
            training_args = TrainingArguments(
                output_dir=f"continual_learning_checkpoints/task_{task_id}_stage_{stage_idx}",
                num_train_epochs=1 if self.cl_config.use_curriculum else 3,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=current_lr,
                warmup_ratio=0.1,
                save_steps=self.cl_config.checkpoint_interval,
                evaluation_strategy="steps",
                eval_steps=self.cl_config.checkpoint_interval,
                logging_steps=50,
                save_total_limit=2,
                load_best_model_at_end=True,
                dataloader_num_workers=4,
                bf16=True,
                remove_unused_columns=False,
                report_to=["wandb"] if self.base_tuner.config.get('use_wandb', False) else [],
            )
            
            # データコレクター
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding="longest",
                label_pad_token_id=-100,
            )
            
            # 継続学習トレーナー
            trainer = ContinualLearningTrainer(
                cl_config=self.cl_config,
                ewc_loss=ewc_loss,
                replay_buffer=self.replay_buffer,
                teacher_model=teacher_model,
                model=model,
                args=training_args,
                train_dataset=stage_dataset,
                eval_dataset=dataset['eval'],
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # 学習実行
            trainer.train()
            
            self.logger.info(f"✅ 学習段階 {stage_idx + 1} 完了")
        
        # 最終モデル保存
        final_model_path = f"continual_learning_checkpoints/task_{task_id}_final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # タスク情報作成
        task_info = TaskInfo(
            task_id=task_id,
            name=task_name,
            description=description,
            data_path=data_path,
            model_path=final_model_path,
            timestamp=datetime.now().isoformat(),
            performance_metrics={"final_loss": trainer.state.log_history[-1].get('eval_loss', 0.0)},
            fisher_diagonal=fisher_diagonal,
            optimal_params=optimal_params
        )
        
        # タスク履歴に追加
        self.task_history.append(task_info)
        self.save_task_info(task_info)
        
        # リプレイバッファ更新
        if self.replay_buffer:
            # 現在のタスクからサンプルを追加
            task_samples = dataset['train'].select(range(min(1000, len(dataset['train']))))
            task_sample_list = [task_samples[i] for i in range(len(task_samples))]
            self.replay_buffer.add(task_sample_list)
            
            # リプレイバッファ保存
            with open(f"replay_buffers/buffer_after_task_{task_id}.pkl", 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        
        # ベースモデルパス更新
        self.cl_config.base_model_path = final_model_path
        
        self.logger.info(f"🎉 タスク {task_name} 完了")
        self.logger.info(f"   モデル保存先: {final_model_path}")
        
        return task_id
    
    def evaluate_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """全タスクでの性能評価"""
        self.logger.info("📊 全タスク性能評価開始")
        
        if not self.task_history:
            self.logger.warning("評価対象のタスクがありません")
            return {}
        
        # 最新モデル読み込み
        latest_task = self.task_history[-1]
        model, tokenizer = self.load_previous_model()
        
        results = {}
        
        for task_info in self.task_history:
            self.logger.info(f"   タスク {task_info.name} 評価中...")
            
            try:
                # タスクデータ読み込み
                dataset = load_dataset("json", data_files={"test": task_info.data_path}, split="test")
                
                # 簡易評価（困惑度計算）
                # 実際の実装では、より詳細な評価指標を使用
                perplexity = self._compute_perplexity(model, dataset, tokenizer)
                
                results[task_info.task_id] = {
                    "task_name": task_info.name,
                    "perplexity": perplexity,
                    "timestamp": task_info.timestamp
                }
                
            except Exception as e:
                self.logger.error(f"タスク {task_info.name} 評価エラー: {e}")
                results[task_info.task_id] = {"error": str(e)}
        
        self.logger.info("✅ 全タスク性能評価完了")
        return results
    
    def _compute_perplexity(self, model: nn.Module, dataset: Dataset, 
                          tokenizer: AutoTokenizer, max_samples: int = 100) -> float:
        """困惑度計算"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i, example in enumerate(dataset.select(range(min(max_samples, len(dataset))))):
                text = example.get('input', '') + example.get('output', '')
                
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    def generate_task_report(self) -> str:
        """タスクレポート生成"""
        if not self.task_history:
            return "タスク履歴がありません。"
        
        report = f"""
# 継続学習タスクレポート
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概要
- 総タスク数: {len(self.task_history)}
- 現在のタスクID: {self.current_task_id - 1}
- リプレイバッファサイズ: {self.replay_buffer.size() if self.replay_buffer else 0}

## タスク履歴
"""
        
        for i, task in enumerate(self.task_history, 1):
            report += f"""
### タスク {i}: {task.name}
- ID: {task.task_id}
- 説明: {task.description}
- 学習日時: {task.timestamp}
- モデル保存先: {task.model_path}
- 最終損失: {task.performance_metrics.get('final_loss', 'N/A')}
"""
        
        # 性能評価結果
        evaluation_results = self.evaluate_all_tasks()
        if evaluation_results:
            report += "\n## 性能評価結果\n"
            for task_id, metrics in evaluation_results.items():
                if 'error' not in metrics:
                    report += f"- {metrics['task_name']}: 困惑度 {metrics['perplexity']:.2f}\n"
        
        return report

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="継続学習ファインチューニング")
    parser.add_argument("--base-config", type=str, help="ベース設定ファイル")
    parser.add_argument("--task-name", type=str, required=True, help="タスク名")
    parser.add_argument("--data-path", type=str, required=True, help="データパス")
    parser.add_argument("--description", type=str, default="", help="タスク説明")
    parser.add_argument("--evaluate", action="store_true", help="評価のみ実行")
    parser.add_argument("--report", action="store_true", help="レポート生成")
    
    args = parser.parse_args()
    
    print("🔄 継続学習ファインチューニング")
    print("=" * 60)
    
    # 継続学習ファインチューナー初期化
    tuner = ContinualLearningFineTuner(args.base_config)
    
    # タスク履歴読み込み
    tuner.load_task_history()
    
    if args.evaluate:
        # 評価のみ
        results = tuner.evaluate_all_tasks()
        print("\n📊 評価結果:")
        for task_id, metrics in results.items():
            if 'error' not in metrics:
                print(f"  {metrics['task_name']}: 困惑度 {metrics['perplexity']:.2f}")
    
    elif args.report:
        # レポート生成
        report = tuner.generate_task_report()
        report_path = f"continual_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📋 レポート生成完了: {report_path}")
    
    else:
        # 新タスク追加
        task_id = tuner.add_new_task(args.task_name, args.data_path, args.description)
        
        print(f"\n🎉 継続学習完了！")
        print(f"タスクID: {task_id}")
        print(f"モデル保存先: continual_learning_checkpoints/task_{task_id}_final")
        print("\n次のステップ:")
        print("1. 性能評価: python continual_learning_finetune.py --evaluate")
        print("2. レポート生成: python continual_learning_finetune.py --report")

if __name__ == "__main__":
    main()