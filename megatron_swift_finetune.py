#!/usr/bin/env python3
# megatron_swift_finetune.py - MS-Swift + Megatron-Core フルスペック版ファインチューニング統合システム
"""
エンタープライズグレードの大規模言語モデルファインチューニングシステム
- MS-Swift + Megatron-Core統合
- 分散学習対応（Tensor/Pipeline/Data Parallel）
- 高度な最適化（Flash Attention, Mixed Precision, Gradient Checkpointing）
- 包括的なモニタリング（Weights & Biases, TensorBoard）
- 自動リソース最適化
"""

import os
import sys
import json
import logging
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import yaml

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb

@dataclass
class ModelConfig:
    """モデル設定"""
    model_id: str = "Qwen/Qwen3-8B-Base"
    model_alternatives: List[str] = None
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    
    def __post_init__(self):
        if self.model_alternatives is None:
            self.model_alternatives = [
                "Qwen/Qwen3-14B-Base",
                "microsoft/DialoGPT-large",
                "elyza/ELYZA-japanese-Llama-2-7b-instruct"
            ]

@dataclass 
class DataConfig:
    """データ設定"""
    dataset_name: str = "kunishou/ApolloCorpus-ja"
    text_column: str = "response_ja"
    output_dir: str = "data"
    output_filename: str = "apollo_cpt.jsonl"
    max_length: int = 32768
    truncation_strategy: str = "right"
    packing: bool = True
    streaming: bool = True
    num_proc: int = 64
    data_format: str = "cpt"  # "cpt" or "chat"

@dataclass
class TrainingConfig:
    """学習設定"""
    # 分散学習設定
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    
    # バッチサイズ設定
    micro_batch_size: int = 2
    global_batch_size: int = 512
    
    # 学習パラメータ
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    warmup_iters: int = 100
    train_iters: int = 1000
    
    # 最適化設定
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    cross_entropy_loss_fusion: bool = True
    
    # 保存・評価設定
    save_interval: int = 50
    eval_interval: int = 100
    log_interval: int = 1
    
    # その他の設定
    finetune: bool = True
    no_save_optim: bool = True
    no_save_rng: bool = True
    num_workers: int = 1

@dataclass
class SystemConfig:
    """システム設定"""
    # 実行環境
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7"
    nproc_per_node: int = 8
    cuda_device_max_connections: int = 1
    
    # 出力設定
    output_base_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    converted_model_dir: str = "converted_models"
    
    # モニタリング
    use_wandb: bool = True
    wandb_project: str = "megatron-swift-finetune"
    use_tensorboard: bool = True
    
    # その他
    seed: int = 42
    debug: bool = False

class MegatronSwiftFineTuner:
    """MS-Swift + Megatron-Core統合ファインチューニングクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        
        # 設定読み込み
        if config_path and os.path.exists(config_path):
            self.load_config_from_file(config_path)
        else:
            self.model_config = ModelConfig()
            self.data_config = DataConfig()
            self.training_config = TrainingConfig()
            self.system_config = SystemConfig()
        
        # ランID生成
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M")
        self.run_name = f"{self.get_safe_model_name()}-{self.run_id}"
        
        # ディレクトリ設定
        self.setup_directories()
        
        # 環境初期化
        self.setup_environment()
        
        self.logger.info("🚀 Megatron-Swift統合ファインチューニング初期化完了")
    
    def setup_logging(self):
        """ログ設定"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'megatron_swift_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config_from_file(self, config_path: str):
        """設定ファイル読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        self.model_config = ModelConfig(**config.get('model', {}))
        self.data_config = DataConfig(**config.get('data', {}))
        self.training_config = TrainingConfig(**config.get('training', {}))
        self.system_config = SystemConfig(**config.get('system', {}))
    
    def get_safe_model_name(self) -> str:
        """モデル名を安全な文字列に変換"""
        model_name = self.model_config.model_id.split('/')[-1]
        return model_name.lower().replace(' ', '-').replace('_', '-')
    
    def setup_directories(self):
        """ディレクトリ作成"""
        base_dirs = [
            self.system_config.output_base_dir,
            self.data_config.output_dir,
        ]
        
        run_dirs = [
            f"{self.system_config.checkpoint_dir}/{self.run_name}",
            f"{self.system_config.log_dir}/{self.run_name}",
            f"{self.system_config.converted_model_dir}/{self.run_name}",
        ]
        
        for dir_path in base_dirs + run_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_environment(self):
        """環境設定"""
        # CUDA設定
        os.environ['CUDA_VISIBLE_DEVICES'] = self.system_config.cuda_visible_devices
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = str(self.system_config.cuda_device_max_connections)
        os.environ['NPROC_PER_NODE'] = str(self.system_config.nproc_per_node)
        
        # その他の環境変数
        if self.system_config.debug:
            os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
            os.environ['NCCL_DEBUG'] = 'INFO'
        
        # Weights & Biases初期化
        if self.system_config.use_wandb:
            wandb.init(
                project=self.system_config.wandb_project,
                name=self.run_name,
                config={
                    "model": asdict(self.model_config),
                    "data": asdict(self.data_config), 
                    "training": asdict(self.training_config),
                    "system": asdict(self.system_config)
                }
            )
    
    def check_environment(self) -> Dict[str, Any]:
        """実行環境チェック"""
        self.logger.info("=== 実行環境チェック ===")
        
        env_info = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
            "python_version": sys.version,
        }
        
        if env_info["cuda_available"]:
            for i in range(env_info["gpu_count"]):
                gpu_props = torch.cuda.get_device_properties(i)
                env_info[f"gpu_{i}"] = {
                    "name": gpu_props.name,
                    "memory": gpu_props.total_memory / 1e9,
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                }
            
            self.logger.info(f"✅ CUDA利用可能 - {env_info['gpu_count']}GPU")
            for i in range(env_info["gpu_count"]):
                gpu_info = env_info[f"gpu_{i}"]
                self.logger.info(f"   GPU{i}: {gpu_info['name']} ({gpu_info['memory']:.1f}GB)")
        else:
            self.logger.error("❌ CUDAが利用できません")
            
        # MS-Swift確認
        try:
            import swift
            env_info["swift_version"] = swift.__version__
            self.logger.info(f"✅ MS-Swift: {swift.__version__}")
        except ImportError:
            self.logger.error("❌ MS-Swiftがインストールされていません")
            env_info["swift_version"] = None
        
        return env_info
    
    def create_cpt_jsonl(self):
        """CPTデータ準備（改良版）"""
        self.logger.info("📊 CPTデータ準備開始")
        
        output_path = os.path.join(self.data_config.output_dir, self.data_config.output_filename)
        
        try:
            self.logger.info(f"データセット '{self.data_config.dataset_name}' をロード中...")
            ds = load_dataset(self.data_config.dataset_name, split="train")
            self.logger.info(f"ロード完了。総エントリ数: {len(ds):,}")
            
            written = 0
            skipped = 0
            
            with open(output_path, "w", encoding="utf-8") as f_out:
                for i, row in enumerate(ds):
                    if i % 10000 == 0 and i > 0:
                        self.logger.info(f"処理進捗: {i:,}/{len(ds):,} ({i/len(ds)*100:.1f}%)")
                    
                    text = row.get(self.data_config.text_column, "")
                    if not text or not text.strip():
                        skipped += 1
                        continue
                    
                    # 長すぎるテキストをフィルタリング
                    if len(text) > 100000:  # 10万文字以上は除外
                        skipped += 1
                        continue
                    
                    if self.data_config.data_format == "cpt":
                        # Continuous Pre-Training形式
                        formatted = {
                            "messages": [
                                {"role": "assistant", "content": text.strip()}
                            ]
                        }
                    else:
                        # Chat形式
                        formatted = {
                            "messages": [
                                {"role": "user", "content": "以下のテキストを要約してください。"},
                                {"role": "assistant", "content": text.strip()}
                            ]
                        }
                    
                    f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                    written += 1
            
            self.logger.info(f"✅ データ準備完了")
            self.logger.info(f"   書き出し: {written:,}件")
            self.logger.info(f"   スキップ: {skipped:,}件")
            self.logger.info(f"   出力: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ データ準備エラー: {e}")
            raise
    
    def convert_to_mcore(self) -> str:
        """HuggingFaceモデルをMegatron-Core形式に変換"""
        self.logger.info("🔄 HF → Megatron-Core変換開始")
        
        model_name = self.get_safe_model_name()
        mcore_dir = f"{model_name}-mcore"
        
        cmd = [
            "swift", "export",
            "--model", self.model_config.model_id,
            "--to_mcore", "true",
            "--torch_dtype", self.model_config.torch_dtype,
            "--output_dir", mcore_dir
        ]
        
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        
        try:
            self.logger.info(f"実行コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            self.logger.info("✅ Megatron-Core変換完了")
            self.logger.info(f"   出力: {mcore_dir}")
            return mcore_dir
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ 変換エラー: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            raise
    
    def optimize_training_config(self, env_info: Dict[str, Any]):
        """環境に応じた学習設定最適化"""
        gpu_count = env_info.get("gpu_count", 1)
        
        # GPU数に応じた最適化
        if gpu_count >= 8:
            self.training_config.tensor_model_parallel_size = 4
            self.training_config.pipeline_model_parallel_size = 2
            self.training_config.micro_batch_size = 4
            self.training_config.global_batch_size = 1024
            self.logger.info("🚀 8GPU以上用設定適用")
        elif gpu_count >= 4:
            self.training_config.tensor_model_parallel_size = 2
            self.training_config.pipeline_model_parallel_size = 2
            self.training_config.micro_batch_size = 2
            self.training_config.global_batch_size = 512
            self.logger.info("⚡ 4-7GPU用設定適用")
        elif gpu_count >= 2:
            self.training_config.tensor_model_parallel_size = 2
            self.training_config.pipeline_model_parallel_size = 1
            self.training_config.micro_batch_size = 1
            self.training_config.global_batch_size = 256
            self.logger.info("⚙️ 2-3GPU用設定適用")
        else:
            self.training_config.tensor_model_parallel_size = 1
            self.training_config.pipeline_model_parallel_size = 1
            self.training_config.micro_batch_size = 1
            self.training_config.global_batch_size = 128
            self.logger.info("💧 1GPU用設定適用")
        
        # VRAM総量に応じた調整
        total_vram = sum(env_info.get(f"gpu_{i}", {}).get("memory", 0) 
                        for i in range(gpu_count))
        
        if total_vram < 48:  # 48GB未満
            self.data_config.max_length = 16384
            self.training_config.recompute_granularity = "selective"
            self.logger.info("💧 VRAM節約設定適用")
    
    def run_megatron_training(self, mcore_model_dir: str):
        """Megatron-Core学習実行"""
        self.logger.info("🎯 Megatron-Core学習開始")
        
        data_path = os.path.join(self.data_config.output_dir, self.data_config.output_filename)
        checkpoint_dir = f"{self.system_config.checkpoint_dir}/{self.run_name}"
        
        cmd = [
            "megatron", "pt",
            "--load", mcore_model_dir,
            "--dataset", data_path,
            
            # 分散設定
            "--tensor_model_parallel_size", str(self.training_config.tensor_model_parallel_size),
            "--pipeline_model_parallel_size", str(self.training_config.pipeline_model_parallel_size),
            "--context_parallel_size", str(self.training_config.context_parallel_size),
            "--sequence_parallel", str(self.training_config.sequence_parallel).lower(),
            
            # バッチ設定
            "--micro_batch_size", str(self.training_config.micro_batch_size),
            "--global_batch_size", str(self.training_config.global_batch_size),
            
            # 最適化設定
            "--recompute_granularity", self.training_config.recompute_granularity,
            "--recompute_method", self.training_config.recompute_method,
            "--recompute_num_layers", str(self.training_config.recompute_num_layers),
            
            # 学習パラメータ
            "--train_iters", str(self.training_config.train_iters),
            "--finetune", str(self.training_config.finetune).lower(),
            "--cross_entropy_loss_fusion", str(self.training_config.cross_entropy_loss_fusion).lower(),
            "--lr", str(self.training_config.learning_rate),
            "--lr_warmup_iters", str(self.training_config.warmup_iters),
            "--min_lr", str(self.training_config.min_lr),
            
            # 保存・ログ設定
            "--save", checkpoint_dir,
            "--save_interval", str(self.training_config.save_interval),
            "--max_length", str(self.data_config.max_length),
            "--truncation_strategy", self.data_config.truncation_strategy,
            "--num_workers", str(self.training_config.num_workers),
            "--no_save_optim", str(self.training_config.no_save_optim).lower(),
            "--no_save_rng", str(self.training_config.no_save_rng).lower(),
            "--dataset_num_proc", str(self.data_config.num_proc),
            "--packing", str(self.data_config.packing).lower(),
            "--streaming", str(self.data_config.streaming).lower(),
            "--use_flash_attn", str(self.model_config.use_flash_attention).lower(),
            
            # モニタリング
            "--log_interval", str(self.training_config.log_interval),
        ]
        
        # Weights & Biases設定
        if self.system_config.use_wandb:
            cmd.extend([
                "--wandb_project", self.system_config.wandb_project,
                "--wandb_exp_name", self.run_name
            ])
        
        # 環境変数設定
        env = os.environ.copy()
        env.update({
            "CUDA_DEVICE_MAX_CONNECTIONS": str(self.system_config.cuda_device_max_connections),
            "CUDA_VISIBLE_DEVICES": self.system_config.cuda_visible_devices,
            "NPROC_PER_NODE": str(self.system_config.nproc_per_node)
        })
        
        try:
            self.logger.info(f"実行コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, check=True, text=True)
            self.logger.info("✅ Megatron-Core学習完了")
            return checkpoint_dir
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ 学習エラー: {e}")
            raise
    
    def convert_to_hf(self, checkpoint_path: str) -> str:
        """Megatron-CoreモデルをHuggingFace形式に変換"""
        self.logger.info("🔄 Megatron-Core → HF変換開始")
        
        converted_path = f"{self.system_config.converted_model_dir}/{self.run_name}"
        
        cmd = [
            "swift", "export",
            "--mcore_model", checkpoint_path,
            "--output_dir", converted_path,
            "--to_hf", "true",
            "--torch_dtype", self.model_config.torch_dtype
        ]
        
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        
        try:
            self.logger.info(f"実行コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            self.logger.info("✅ HuggingFace変換完了")
            self.logger.info(f"   出力: {converted_path}")
            return converted_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ 変換エラー: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            raise
    
    def create_inference_script(self, model_path: str):
        """推論スクリプト生成"""
        inference_script = f'''#!/usr/bin/env python3
# inference_{self.run_name}.py - 推論スクリプト
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import readline

model_path = "{model_path}"

print("🤖 モデル読み込み中...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.{self.model_config.torch_dtype},
    trust_remote_code=True
).eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# 停止条件設定
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
class EndOfAssistant(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == im_end_id

stop_list = StoppingCriteriaList([EndOfAssistant()])
system_prompt = "You are a helpful assistant."

print("✅ モデル読み込み完了")

while True:
    q = input("\\n プロンプト > ").strip()
    if q.lower() in {{"exit", "quit"}}:
        break

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": q}}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_new_tokens=256,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
            stopping_criteria=stop_list
        )

    gen_tokens = output[0, inputs.input_ids.shape[-1]:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).split("<|im_end|>")[0].strip()

    print("\\n 応答:\\n" + reply)
'''
        
        script_path = f"inference_{self.run_name}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(inference_script)
        
        os.chmod(script_path, 0o755)  # 実行権限付与
        self.logger.info(f"✅ 推論スクリプト生成: {script_path}")
        return script_path
    
    def save_config(self):
        """設定保存"""
        config_dict = {
            "model": asdict(self.model_config),
            "data": asdict(self.data_config),
            "training": asdict(self.training_config),
            "system": asdict(self.system_config),
            "run_info": {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        config_path = f"config_{self.run_name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"✅ 設定保存: {config_path}")
    
    def run_full_pipeline(self):
        """フルパイプライン実行"""
        self.logger.info("🚀 フルスペック ファインチューニングパイプライン開始")
        
        try:
            # 1. 環境チェック
            env_info = self.check_environment()
            
            # 2. 設定最適化
            self.optimize_training_config(env_info)
            
            # 3. データ準備
            self.create_cpt_jsonl()
            
            # 4. モデル変換（HF → Megatron-Core）
            mcore_model_dir = self.convert_to_mcore()
            
            # 5. Megatron-Core学習
            checkpoint_dir = self.run_megatron_training(mcore_model_dir)
            
            # 6. 最新チェックポイント取得
            checkpoint_files = list(Path(checkpoint_dir).glob("**/pytorch_model.bin"))
            if not checkpoint_files:
                checkpoint_files = list(Path(checkpoint_dir).glob("**/model_optim_rng.pt"))
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                checkpoint_path = str(latest_checkpoint.parent)
            else:
                raise FileNotFoundError("チェックポイントが見つかりません")
            
            # 7. モデル変換（Megatron-Core → HF）
            converted_model_path = self.convert_to_hf(checkpoint_path)
            
            # 8. 推論スクリプト生成
            inference_script = self.create_inference_script(converted_model_path)
            
            # 9. 設定保存
            self.save_config()
            
            # 完了報告
            self.logger.info("🎉 フルスペックファインチューニング完了！")
            self.logger.info(f"📁 変換済みモデル: {converted_model_path}")
            self.logger.info(f"🔧 推論スクリプト: {inference_script}")
            self.logger.info(f"📋 設定ファイル: config_{self.run_name}.yaml")
            
            if self.system_config.use_wandb:
                wandb.finish()
            
            return {
                "model_path": converted_model_path,
                "inference_script": inference_script,
                "run_name": self.run_name
            }
            
        except Exception as e:
            self.logger.error(f"❌ パイプラインエラー: {e}")
            if self.system_config.use_wandb:
                wandb.finish(exit_code=1)
            raise

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="MS-Swift + Megatron-Core フルスペック版ファインチューニング")
    parser.add_argument("--config", type=str, help="設定ファイルパス (.json or .yaml)")
    parser.add_argument("--model", type=str, help="モデルID (例: Qwen/Qwen3-8B-Base)")
    parser.add_argument("--dataset", type=str, help="データセット名")
    parser.add_argument("--no-wandb", action="store_true", help="Weights & Biasesを無効化")
    
    args = parser.parse_args()
    
    print("🚀 MS-Swift + Megatron-Core フルスペック版ファインチューニング")
    print("=" * 80)
    
    # ファインチューニング実行
    tuner = MegatronSwiftFineTuner(args.config)
    
    # CLI引数で設定上書き
    if args.model:
        tuner.model_config.model_id = args.model
    if args.dataset:
        tuner.data_config.dataset_name = args.dataset
    if args.no_wandb:
        tuner.system_config.use_wandb = False
    
    result = tuner.run_full_pipeline()
    
    print("\\n🎉 すべて完了！")
    print("\\n次の手順:")
    print(f"1. 推論テスト: python3 {result['inference_script']}")
    print("2. RAGシステムに統合")
    print("3. 性能評価・ベンチマーク実行")

if __name__ == "__main__":
    main()