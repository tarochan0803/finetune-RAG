# vllm_optimizer.py - vLLMによる推論高速化
# RTX 5070向け最適化推論エンジン

import torch
import os
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import json

# vLLMインポート（インストール確認付き）
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
    print("✅ vLLM利用可能")
except ImportError:
    VLLM_AVAILABLE = False
    print("❌ vLLM未インストール - 標準推論を使用")

@dataclass
class OptimizationConfig:
    """推論最適化設定"""
    model_path: str
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048

class VLLMOptimizer:
    """vLLM推論最適化クラス"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.llm = None
        self.async_engine = None
        self.sampling_params = None
        
    def setup_vllm_engine(self):
        """vLLMエンジン初期化"""
        if not VLLM_AVAILABLE:
            print("❌ vLLMが利用できません")
            return False
            
        try:
            print("🔧 vLLMエンジン初期化中...")
            
            # サンプリングパラメータ
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=256,
                repetition_penalty=1.15,
                stop=["<|endoftext|>", "###"]
            )
            
            # 同期エンジン
            self.llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                trust_remote_code=True,
                enforce_eager=True,  # RTX 5070対応
            )
            
            print("✅ vLLMエンジン初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ vLLMエンジン初期化失敗: {e}")
            return False
    
    def setup_async_engine(self):
        """非同期vLLMエンジン初期化"""
        try:
            print("🔧 非同期vLLMエンジン初期化中...")
            
            engine_args = AsyncEngineArgs(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
                trust_remote_code=True,
                enforce_eager=True,
            )
            
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            print("✅ 非同期vLLMエンジン初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 非同期vLLMエンジン初期化失敗: {e}")
            return False
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """バッチ推論（同期）"""
        if not self.llm:
            print("❌ vLLMエンジンが初期化されていません")
            return []
        
        try:
            start_time = time.time()
            outputs = self.llm.generate(prompts, self.sampling_params)
            end_time = time.time()
            
            results = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                results.append(generated_text)
            
            throughput = len(prompts) / (end_time - start_time)
            print(f"⚡ バッチ推論完了: {len(prompts)}件, {throughput:.2f}件/秒")
            
            return results
            
        except Exception as e:
            print(f"❌ バッチ推論エラー: {e}")
            return []
    
    async def generate_async(self, prompt: str) -> AsyncGenerator[str, None]:
        """非同期ストリーミング推論"""
        if not self.async_engine:
            print("❌ 非同期vLLMエンジンが初期化されていません")
            return
        
        try:
            request_id = f"req_{time.time()}"
            
            # 推論開始
            results_generator = self.async_engine.generate(
                prompt, 
                self.sampling_params, 
                request_id
            )
            
            # ストリーミング出力
            async for request_output in results_generator:
                if request_output.outputs:
                    new_text = request_output.outputs[0].text
                    yield new_text
                    
        except Exception as e:
            print(f"❌ 非同期推論エラー: {e}")
            yield f"エラー: {e}"

class OptimizedRAGInference:
    """最適化RAG推論システム"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.optimizer = None
        self.fallback_model = None
        
    def initialize(self):
        """推論エンジン初期化"""
        print("🚀 最適化RAG推論システム初期化")
        
        # GPU VRAM確認
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU VRAM: {vram_gb:.1f}GB")
            
            # VRAMに応じた設定調整
            if vram_gb >= 12:
                config = OptimizationConfig(
                    model_path=self.model_path,
                    max_model_len=1024,
                    gpu_memory_utilization=0.85,
                    max_num_seqs=8,
                    max_num_batched_tokens=1024
                )
            else:
                config = OptimizationConfig(
                    model_path=self.model_path,
                    max_model_len=512,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=4,
                    max_num_batched_tokens=512
                )
        else:
            print("❌ GPU未検出 - CPU推論は非対応")
            return False
        
        # vLLM初期化試行
        self.optimizer = VLLMOptimizer(config)
        
        if self.optimizer.setup_vllm_engine():
            print("✅ vLLM高速推論エンジン準備完了")
            return True
        else:
            print("⚠️ vLLM初期化失敗 - フォールバック推論を使用")
            return self.setup_fallback()
    
    def setup_fallback(self):
        """フォールバック推論設定"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from peft import PeftModel
            
            print("🔄 フォールバック推論設定中...")
            
            # ベースモデル読み込み
            base_model = AutoModelForCausalLM.from_pretrained(
                "elyza/ELYZA-japanese-Llama-2-7b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRAアダプタ適用
            if os.path.exists(self.model_path):
                model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                model = base_model
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.fallback_model = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            print("✅ フォールバック推論準備完了")
            return True
            
        except Exception as e:
            print(f"❌ フォールバック推論設定失敗: {e}")
            return False
    
    def generate_optimized(self, prompts: List[str]) -> List[str]:
        """最適化推論実行"""
        if self.optimizer and self.optimizer.llm:
            return self.optimizer.generate_batch(prompts)
        elif self.fallback_model:
            results = []
            for prompt in prompts:
                output = self.fallback_model(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False
                )
                results.append(output[0]['generated_text'][len(prompt):].strip())
            return results
        else:
            print("❌ 推論エンジンが利用できません")
            return []
    
    def benchmark_performance(self):
        """性能ベンチマーク"""
        print("📊 性能ベンチマーク実行")
        
        test_prompts = [
            "### 入力:\n株式会社三建の壁仕様について\n### 応答:\n",
            "### 入力:\n仮筋交の標準仕様\n### 応答:\n",
            "### 入力:\n鋼製束の使用基準\n### 応答:\n"
        ] * 5  # 15プロンプト
        
        # ウォームアップ
        _ = self.generate_optimized(test_prompts[:1])
        
        # ベンチマーク実行
        start_time = time.time()
        results = self.generate_optimized(test_prompts)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(test_prompts) / total_time
        avg_latency = total_time / len(test_prompts)
        
        print(f"📈 性能結果:")
        print(f"   処理件数: {len(test_prompts)}")
        print(f"   総処理時間: {total_time:.2f}秒")
        print(f"   スループット: {throughput:.2f}件/秒")
        print(f"   平均レイテンシ: {avg_latency:.3f}秒")
        
        return {
            "throughput": throughput,
            "latency": avg_latency,
            "total_time": total_time
        }

def install_vllm():
    """vLLMインストール"""
    print("📦 vLLMインストール中...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "vllm", "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print("✅ vLLMインストール完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ vLLMインストール失敗: {e}")
        return False

def main():
    print("🔥 推論最適化システム")
    
    # vLLMインストール確認
    if not VLLM_AVAILABLE:
        response = input("vLLMをインストールしますか？ (y/N): ")
        if response.lower() in ['y', 'yes']:
            if install_vllm():
                print("🔄 スクリプトを再実行してください")
                return
    
    # モデルパス確認
    model_candidates = [
        "./rag_model_13b",
        "./rag_finetuned_model", 
        "./rag_model_light",
        "./optimized_rag_model"
    ]
    
    model_path = None
    for path in model_candidates:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ ファインチューニング済みモデルが見つかりません")
        print("先にファインチューニングを実行してください")
        return
    
    print(f"🤖 使用モデル: {model_path}")
    
    # 推論システム初期化
    inference_system = OptimizedRAGInference(model_path)
    
    if inference_system.initialize():
        # 性能ベンチマーク
        benchmark_results = inference_system.benchmark_performance()
        
        print("\n🎉 推論最適化完了！")
        print("次の手順:")
        print("1. RAGアプリでこの最適化システムを使用")
        print("2. 本番環境での性能測定")
        print("3. 必要に応じてさらなる最適化")
    else:
        print("❌ 推論最適化に失敗しました")

if __name__ == "__main__":
    main()