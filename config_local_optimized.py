# config_local_optimized.py - ローカルファインチューニングモデル専用設定
# APIコストゼロ・高速・プライバシー保護

import os
import logging
import sys
import torch

class Config:
    """ローカルファインチューニングモデル最適化設定"""
    def __init__(self):
        # --- ローカルファインチューニングモデル設定 ---
        self.base_model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        # ★★★ ファインチューニング完了後、ここを更新 ★★★
        self.lora_adapter_path = "./optimized_rag_model"  # run_finetune.py完了後に生成
        
        # --- モデル最適化設定 ---
        self.use_4bit_quant = True                # 4bit量子化（VRAM節約）
        self.quant_compute_dtype = torch.bfloat16 # 計算精度（安定性重視）
        self.model_load_dtype = torch.bfloat16    # ロード時データ型
        self.use_flash_attention_2 = True         # Flash Attention 2（高速化）
        
        # --- 埋め込みモデル（軽量・高性能） ---
        self.embeddings_model = "intfloat/multilingual-e5-base"
        
        # --- RAG設定 ---
        self.chunk_size = 800
        self.chunk_overlap = 100
        
        # --- 検索精度設定 ---
        self.enable_query_expansion = True   # クエリ拡張有効化
        self.enable_reranking = False        # リランキング（重いので無効）
        self.query_expansion_dict = {
            "会社": ["企業", "法人", "工務店"],
            "条件": ["要件", "規定", "仕様"],
            "壁": ["壁面", "壁材", "外壁", "内壁"],
            "構造": ["フレーム", "骨組み", "躯体"]
        }
        
        # --- データとDB設定 ---
        self.persist_directory = "./chroma_db"
        self.collection_name = "my_collection"
        self.csv_file = "/home/ncnadmin/rag_data3.csv"
        self.required_columns = [
            "company", "conditions", "type", "category",
            "major_item", "middle_item", "small_item", "text"
        ]
        
        # --- ローカルLLM生成パラメータ（高品質・高速化） ---
        self.max_new_tokens = 256           # トークン数（品質重視）
        self.temperature = 0.1              # 低温度（一貫性重視）
        self.top_p = 0.9                    # 適度な多様性
        self.repetition_penalty = 1.15      # 繰り返し抑制
        self.do_sample = False              # 決定論的生成（高速化）
        self.num_beams = 1                  # ビームサーチ無効（高速化）
        
        # --- 外部API設定（緊急時バックアップ用）---
        self.enable_api_fallback = False    # APIフォールバック無効
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")  # 緊急時用
        
        # --- アンサンブルRAG設定 ---
        self.rag_variant_k = [5, 8, 10]    # 検索数バリエーション
        self.max_parallel_variants = 3     # 並列処理数（ローカルなので積極的）
        
        # --- プロンプトテンプレート（ファインチューニング済みモデル用）---
        self.intermediate_prompt_template = """### 指示:
与えられた参考情報を基に、質問に正確に答えてください。参考情報にない内容は推測せず、「情報不足」と回答してください。

### 参考情報:
{context}

### 質問:
{question}

### 応答:
"""

        # --- 統合用プロンプト（ローカルモデル統合版）---
        self.synthesis_prompt_template = """### 指示:
以下の3つの回答案を統合し、最も適切で一貫した最終回答を生成してください。

### 質問:
{original_question}

### 回答案1:
{answer_1}

### 回答案2:
{answer_2}

### 回答案3:
{answer_3}

### 最終統合回答:
"""
        
        # --- UI表示設定 ---
        self.metadata_display_columns = ["source", "type", "major_item", "middle_item", "small_item"]
        
        # --- パフォーマンス設定 ---
        self.pipeline_batch_size = 4       # バッチサイズ（ローカルなので大きめ）
        self.stream_buffer_size = 5        # ストリーミングバッファ
        self.stream_delay = 0.005          # ストリーミング遅延（高速化）
        
        # --- システム設定 ---
        self.log_level = logging.INFO
        self.hf_token = os.getenv("HF_API_TOKEN")

# --- ロギング設定（ローカル専用最適化）---
def setup_logging(config: Config, log_filename: str = "rag_local_optimized.log") -> logging.Logger:
    """ローカル環境用最適化ロギング設定"""
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    logger = logging.getLogger("RAGLocal")
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(config.log_level)
    formatter = logging.Formatter(log_format)
    
    # ファイルハンドラ
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(config.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: ログファイル作成失敗 '{log_filename}': {e}", file=sys.stderr)
    
    # コンソールハンドラ
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(config.log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.propagate = False
    logger.info("ローカル最適化ロギング設定完了")
    
    return logger