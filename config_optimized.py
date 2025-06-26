# config_optimized.py - 最適化完了版
import os
import logging
import torch

class Config:
    def __init__(self):
        # --- 最適化済みモデル ---
        self.base_model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        self.lora_adapter_path = "./rag_optimized_model"  # ファインチューニング済み
        
        # --- 高性能設定 ---
        self.use_4bit_quant = True
        self.quant_compute_dtype = torch.bfloat16
        self.model_load_dtype = torch.bfloat16
        self.use_flash_attention_2 = True
        
        # --- RAG設定 ---
        self.embeddings_model = "intfloat/multilingual-e5-base"
        self.chunk_size = 800
        self.chunk_overlap = 100
        self.rag_variant_k = [5, 8, 10]
        
        # --- 推論最適化 ---
        self.max_new_tokens = 256
        self.temperature = 0.1
        self.top_p = 0.9
        self.repetition_penalty = 1.15
        self.do_sample = False
        self.num_beams = 1
        
        # --- データ設定 ---
        self.persist_directory = "./chroma_db"
        self.collection_name = "my_collection"
        self.csv_file = "/home/ncnadmin/rag_data3.csv"
        self.required_columns = [
            "company", "conditions", "type", "category",
            "major_item", "middle_item", "small_item", "text"
        ]
        
        # --- プロンプト最適化 ---
        self.intermediate_prompt_template = """### 指示:
以下の参考情報を基に、正確で専門的な回答を提供してください。

### 参考情報:
{context}

### 質問:
{question}

### 応答:
"""
        
        self.synthesis_prompt_template = """### 指示:
複数の回答案を統合し、最も正確な最終回答を生成してください。

### 質問:
{original_question}

### 回答案1:
{answer_1}

### 回答案2:
{answer_2}

### 回答案3:
{answer_3}

### 最終回答:
"""
        
        # --- その他設定 ---
        self.max_parallel_variants = 3
        self.pipeline_batch_size = 4
        self.metadata_display_columns = ["source", "type", "major_item", "middle_item", "small_item"]
        self.log_level = logging.INFO
        self.hf_token = os.getenv("HF_API_TOKEN")

def setup_logging(config: Config, log_filename: str = "rag_optimized.log") -> logging.Logger:
    logger = logging.getLogger("RAGOptimized")
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(config.log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(config.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"ログファイル作成失敗: {e}")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(config.log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.propagate = False
    return logger
