# config_check.py (外部DB参照・ローカルLLM設定削除版)
# -*- coding: utf-8 -*-

import os
import logging
import sys
from typing import Optional, List # List は現状使わないが念のため残す

class Config:
    """アプリケーション設定クラス (外部DB参照版)"""
    def __init__(self):
        # --- Synthesizer (Gemini) 設定 ---
        self.synthesizer_api: str = "gemini" # 現在は gemini 固定
        # 環境変数 GEMINI_MODEL があればそれを使い、なければ gemini-1.5-flash-latest を使う
        self.synthesizer_model_name: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY environment variable not set. Gemini API calls will fail.", file=sys.stderr)

        # --- Embedding モデル設定 ---
        # 環境変数 EMBEDDING_MODEL があればそれを使い、なければ intfloat/multilingual-e5-base を使う
        self.embeddings_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

        # --- データとDB設定 (外部DBを使用) ---
        self.use_memory_db: bool = False # ★★★ 外部DBを使用するため False に設定 ★★★
        # 環境変数 CHROMA_DB_PATH があればそれを使い、なければ ./chroma_db を使う
        self.persist_directory: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        # 環境変数 CHROMA_COLLECTION があればそれを使い、なければ my_collection を使う
        self.collection_name: str = os.getenv("CHROMA_COLLECTION", "my_collection")

        # --- RAG設定 (検索時) ---
        # 環境変数 RAG_K があればそれを使い、なければ 20 を使う (外部DBは情報量が多い可能性を考慮)
        self.rag_k: int = int(os.getenv("RAG_K", "20"))

        # --- チェック機能用設定 ---
        self.ocr_language: str = 'jpn' # Geminiでは通常不要だが念のため
        self.ocr_dpi: int = int(os.getenv("OCR_DPI", "300")) # PDF->Image変換時の解像度

        # --- LLM生成パラメータ (Gemini Synthesizer用) ---
        # 環境変数 GEMINI_MAX_TOKENS があればそれを使い、なければ 1024 を使う
        self.synthesizer_max_new_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
        # 環境変数 GEMINI_TEMPERATURE があればそれを使い、なければ 0.4 を使う
        self.synthesizer_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))
        # 環境変数 GEMINI_TOP_P があればそれを使い、なければ 1.0 を使う
        self.synthesizer_top_p: float = float(os.getenv("GEMINI_TOP_P", "1.0"))

        # --- API 待機時間 ---
        # 環境変数 API_DELAY_SECONDS があればそれを使い、なければ 1.5 を使う
        self.api_call_delay_seconds: float = float(os.getenv("API_DELAY_SECONDS", "1.5"))

        # --- その他 ---
        # 環境変数 LOG_LEVEL があればそれを使い、なければ INFO を使う
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_level = getattr(logging, log_level_str, logging.INFO)

# --- setup_logging 関数 ---
def setup_logging(config: Config, log_filename: str = "rag_pipeline_check.log") -> logging.Logger:
    """ロギング設定を行う"""
    logger_name = "RAGapp_check" # アプリケーション名に合わせる
    logger = logging.getLogger(logger_name)

    # ハンドラが既に追加されている場合はクリア（Streamlit再実行時の重複防止）
    if logger.hasHandlers():
        logger.handlers.clear()

    log_level = config.log_level
    logger.setLevel(log_level)

    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(log_format)

    log_to_file = False
    # ファイルハンドラ設定 (エラーハンドリング強化)
    try:
        log_dir = os.path.dirname(log_filename)
        # ディレクトリが存在しない場合に作成
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}", file=sys.stderr)
            except OSError as e:
                print(f"Warning: Could not create log directory '{log_dir}': {e}", file=sys.stderr)
                log_filename = os.path.basename(log_filename) # カレントディレクトリに変更

        # ファイルハンドラの追加
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_to_file = True
    except Exception as e:
        print(f"Warning: Failed to set up file logging to '{log_filename}': {e}", file=sys.stderr)

    # ストリームハンドラ (標準出力) 設定
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 親ロガーへの伝播を防ぐ (重複ログ防止)
    logger.propagate = False

    logger.info(f"Logging setup complete for '{logger_name}'. Level: {logging.getLevelName(log_level)}. Log to console: True, Log to file: {log_to_file} ('{log_filename if log_to_file else 'N/A'}')")
    return logger