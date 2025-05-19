# config_micheck.py (ルールベース抽出対応 + RAG質問削減版)
# -*- coding: utf-8 -*-

import os
import logging
import sys
from typing import Optional, List, Dict, Any, Union

# --- 基本設定 ---
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_CHROMA_DB_PATH = "./chroma_db"
DEFAULT_CHROMA_COLLECTION = "my_collection"
DEFAULT_RAG_K = 15 # RAG検索時の取得ドキュメント数 (少し減らしてみる)
DEFAULT_OCR_DPI = 300
DEFAULT_GEMINI_MAX_TOKENS = 1024
DEFAULT_GEMINI_TEMPERATURE = 0.3 # 回答の多様性を抑える
DEFAULT_GEMINI_TOP_P = 1.0
# ★ APIレート制限対策: 待機時間を増やす (Gemini無料枠は1分あたり15リクエスト程度)
DEFAULT_API_DELAY_SECONDS = 5.0 # 例: 5秒 (60秒 / 15回 ≒ 4秒以上)
DEFAULT_LOG_LEVEL = "DEBUG" # 通常は INFO

class Config:
    """
    RAG Document Check (MiCheck) アプリケーション設定クラス。
    (ルールベース抽出導入 + RAG質問削減版)
    """
    def __init__(self):
        # --- Gemini API 設定 ---
        self.synthesizer_api: str = "gemini"
        self.synthesizer_model_name: str = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY environment variable not set.", file=sys.stderr)

        # --- Embedding モデル設定 ---
        self.embeddings_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

        # --- 外部 Chroma DB 設定 ---
        self.use_memory_db: bool = False
        self.persist_directory: str = os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA_DB_PATH)
        self.collection_name: str = os.getenv("CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION)

        # --- RAG 検索設定 ---
        self.rag_k: int = int(os.getenv("RAG_K", DEFAULT_RAG_K))

        # --- OCR 設定 ---
        self.ocr_language: str = 'jpn'
        self.ocr_dpi: int = int(os.getenv("OCR_DPI", DEFAULT_OCR_DPI))

        # --- LLM 生成パラメータ ---
        self.synthesizer_max_new_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", DEFAULT_GEMINI_MAX_TOKENS))
        self.synthesizer_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", DEFAULT_GEMINI_TEMPERATURE))
        self.synthesizer_top_p: float = float(os.getenv("GEMINI_TOP_P", DEFAULT_GEMINI_TOP_P))

        # --- API レート制限対策 ---
        self.api_call_delay_seconds: float = float(os.getenv("API_DELAY_SECONDS", DEFAULT_API_DELAY_SECONDS))

        # --- ログ設定 ---
        log_level_str = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
        self.log_level = getattr(logging, log_level_str, logging.INFO)
        self.log_filename: str = "rag_pipeline_micheck.log"

        # === 見積明細 CSV チェック設定 (変更なし) ===
        self.csv_required_columns: List[str] = [
            "階", "名称", "巾", "成", "長", "数量", "材積", "単位", "単価", "金額", "備考"
        ]
        self.csv_dtype_map: Dict[str, type] = {
            "巾": float, "成": float, "長": float, "数量": float, "材積": float, "単価": float, "金額": float
        }
        self.csv_material_name_column = "名称"
        self.csv_spec_width_column = "巾"
        self.csv_spec_height_column = "成"
        self.csv_spec_length_column = "長"
        self.csv_quantity_column = "数量"
        self.csv_unit_column = "単位"
        self.csv_unit_price_column = "単価"
        self.csv_amount_column = "金額"
        self.csv_remarks_column = "備考"

        # === 標準仕様確認用 RAG 設定 ===
        # RAGに問い合わせる項目リスト (ルールベースで抽出するものを除外)
        # ★★★ このリストは実際の運用に合わせて見直し・拡充してください ★★★
        self.specification_check_items: List[Dict[str, Any]] = [
            # --- RAGで確認が必要そうな項目例 ---
            {"category": "構造材", "item_name": "土台", "ask_spec": True}, # 樹種・等級などルール化しにくいもの
            {"category": "構造材", "item_name": "大引", "ask_spec": True},
            {"category": "構造材", "item_name": "横架材", "ask_spec": True},
            {"category": "構造材", "item_name": "柱", "ask_spec": True},
            # {"category": "構造材", "item_name": "垂木", "ask_spec": True}, # 例: 米松KDなど固定なら不要かも
            {"category": "金物", "item_name": "ジョイントスクリュー", "ask_standard": True}, # 標準使用するか？
            {"category": "金物", "item_name": "カットスクリュー", "ask_standard": True}, # 標準使用するか？
            {"category": "金物", "item_name": "ホールダウン金物", "ask_standard": True, "ask_spec": True}, # 標準使用するか？種類は？
            {"category": "金物", "item_name": "火打金物", "ask_standard": True}, # 標準使用するか？
            # {"category": "面材", "item_name": "床合板", "ask_spec": True}, # 厚みや等級など (壁仕様と関連なければ聞く)
            # {"category": "面材", "item_name": "壁合板", "ask_spec": True}, # ルールベース抽出するので不要
            {"category": "加工費", "item_name": "UD登梁端部加工", "ask_is_extra": True}, # 標準外か？
            {"category": "加工費", "item_name": "PF溝切加工", "ask_is_extra": True},
            {"category": "加工費", "item_name": "ケラバカット加工", "ask_is_extra": True},
            {"category": "加工費", "item_name": "梁貫通加工", "ask_is_extra": True},
            {"category": "その他", "item_name": "採用単価", "ask_spec": True}, # 一般・基準単価 / 重木店単価 など
            # {"category": "その他", "item_name": "壁仕様", "ask_spec": True}, # ルールベース抽出するので不要
        ]
        # ask_spec: 具体的な仕様（樹種、等級、厚み、種類など）を質問するか
        # ask_standard: その項目自体が標準的に使われるかを質問するか (例: 金物)
        # ask_is_extra: その項目が標準外（追加費用扱い）かを質問するか (例: 加工費)

        # === 詳細項目ルール (基本的な寸法チェックなど) ===
        # (変更なし、必要なら調整)
        self.structural_material_rules: Dict[str, Dict[str, Any]] = {
            "土台": { "max_width": 150, "max_height": 150 },
            "大引": { "max_width": 150, "max_height": 150 },
            "横架材": { "max_width": 150, "max_height": 600 },
            "柱": { "max_width": 150, "max_height": 150 },
        }
        self.metal_fittings_rules: Dict[str, Any] = {
            "required_items": [],
        }
        self.floor_panel_rules: Dict[str, Dict[str, Any]] = {
             "床合板": {"min_thickness_mm": 12, "max_thickness_mm": 30, "thickness_keyword": "mm"},
        }
        # 壁用面材ルールは check_detail_rules_with_rag 内で申し送り書情報と比較するため、
        # ここでの定義はシンプルにするか、基本的なチェックのみにする
        self.wall_panel_rules: Dict[str, Dict[str, Any]] = {
             "壁合板": {"min_thickness_mm": 9, "max_thickness_mm": 15, "thickness_keyword": "mm"},
        }
        self.excluded_processing_items: List[str] = [] # 基本的にRAGで確認


# --- ロギング設定関数 (変更なし) ---
def setup_logging(config: Config) -> logging.Logger:
    logger_name = "RAGapp_micheck"
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): logger.handlers.clear()
    log_level = config.log_level
    logger.setLevel(log_level)
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d %(funcName)s] - %(message)s"
    formatter = logging.Formatter(log_format)
    log_to_file = False
    try:
        log_filename = config.log_filename
        log_dir = os.path.dirname(log_filename)
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir); print(f"Created log dir: {log_dir}", file=sys.stderr)
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setLevel(log_level); file_handler.setFormatter(formatter)
        logger.addHandler(file_handler); log_to_file = True
    except Exception as e: print(f"Warning: Failed to create file handler for '{config.log_filename}': {e}", file=sys.stderr)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level); stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info(f"Logging setup complete for '{logger_name}'. Level: {logging.getLevelName(log_level)}. Log to console: True, Log to file: {log_to_file} ('{config.log_filename if log_to_file else 'N/A'}')")
    return logger