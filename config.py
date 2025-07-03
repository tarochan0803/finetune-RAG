# config.py (最適化設定反映版)

import os
import logging
import sys
import torch # データ型指定のために追加

# Disable ChromaDB telemetry to prevent capture() error
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "True"

# Additional telemetry suppression
try:
    import chromadb
    # Monkey patch to suppress telemetry errors
    original_capture = None
    if hasattr(chromadb, 'telemetry'):
        telemetry_module = chromadb.telemetry
        if hasattr(telemetry_module, 'capture'):
            original_capture = telemetry_module.capture
            telemetry_module.capture = lambda *_args, **_kwargs: None
except ImportError:
    pass

class Config:
    """アプリケーション設定クラス"""
    def __init__(self):
        # --- モデル設定 ---
        # 中間回答生成用 ローカルモデル
        self.base_model_name = "Qwen/Qwen1.5-1.8B"
        # ★★★ LoRA アダプターのパスは環境に合わせて設定 ★★★
        self.lora_adapter_path = "./tourokuten_finetune_model_full"  # Qwen3ファインチューニング済みモデル

        # <<< NEW: 最適化関連フラグ >>>
        self.use_4bit_quant = False               # 4bit量子化を使用するかどうか (互換性のため無効)
        self.quant_compute_dtype = torch.float32  # 量子化計算時のデータ型 
        self.model_load_dtype = torch.float32     # モデルロード時のデータ型 (CPU用)
        self.use_flash_attention_2 = False        # Flash Attention 2 を使用試行するかどうか (対応環境の場合)
        self.force_cpu = True                     # CPU強制使用フラグ (CUDA互換性問題のため)

        # <<< NEW: 最終統合用 外部APIモデル設定 >>>
        self.synthesizer_api = "gemini" # 使用するAPI ('gemini')
        self.synthesizer_model_name = "gemini-1.5-pro-latest" # Gemini のモデル名
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") # 環境変数から読み込む
        
        # 自動入力用Gemini設定
        self.auto_fill_use_gemini = False  # 自動入力でGemini APIを使用
        self.auto_fill_model = "gemini-1.5-flash-latest"  # 自動入力用モデル（高速・高精度）

        # 埋め込みモデル (RAG用) - 768次元対応
        self.embeddings_model = "intfloat/multilingual-e5-base"

        # --- RAG設定 ---
        self.chunk_size = 800
        self.chunk_overlap = 100

        # --- 検索精度向上用設定 ---
        self.enable_query_expansion = True  # クエリ拡張を有効化
        self.enable_reranking = True        # リランキングを有効化
        self.rerank_weight = 0.3            # リランキング重み
        self.query_expansion_dict = {
            # 企業・工務店関連(最重要)
            "会社": ["企業", "法人", "業者", "店舗", "事業者", "工務店", "株式会社", "有限会社"],
            "工務店": ["会社", "企業", "建設会社", "建築会社", "施工会社", "業者"],
            "建設": ["建築", "工事", "施工", "土木", "工務店"],
            "登録店": ["登録会社", "加盟店", "取引先", "提携先"],
            # 仕様・条件関連
            "条件": ["要件", "規定", "仕様", "基準", "規格"],
            "設計": ["プランニング", "計画", "図面", "構造"],
            "材料": ["部材", "資材", "素材", "製品"],
            "登録": ["認定", "承認", "許可", "資格"]
        }
        # RAGapp.py の詳細表示で使うメタデータカラム名リスト(企業名優先)
        self.metadata_display_columns = ["company", "source", "type", "major_item", "middle_item", "small_item"]
        
        # 企業名検索強化設定
        self.company_boost_factor = 2.0  # 企業名マッチ時のスコア増強率
        self.company_keywords = ["株式会社", "有限会社", "合同会社", "協同組合", "工務店", "建設"]

        # --- データとDB設定 ---
        self.persist_directory = "./chroma_db"
        self.collection_name = "my_collection"
        # ★★★ CSVファイルのパスは環境に合わせて設定 ★★★
        self.csv_file = "/home/ncnadmin/rag_data3.csv"
        self.required_columns = [
            "company", "conditions", "type", "category",
            "major_item", "middle_item", "small_item", "text"
        ]

        # --- LLM生成パラメータ (中間回答生成用 - 精度向上) ---
        self.max_new_tokens = 256      # トークン数復元
        self.temperature = 0.2         # 少し創造性を追加
        self.top_p = 0.95              # サンプリング範囲拡大
        self.repetition_penalty = 1.1  # 繰り返し制限緩和
        self.do_sample = True          # サンプリング有効化
        self.num_beams = 1             # ビームサーチ維持

        # --- LLM生成パラメータ (最終統合用 API - 変更なし) ---
        self.synthesizer_max_new_tokens = 1024
        self.synthesizer_temperature = 0.4
        self.synthesizer_top_p = 1.0

        # --- アンサンブルRAG用設定 ---
        self.rag_variant_k = [7, 10, 12] # 各バリアントの検索数を増加 (変更)
        # <<< NEW: 並列処理関連設定 >>>
        self.max_parallel_variants = 3 # 並列処理数を増加

        # 中間回答生成用プロンプトテンプレート (変更)
        self.intermediate_prompt_template = """与えられた「参考情報」のみを根拠とし、「質問」に答えてください。
* 参考情報の中に質問の答えとなる情報がある場合は、その情報を簡潔にまとめて回答してください。
* 参考情報の中に質問の答えとなる情報がない場合は、他の知識や推測を一切交えず、必ず「提供された情報からは判断できません。」という一文のみで回答してください。

# 参考情報:
{context}

# 質問:
{question}

# 回答:
"""

        # 最終統合用プロンプトテンプレート (変更なし)
        self.synthesis_prompt_template = """以下の「質問」に対する、3つの異なる情報源からの「回答案」と、それぞれの「根拠となった情報」があります。

# 指示:
1. 各回答案とその根拠、そして元の「質問」を注意深く分析してください。特に、質問が否定形（「〜ではない場合」など）や特定の条件を要求しているか確認してください。
2. 各回答案が、提示された「根拠となった情報」に基づいており、かつ質問の条件（肯定形・否定形を含む）に合致しているか評価してください。
3. 回答案が「判断できません」やエラーを示している場合は、その情報源からは結論が得られなかったと判断してください。
4. **重要:** もし質問が「条件Xではない場合」を問い、かつ有効な回答案の根拠が全て「条件Xの場合」の情報しか含んでいない場合、または回答案が「判断できません」である場合は、『質問の条件（例：条件Xではない場合）に関する直接的な情報は見つかりませんでした。』のように回答してください。根拠情報に基づいていない推測（「いいえ」や「なし」など）は絶対にしないでください。
5. 上記を踏まえ、根拠があり、かつ論理的に矛盾しない有効な回答案があれば、それらを統合して最も確からしい最終回答を生成してください。複数の有効な回答案がない場合は、最も信頼できる単一の回答案を採用してください。最終回答のみを出力してください。

# 質問:
{original_question}

# 回答案 1 (根拠1: {context_1_snippet})
{answer_1}

# 回答案 2 (根拠2: {context_2_snippet})
{answer_2}

# 回答案 3 (根拠3: {context_3_snippet})
{answer_3}

# 最終回答:
"""

        # --- その他 ---
        self.log_level = logging.INFO
        self.hf_token = os.getenv("HF_API_TOKEN") # HuggingFace Hubトークン (オプション)

# --- setup_logging 関数 (変更なし) ---
def setup_logging(config: Config, log_filename: str = "rag_pipeline.log") -> logging.Logger:
    """ロギング設定を行う"""
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    log_level = config.log_level
    logger = logging.getLogger("RAGApp") # アプリケーション共通のロガー名
    logger.setLevel(log_level)

    # ハンドラが既に追加されている場合はクリア（再実行時の重複防止）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(log_format)
    log_to_file = False

    # ファイルハンドラ設定
    try:
        # ログファイル名に注意 (Streamlitアプリ用と分ける場合など)
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_to_file = True
    except Exception as e:
        print(f"Warning: Failed to create file handler for '{log_filename}': {e}", file=sys.stderr)

    # ストリームハンドラ（コンソール出力）設定
    stream_handler = logging.StreamHandler(sys.stdout) # 標準出力へ
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 上位ロガーへの伝播を防ぐ
    logger.propagate = False

    logger.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}. Output to console and file: {log_to_file}")
    return logger