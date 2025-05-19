# rag_query_utils_check.py (外部DBロード関数追加版)
# -*- coding: utf-8 -*-

import sys
import traceback
import logging
import os
import time
from typing import Tuple, Dict, Any, Optional, List, Generator

try:
    from config_check import Config
except ImportError as e:
    sys.exit(f"Missing config_check.py: {e}")

# Chroma と Embedding のみをインポート
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    # utils から必要な関数のみインポート
    from utils_check import format_document_snippet
except ImportError as e:
    sys.exit(f"Missing utils_check.py: {e}")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not found. Gemini calls will fail.", file=sys.stderr)

logger = logging.getLogger("RAGapp_check") # Streamlitアプリと同じロガーを使用

# --- Embedding関数初期化 ---
def initialize_embedding(config: Config, logger_instance: logging.Logger) -> Optional[HuggingFaceEmbeddings]:
    """Embedding関数のみを初期化"""
    if not logger_instance: # フォールバックロガー
        logger_instance = logging.getLogger("RAGapp_check_Init_Embed")
        logger_instance.addHandler(logging.StreamHandler(sys.stderr))
        logger_instance.setLevel(logging.INFO)
        logger_instance.warning("Logger not provided to initialize_embedding, using basic logger.")

    logger_instance.info(f"Initializing Embedding function using model: {config.embeddings_model}...")
    embedding = None
    try:
        embedding = HuggingFaceEmbeddings(model_name=config.embeddings_model)
        logger_instance.info(f"Embeddings model '{config.embeddings_model}' loaded successfully.")
    except Exception as emb_e:
        logger_instance.critical(f"CRITICAL ERROR loading Embedding model: {emb_e}", exc_info=True)
        return None
    logger_instance.info("Embedding function initialization finished.")
    return embedding

# --- ★★★ NEW: 外部Chroma DBをロードする関数 ★★★ ---
def load_external_chroma_db(config: Config, embedding_function: HuggingFaceEmbeddings, logger_instance: logging.Logger) -> Optional[Chroma]:
    """永続化されたChromaデータベースをロードする"""
    if config.use_memory_db:
        logger_instance.error("Configuration specifies use_memory_db=True, cannot load external DB.")
        return None
    if not config.persist_directory or not config.collection_name:
        logger_instance.error("External DB loading requires 'persist_directory' and 'collection_name' in config.")
        return None
    if not embedding_function:
        logger_instance.error("Embedding function is required to load Chroma DB.")
        return None

    db_path = config.persist_directory
    collection = config.collection_name
    logger_instance.info(f"Attempting to load persistent Chroma DB from: {db_path}, collection: {collection}")

    if not os.path.isdir(db_path):
        logger_instance.error(f"Chroma DB directory not found at specified path: {db_path}")
        return None

    try:
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=collection
        )
        # DB内のドキュメント数をログに出力（オプションだが有用）
        count = vectordb._collection.count()
        logger_instance.info(f"Successfully loaded persistent Chroma DB. Found {count} documents.")
        return vectordb
    except Exception as e:
        # DB接続に関するエラーはクリティカルな場合が多い
        logger_instance.critical(f"CRITICAL ERROR loading Chroma DB from {db_path} (Collection: {collection}): {e}", exc_info=True)
        return None

# --- Gemini API呼び出し関数 (変更なし) ---
def generate_answer_with_gemini(question: str, context: str, config: Config, logger_instance: logging.Logger) -> str:
    """Gemini APIを使用して、提供されたコンテキストに基づいて質問に回答する"""
    if not context:
        logger_instance.warning("Context is empty for Gemini inference. Returning 'no info'.")
        return "関連情報が見つかりませんでした。"

    prompt = f"""以下の「参考情報」のみに基づいて、「質問」に答えてください。参考情報から答えが明確に分からない場合は、「関連情報なし」とだけ回答してください。

# 参考情報:
{context}

# 質問:
{question}

# 回答:"""

    logger_instance.debug(f"Generating answer via Gemini for question: '{question[:50]}...' using context snippet: '{format_document_snippet(context, 100)}'")
    if not GEMINI_AVAILABLE or not genai:
        logger_instance.error("Gemini library is not available.")
        return "[Error: Gemini library not found]"
    if not config.gemini_api_key:
        logger_instance.error("Gemini API key is not configured.")
        return "[Error: Gemini API key not set]"

    final_answer = "[Error: Gemini call failed]"
    try:
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.synthesizer_model_name)
        gen_config = genai.types.GenerationConfig(
            temperature=config.synthesizer_temperature,
            top_p=config.synthesizer_top_p,
            max_output_tokens=config.synthesizer_max_new_tokens
        )
        safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        response = model.generate_content(prompt, generation_config=gen_config, safety_settings=safety)

        logger_instance.debug(f"Waiting {config.api_call_delay_seconds}s after Gemini API call...")
        time.sleep(config.api_call_delay_seconds)

        final_answer = response.text.strip() if hasattr(response, 'text') else ""
        if not final_answer:
            logger_instance.warning("Gemini API returned an empty answer.")
            final_answer = "関連情報なし" # または "[Error: Empty response from API]"

        logger_instance.debug(f"Gemini generated answer: '{final_answer[:100]}...'")
        return final_answer

    except Exception as e:
        logger_instance.error(f"Error during Gemini API call: {e}", exc_info=True)
        if "API key not valid" in str(e): return "[Error: Invalid Gemini API key]"
        elif "ResourceExhausted" in str(e) or ("429" in str(e)): return "[Error: Gemini API quota exceeded or rate limit hit]" # 429エラーも捕捉
        else: return f"[Error: Gemini API call failed ({type(e).__name__})]"

# --- 単一Variant実行 (検索 -> Gemini推論) (変更なし) ---
def process_single_variant(
    query: str,
    config: Config,
    vectordb: Chroma, # ここに外部DBインスタンスが渡される想定
    embedding_function: HuggingFaceEmbeddings,
    logger_instance: logging.Logger,
    metadata_filter: Optional[dict] = None,
    k_value: int = 5
) -> Tuple[str, str, List[Document]]:
    """単一のRAGプロセス（検索->Gemini推論）を実行"""
    log_prefix = "[SingleVariant]"
    logger_instance.info(f"{log_prefix} Processing query '{query[:50]}...' with k={k_value} against provided DB.")

    retrieved_docs: List[Document] = []
    context: str = ""
    answer: str = "[Error: Variant processing failed]"
    context_snippet: str = "N/A"
    top_docs: List[Document] = []

    try:
        # 1. 検索 (Vector Search)
        if not vectordb: raise ValueError("VectorDB instance is required for search.")
        search_kwargs = {"k": k_value}
        if metadata_filter: search_kwargs["filter"] = metadata_filter

        retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
        logger_instance.debug(f"{log_prefix} Retriever created with k={k_value}")
        docs = retriever.invoke(query)

        seen_content = set()
        unique_docs = []
        for doc in docs:
            page_content = getattr(doc, 'page_content', None)
            if isinstance(page_content, str) and page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(page_content)
        retrieved_docs = unique_docs
        logger_instance.info(f"{log_prefix} Retrieved {len(retrieved_docs)} unique documents from DB.")

        # リランキングはスキップ (configで無効化されている前提)
        top_docs = retrieved_docs

        # 2. コンテキスト作成
        context = "\n\n".join([d.page_content for d in top_docs if hasattr(d, 'page_content') and isinstance(d.page_content, str)])
        if not context:
            context_snippet = "(Context is empty)"
            logger_instance.warning(f"{log_prefix} Context is empty after retrieval from DB.")
        else:
            context_snippet = format_document_snippet(context, 150)
        logger_instance.debug(f"{log_prefix} Context length: {len(context)}, Snippet: '{context_snippet}'")

        # 3. 推論生成 (Gemini API)
        logger_instance.info(f"{log_prefix} Generating answer using Gemini based on retrieved context...")
        answer = generate_answer_with_gemini(query, context, config, logger_instance)

        logger_instance.info(f"{log_prefix} Answer generation finished.")
        logger_instance.debug(f"{log_prefix} Generated Answer: {answer[:100]}...")

    except Exception as e:
        logger_instance.error(f"{log_prefix} Error during variant processing: {e}", exc_info=True)
        answer = f"[Error: Variant failed - {type(e).__name__}]"
        context_snippet = context_snippet if context else "Error during retrieval"
        top_docs = top_docs if retrieved_docs else []

    logger_instance.info(f"{log_prefix} Variant processing finished.")
    return answer, context_snippet, top_docs


# --- 質問応答関数 (単一Variant版) (変更なし) ---
def ask_question_single_variant(
    vectordb: Optional[Chroma], # ここに外部DBインスタンスが渡される想定
    embedding_function: Optional[HuggingFaceEmbeddings],
    config: Config,
    query: str,
    logger_instance: logging.Logger,
    metadata_filter: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    単一のRAG Variantを実行して質問に回答する（検索→Gemini推論）。
    ストリーム形式の戻り値構造を維持する。
    """
    start_time_total = time.time()

    # 入力チェック
    if not all([embedding_function, config, logger_instance]):
        err_msg="System Error: Missing essential components (embedding, config, logger)."
        print(err_msg, file=sys.stderr); return {"result_stream": iter([err_msg]), "source_documents": [], "variant_answers": []}
    if vectordb is None: # ★★★ 外部DBが渡されることが前提 ★★★
        err_msg="System Error: VectorDB instance is None. External DB might not be loaded."
        logger_instance.error(err_msg); return {"result_stream": iter([err_msg]), "source_documents": [], "variant_answers": []}
    if not isinstance(query, str) or not query.strip():
        return {"result_stream": iter(["質問が空です。"]), "source_documents": [], "variant_answers": []}

    # APIチェック (Gemini)
    if not GEMINI_AVAILABLE: err_msg="Error: Gemini library (google-generativeai) is not installed."; logger_instance.error(err_msg); return {"result_stream": iter([err_msg]), "source_documents": [], "variant_answers": []}
    if not config.gemini_api_key: err_msg="Error: GEMINI_API_KEY is not set."; logger_instance.error(err_msg); return {"result_stream": iter([err_msg]), "source_documents": [], "variant_answers": []}

    logger_instance.info(f"Processing query (Single Variant against external DB): '{query[:100]}...'")

    final_answer_text = "[Error: Initialization failed]"
    context_snippet = "N/A"
    source_documents: List[Document] = []

    try:
        logger_instance.info("Starting single variant processing using external DB...")
        answer, context_snippet, source_documents = process_single_variant(
            query=query,
            config=config,
            vectordb=vectordb, # 外部DBインスタンスを使用
            embedding_function=embedding_function,
            logger_instance=logger_instance,
            metadata_filter=metadata_filter,
            k_value=config.rag_k # configからk値を取得
        )
        final_answer_text = answer
        logger_instance.info("Single variant processing finished.")

    except Exception as e:
        logger_instance.error(f"Error during single variant execution: {e}", exc_info=True)
        final_answer_text = f"[Error: QA process failed - {type(e).__name__}]"
        source_documents = []

    # 最終回答をイテレータに変換
    final_answer_stream = iter([final_answer_text])

    time_end_total = time.time()
    logger_instance.info(f"Total query processing time: {time_end_total - start_time_total:.2f} sec.")

    # 戻り値の辞書構造は維持
    return {
        "result_stream": final_answer_stream,
        "source_documents": source_documents,
        "variant_answers": [final_answer_text]
    }