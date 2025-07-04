# utils_gemeni.py (キャッシュ最適化 反映版)

import re
import csv
import traceback
import pandas as pd
import logging
import unicodedata
from functools import lru_cache # キャッシュ用に追加
from typing import List, Dict, Any, Optional # Optional 追加
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # 型ヒント用に追加
from langchain_core.vectorstores import VectorStoreRetriever # 型ヒント用に追加

# --- ロガー取得 ---
# ロガー名を変更
logger = logging.getLogger("RAGapp_gemeni")

# --- CSV読み込み関数 (変更なし) ---
def load_csv(file_path: str, required_columns: List[str], logger: logging.Logger) -> pd.DataFrame:
    """CSVファイルを読み込み、指定された必須カラムを持つDataFrameを返す"""
    logger.info(f"Loading CSV file: {file_path}")
    try:
        # UTF-8-sigでBOM付きファイルに対応、dtype=strで型を統一、欠損値を空文字に
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
            logger.info(f"Pandas loaded {len(df)} rows.")
        except Exception as read_err:
            # Pandasで読めない場合 (例: クォート処理の問題など) フォールバック
            logger.warning(f"Pandas read failed: {read_err}. Retrying with basic csv reader...")
            rows = []
            header = []
            try:
                with open(file_path, "r", encoding="utf-8-sig", newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader) # ヘッダー行読み込み
                    # データ行読み込み
                    raw_data = [row for row in reader]
                if not header:
                    raise ValueError("CSV header is empty.")
                # DataFrame作成と型変換
                df = pd.DataFrame(raw_data, columns=header)
                df = df.fillna("").astype(str) # 欠損値を空文字に、型を文字列に
                logger.info(f"Basic csv reader loaded {len(df)} rows.")
            except Exception as fallback_err:
                logger.error(f"Fallback csv reader also failed: {fallback_err}", exc_info=True)
                raise fallback_err # エラーを再発生させる

        # 必須カラムの存在チェック
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Required columns are missing from CSV: {missing_cols}")

        # 必須カラムのみを選択し、順序を保証
        df = df[required_columns]

        # 'text' カラムの空白文字を正規化 (存在する場合)
        if "text" in df.columns:
            df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

        logger.info(f"CSV processing complete. Returning {len(df)} rows.")
        # インデックスをリセットして返す
        return df.reset_index(drop=True)

    except FileNotFoundError:
        logger.error(f"CSV file not found at path: {file_path}")
        raise # エラーを再発生させる
    except ValueError as ve:
        logger.error(f"Data validation error in CSV: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV processing: {e}")
        logger.debug(traceback.format_exc()) # デバッグ用にトレースバックを出力
        raise

# --- 文字列正規化関数 (変更なし) ---
def normalize_str(text: str) -> str:
    """文字列をNFKC正規化し、小文字化、連続空白を単一スペースに置換"""
    if not isinstance(text, str):
        return "" # 文字列以外は空文字を返す
    try:
        # NFKC正規化 (全角英数などを半角に、環境依存文字を変換)
        normalized_text = unicodedata.normalize('NFKC', text)
        # 小文字化
        lower_text = normalized_text.lower()
        # 連続する空白文字（スペース、タブ、改行など）を単一スペースに置換し、前後の空白を削除
        space_normalized = re.sub(r'\s+', ' ', lower_text).strip()
        return space_normalized
    except Exception as e:
        logger.warning(f"String normalization error for text '{text[:50]}...': {e}")
        return text # エラー時は元のテキストを返す

# --- Document作成関数 (変更なし) ---
def create_documents(df: pd.DataFrame, logger: logging.Logger, source_identifier: str = "csv") -> List[Document]:
    """DataFrameからLangChainのDocumentオブジェクトリストを作成"""
    documents: List[Document] = []
    seen_content_hashes = set() # 重複コンテンツ検出用
    processed_rows = 0
    skipped_empty = 0
    skipped_duplicate = 0
    logger.info(f"Creating documents from {len(df)} DataFrame rows...")

    # メタデータとして含めるカラム (text以外)
    required_meta_cols = [col for col in df.columns if col != 'text']
    # textカラムの存在確認
    if 'text' not in df.columns:
        logger.error("'text' column not found in DataFrame. Cannot create documents.")
        return []

    for idx, row in df.iterrows():
        try:
            text_data = str(row.get("text", "")).strip()
            if not text_data: # textが空ならスキップ
                skipped_empty += 1
                continue

            # メタデータ作成 (各値を正規化)
            metadata: Dict[str, Any] = {
                "source": f"{source_identifier}_row_{idx + 1}" # 行番号ベースのソース情報
            }
            for col in required_meta_cols:
                metadata[col] = normalize_str(str(row.get(col, "")))

            page_content = text_data

            # コンテンツとメタデータに基づいたハッシュで重複チェック
            # メタデータも考慮することで、全く同じテキストでも由来が違えば別ドキュメント扱い
            hash_input = page_content + "|" + "|".join(f"{k}:{v}" for k, v in sorted(metadata.items()))
            content_hash = hash(hash_input)

            if content_hash in seen_content_hashes:
                skipped_duplicate += 1
                continue

            seen_content_hashes.add(content_hash)
            documents.append(Document(page_content=page_content, metadata=metadata))
            processed_rows += 1

        except Exception as e:
            logger.error(f"Error processing DataFrame row {idx + 1}: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"Document creation finished. "
                f"Created: {processed_rows}, Skipped (Empty Text): {skipped_empty}, Skipped (Duplicate): {skipped_duplicate}")

    if not documents:
        logger.warning("No documents were created from the DataFrame.")

    return documents

# --- ドキュメントスニペット表示用関数 (変更なし) ---
def format_document_snippet(content: str, max_length: int = 150) -> str:
    """ドキュメントの内容を短縮して表示用スニペットを作成"""
    if not isinstance(content, str):
        return ""
    # 連続する空白を単一スペースに置換し、前後の空白を削除
    normalized_content = re.sub(r'\s+', ' ', content).strip()
    if len(normalized_content) > max_length:
        return normalized_content[:max_length] + '...'
    else:
        return normalized_content

# --- クエリ前処理関数 (変更なし) ---
def preprocess_query(query: str) -> str:
    """クエリの前処理（現状はstripのみ、必要なら正規化追加）"""
    logger.debug(f"Preprocessing query: '{query}'")
    if not isinstance(query, str):
        return ""
    # 前後の空白削除
    processed = query.strip()
    # 必要であればここで normalize_str(processed) を呼ぶ
    # processed = normalize_str(processed)
    logger.debug(f"Preprocessed query: '{processed}'")
    return processed

# --- クエリ拡張関数 (変更なし) ---
def expand_query(query: str, expansion_dict: Optional[dict]) -> List[str]:
    """クエリ文字列を辞書に基づき基本的なキーワード置換で拡張"""
    norm_query = normalize_str(query) # 正規化してから比較
    expanded_queries = [norm_query] # 元のクエリ(正規化済み)は必ず含める
    logger.debug(f"Original query for expansion: '{norm_query}'")

    if not isinstance(expansion_dict, dict):
        logger.warning("Expansion dictionary is not a valid dict. Skipping expansion.")
        return expanded_queries

    try:
        for key, synonyms in expansion_dict.items():
            # キーが文字列で、正規化クエリ内に存在し、シノニムがリスト形式の場合
            if isinstance(key, str) and key in norm_query and isinstance(synonyms, list):
                for syn in synonyms:
                    if isinstance(syn, str):
                        # キーをシノニムで置換したクエリを作成
                        expanded_query = norm_query.replace(key, syn)
                        # 置換によって変化があった場合のみ追加
                        if expanded_query != norm_query:
                            expanded_queries.append(expanded_query)
    except Exception as e:
        logger.error(f"Error during query expansion: {e}", exc_info=True)

    # 重複を除去して返す
    unique_expanded = list(set(expanded_queries))
    if len(unique_expanded) > 1:
        logger.debug(f"Expanded queries: {unique_expanded}")
    return unique_expanded

# --- リランキング関数 (基本的なキーワードスコアリング - 変更なし) ---
def rerank_documents(query: str, docs: List[Document], embedding_function: Any = None, logger: logging.Logger = None, k: int = 10) -> List[Document]:
    """取得ドキュメントをリランキング（簡易キーワードスコア例）。上位k件を返す"""
    if not docs:
        return []
    if logger is None:
        # ロガー名を変更
        logger = logging.getLogger("RAGapp_gemeni")

    logger.debug(f"Reranking {len(docs)} documents for query: '{query[:50]}...' (using simple keyword scoring)")
    norm_query = normalize_str(query) # クエリを正規化

    scored_docs = []
    for doc in docs:
        text = doc.page_content
        norm_text = normalize_str(text) # ドキュメントテキストも正規化

        # シンプルなキーワードスコアリングロジック
        score_kw = 0.0
        if norm_query in norm_text:
            score_kw = 1.0 + norm_text.count(norm_query) * 0.1 # クエリが含まれていたら基本スコア1、出現回数で加点

        # スコアをメタデータに追加
        doc.metadata['rerank_score'] = score_kw
        scored_docs.append((score_kw, doc))

    # スコアで降順ソート
    reranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)

    # 上位k件のドキュメントのみを抽出して返す
    final_docs = [doc for score, doc in reranked][:k]
    if final_docs and reranked:
        logger.debug(f"Reranked top {len(final_docs)} docs. Highest score: {reranked[0][0]:.4f}")
    elif not final_docs:
         logger.debug("Reranking resulted in empty list (or k=0).")

    return final_docs

# --- セマンティック類似度計算 (ダミー - 変更なし) ---
def calculate_semantic_similarity(query: str, answer: str) -> float:
    """クエリと回答のセマンティック類似度を計算（ダミー実装）"""
    logger.warning("calculate_semantic_similarity function is a dummy.")
    return 0.85 # 例: 固定値

# <<< キャッシュ関数 (変更なし) >>>
@lru_cache(maxsize=1024)
def cached_embedding(text: str, embedding_function: Embeddings) -> List[float]:
    """Embedding計算をキャッシュするラッパー関数"""
    logger.debug(f"Cache miss for embedding: '{text[:50]}...' (Using '{type(embedding_function).__name__}')")
    try:
        return embedding_function.embed_query(text)
    except Exception as e:
        logger.error(f"Error during cached embedding generation: {e}", exc_info=True)
        return []

@lru_cache(maxsize=128)
def cached_retrieval(query: str, k: int, retriever_hash: int, retriever: VectorStoreRetriever) -> List[Document]:
    """VectorStore検索をキャッシュするラッパー関数"""
    logger.debug(f"Cache miss for retrieval: query='{query[:50]}...', k={k}, retriever_hash={retriever_hash}")
    try:
        current_k = retriever.search_kwargs.get('k')
        if current_k != k:
            logger.warning(f"Retriever k ({current_k}) differs from cache key k ({k}). Consider aligning them.")
            # retriever.search_kwargs['k'] = k # 必要に応じて設定変更（副作用に注意）
        return retriever.invoke(query)
    except Exception as e:
        logger.error(f"Error during cached retrieval: {e}", exc_info=True)
        return []

def generate_retriever_hash(retriever: VectorStoreRetriever) -> int:
    """Retrieverの設定に基づいて簡易的なハッシュ値を生成する（キャッシュキー用）"""
    kwargs_items = tuple(sorted(retriever.search_kwargs.items(), key=lambda item: str(item[0])))
    stable_kwargs_items = []
    for key, value in kwargs_items:
        if isinstance(value, dict):
            stable_value = frozenset(sorted(value.items()))
        else:
            stable_value = value
        stable_kwargs_items.append((key, stable_value))

    hash_input = f"{retriever.vectorstore.__class__.__name__}|{retriever.search_type}|{tuple(stable_kwargs_items)}"
    return hash(hash_input)

# --- ここまで utils_gemeni.py ---