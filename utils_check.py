# utils_check.py (Gemini OCR + 施工店名抽出 + 待機時間追加版)
# -*- coding: utf-8 -*-

import re
import csv
import traceback
import pandas as pd
import logging
import unicodedata
import os
import time # ★ time モジュールをインポート
# import yaml # YAML不要
# import cv2
# import pytesseract
import numpy as np
from pdf2image import convert_from_path # PDF処理には依然として必要
from functools import lru_cache
from typing import List, Dict, Any, Optional

# LangChain 関連
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ★ Gemini API と画像処理ライブラリを追加 ★
import google.generativeai as genai
from PIL import Image # Pillow

# config_check から Config 型をインポート (型ヒント用)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config_check import Config

logger = logging.getLogger("RAGapp_check")

# === 元の utils_gemeni.py 由来の関数群 ===
# (実装は変更なし)
def load_csv(file_path: str, required_columns: List[str], logger_instance: logging.Logger) -> pd.DataFrame:
    """CSVファイルを読み込み"""
    logger_instance.info(f"Loading CSV file: {file_path}")
    try:
        try: df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        except Exception as read_err: logger_instance.warning(f"Pandas read failed: {read_err}. Retrying..."); header, raw_data = [], []
        try:
            with open(file_path, "r", encoding="utf-8-sig", newline='') as f: reader = csv.reader(f); header = next(reader); raw_data = [row for row in reader]
            if not header: raise ValueError("CSV header empty."); df = pd.DataFrame(raw_data, columns=header); df = df.fillna("").astype(str)
        except Exception as fallback_err: logger_instance.error(f"Fallback read failed: {fallback_err}", exc_info=True); raise fallback_err
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols: raise ValueError(f"Missing columns: {missing_cols}")
        df = df[required_columns];
        if "text" in df.columns: df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()
        logger_instance.info(f"CSV processed: {len(df)} rows."); return df.reset_index(drop=True)
    except FileNotFoundError: logger_instance.error(f"CSV not found: {file_path}"); raise
    except ValueError as ve: logger_instance.error(f"CSV validation error: {ve}"); raise
    except Exception as e: logger_instance.error(f"CSV processing error: {e}"); logger_instance.debug(traceback.format_exc()); raise

def normalize_str(text: str) -> str:
    """文字列を正規化"""
    if not isinstance(text, str): return ""
    try: t = unicodedata.normalize('NFKC', text); t = t.lower(); t = re.sub(r'\s+', ' ', t).strip(); return t
    except Exception as e: logger.warning(f"Normalization error: {e}"); return text

def create_documents_from_df(df: pd.DataFrame, logger_instance: logging.Logger, source_identifier: str = "csv") -> List[Document]:
    """DataFrameからDocumentリストを作成"""
    docs: List[Document] = []; hashes = set(); rows=0; skip_e=0; skip_d=0; logger_instance.info(f"Creating docs from DF {len(df)} rows...")
    meta_cols = [c for c in df.columns if c != 'text'];
    if 'text' not in df.columns: logger_instance.error("'text' col missing."); return []
    for idx, row in df.iterrows():
        try:
            txt = str(row.get("text", "")).strip();
            if not txt: skip_e += 1; continue
            meta: Dict[str, Any] = {"source": f"{source_identifier}_row_{idx + 1}"};
            for col in meta_cols: meta[col] = normalize_str(str(row.get(col, "")))
            h_in = txt + "|" + "|".join(f"{k}:{v}" for k, v in sorted(meta.items())); h = hash(h_in)
            if h in hashes: skip_d += 1; continue
            hashes.add(h); docs.append(Document(page_content=txt, metadata=meta)); rows += 1
        except Exception as e: logger_instance.error(f"Error row {idx + 1}: {e}"); logger_instance.debug(traceback.format_exc())
    logger_instance.info(f"Doc creation(DF): Created:{rows}, SkipEmpty:{skip_e}, SkipDup:{skip_d}");
    if not docs: logger_instance.warning("No docs created from DF."); return docs

def format_document_snippet(content: str, max_length: int = 150) -> str:
    """表示用スニペット作成"""
    if not isinstance(content, str): return ""
    t = re.sub(r'\s+', ' ', content).strip(); return t[:max_length] + '...' if len(t) > max_length else t

def preprocess_query(query: str) -> str:
    """クエリ前処理"""
    logger.debug(f"Preprocessing query: '{query}'")
    if not isinstance(query, str): return ""
    p = query.strip(); logger.debug(f"Preprocessed query: '{p}'"); return p

def rerank_documents(query: str, docs: List[Document], embedding_function: Any = None, logger_instance: logging.Logger = None, k: int = 10) -> List[Document]:
    """リランキング（簡易キーワードスコア）"""
    if not docs: return []
    if logger_instance is None: logger_instance = logging.getLogger("RAGapp_check")
    logger_instance.debug(f"Reranking {len(docs)} docs (keyword scoring)...")
    norm_q = normalize_str(query); scored = []
    for idx, doc in enumerate(docs):
        if not hasattr(doc, 'page_content') or not isinstance(doc.page_content, str): logger_instance.warning(f"Skipping doc idx {idx} in rerank."); continue
        txt = doc.page_content; norm_txt = normalize_str(txt); score = 0.0
        if norm_q in norm_txt: score = 1.0 + norm_txt.count(norm_q) * 0.1
        if not hasattr(doc, 'metadata'): doc.metadata = {}
        doc.metadata['rerank_score'] = round(score, 4); scored.append((score, doc))
    reranked = sorted(scored, key=lambda x: x[0], reverse=True); final = [d for s, d in reranked][:k]
    if final and reranked: logger_instance.debug(f"Reranked top {len(final)}. Score range: {reranked[0][0]:.4f} - {reranked[min(k, len(reranked)) - 1][0]:.4f}")
    elif not final: logger_instance.debug("Reranking empty.")
    return final

def calculate_semantic_similarity(query: str, answer: str) -> float: pass # 実装省略 (Dummy)

@lru_cache(maxsize=1024)
def cached_embedding(text: str, embedding_function: Embeddings) -> List[float]:
    """Embedding計算キャッシュ"""
    logger.debug(f"Cache miss/hit embedding: '{text[:50]}...'")
    try:
        if hasattr(embedding_function,'embed_query') and callable(embedding_function.embed_query): return embedding_function.embed_query(text)
        else: logger.error(f"Invalid embedding_function."); return []
    except Exception as e: logger.error(f"Cached embedding error: {e}", exc_info=True); return []

@lru_cache(maxsize=128)
def cached_retrieval(query: str, k: int, retriever_hash: int, retriever: VectorStoreRetriever) -> List[Document]: pass # 実装省略

def generate_retriever_hash(retriever: VectorStoreRetriever) -> int: pass # 実装省略


# === OCR 関数 (Gemini API 版 + 待機時間) ===
API_CALL_DELAY_SECONDS = 1.5 # ★ 待機時間を 1.5 秒に延長 ★

def ocr_image_with_gemini(img_path: str, config: 'Config') -> str:
    """画像ファイルパスから Gemini API を使って OCR テキストを抽出 + 待機時間。"""
    logger.info(f"Performing OCR using Gemini API on image: {img_path}")
    if not genai: return "[OCR Error: Gemini library not installed]"
    if not config.gemini_api_key: return "[OCR Error: Gemini API Key not set]"
    try:
        img = Image.open(img_path)
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.synthesizer_model_name)
        prompt = "この画像から全てのテキストを抽出して、できるだけ元のレイアウトに近い形で書き出してください。"
        contents = [prompt, img]
        safety_settings=[ {"category": c, "threshold": "BLOCK_NONE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(contents, stream=False, safety_settings=safety_settings)
        time.sleep(API_CALL_DELAY_SECONDS); logger.debug(f"Waited {API_CALL_DELAY_SECONDS}s after Gemini OCR call.")
        ocr_text = ""
        if response and hasattr(response, 'text'): ocr_text = response.text.strip()
        elif response and hasattr(response, 'parts') and response.parts: ocr_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
        if not ocr_text:
             if hasattr(response, 'candidates') and response.candidates:
                  candidate_content = getattr(response.candidates[0], 'content', None)
                  if candidate_content and hasattr(candidate_content, 'parts'): ocr_text = "".join(part.text for part in candidate_content.parts if hasattr(part, 'text')).strip()
             if not ocr_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                  feedback = response.prompt_feedback; logger.error(f"Gemini OCR blocked: {feedback}"); return f"[OCR Error: Blocked - {feedback}]"
        if ocr_text: logger.info(f"Gemini OCR successful. Length: {len(ocr_text)}"); return ocr_text
        else: logger.error(f"Gemini OCR failed: No text extracted."); return "[OCR Error: Gemini returned no text]"
    except FileNotFoundError: logger.error(f"Image not found: {img_path}"); return "[OCR Error: Image file not found]"
    except ImportError: logger.error("Pillow not installed."); return "[OCR Error: Pillow library not installed]"
    except Exception as e: logger.error(f"Gemini OCR error for image {img_path}: {e}", exc_info=True); return f"[OCR Error: {type(e).__name__}]"


def ocr_pdf_with_gemini(pdf_path: str, dpi: int, config: 'Config') -> str:
    """PDFからページ画像を抽出し、Gemini API で OCR する + 待機時間。"""
    logger.info(f"Performing OCR using Gemini API on PDF: {pdf_path}")
    all_text = ""
    if not genai: return "[OCR Error: Gemini library not installed]"
    if not config.gemini_api_key: return "[OCR Error: Gemini API Key not set]"
    try:
        logger.debug(f"Converting PDF to images (DPI: {dpi})...")
        pages = convert_from_path(pdf_path, dpi=dpi)
        if not pages: return "[OCR Error: No pages found in PDF via pdf2image]"
        logger.info(f"Converted PDF to {len(pages)} images. OCRing page by page...")
        genai.configure(api_key=config.gemini_api_key); model = genai.GenerativeModel(config.synthesizer_model_name)
        safety_settings=[ {"category": c, "threshold": "BLOCK_NONE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        prompt = "この画像からテキストを抽出してください。 Scan this image and extract all text content."
        for i, page_image in enumerate(pages):
            page_num = i + 1; logger.debug(f"Processing page {page_num}/{len(pages)}...")
            page_text = f"[OCR Error on page {page_num}: Unknown]"
            try:
                contents = [prompt, page_image]
                response = model.generate_content(contents, stream=False, safety_settings=safety_settings)
                time.sleep(API_CALL_DELAY_SECONDS); logger.debug(f"Waited {API_CALL_DELAY_SECONDS}s after Gemini OCR call for page {page_num}.")
                current_page_text = ""
                if response and hasattr(response, 'text'): current_page_text = response.text.strip()
                elif response and hasattr(response, 'parts') and response.parts: current_page_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
                if not current_page_text and hasattr(response, 'candidates') and response.candidates:
                    candidate_content = getattr(response.candidates[0], 'content', None)
                    if candidate_content and hasattr(candidate_content, 'parts'): current_page_text = "".join(part.text for part in candidate_content.parts if hasattr(part, 'text')).strip()
                if not current_page_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    feedback = response.prompt_feedback; logger.error(f"Gemini OCR page {page_num} blocked: {feedback}"); page_text = f"[OCR Error: Blocked - {feedback}]"
                elif current_page_text: logger.debug(f"Page {page_num} OCR completed."); page_text = current_page_text
                else: logger.warning(f"No text on page {page_num}."); page_text = f"[OCR Info: No text detected on page {page_num}]"
            except Exception as page_e: logger.error(f"Gemini OCR error page {page_num}: {page_e}", exc_info=True); page_text = f"[OCR Error on page {page_num}: {type(page_e).__name__}]"
            all_text += page_text + "\n--- Page Break ---\n"
        logger.info(f"Gemini OCR finished for PDF {pdf_path}."); return all_text
    except ImportError: logger.error("pdf2image or Pillow not found."); return "[OCR Error: pdf2image or Pillow not installed]"
    except FileNotFoundError: logger.error("Poppler not found."); return "[OCR Error: Poppler not found]"
    except Exception as e: logger.error(f"Gemini PDF OCR error: {e}", exc_info=True); return f"[OCR Error: {type(e).__name__}]"

def split_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100, source_name: str = "uploaded_file") -> List[Document]:
    """テキストをチャンク化"""
    logger.info(f"Splitting text (size={chunk_size}, overlap={chunk_overlap})...");
    if not text or not isinstance(text, str): return []
    docs = [];
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
        docs = splitter.create_documents([text], metadatas=[{"source": source_name}])
        for i, doc in enumerate(docs):
            if not hasattr(doc, 'metadata'): doc.metadata = {}
            doc.metadata["chunk_index"] = i
        logger.info(f"Split into {len(docs)} chunks.")
        return docs
    except Exception as e: logger.error(f"Splitting error: {e}", exc_info=True); return [Document(page_content=text, metadata={"source": source_name, "error": "splitting_failed"})]

# --- ★施工店名抽出関数 (Gemini API 版 + 待機時間)★ ---
def extract_store_name_with_gemini(ocr_text: str, config: 'Config') -> Optional[str]:
    """OCRテキスト全体から Gemini API を使って施工店名を抽出する。"""
    logger.info("Attempting to extract store name from OCR text using Gemini API...")
    if not ocr_text or not isinstance(ocr_text, str): logger.warning("OCR text empty/invalid."); return None
    if not genai: logger.error("Gemini library missing."); return None
    if not config.gemini_api_key: logger.error("GEMINI_API_KEY missing."); return None
    prompt = f"""以下のテキストはOCRで読み取った文書の内容です。\nこの中から「登録施工店」または「工務店」として記載されている会社名・組織名を一つだけ抽出してください。\n名前の部分だけを回答し、余計な説明は不要です。\nもし見つからない場合は、必ず「不明」とだけ回答してください。\n\n--- OCRテキスト ---\n{ocr_text[:4000]}\n--- ここまで ---\n\n抽出結果:"""
    try:
        genai.configure(api_key=config.gemini_api_key); model = genai.GenerativeModel(config.synthesizer_model_name)
        gen_config = genai.types.GenerationConfig(temperature=0.1); safety_settings=[ {"category": c, "threshold": "BLOCK_NONE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, generation_config=gen_config, safety_settings=safety_settings)
        time.sleep(API_CALL_DELAY_SECONDS); logger.debug(f"Waited {API_CALL_DELAY_SECONDS}s after Gemini Store Name extraction.")
        extracted_name = response.text.strip() if hasattr(response, 'text') else ""
        if extracted_name and extracted_name != "不明":
             cleaned_name = extracted_name # 必要なら前後の"株式会社"などを除去
             logger.info(f"Extracted store name via Gemini: '{cleaned_name}'")
             return cleaned_name
        elif extracted_name == "不明": logger.info("Gemini could not find store name."); return None
        else: logger.warning(f"Gemini unexpected response for store name: '{extracted_name}'"); return None
    except Exception as e: logger.error(f"Gemini store name extraction error: {e}", exc_info=True); return None

# --- 質問生成関数 (ハードコード版) ---
def generate_questions(store_name: Optional[str]) -> List[str]:
    """ハードコードされたテンプレートから質問リストを生成。"""
    store = store_name if store_name and store_name.strip() else "施工店不明"
    logger.info(f"Generating questions for store: '{store}'")
    templates = [
        f"{store}の壁面材仕様で耐力壁について記載はありますか？ 具体的な仕様は何ですか？", f"{store}の壁面材仕様で一般壁について記載はありますか？ 具体的な仕様は何ですか？",
        f"{store}の壁面材仕様でその他の仕様について記載はありますか？", f"{store}の壁面材は大壁ですか？真壁ですか？あるいは記載はありますか？",
        f"{store}の羽柄材の供給について記載はありますか？", f"{store}の金物取付について記載はありますか？",
        f"{store}の副資材として仮筋交いの記載はありますか？", f"{store}の副資材として基礎パッキンの記載はありますか？ 仕様（厚みなど）も分かれば教えてください。",
        f"{store}の副資材として気密パッキンの記載はありますか？ 仕様（厚みなど）も分かれば教えてください。", f"{store}の副資材として鋼製束の記載はありますか？",
        f"{store}の副資材としてカットスクリューの記載はありますか？", f"{store}の副資材として上記以外の記載はありますか？",
        f"{store}に関する特記事項はありますか？ あれば内容を教えてください。", f"{store}に関する設計上の仕様や条件を全て書き出してください。", ]
    logger.info(f"Generated {len(templates)} questions for store '{store}'."); return templates