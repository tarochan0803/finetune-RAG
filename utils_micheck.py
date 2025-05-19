# utils_micheck.py (ルールベース抽出導入 + RAG質問削減対応版)
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import os
import sys
import time
import logging
import re # 正規表現
from io import StringIO # メモリ上でCSVを扱う
from typing import Optional, Tuple, List, Dict, Any, Union
import pandas as pd
from io import StringIO
import chardet
import csv # csvモジュールをインポート

# --- 設定ファイルとGeminiライブラリ ---
try:
    from config_micheck import Config
except ImportError as e:
    print(f"Fatal Error: config_micheck.py not found. {e}", file=sys.stderr)
    sys.exit(f"Fatal Error: config_micheck.py not found. {e}")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not found. Some functions may fail.", file=sys.stderr)

try:
    import chardet
except ImportError:
    chardet = None
    print("Warning: chardet library not found. CSV encoding detection will be limited.", file=sys.stderr)

# --- ロガー ---
logger = logging.getLogger("RAGapp_micheck")

# === OCR 関連関数 (変更なし) ===
# (前の回答の ocr_with_gemini, ocr_image_with_gemini, ocr_pdf_with_gemini をここに挿入)
def ocr_with_gemini(file_content: bytes, mime_type: str, config: Config, prompt: str = "この文書の内容を、レイアウトを考慮してできる限り正確にテキストで書き起こしてください。") -> str:
    if not GEMINI_AVAILABLE or not genai: logger.error("Gemini library not available for OCR."); return "[OCR Error: Gemini library not found]"
    if not config.gemini_api_key: logger.error("Gemini API key is not configured for OCR."); return "[OCR Error: Gemini API key not set]"
    try:
        genai.configure(api_key=config.gemini_api_key)
        safety = [{"category":c,"threshold":"BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        gen_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=8192)
        model = genai.GenerativeModel(config.synthesizer_model_name)
        logger.info(f"Performing OCR using Gemini API (mime_type: {mime_type})...")
        start_time = time.time()
        file_part = {"mime_type": mime_type, "data": file_content}
        response = model.generate_content([prompt, file_part], generation_config=gen_config, safety_settings=safety, stream=False)
        ocr_text = response.text.strip() if hasattr(response, 'text') else ""
        duration = time.time() - start_time
        logger.info(f"Gemini OCR completed in {duration:.2f}s. Text length: {len(ocr_text)}")
        logger.debug(f"Waiting {config.api_call_delay_seconds}s after Gemini OCR call...")
        time.sleep(config.api_call_delay_seconds)
        if not ocr_text: logger.warning("Gemini OCR returned empty text."); return "[OCR Warning: Empty result]"
        return ocr_text
    except Exception as e:
        logger.error(f"Error during Gemini OCR API call: {e}", exc_info=True)
        if "API key not valid" in str(e): return "[OCR Error: Invalid API key]"
        elif "ResourceExhausted" in str(e) or ("429" in str(e)): return "[OCR Error: API quota/rate limit exceeded]"
        else: return f"[OCR Error: {type(e).__name__}]"

def ocr_image_with_gemini(image_path: str, config: Config) -> str:
    logger.info(f"Starting OCR for image file: {image_path}")
    try:
        with open(image_path, "rb") as f: image_bytes = f.read()
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif"}
        mime_type = mime_map.get(ext)
        if not mime_type: logger.error(f"Unsupported image type for OCR: {ext}"); return "[OCR Error: Unsupported image type]"
        return ocr_with_gemini(image_bytes, mime_type, config)
    except FileNotFoundError: logger.error(f"Image file not found for OCR: {image_path}"); return "[OCR Error: File not found]"
    except Exception as e: logger.error(f"Error reading image file {image_path}: {e}", exc_info=True); return f"[OCR Error: Cannot read image file ({type(e).__name__})]"

def ocr_pdf_with_gemini(pdf_path: str, config: Config) -> str:
    logger.info(f"Starting OCR for PDF file: {pdf_path}")
    try:
        with open(pdf_path, "rb") as f: pdf_bytes = f.read()
        mime_type = "application/pdf"
        prompt = "このPDF文書の内容を、ページ構造やレイアウト（表形式など）を考慮して、できる限り正確にテキストで書き起こしてください。"
        return ocr_with_gemini(pdf_bytes, mime_type, config, prompt=prompt)
    except FileNotFoundError: logger.error(f"PDF file not found for OCR: {pdf_path}"); return "[OCR Error: File not found]"
    except Exception as e: logger.error(f"Error reading PDF file {pdf_path}: {e}", exc_info=True); return f"[OCR Error: Cannot read PDF file ({type(e).__name__})]"


# === 申し送り書詳細情報抽出 (ルールベース) ===
def extract_moushikomi_details_rule_based(ocr_text: str) -> Dict[str, Any]:
    """
    申し送り書OCRテキストから、固定レイアウト部分（赤枠内など）の情報を
    ルールベース（正規表現）で抽出する。
    """
    details = {
        "wall_spec": None, # 壁仕様 (大壁/真壁)
        "wall_panel_exterior": None, # 壁用面材 耐力壁(外部)
        "wall_panel_interior": None, # 壁用面材 耐力壁(内部)
        "wall_panel_non_bearing": None, # 壁用面材 非耐力壁
        "cladding_supply": None, # 羽柄材の供給
        "base_packing": None, # 基礎パッキン (厚み含む)
        "airtight_packing": None, # 気密パッキン (厚み含む)
        "steel_post": None, # 鋼製束 (有無)
        "truck_size": None, # トラックの大きさ
        # 必要に応じて他の項目を追加
    }
    logger.info("Extracting details from Moushikomi OCR text using rule-based approach...")

    if not isinstance(ocr_text, str) or len(ocr_text) < 10:
        logger.warning("Moushikomi OCR text is invalid or too short for rule-based extraction.")
        return details

    # 壁仕様 (外壁仕様の行から抽出)
    match = re.search(r"外壁仕様\s*[:：]?\s*(大壁|真壁)", ocr_text, re.IGNORECASE)
    if match:
        details["wall_spec"] = match.group(1)
        logger.debug(f"  Rule-based extraction: wall_spec = {details['wall_spec']}")
    else: # 見つからない場合、単独のキーワードも探す (extract_wall_spec_from_text と同様)
        if re.search(r"大壁", ocr_text): details["wall_spec"] = "大壁"
        elif re.search(r"真壁", ocr_text): details["wall_spec"] = "真壁"
        if details["wall_spec"]:
             logger.debug(f"  Rule-based extraction (keyword search): wall_spec = {details['wall_spec']}")


    # 壁用面材 (各行から抽出) - 正規表現は要調整
    # 例: "耐力壁(外部)\s+合板1級\(9mm\)" のようなパターン
    match = re.search(r"耐力壁\(外部\)\s+(.*)", ocr_text, re.IGNORECASE | re.MULTILINE)
    if match: details["wall_panel_exterior"] = match.group(1).strip(); logger.debug(f"  Rule-based extraction: wall_panel_exterior = {details['wall_panel_exterior']}")
    match = re.search(r"耐力壁\(内部\)\s+(.*)", ocr_text, re.IGNORECASE | re.MULTILINE)
    if match: details["wall_panel_interior"] = match.group(1).strip(); logger.debug(f"  Rule-based extraction: wall_panel_interior = {details['wall_panel_interior']}")
    match = re.search(r"非耐力壁\s+(.*)", ocr_text, re.IGNORECASE | re.MULTILINE)
    if match: details["wall_panel_non_bearing"] = match.group(1).strip(); logger.debug(f"  Rule-based extraction: wall_panel_non_bearing = {details['wall_panel_non_bearing']}")

    # 羽柄材の供給
    match = re.search(r"羽柄材の供給\s*[:：]?\s*(.*)", ocr_text, re.IGNORECASE | re.MULTILINE)
    if match: details["cladding_supply"] = match.group(1).strip(); logger.debug(f"  Rule-based extraction: cladding_supply = {details['cladding_supply']}")

    # 副資材 (1行にまとまっていると仮定)
    match = re.search(r"副資材\s*[:：]?\s*(.*)", ocr_text, re.IGNORECASE | re.MULTILINE)
    if match:
        line_content = match.group(1).lower() # 小文字で比較
        if "基礎パッキン" in line_content:
             m_pack = re.search(r"基礎パッキン(\d+mm?)", line_content) # 厚みも抽出試行
             details["base_packing"] = m_pack.group(1) if m_pack else True
             logger.debug(f"  Rule-based extraction: base_packing = {details['base_packing']}")
        if "気密パッキン" in line_content:
             m_pack = re.search(r"気密パッキン(\d+mm?)", line_content)
             details["airtight_packing"] = m_pack.group(1) if m_pack else True
             logger.debug(f"  Rule-based extraction: airtight_packing = {details['airtight_packing']}")
        if "鋼製束" in line_content:
            details["steel_post"] = True
            logger.debug(f"  Rule-based extraction: steel_post = {details['steel_post']}")

    # トラックの大きさ
    match = re.search(r"トラックの大きさ\s*[:：]?\s*(.*)", ocr_text, re.IGNORECASE | re.MULTILINE)
    if match: details["truck_size"] = match.group(1).strip(); logger.debug(f"  Rule-based extraction: truck_size = {details['truck_size']}")


    logger.info(f"Rule-based extraction from Moushikomi finished. Result: {details}")
    return details


# === 見積明細 CSV 処理関数 (dtype指定追加) ===
def read_estimate_csv(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile], config: Config) -> Optional[pd.DataFrame]:
    """
    アップロードされた見積明細CSVファイルを読み込み、DataFrameに変換する関数。
    10行目からデータを読み込み、カラム名はconfigから指定する。
    名称列は文字列として読み込む。
    """
    # (コードは前回の回答と同じだが、pd.read_csv に dtype を追加)
    if not uploaded_file: logger.error("No CSV file provided."); return None
    file_name = uploaded_file.name
    logger.info(f"Reading estimate CSV file: {file_name}")
    df = None
    try:
        csv_bytes = uploaded_file.getvalue()
        if not csv_bytes: logger.error(f"CSV empty: {file_name}"); st.error(f"CSV空: {file_name}"); return None
    except Exception as e: logger.error(f"Read bytes failed: {e}", exc_info=True); st.error(f"読込失敗: {e}"); return None

    # --- 文字コード判定 ---
    detected_encoding = None
    encodings_to_try = ['utf-8-sig', 'cp932', 'utf-8', 'shift_jis', 'euc_jp']
    if chardet:
        try:
            if not csv_bytes.startswith(b'\xef\xbb\xbf'):
                detection_result = chardet.detect(csv_bytes[:20000])
                detected_encoding = detection_result['encoding']
                confidence = detection_result['confidence']
                logger.debug(f"Chardet detected encoding: {detected_encoding} with confidence: {confidence}")
                if detected_encoding and confidence > 0.7:
                    detected_encoding_lower = detected_encoding.lower()
                    encodings_to_try = [enc for enc in encodings_to_try if enc.lower() != detected_encoding_lower]
                    encodings_to_try.insert(1, detected_encoding)
            else:
                 logger.debug("UTF-8 BOM detected.")
        except Exception as char_e: logger.warning(f"Chardet failed: {char_e}")
    logger.debug(f"Encodings to try: {encodings_to_try}")

    # --- デコード試行 ---
    string_data = None
    used_encoding = None
    for encoding in encodings_to_try:
        try:
            logger.debug(f"Attempting decode with: {encoding}")
            string_data = csv_bytes.decode(encoding)
            used_encoding = encoding
            logger.info(f"Successfully decoded with: '{encoding}'.")
            break
        except: continue
    if string_data is None:
        logger.error(f"Could not decode with any tried encodings: {encodings_to_try}")
        st.error(f"デコード失敗。文字コード確認(UTF-8, Shift-JIS等)。")
        return None

    # --- Pandasでの読み込み (skiprows, names, dtype指定) ---
    rows_to_skip = list(range(9)) # 1行目から9行目までスキップ
    expected_columns = config.csv_required_columns
    # ★ dtype={'名称': str} を追加 ★
    dtypes_for_read = {config.csv_material_name_column: str}
    logger.info(f"Attempting to read CSV data starting from row 10 (skipping first {len(rows_to_skip)} rows), applying predefined column names and dtypes.")
    logger.debug(f"Rows to skip: {rows_to_skip}")
    logger.debug(f"Expected columns: {expected_columns}")
    logger.debug(f"Explicit dtypes for read: {dtypes_for_read}")

    try:
        string_io_for_read = StringIO(string_data)
        df = pd.read_csv(
            string_io_for_read,
            header=None,
            names=expected_columns,
            skiprows=rows_to_skip,
            sep=',',
            dtype=dtypes_for_read, # ★ dtype指定を追加
            skipinitialspace=True,
            on_bad_lines='warn'
        )
        logger.info(f"CSV parsed successfully. DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns assigned: {df.columns.tolist()}")
        if df.empty: logger.warning("Parsed DataFrame is empty."); st.error("CSVデータ空？"); return None
        else: logger.debug(f"DataFrame head:\n{df.head().to_string()}")

        actual_cols = df.columns.tolist()
        missing_cols = [col for col in expected_columns if col not in actual_cols]
        if missing_cols:
            error_msg = f"Mismatch expected/actual columns: Missing: {missing_cols}. Actual: {actual_cols}"
            logger.error(error_msg)
            st.error(f"内部エラー: カラム名不一致。")
            return None
        logger.info("Required columns check passed.")

    except Exception as e:
        logger.error(f"Error parsing CSV data using skiprows (Encoding: {used_encoding}, Sep=','): {e}", exc_info=True)
        st.error(f"CSVパースエラー (10行目以降): {e}")
        logger.warning("Check CSV delimiter and structure after row 9.")
        return None

    # --- データ型の変換とクリーニング ---
    # (名称列以外を処理)
    try:
        logger.debug(f"Starting data type conversion and cleaning. Dtypes BEFORE:\n{df.dtypes.to_string()}")
        dtype_map = config.csv_dtype_map

        for col, target_dtype in dtype_map.items():
            # ★ 名称列は dtype で指定済みなのでスキップ ★
            if col == config.csv_material_name_column: continue

            if col in df.columns:
                current_dtype = df[col].dtype
                logger.debug(f"Processing column '{col}' (current dtype: {current_dtype}, target: {target_dtype})...")
                try:
                    if pd.api.types.is_numeric_dtype(target_dtype) or target_dtype in [int, float]:
                        df[col] = df[col].astype(str).replace(['', '-', ' ', '\t', '#DIV/0!', 'N/A', 'None', 'nan', 'NaN'], pd.NA)
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    if df[col].isnull().all(): continue

                    if target_dtype == float and not pd.api.types.is_float_dtype(df[col]):
                        df[col] = df[col].astype(float)
                    elif target_dtype == int:
                        if df[col].isnull().any():
                            try:
                                if hasattr(pd, 'Int64Dtype'): df[col] = df[col].astype('Int64')
                                else: logger.warning(f"Cannot convert '{col}' to Int64..."); df[col] = df[col].astype(float) # fallback
                            except Exception as int_e: logger.warning(f"Could not convert '{col}' to Int64: {int_e}.")
                        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().mod(1).eq(0).all():
                             try: df[col] = df[col].astype(float).astype(int)
                             except Exception as int_conv_e: logger.warning(f"Could not convert '{col}' to standard int: {int_conv_e}.")
                        else: logger.warning(f"Column '{col}' not converted to int.")

                except Exception as e: logger.warning(f"Could not process/convert column '{col}': {e}.", exc_info=False)

        logger.debug(f"DataFrame dtypes AFTER conversion:\n{df.dtypes.to_string()}")
        logger.debug("Stripping whitespace from object type columns.")
        for col in df.select_dtypes(include='object').columns:
            try: df[col] = df[col].str.strip()
            except Exception as strip_e: logger.warning(f"Could not strip '{col}': {strip_e}.")

        logger.info("CSV cleaning/conversion finished.")
        return df
    except Exception as e:
        logger.error(f"Error during CSV cleaning: {e}", exc_info=True)
        st.error(f"CSVデータ処理エラー: {e}")
        return None


# === 詳細ルールチェック関数 (ルールベース抽出情報とRAG結果を統合) ===
def check_detail_rules_with_rag(
    df: pd.DataFrame,
    rag_results: List[Dict[str, Any]],
    moushikomi_details: Dict[str, Any], # ルールベース抽出結果を受け取る
    config: Config
) -> List[str]:
    """
    DataFrame(見積明細)、RAG結果(標準仕様)、申し送り書詳細情報 を照合し、
    ルール違反や不整合点のリストを返す。
    """
    violations = []
    logger.info(f"Checking detail rules against RAG results and Moushikomi rule-based details...")

    if df is None or df.empty:
        logger.warning("DataFrame is None or empty, skipping detail rule check.")
        return ["明細データが見つからないため、詳細ルールチェックをスキップしました。"]

    # --- RAG結果の解析 (標準仕様の特定) ---
    standard_specs = {} # 工務店の標準仕様を格納 (例: {'カットスクリュー': False, '土台仕様': '桧EW E95'})
    rag_errors = False
    for result in rag_results:
        q = result.get('q', '').lower()
        a = result.get('a', '') # 回答は小文字にしない (固有名詞など保持)

        if "[Error:" in a or "[RAG Error:" in a: # RAGエラーを検出
             rag_errors = True
             logger.warning(f"RAG error detected for question '{q}': {a}")
             continue # エラー回答は無視

        # キーワードやパターンで回答を解釈 (より洗練させる必要あり)
        item_match = re.search(r"「(.*?)」|「?(.*?)」?を標準|「?(.*?)」?は標準外", q) # 質問から項目名抽出
        if item_match:
            item_name = next((g for g in item_match.groups() if g is not None), None) # 最初に見つかったグループ
            if item_name:
                item_name = item_name.strip()
                a_lower = a.lower() # 判定用に小文字化
                if "標準で使用しますか" in q or "使いますか" in q:
                    if "はい" in a_lower or "使用します" in a_lower or "標準です" in a_lower: standard_specs[item_name] = {"standard": True, "detail": a}
                    elif "いいえ" in a_lower or "使用しません" in a_lower or "標準ではありません" in a_lower: standard_specs[item_name] = {"standard": False, "detail": a}
                elif "標準外の追加費用扱いとなりますか" in q:
                     if "はい" in a_lower or "標準外です" in a_lower or "追加費用" in a_lower: standard_specs[item_name] = {"standard": False, "is_extra": True, "detail": a} # 標準外フラグ
                     elif "いいえ" in a_lower or "標準です" in a_lower or "標準内" in a_lower: standard_specs[item_name] = {"standard": True, "is_extra": False, "detail": a}
                elif "樹種や等級" in q or "厚みや等級" in q or "種類や仕様" in q or "標準的な仕様" in q:
                     if "関連情報なし" not in a_lower and "判断できません" not in a_lower: standard_specs[item_name] = {"spec_detail": a} # 具体的な仕様詳細
                elif "採用単価はどれ" in q:
                     if "一般・基準単価" in a: standard_specs["採用単価"] = "一般・基準単価"
                     elif "重木店単価" in a: standard_specs["採用単価"] = "重木店単価"

    if rag_errors:
         violations.append("[注意] RAGによる標準仕様確認中にAPIエラーが発生しました。一部のチェックが不完全な可能性があります。")
    logger.info(f"Parsed standard specs from RAG: {standard_specs}")

    # --- 申し送り書情報とRAG情報の比較 ---
    m_wall_spec = moushikomi_details.get("wall_spec")
    r_wall_spec = standard_specs.get("壁仕様") # RAGは壁仕様の質問に答えていない可能性あり
    if m_wall_spec and r_wall_spec and m_wall_spec != r_wall_spec:
         violations.append(f"[申送/RAG不一致] 壁仕様: 申送='{m_wall_spec}', RAG='{r_wall_spec}'")
    # 他のルールベース抽出情報とRAG結果の比較も追加可能

    # --- CSV明細の各行チェック ---
    name_col = config.csv_material_name_column
    w_col = config.csv_spec_width_column
    h_col = config.csv_spec_height_column
    l_col = config.csv_spec_length_column
    # カラム名を安全化
    safe_name_col = re.sub(r'\W|^(?=\d)', '_', name_col)
    safe_w_col = re.sub(r'\W|^(?=\d)', '_', w_col)
    safe_h_col = re.sub(r'\W|^(?=\d)', '_', h_col)
    safe_l_col = re.sub(r'\W|^(?=\d)', '_', l_col)

    checked_required_items_rag = set() # RAGで標準とされた項目のうち、CSVで見つかったもの

    for row_index, row_tuple in enumerate(df.itertuples(index=True, name='EstimateRow')):
        try:
            excel_row_num = row_index + 10
            item_name = str(getattr(row_tuple, safe_name_col, "")).strip()
            if not item_name: continue

            w = pd.to_numeric(getattr(row_tuple, safe_w_col, pd.NA), errors='coerce')
            h = pd.to_numeric(getattr(row_tuple, safe_h_col, pd.NA), errors='coerce')
            l = pd.to_numeric(getattr(row_tuple, safe_l_col, pd.NA), errors='coerce')

            # (1) RAG標準仕様との比較
            for spec_item, spec_info in standard_specs.items():
                is_standard = spec_info.get("standard")
                spec_detail = spec_info.get("spec_detail")

                if spec_item.lower() in item_name.lower():
                    # RAG必須項目の存在確認
                    if is_standard is True:
                        checked_required_items_rag.add(spec_item)

                    # RAG不許可項目のチェック
                    if is_standard is False:
                         violations.append(f"[明細/仕様不一致](L{excel_row_num}): '{item_name}' は標準仕様外(RAG確認)ですが、見積に含まれています。")

                    # RAG仕様詳細との比較 (例: 樹種/等級など - より詳細な実装が必要)
                    if spec_detail:
                         # 例: RAG回答のスペック詳細と item_name が一致するか簡易チェック
                         if spec_detail.split('/')[0].lower() not in item_name.lower(): # 簡易的な比較
                             # violations.append(f"[明細/仕様不一致](L{excel_row_num}): '{item_name}' の仕様がRAG確認結果 ('{spec_detail}') と異なる可能性があります。")
                             logger.debug(f"Potential spec mismatch (L{excel_row_num}): Item='{item_name}', RAG detail='{spec_detail}'")


            # (2) 申し送り書ルールベース情報との比較
            # 壁仕様と壁面材の比較
            current_wall_spec = moushikomi_details.get("wall_spec") # ルールベース抽出結果を使用
            if current_wall_spec and "壁合板" in item_name: # 壁合板の行の場合
                 # 例: 申し送り書の壁面材情報と比較 (moushikomi_details から取得)
                 panel_type_ext = moushikomi_details.get("wall_panel_exterior", "")
                 panel_type_int = moushikomi_details.get("wall_panel_interior", "")
                 panel_type_non = moushikomi_details.get("wall_panel_non_bearing", "")
                 # item_name が申し送り書の指定と合っているかチェック (より詳細なロジックが必要)
                 # if not (panel_type_ext in item_name or panel_type_int in item_name or panel_type_non in item_name):
                 #     violations.append(f"[明細/申送不一致](L{excel_row_num}): 壁面材 '{item_name}' が申し送り書の指定と異なります。")
                 pass # 詳細な比較ロジックをここに追加

            # 副資材の比較
            if moushikomi_details.get("base_packing") and "基礎パッキン" in item_name: pass # OK
            if moushikomi_details.get("airtight_packing") and "気密パッキン" in item_name: pass # OK
            if moushikomi_details.get("steel_post") and "鋼製束" in item_name: pass # OK
            # 逆（申し送り書にあるのにCSVにない）はループ外でチェック

            # (3) 基本的な寸法ルールチェック (変更なし)
            for rule_key, rule_values in config.structural_material_rules.items():
                if rule_key.lower() in item_name.lower():
                    if "max_width" in rule_values and pd.notna(w) and w > rule_values["max_width"]:
                        violations.append(f"[基本ルール違反](L{excel_row_num}): '{item_name}' の巾({w})が最大値({rule_values['max_width']})超過。")
                    if "max_height" in rule_values and pd.notna(h) and h > rule_values["max_height"]:
                         violations.append(f"[基本ルール違反](L{excel_row_num}): '{item_name}' の成({h})が最大値({rule_values['max_height']})超過。")
                    break

        except Exception as row_e:
            error_line = row_index + 10
            logger.error(f"Error processing row {error_line} in detail check: {row_e}", exc_info=True)
            violations.append(f"明細チェックエラー(L{error_line}): 処理中にエラー - {type(row_e).__name__}: {row_e}")
            problematic_item_name = "N/A"
            try: problematic_item_name = getattr(row_tuple, safe_name_col, "N/A")
            except: pass
            logger.error(f"  Error occurred while processing item: '{problematic_item_name}'")

    # --- ループ外でのチェック ---
    # (4) RAG必須項目の最終チェック
    required_items_from_rag = {item for item, spec in standard_specs.items() if spec.get("standard") is True}
    missing_required_rag = required_items_from_rag - checked_required_items_rag
    if missing_required_rag:
        for item in missing_required_rag:
             violations.append(f"[明細/仕様不一致]: 標準仕様の必須項目 '{item}' (RAG確認) が見積明細に見つかりません。")

    # (5) 申し送り書必須項目 (副資材など) のチェック
    csv_items_lower = {str(name).lower() for name in df[name_col].dropna()}
    if moushikomi_details.get("base_packing") and not any("基礎パッキン" in name for name in csv_items_lower):
         violations.append(f"[明細/申送不一致]: 申し送り書記載の「基礎パッキン」が見積明細に見つかりません。")
    if moushikomi_details.get("airtight_packing") and not any("気密パッキン" in name for name in csv_items_lower):
         violations.append(f"[明細/申送不一致]: 申し送り書記載の「気密パッキン」が見積明細に見つかりません。")
    if moushikomi_details.get("steel_post") and not any("鋼製束" in name for name in csv_items_lower):
         violations.append(f"[明細/申送不一致]: 申し送り書記載の「鋼製束」が見積明細に見つかりません。")


    if violations:
        logger.warning(f"Detail rule check finished with {len(violations)} violation(s).")
    else:
        logger.info("Detail rule check passed with no violations.")
    return violations

# === 申し送り書 vs RAG 比較関数 (修正版) ===
def compare_rag_and_moushikomi(
    rag_results: List[Dict[str, Any]],
    moushikomi_details: Dict[str, Any], # ルールベース抽出結果を使用
    config: Config
) -> List[str]:
    """
    RAG結果 と 申し送り書詳細情報 を比較し、不一致点のリストを返す。
    """
    discrepancies: List[str] = []
    logger.info("Comparing RAG results with rule-based Moushikomi details...")

    if not moushikomi_details: # 抽出結果がない場合は比較不可
        logger.warning("Moushikomi details not available for comparison.")
        return discrepancies

    # --- 壁仕様の比較 ---
    m_wall_spec = moushikomi_details.get("wall_spec")
    # RAG結果から壁仕様を再解析 (check_details_rules_with_ragと同様のロジック)
    r_wall_spec = None
    rag_wall_answer = ""
    for r in rag_results:
        if "壁仕様" in r.get('q',''):
            rag_wall_answer = r.get('a','')
            if "[Error" not in rag_wall_answer: # エラー回答は無視
                if "大壁" in rag_wall_answer: r_wall_spec = "大壁"
                elif "真壁" in rag_wall_answer: r_wall_spec = "真壁"
            break # 壁仕様の質問は一つと仮定

    if m_wall_spec and r_wall_spec and m_wall_spec != r_wall_spec:
        discrepancy = (f"壁仕様不一致(申送ルール/RAG): 申送='{m_wall_spec}', "
                       f"RAG判定='{r_wall_spec}' (回答抜粋:'{rag_wall_answer[:30]}...')")
        discrepancies.append(discrepancy)
        logger.warning(discrepancy)
    elif m_wall_spec and not r_wall_spec and rag_wall_answer and "[Error" not in rag_wall_answer \
         and "関連情報なし" not in rag_wall_answer and "判断できません" not in rag_wall_answer:
         logger.debug(f"Wall spec compare: Moushikomi='{m_wall_spec}', RAG answered but couldn't determine spec.")
    # --- 他の項目比較 ---
    # 例: カットスクリュー (RAGで確認した場合)
    r_cutscrew_standard = None
    r_cutscrew_answer = ""
    for r in rag_results:
         q_lower = r.get('q','').lower()
         a_lower = r.get('a','').lower()
         if "カットスクリュー" in q_lower and ("標準ですか" in q_lower or "使いますか" in q_lower):
             r_cutscrew_answer = r.get('a','')
             if "[Error" not in r_cutscrew_answer:
                 if "はい" in a_lower or "使用します" in a_lower or "標準です" in a_lower: r_cutscrew_standard = True
                 elif "いいえ" in a_lower or "使用しません" in a_lower or "標準ではありません" in a_lower: r_cutscrew_standard = False
             break
    # 申し送り書にカットスクリューに関する記載があるかチェック (より詳細な抽出が必要かも)
    m_mentions_cutscrew = "カットスクリュー" in moushikomi_details.get("remarks_text","").lower() # 仮: 備考欄を見るなど

    if r_cutscrew_standard is False and m_mentions_cutscrew:
         discrepancy = "カットスクリュー不一致(申送/RAG): 申し送り書に記載があるが、RAGは標準使用しないと回答。"
         discrepancies.append(discrepancy)
         logger.warning(discrepancy)
    elif r_cutscrew_standard is True and not m_mentions_cutscrew:
         logger.debug("Screw check: RAG suggests standard use, but not mentioned in Moushikomi details.")


    if discrepancies:
        logger.warning(f"Comparison between Moushikomi details and RAG finished with {len(discrepancies)} potential discrepancies.")
    else:
        logger.info("Comparison between Moushikomi details and RAG finished with no major discrepancies found.")
    return discrepancies


# === 壁仕様抽出関数 (ルールベース抽出があるので優先度は低いが残す) ===
def extract_wall_spec_from_text(ocr_text: Optional[str]) -> Optional[str]:
    # (変更なし)
    if not isinstance(ocr_text, str): logger.debug("Wall spec extraction skipped."); return None
    logger.debug("Attempting to extract wall specification (keyword search)...")
    if re.search(r"(?:壁仕様\s*[:：]?\s*)?大壁", ocr_text, re.IGNORECASE): logger.debug("Found '大壁' keyword."); return "大壁"
    if re.search(r"(?:壁仕様\s*[:：]?\s*)?真壁", ocr_text, re.IGNORECASE): logger.debug("Found '真壁' keyword."); return "真壁"
    logger.debug("Wall specification keyword not found.")
    return None

# === その他ユーティリティ (変更なし) ===
def format_document_snippet(text: str, max_length: int = 300) -> str:
    # (変更なし)
    if not isinstance(text, str): return ""
    text_single_line = ' '.join(text.splitlines())
    return text_single_line[:max_length] + "..." if len(text_single_line) > max_length else text_single_line

# === RAG質問生成関数 (configのリストに基づき、ルール抽出項目は除外) ===
def generate_questions_for_spec(store_name: Optional[str], config: Config) -> List[str]:
    """工務店の標準仕様を確認するためのRAG質問リストを生成する (ルール抽出項目除く)"""
    questions = []
    logger.info(f"Generating RAG questions for store specifications (Store Name='{store_name}')...")

    if not store_name or "[Error" in store_name:
        logger.warning("Store name is missing or invalid, cannot generate specific questions.")
        return [] # 工務店名がないと仕様を特定できない

    check_items = config.specification_check_items # configからRAG確認項目リストを取得

    # ★ ルールベースで抽出する項目は質問しないようにする ★
    #    (config.specification_check_items自体を絞り込むのがより直接的)
    items_to_ask_rag = [
        item for item in check_items
        # 例: 壁仕様、壁面材、特定の副資材などはルールベース抽出するので除外
        # if item.get("item_name") not in ["壁仕様", "壁合板", "基礎パッキン", "気密パッキン", "鋼製束"]
        # このフィルタリングは config 側で行う方が管理しやすい
    ]
    logger.debug(f"Items to ask RAG (filtered): {[item['item_name'] for item in items_to_ask_rag]}")


    for item in items_to_ask_rag: # 絞り込んだリストでループ
        item_name = item.get("item_name")
        category = item.get("category")
        if not item_name: continue

        # 質問テンプレート (項目の種類に応じて調整)
        if item.get("ask_standard"):
            q = f"「{store_name}」では「{item_name}」を標準で使用しますか？"
            questions.append(q)
        if item.get("ask_spec"):
            if category == "構造材": q = f"「{store_name}」で標準的に使用される「{item_name}」の樹種や等級を教えてください。"
            elif category == "面材": q = f"「{store_name}」で標準的に使用される「{item_name}」の厚みや等級を教えてください。"
            elif category == "金物": q = f"「{store_name}」で標準的に使用される「{item_name}」の種類や仕様を教えてください。"
            elif category == "その他" and item_name == "採用単価": q = f"「{store_name}」向けの標準的な採用単価はどれ（一般・基準単価、重木店単価など）ですか？"
            else: q = f"「{store_name}」における「{item_name}」の標準的な仕様を教えてください。"
            questions.append(q)
        if item.get("ask_is_extra"):
            q = f"「{store_name}」において、「{item_name}」は標準外の追加費用扱いとなりますか？"
            questions.append(q)

    questions = sorted(list(set(questions)))
    logger.info(f"Generated {len(questions)} questions for RAG.")
    logger.debug(f"Generated questions list (first 10): {questions[:10]}")
    return questions


# === 店名抽出関数 (変更なし) ===
def extract_store_name_with_gemini(ocr_text: str, config: Config) -> Optional[str]:
    # (前の回答の extract_store_name_with_gemini をここに挿入)
    if not GEMINI_AVAILABLE or not genai: logger.error("Cannot extract store name: Gemini library not available."); return "[Error: Gemini not available]"
    if not config.gemini_api_key: logger.error("Cannot extract store name: Gemini API key missing."); return "[Error: API key missing]"
    if not ocr_text or len(ocr_text) < 10: logger.warning("Cannot extract store name: OCR text is too short or empty."); return None
    prompt = f"""以下のテキストから、「施工登録店」または「提出先」として記載されている会社名（株式会社、有限会社、合資会社、合同会社などを含む正式名称）を一つだけ抽出してください。会社名が見つからない場合は「不明」とだけ答えてください。他の説明や前置きは不要です。\n\n# テキスト (先頭部分):\n{ocr_text[:2000]}\n\n# 会社名:"""
    logger.info("Extracting store name using Gemini API...")
    try:
        genai.configure(api_key=config.gemini_api_key)
        gen_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=100)
        safety = [{"category":c,"threshold":"BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        model = genai.GenerativeModel(config.synthesizer_model_name)
        response = model.generate_content(prompt, generation_config=gen_config, safety_settings=safety)
        logger.debug(f"Waiting {config.api_call_delay_seconds}s after Gemini store name extraction call...")
        time.sleep(config.api_call_delay_seconds)
        store_name = response.text.strip() if hasattr(response, 'text') else ""
        logger.debug(f"Gemini raw response for store name: '{store_name}'")
        if store_name and store_name != "不明" and len(store_name) > 2 and "[Error" not in store_name:
            logger.info(f"Successfully extracted store name: '{store_name}'")
            return store_name
        else:
            logger.warning(f"Could not extract a valid store name. Gemini response: '{store_name}'")
            return None
    except Exception as e:
        logger.error(f"Error during Gemini store name extraction API call: {e}", exc_info=True)
        if "API key not valid" in str(e): return "[Error: Invalid Gemini API key for store name extraction]"
        elif "ResourceExhausted" in str(e) or ("429" in str(e)): return "[Error: Gemini API quota/rate limit for store name extraction]"
        else: return f"[Error: Store name extraction failed ({type(e).__name__})]"