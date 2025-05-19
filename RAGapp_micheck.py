# RAGapp_micheck.py (ステップ分割対応版)
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import os
import sys
import logging
import time
import tempfile
import traceback
from typing import List, Dict, Any, Optional

# --- モジュールのインポート ---
try:
    # config は最初に読み込む
    from config_micheck import Config, setup_logging
    config = Config() # グローバル設定としてインスタンス化
    logger = setup_logging(config) # ロガーも早期に設定

    from rag_query_utils_micheck import (
        initialize_embedding, load_external_chroma_db, ask_question_single_variant
    )
    from utils_micheck import (
        ocr_image_with_gemini, ocr_pdf_with_gemini,
        read_estimate_csv,
        extract_moushikomi_details_rule_based,
        check_detail_rules_with_rag,
        compare_rag_and_moushikomi,
        generate_questions_for_spec,
        format_document_snippet,
        extract_store_name_with_gemini,
        extract_wall_spec_from_text # 補助的に使用
    )
    from langchain_core.documents import Document

except ImportError as e:
    # Streamlitが起動する前にエラーが発生する可能性があるため、標準エラー出力にも出す
    print(f"Import Error: {e}", file=sys.stderr)
    detailed_error = traceback.format_exc()
    print(detailed_error, file=sys.stderr)
    # Streamlit起動後であれば st.error を使いたいが、ここでは sys.exit する
    sys.exit(f"Import Error: Required module not found. Check installation and paths.\nError: {e}\n{detailed_error}")
except Exception as global_e:
    print(f"Global Setup Error: {global_e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(f"Global Setup Error: {global_e}")


# --- Embedding/DB初期化 (キャッシュ) ---
@st.cache_resource
def load_components_cached():
    """Embedding関数と外部Chroma DBをキャッシュしてロード"""
    logger.info("Initializing components (Embedding and External DB)...")
    embedding_function = initialize_embedding(config, logger)
    if embedding_function is None:
        logger.critical("Embedding initialization failed.")
        st.error("Embeddingモデルの初期化に失敗しました。設定を確認してください。")
        return None, None

    external_db = None
    if not config.use_memory_db:
        external_db = load_external_chroma_db(config, embedding_function, logger)
        if external_db is None:
            logger.critical(f"Failed to load external DB from {config.persist_directory}.")
            st.error(f"外部データベース({config.persist_directory})の読み込みに失敗しました。パスや設定を確認してください。")
            return embedding_function, None # Embeddingは成功している可能性があるので返す
        logger.info("External Chroma DB loaded successfully.")
    else:
        # このアプリでは外部DBが必須なのでエラーとする
        logger.critical("Configuration error: use_memory_db is True, but external DB is required.")
        st.error("設定エラー: このアプリケーションには外部データベースが必要です (use_memory_db=False)。")
        return embedding_function, None

    logger.info("Essential components (Embedding and DB) initialized.")
    return embedding_function, external_db


# --- セッション状態初期化 ---
def initialize_session_state():
    """セッション状態変数を初期化"""
    default_values = {
        # ファイル関連
        "uploaded_moushikomi": None,
        "uploaded_estimate_csv": None,
        "moushikomi_file_name": None,
        "estimate_csv_file_name": None,
        # ステップ管理
        "current_step": 0, # 0: 初期, 1: ファイル読込済, 2: 申送/RAG比較済, 3: 見積チェック済
        # ステップ1の結果
        "moushikomi_ocr_text": None,
        "moushikomi_details": {}, # ルールベース抽出結果
        "store_name": None,
        "estimate_dataframe": None,
        # ステップ2の結果
        "generated_questions": [],
        "rag_qa_results": [],
        "comparison_discrepancies": [], # 申送 vs RAG比較結果
        # ステップ3の結果
        "detail_check_violations": [], # 見積 vs 仕様チェック結果
        # その他
        "error_message": None,
        "processing": False, # 処理中フラグ
        "temp_file_paths": set(), # 一時ファイル管理
    }
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- 一時ファイル管理 ---
def add_temp_file_path(path: str):
    if path:
        st.session_state.temp_file_paths.add(path)
        logger.debug(f"Added temp file path: {path}")

def cleanup_temp_files():
    paths_to_remove = set(st.session_state.temp_file_paths)
    if not paths_to_remove:
        logger.debug("No temporary files to clean up.")
        return
    logger.debug(f"Attempting to clean up {len(paths_to_remove)} temp files: {paths_to_remove}")
    cleaned_paths = set()
    for path in paths_to_remove:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
                logger.info(f"Temp file deleted: {path}")
            cleaned_paths.add(path)
        except Exception as e:
            logger.error(f"Failed to delete temp file {path}: {e}")
            # 削除に失敗してもリストからは消す（再試行しない）
            cleaned_paths.add(path)
    st.session_state.temp_file_paths -= cleaned_paths
    logger.debug(f"Temp files remaining: {st.session_state.temp_file_paths}")


# --- ステップ実行関数 ---

# ステップ1: ファイル読み込みと基本情報抽出
def run_step1_load_and_extract():
    """申し送り書OCR、基本情報抽出、CSV読み込みを実行"""
    st.session_state.processing = True
    st.session_state.error_message = None
    # 前回の結果をクリア（ステップ1関連）
    st.session_state.moushikomi_ocr_text = None
    st.session_state.moushikomi_details = {}
    st.session_state.store_name = None
    st.session_state.estimate_dataframe = None
    st.session_state.current_step = 0 # 初期状態に戻す

    moushikomi_file = st.session_state.get("uploaded_moushikomi")
    estimate_csv_file = st.session_state.get("uploaded_estimate_csv")

    if not moushikomi_file or not estimate_csv_file:
        st.session_state.error_message = "申し送り書と見積明細CSVの両方をアップロードしてください。"
        st.session_state.processing = False
        logger.error("Step 1 Error: Files not uploaded.")
        st.rerun()
        return

    progress_bar = st.progress(0.0, text="ステップ1: ファイル読み込みと基本情報抽出を開始...")
    moushikomi_temp_path = None

    try:
        # 1-1. 申し送り書処理 (OCRとルールベース抽出)
        progress_bar.progress(0.1, text="申し送り書を読み込み中...")
        logger.info(f"Step 1: Processing Moushikomi file: {moushikomi_file.name}")
        st.session_state.moushikomi_file_name = moushikomi_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(moushikomi_file.name)[1]) as tmp_file:
            tmp_file.write(moushikomi_file.getvalue())
            moushikomi_temp_path = tmp_file.name
            add_temp_file_path(moushikomi_temp_path) # 一時ファイルパスを記録
        logger.info(f"Moushikomi temp file created: {moushikomi_temp_path}")

        progress_bar.progress(0.2, text="申し送り書のOCRを実行中 (Gemini)...")
        ocr_text = None
        if moushikomi_file.type == "application/pdf":
            ocr_text = ocr_pdf_with_gemini(moushikomi_temp_path, config)
        elif moushikomi_file.type in ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]:
             ocr_text = ocr_image_with_gemini(moushikomi_temp_path, config)
        else:
            raise ValueError(f"サポートされていない申し送り書のファイル形式です: {moushikomi_file.type}")

        if not ocr_text or "[OCR Error" in ocr_text:
             raise RuntimeError(f"申し送り書のOCRに失敗しました: {ocr_text}")
        st.session_state.moushikomi_ocr_text = ocr_text
        logger.info(f"Moushikomi OCR completed. Text length: {len(ocr_text)}")

        progress_bar.progress(0.4, text="申し送り書から情報を抽出中 (Gemini & Rule)...")
        # 工務店名抽出 (Gemini)
        store_name = extract_store_name_with_gemini(ocr_text, config)
        st.session_state.store_name = store_name if store_name and "[Error" not in store_name else "抽出失敗"
        logger.info(f"Store name extracted: {st.session_state.store_name}")
        if st.session_state.store_name == "抽出失敗":
            st.warning("工務店名の抽出に失敗しました。RAGによる仕様確認の精度が低下する可能性があります。")

        # 詳細情報抽出 (ルールベース)
        moushikomi_details = extract_moushikomi_details_rule_based(ocr_text)
        st.session_state.moushikomi_details = moushikomi_details
        logger.info(f"Rule-based details extracted: {moushikomi_details}")

        progress_bar.progress(0.6, text="見積明細CSVを読み込み中...")
        logger.info(f"Step 1: Processing Estimate CSV file: {estimate_csv_file.name}")
        st.session_state.estimate_csv_file_name = estimate_csv_file.name
        # CSV読み込み (utils_micheckの関数を使用)
        df = read_estimate_csv(estimate_csv_file, config)
        if df is None:
             raise RuntimeError("見積明細CSVの読み込みまたは検証に失敗しました。")
        st.session_state.estimate_dataframe = df
        logger.info(f"Estimate CSV loaded successfully. Shape: {df.shape}")

        progress_bar.progress(1.0, text="ステップ1: 完了！")
        st.session_state.current_step = 1 # ステップ1完了
        logger.info("Step 1 completed successfully.")
        time.sleep(0.5) # 完了メッセージ表示用

    except Exception as e:
        logger.error(f"Error during Step 1: {e}", exc_info=True)
        st.session_state.error_message = f"ステップ1エラー: {e}\n{traceback.format_exc()}"
        st.session_state.current_step = 0 # エラー時はステップを戻す
    finally:
        progress_bar.empty()
        # 一時ファイルは最後（全ステップ完了後かアプリ終了時）に消すので、ここでは消さない
        st.session_state.processing = False
        st.rerun() # 画面更新

# ステップ2: 申し送り書と標準仕様（DB）の整合性確認
def run_step2_check_moushikomi_vs_rag(embedding_function, external_vdb):
    """RAGを実行し、申し送り書の情報と比較"""
    st.session_state.processing = True
    st.session_state.error_message = None
    # 前回の結果をクリア（ステップ2関連）
    st.session_state.generated_questions = []
    st.session_state.rag_qa_results = []
    st.session_state.comparison_discrepancies = []
    st.session_state.current_step = 1 # ステップ1完了状態に戻す

    store_name = st.session_state.get("store_name")
    moushikomi_details = st.session_state.get("moushikomi_details", {})

    if not store_name or store_name == "抽出失敗":
        st.warning("工務店名が不明なため、RAGによる標準仕様確認をスキップします。")
        st.session_state.current_step = 2 # RAGスキップでもステップ2は完了扱い
        st.session_state.processing = False
        st.rerun()
        return
    if not external_vdb:
        st.session_state.error_message = "外部データベースがロードされていません。RAGを実行できません。"
        st.session_state.processing = False
        logger.error("Step 2 Error: External VDB not loaded.")
        st.rerun()
        return

    progress_bar = st.progress(0.0, text="ステップ2: RAGによる標準仕様確認を開始...")

    try:
        # 2-1. RAG質問生成
        progress_bar.progress(0.1, text="標準仕様確認のための質問を生成中...")
        questions = generate_questions_for_spec(store_name, config)
        st.session_state.generated_questions = questions
        num_q = len(questions)
        logger.info(f"Generated {num_q} questions for RAG (Store: {store_name}).")

        # 2-2. RAG実行
        temp_qa_results = []
        if questions:
            progress_bar.progress(0.2, text=f"RAGを実行中 (0/{num_q})...")
            for i, q in enumerate(questions):
                q_num = i + 1
                logger.info(f"Running RAG QA {q_num}/{num_q}: {q}")
                # 進捗計算 (0.2から0.8までを使用)
                progress_value = 0.2 + (0.6 * (q_num / num_q))
                progress_bar.progress(progress_value, text=f"RAGを実行中 ({q_num}/{num_q})")
                try:
                    # ask_question_single_variant を使用
                    response_dict = ask_question_single_variant(
                        external_vdb, embedding_function, config, q, logger
                    )
                    # result_stream はイテレータなのでリスト化して結合
                    answer = "".join(list(response_dict.get("result_stream", iter(["[Error: No stream]"]))))
                    sources = response_dict.get("source_documents", [])
                    temp_qa_results.append({"q": q, "a": answer, "sources": sources})
                except Exception as rag_e:
                    logger.error(f"Error during RAG QA for question '{q}': {rag_e}", exc_info=True)
                    temp_qa_results.append({"q": q, "a": f"[RAG Error: {type(rag_e).__name__}]", "sources": []})
            st.session_state.rag_qa_results = temp_qa_results
            logger.info("RAG QA finished.")
        else:
            logger.warning("No questions generated for RAG. Skipping RAG execution.")
            progress_bar.progress(0.8, text="RAG質問がないためスキップしました。")

        # 2-3. 申し送り書(ルール抽出結果) vs RAG比較
        progress_bar.progress(0.9, text="申し送り書の情報とRAGの結果を比較中...")
        discrepancies = compare_rag_and_moushikomi(
            st.session_state.rag_qa_results,
            moushikomi_details, # ステップ1のルールベース抽出結果
            config
        )
        st.session_state.comparison_discrepancies = discrepancies
        logger.info(f"Comparison between Moushikomi (Rule) and RAG finished. Found {len(discrepancies)} discrepancies.")

        progress_bar.progress(1.0, text="ステップ2: 完了！")
        st.session_state.current_step = 2 # ステップ2完了
        logger.info("Step 2 completed successfully.")
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error during Step 2: {e}", exc_info=True)
        st.session_state.error_message = f"ステップ2エラー: {e}\n{traceback.format_exc()}"
        st.session_state.current_step = 1 # エラー時はステップ1完了状態に戻す
    finally:
        progress_bar.empty()
        st.session_state.processing = False
        st.rerun()

# ステップ3: 見積明細と仕様の整合性確認
def run_step3_check_estimate_vs_spec():
    """見積明細と（申し送り書＋RAG）仕様を比較"""
    st.session_state.processing = True
    st.session_state.error_message = None
    # 前回の結果をクリア（ステップ3関連）
    st.session_state.detail_check_violations = []
    st.session_state.current_step = 2 # ステップ2完了状態に戻す

    df = st.session_state.get("estimate_dataframe")
    rag_results = st.session_state.get("rag_qa_results", [])
    moushikomi_details = st.session_state.get("moushikomi_details", {})

    if df is None:
        st.session_state.error_message = "見積明細データが読み込まれていません。ステップ1を先に実行してください。"
        st.session_state.processing = False
        logger.error("Step 3 Error: Estimate DataFrame not found.")
        st.rerun()
        return
    # RAG結果がなくてもチェックは実行可能（RAG由来のチェックはスキップされる）

    progress_bar = st.progress(0.0, text="ステップ3: 見積明細と仕様の整合性確認を開始...")

    try:
        # check_detail_rules_with_rag を使用
        progress_bar.progress(0.2, text="見積明細データと仕様情報を照合中...")
        violations = check_detail_rules_with_rag(
            df,
            rag_results,
            moushikomi_details, # ステップ1のルールベース抽出結果も渡す
            config
        )
        st.session_state.detail_check_violations = violations
        logger.info(f"Check between CSV details and specifications finished. Found {len(violations)} violations.")

        progress_bar.progress(1.0, text="ステップ3: 完了！")
        st.session_state.current_step = 3 # ステップ3完了
        logger.info("Step 3 completed successfully.")
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error during Step 3: {e}", exc_info=True)
        st.session_state.error_message = f"ステップ3エラー: {e}\n{traceback.format_exc()}"
        st.session_state.current_step = 2 # エラー時はステップ2完了状態に戻す
    finally:
        progress_bar.empty()
        st.session_state.processing = False
        st.rerun()

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="MiCheck Step-by-Step", layout="wide", initial_sidebar_state="expanded")
    st.title("📄 MiCheck: 申し送り書 & 見積明細 検証システム (ステップ実行版)")
    st.caption(f"参照DB: {config.persist_directory} | Collection: {config.collection_name}")

    initialize_session_state()

    # --- グローバルコンポーネントのロード ---
    embedding_function, external_vdb = load_components_cached()
    # EmbeddingかDBのロード失敗時はエラー表示して停止
    if embedding_function is None:
        st.error("Embeddingモデルのロードに失敗しました。アプリケーションを続行できません。")
        st.stop()
    if external_vdb is None and not config.use_memory_db:
         st.error(f"外部データベース ({config.persist_directory}) のロードに失敗しました。アプリケーションを続行できません。")
         st.stop()

    # --- サイドバー ---
    with st.sidebar:
        st.header("📂 ファイルアップロード")
        # 申し送り書
        uploaded_m_file = st.file_uploader(
            "1. 申し送り書 (PDF/Image)",
            type=["pdf", "png", "jpg", "jpeg", "webp", "heic", "heif"],
            key="moushikomi_uploader",
            help="対応形式: PDF, PNG, JPG, WEBP, HEIC, HEIF"
        )
        if uploaded_m_file is not None and st.session_state.uploaded_moushikomi != uploaded_m_file:
            st.session_state.uploaded_moushikomi = uploaded_m_file
            st.session_state.moushikomi_file_name = uploaded_m_file.name
            st.session_state.current_step = 0 # ファイル変更時はリセット
            logger.debug(f"Moushikomi file uploaded: {uploaded_m_file.name}")
            st.rerun()
        elif uploaded_m_file is None and st.session_state.uploaded_moushikomi is not None:
             st.session_state.uploaded_moushikomi = None
             st.session_state.moushikomi_file_name = None
             st.session_state.current_step = 0
             logger.debug("Moushikomi file removed.")
             st.rerun()

        # 見積明細CSV
        uploaded_e_file = st.file_uploader(
            "2. 見積明細 CSV",
            type="csv",
            key="estimate_csv_uploader",
            help="対応形式: CSV (UTF-8 or Shift-JIS)"
        )
        if uploaded_e_file is not None and st.session_state.uploaded_estimate_csv != uploaded_e_file:
            st.session_state.uploaded_estimate_csv = uploaded_e_file
            st.session_state.estimate_csv_file_name = uploaded_e_file.name
            st.session_state.current_step = 0
            logger.debug(f"Estimate CSV file uploaded: {uploaded_e_file.name}")
            st.rerun()
        elif uploaded_e_file is None and st.session_state.uploaded_estimate_csv is not None:
             st.session_state.uploaded_estimate_csv = None
             st.session_state.estimate_csv_file_name = None
             st.session_state.current_step = 0
             logger.debug("Estimate CSV file removed.")
             st.rerun()

        st.divider()

        # --- ステップ実行ボタン ---
        st.header("⚙️ 実行ステップ")
        files_ready = st.session_state.uploaded_moushikomi is not None and \
                      st.session_state.uploaded_estimate_csv is not None
        processing_now = st.session_state.processing

        # ステップ1ボタン
        st.button(
            "▶️ ステップ1: 読込＆基本情報抽出",
            key="run_step1_button",
            on_click=run_step1_load_and_extract,
            disabled=not files_ready or processing_now or st.session_state.current_step > 0, # ファイル未選択、処理中、完了済なら非活性
            use_container_width=True,
            type="primary" if files_ready and st.session_state.current_step == 0 else "secondary"
        )

        # ステップ2ボタン
        st.button(
            "▶️ ステップ2: 申送/RAG整合性確認",
            key="run_step2_button",
            on_click=run_step2_check_moushikomi_vs_rag,
            args=(embedding_function, external_vdb), # 引数を渡す
            disabled=st.session_state.current_step < 1 or processing_now or st.session_state.current_step > 1, # Step1未完了、処理中、完了済なら非活性
            use_container_width=True,
            type="primary" if st.session_state.current_step == 1 else "secondary"
        )

        # ステップ3ボタン
        st.button(
            "▶️ ステップ3: 見積整合性確認",
            key="run_step3_button",
            on_click=run_step3_check_estimate_vs_spec,
            disabled=st.session_state.current_step < 2 or processing_now or st.session_state.current_step > 2, # Step2未完了、処理中、完了済なら非活性
            use_container_width=True,
            type="primary" if st.session_state.current_step == 2 else "secondary"
        )

        if processing_now:
            st.warning("処理を実行中です...")
        elif st.session_state.error_message:
            st.error(f"エラーが発生しました:\n{st.session_state.error_message.splitlines()[0]}") # エラーメッセージ短縮表示

        st.divider()
        # --- デバッグ情報など ---
        if logger.level <= logging.DEBUG:
            st.sidebar.subheader("Debug Info")
            st.sidebar.write(f"Current Step: {st.session_state.current_step}")
            st.sidebar.write(f"Processing: {st.session_state.processing}")
            st.sidebar.write(f"M File: {st.session_state.moushikomi_file_name}")
            st.sidebar.write(f"E File: {st.session_state.estimate_csv_file_name}")
            st.sidebar.write(f"Store: {st.session_state.store_name}")
            # 他のセッション状態も必要なら表示

    # === メインエリア ===
    st.header("📊 検証結果")

    # --- エラー表示 ---
    if st.session_state.error_message and not processing_now:
        st.error("前回の処理でエラーが発生しました。")
        with st.expander("エラー詳細"):
            st.code(st.session_state.error_message, language=None)

    # --- ステータス表示 ---
    current_step = st.session_state.current_step
    if current_step == 0 and not files_ready:
        st.info("サイドバーから申し送り書と見積明細CSVをアップロードしてください。")
    elif current_step == 0 and files_ready:
         st.info("ファイルの準備ができました。「ステップ1: 読込＆基本情報抽出」ボタンを押してください。")
    elif current_step == 1:
        st.success("ステップ1完了: ファイル読み込みと基本情報抽出が完了しました。")
        st.info("「ステップ2: 申送/RAG整合性確認」ボタンを押してください。")
    elif current_step == 2:
         st.success("ステップ2完了: 申し送り書と標準仕様(RAG)の整合性確認が完了しました。")
         st.info("「ステップ3: 見積整合性確認」ボタンを押してください。")
    elif current_step == 3:
         st.success("ステップ3完了: 見積明細と仕様の整合性確認が完了しました。")
         st.info("最終結果を確認してください。")

    # --- 結果表示エリア ---
    if current_step > 0:
        tab_titles = [
            "① 基本情報",
            f"② 申送/RAG比較 ({len(st.session_state.comparison_discrepancies)})" if current_step >= 2 else "② 申送/RAG比較",
            f"③ 見積チェック ({len(st.session_state.detail_check_violations)})" if current_step >= 3 else "③ 見積チェック",
            f"🚨 最終問題点" if current_step >= 3 else "🚨 問題点 (待機中)",
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1: # 基本情報 (ステップ1の結果)
            st.subheader("① 読み込みファイルと抽出情報")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**申し送り書:** `{st.session_state.moushikomi_file_name or 'N/A'}`")
                with st.expander("OCRテキスト (抜粋)", expanded=False):
                    ocr = st.session_state.moushikomi_ocr_text or "(未処理)"
                    st.text_area("m_ocr_disp_tab1", format_document_snippet(ocr, 500), height=150, disabled=True)
                st.markdown(f"**抽出された工務店名:** `{st.session_state.store_name or '(未抽出)'}`")
                with st.expander("申し送り書からのルールベース抽出情報", expanded=False):
                    st.json(st.session_state.moushikomi_details or {})

            with col2:
                st.markdown(f"**見積明細CSV:** `{st.session_state.estimate_csv_file_name or 'N/A'}`")
                if st.session_state.estimate_dataframe is not None:
                    st.dataframe(st.session_state.estimate_dataframe.head(), height=250, use_container_width=True)
                    st.caption(f"全{len(st.session_state.estimate_dataframe)}行 (先頭5行表示)")
                else:
                    st.info("見積明細CSVはまだ読み込まれていません。")

        with tab2: # 申送/RAG比較 (ステップ2の結果)
            st.subheader("② 申し送り書とRAGによる標準仕様の比較")
            if current_step < 2:
                st.info("ステップ2を完了すると、ここに比較結果が表示されます。")
            else:
                if st.session_state.comparison_discrepancies:
                    st.warning(f"{len(st.session_state.comparison_discrepancies)} 件の不一致が見つかりました:")
                    for d in st.session_state.comparison_discrepancies:
                        st.markdown(f"- {d}")
                else:
                    st.success("申し送り書の情報とRAGで確認した標準仕様の間に、定義された不一致はありませんでした。")

                with st.expander("RAGによる標準仕様確認結果 (質問と回答)", expanded=False):
                    if st.session_state.rag_qa_results:
                        st.markdown(f"**実行質問数:** {len(st.session_state.rag_qa_results)}")
                        for i, result in enumerate(st.session_state.rag_qa_results):
                            q = result.get('q', '質問不明')
                            answer = result.get('a', '[回答なし]')
                            expanded_state = "[Error" in answer or "判断できません" in answer or "関連情報なし" in answer
                            with st.expander(f"Q{i+1}: {q}", expanded=expanded_state):
                                st.markdown("**外部DB回答:**")
                                if "[Error" in answer or "[RAG Error" in answer: st.error(answer)
                                elif "判断できません" in answer or "関連情報なし" in answer: st.warning(answer)
                                else: st.info(answer)
                                # 必要ならソース表示を追加
                    elif st.session_state.store_name and st.session_state.store_name != "抽出失敗":
                         st.info("RAGによる仕様確認は実行されましたが、結果がありません。")
                    else:
                         st.info("工務店名不明等のため、RAGによる仕様確認はスキップされました。")

        with tab3: # 見積チェック (ステップ3の結果)
            st.subheader("③ 見積明細と仕様（申し送り書＋RAG）の整合性チェック")
            if current_step < 3:
                st.info("ステップ3を完了すると、ここにチェック結果が表示されます。")
            else:
                if st.session_state.detail_check_violations:
                     st.warning(f"{len(st.session_state.detail_check_violations)} 件の違反または不整合が見つかりました:")
                     for v in st.session_state.detail_check_violations[:50]: # 表示件数制限
                         st.markdown(f"- {v}")
                     if len(st.session_state.detail_check_violations) > 50:
                         st.caption("... (表示件数上限)")
                else:
                     st.success("見積明細と仕様の間に、定義された違反や不整合はありませんでした。")

                with st.expander("チェックに使用した見積明細データ (全体表示)", expanded=False):
                    if st.session_state.estimate_dataframe is not None:
                         st.dataframe(st.session_state.estimate_dataframe, height=300, use_container_width=True)
                    else:
                         st.info("見積明細データがありません。")

        with tab4: # 最終問題点 (ステップ3完了後)
            st.subheader("🚨 最終的な問題点のまとめ")
            if current_step < 3:
                st.info("ステップ3まで完了すると、ここに問題点の最終リストが表示されます。")
            else:
                final_issue_list = []
                # ステップ2と3の結果を結合
                if st.session_state.comparison_discrepancies:
                    final_issue_list.extend([f"[申送/RAG不一致] {d}" for d in st.session_state.comparison_discrepancies])
                if st.session_state.detail_check_violations:
                    final_issue_list.extend([f"[明細/仕様不一致] {v}" for v in st.session_state.detail_check_violations])

                if final_issue_list:
                    st.warning(f"{len(final_issue_list)} 件の問題点が検出されました。")
                    st.markdown("##### 申し送り書/RAG不一致:")
                    compare_issues = [s for s in final_issue_list if s.startswith("[申送/RAG不一致]")]
                    if compare_issues:
                        for issue in compare_issues: st.markdown(f"- {issue.replace('[申送/RAG不一致] ','')}")
                    else: st.caption("なし")

                    st.markdown("##### 明細/仕様不一致:")
                    detail_issues = [s for s in final_issue_list if s.startswith("[明細/仕様不一致]")]
                    if detail_issues:
                        for issue in detail_issues: st.markdown(f"- {issue.replace('[明細/仕様不一致] ','')}")
                    else: st.caption("なし")
                else:
                    st.success("問題点は検出されませんでした。")

    # アプリケーション終了時に一時ファイルをクリーンアップ
    # Streamlitの on_session_end はないので、この位置で毎回チェックする（やや非効率）
    # または、より高度な管理（例: 終了ボタンを設ける）が必要
    # cleanup_temp_files() # ここで呼ぶと rerun 時に消えてしまう可能性

# --- アプリケーション実行 ---
if __name__ == "__main__":
    logger.info("Starting RAGapp_micheck (Step-by-Step version)...")
    main()
    # アプリ終了時のクリーンアップは Streamlit の標準機能では難しい
    # 必要であれば、セッション終了を検知するハックや、
    # 定期的なクリーンアップスクリプトを検討する。
    # logger.info("Application main function finished.")