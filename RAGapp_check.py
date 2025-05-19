# RAGapp_check.py (外部DB参照・検証フロー版)
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd # QA結果表示で使う可能性
import os
import sys
import logging
import time
import tempfile
import traceback
from typing import List, Dict, Any # 型ヒントで使用

# --- モジュールのインポート ---
try:
    # ★★★ 設定ファイルとユーティリティをインポート ★★★
    from config_check import Config, setup_logging # 修正されたconfigを使う
    from rag_query_utils_check import (
        initialize_embedding,       # Embedding初期化
        load_external_chroma_db,    # 外部DBロード関数
        ask_question_single_variant # QA実行関数
    )
    from utils_check import (
        ocr_image_with_gemini, ocr_pdf_with_gemini, # OCR関数
        extract_store_name_with_gemini,             # 店名抽出関数
        generate_questions,                         # 質問生成関数
        format_document_snippet,                    # スニペット表示関数
    )
    # Chromaはrag_query_utils_check内で使われるので直接インポートは不要かも
    # from langchain_chroma import Chroma
    from langchain_core.documents import Document # ソース表示で型チェックに使う可能性

except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr); detailed_error = traceback.format_exc(); print(detailed_error, file=sys.stderr)
    try: st.error(f"Import Error: {e}\n\nTraceback:\n{detailed_error}\n\n確認してください。"); st.stop()
    except Exception: sys.exit(f"Import Error: {e}\n{detailed_error}")

# --- グローバル設定 ---
try:
    config = Config()
    logger = setup_logging(config, log_filename="rag_pipeline_check.log") # ログファイル名指定
except Exception as global_e: print(f"Global Setup Error: {global_e}", file=sys.stderr); traceback.print_exc(); sys.exit(f"Global Setup Error: {global_e}")

# --- ★★★ Embedding関数と外部DBの初期化（キャッシュ利用）★★★ ---
@st.cache_resource
def load_components_cached():
    """Embedding関数と外部Chroma DBをロード（アプリ起動時に1回）"""
    logger.info("Initializing components (Embedding and External DB)...")
    embedding_function = initialize_embedding(config, logger)
    if embedding_function is None:
        logger.critical("Embedding function failed to initialize. Cannot proceed.")
        return None, None # EmbeddingがないとDBもロードできない

    external_db = None
    if not config.use_memory_db: # configで外部DBを使う設定の場合
        external_db = load_external_chroma_db(config, embedding_function, logger)
        if external_db is None:
            # 外部DBのロード失敗は致命的エラーとする
            logger.critical(f"Failed to load external Chroma DB from '{config.persist_directory}'. Cannot proceed.")
            return None, None # DBがないとQAできない
        logger.info("External Chroma DB loaded successfully.")
    else:
        # このアプリは外部DB前提なので、use_memory_db=True は設定ミスとする
        logger.critical("Configuration error: 'use_memory_db' is True, but this application requires an external database. Set use_memory_db to False in config_check.py.")
        return None, None

    logger.info("Essential components (Embedding and External DB) initialized.")
    return embedding_function, external_db

# --- QA ループ内待機時間 定数 ---
QA_LOOP_DELAY_SECONDS = config.api_call_delay_seconds # configから取得

# --- Streamlit アプリのメイン処理 ---
def main():
    st.set_page_config(page_title="RAG Document Check (External DB)", layout="wide", initial_sidebar_state="expanded")
    st.title("📝 RAG 文書内容チェック AI (外部DB検証版)")
    st.caption(f"アップロードされた文書を読み取り、内容に基づいて生成した質問を、外部DB({config.persist_directory})に問い合わせて検証します。")

    # --- ★★★ コンポーネント（Embeddingと外部DB）をロード ★★★ ---
    embedding_function, external_vdb = load_components_cached()
    if embedding_function is None or external_vdb is None:
        st.error("システム初期化エラー (Embeddingまたは外部DB)。設定ファイル(config_check.py)のDBパスや、DB自体が存在するか確認してください。")
        logger.critical("Failed to get embedding_function or external_vdb during startup.")
        st.stop() # アプリケーションを停止
    logger.info("Embedding function and external VectorDB are ready.")

    # --- セッション状態 ---
    default_session_state = {
        "uploaded_file_info": None,
        "ocr_text": None,
        "extracted_store_name": None,
        "generated_questions": None,
        "qa_results": [],
        "processing_state": "", # "processing", "done", "error"
        # "current_vdb" は不要
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state: st.session_state[key] = default_value

    # === サイドバー ===
    with st.sidebar:
        st.header("📄 ファイル選択")
        uploaded_file = st.file_uploader("チェック対象ファイルを選択", type=["pdf", "png", "jpg", "jpeg"], key="file_uploader")

        # --- ファイルアップロード時のリセット処理 ---
        if uploaded_file is not None:
            if st.session_state.uploaded_file_info != uploaded_file.name:
                logger.info(f"New file uploaded: {uploaded_file.name}. Resetting application state.")
                st.session_state.uploaded_file_info = uploaded_file.name
                st.session_state.ocr_text = None
                st.session_state.extracted_store_name = None
                st.session_state.generated_questions = None
                st.session_state.qa_results = []
                st.session_state.processing_state = "processing" # 処理開始状態へ
                st.rerun() # 状態をリセットして再描画

        # --- ファイル処理（OCR、質問生成） ---
        if st.session_state.processing_state == "processing" and uploaded_file is not None:
            # ★★★ 処理フローからDB作成を削除 ★★★
            with st.spinner(f"ファイルを処理中 (OCR & 質問生成)..."):
                tmp_file_path = None
                try:
                    # 1. 一時ファイル作成
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    logger.info(f"Temporary file created: {tmp_file_path}")

                    # 2. OCR処理
                    ocr_start_time = time.time()
                    ocr_text_result = ""
                    file_ext = os.path.splitext(tmp_file_path)[1].lower()
                    if file_ext == ".pdf":
                        ocr_text_result = ocr_pdf_with_gemini(tmp_file_path, config.ocr_dpi, config)
                    elif file_ext in [".png",".jpg",".jpeg"]:
                        ocr_text_result = ocr_image_with_gemini(tmp_file_path, config)
                    else:
                        raise ValueError("Unsupported file type for OCR")

                    if "[OCR Error" in ocr_text_result or not ocr_text_result.strip():
                        raise RuntimeError(f"OCR failed or returned empty: {ocr_text_result}")
                    st.session_state.ocr_text = ocr_text_result
                    logger.info(f"OCR completed in {time.time()-ocr_start_time:.2f}s. Text length: {len(ocr_text_result)}")

                    # 3. 施工店名抽出 (質問生成の材料として)
                    extract_start_time = time.time()
                    # OCRテキストが短い場合はスキップするなどの考慮も可能
                    if len(st.session_state.ocr_text) > 10: # 例: 短すぎるテキストは処理しない
                         extracted_store_name_result = extract_store_name_with_gemini(st.session_state.ocr_text, config)
                         st.session_state.extracted_store_name = extracted_store_name_result
                         logger.info(f"Store name extraction done in {time.time()-extract_start_time:.2f}s. Result: '{extracted_store_name_result}'")
                    else:
                         st.session_state.extracted_store_name = None
                         logger.info("OCR text too short, skipping store name extraction.")


                    # ★★★ チャンク化 と VectorDB作成 は行わない ★★★

                    # 4. 質問生成 (OCRテキストや抽出情報から)
                    qgen_start_time = time.time()
                    # generate_questions に渡す情報を調整 (例: 店名だけでなくOCRテキスト全体も渡すなど)
                    st.session_state.generated_questions = generate_questions(st.session_state.extracted_store_name)
                    if st.session_state.generated_questions:
                         logger.info(f"Question generation done in {time.time()-qgen_start_time:.2f}s. Generated {len(st.session_state.generated_questions)} questions.")
                    else:
                         logger.warning("Question generation resulted in no questions.")
                         # 質問が生成されなかった場合のエラーハンドリングも検討

                    st.session_state.processing_state = "done" # 処理完了状態へ
                    st.success("ファイル読み取りと質問生成が完了しました。")

                except Exception as proc_e:
                    logger.error(f"Error during file processing (OCR/Question Gen): {proc_e}", exc_info=True)
                    st.error(f"ファイル処理中にエラーが発生しました: {proc_e}")
                    st.session_state.processing_state = "error" # エラー状態へ
                finally:
                    # 一時ファイル削除
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try: os.unlink(tmp_file_path); logger.info("Temporary file deleted.")
                        except Exception as del_e: logger.error(f"Failed to delete temporary file: {del_e}")
                    st.rerun() # スピナーを消し、状態を更新するために再実行

        # --- サイドバー 処理結果概要表示 ---
        if st.session_state.uploaded_file_info:
            st.divider()
            st.markdown(f"**対象ファイル:** `{st.session_state.uploaded_file_info}`")
            if st.session_state.processing_state == "done":
                st.success("ステータス: 読み取り完了")
                store_display = st.session_state.extracted_store_name or "(不明)"
                st.markdown(f"**抽出施工店名:** {store_display}")
                if st.session_state.ocr_text:
                     with st.expander("OCRテキスト(抜粋)", expanded=False):
                          ocr_preview = format_document_snippet(st.session_state.ocr_text, 300)
                          st.text_area("OCRPreviewSidebar", ocr_preview, height=100, disabled=True, label_visibility="collapsed")
            elif st.session_state.processing_state == "error":
                st.error("ステータス: 処理エラー")

        st.divider()
        st.markdown(f"**参照DB:** `{config.persist_directory}`")
        # リセットボタンは省略

    # === メインエリア ===
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("✅ 内容チェック QA 結果 (外部DB参照)")
        if st.session_state.processing_state == "done":
            # ★★★ QA実行時に外部DB (external_vdb) を渡す ★★★
            if st.session_state.generated_questions and external_vdb:
                if not st.session_state.qa_results: # まだQAを実行していない場合
                    st.info(f"**{len(st.session_state.generated_questions)}** 件のチェック項目について、外部DBに対してQAを実行します。")
                    if st.button("▶️ QA チェック開始 (外部DB検証)", type="primary", key="start_qa_button"):
                        qa_start_time = time.time()
                        num_q = len(st.session_state.generated_questions)
                        progress_bar = st.progress(0)
                        temp_results = []

                        for i, q in enumerate(st.session_state.generated_questions):
                            q_num = i + 1
                            logger.info(f"Running QA {q_num}/{num_q} against external DB: {q}")
                            progress_bar.progress(q_num / num_q, text=f"外部DB検証中 ({q_num}/{num_q})")
                            answer = "[QA Error]"
                            sources = []
                            response_dict = None

                            try:
                                # ★★★ ask_question_single_variant に external_vdb を渡す ★★★
                                response_dict = ask_question_single_variant(
                                    vectordb=external_vdb, # ロード済みの外部DBインスタンス
                                    embedding_function=embedding_function,
                                    config=config,
                                    query=q,
                                    logger_instance=logger
                                    # metadata_filter は必要なら追加
                                )
                                # レスポンス処理
                                if isinstance(response_dict, dict):
                                    answer_stream = response_dict.get("result_stream", iter(["[Error: streamなし]"]))
                                    final_answer = "".join(list(answer_stream)) # ストリームを結合
                                    answer = final_answer if final_answer else "(空の回答)"
                                    sources = response_dict.get("source_documents", [])
                                    logger.info(f"QA Q{q_num} completed.")
                                else:
                                    logger.error(f"QA error Q{q_num}: unexpected response type {type(response_dict)}")
                                    answer = "[QA System Error: Unexpected response type]"
                                    sources = []
                            except Exception as qa_e:
                                logger.error(f"QA error during Q{q_num}: {qa_e}", exc_info=True)
                                answer = f"[QA Execution Error: {type(qa_e).__name__}]"
                                sources = []

                            temp_results.append({"q": q, "a": answer, "sources": sources})

                            # APIレート制限のための待機
                            logger.debug(f"Waiting {QA_LOOP_DELAY_SECONDS}s after QA for Q{q_num}...")
                            time.sleep(QA_LOOP_DELAY_SECONDS)

                        st.session_state.qa_results = temp_results
                        progress_bar.empty()
                        logger.info(f"External DB QA process completed in {time.time() - qa_start_time:.2f} seconds.")
                        st.success("外部DBへのQAチェックが完了しました。")
                        st.rerun() # 結果を表示するために再実行

                # --- QA結果表示 ---
                if st.session_state.qa_results:
                    st.markdown("---")
                    st.markdown(f"**検証結果 ({len(st.session_state.qa_results)} 件):**")
                    for i, result in enumerate(st.session_state.qa_results):
                        with st.expander(f"Q{i+1}: {result['q']}", expanded=False):
                            st.markdown("**外部DBからの回答:**")
                            st.info(result['a']) # 回答を表示
                            st.markdown("**関連情報 (外部DBより):**")
                            if result['sources']:
                                for idx, src_doc in enumerate(result['sources']):
                                    if isinstance(src_doc, Document): # Documentオブジェクトかチェック
                                        # メタデータからソース情報を取得 (キーは実際のDB構造に合わせる)
                                        source_info = src_doc.metadata.get('source', f'doc_{idx}') # 'source'キーがない場合のフォールバック
                                        display_text = f"Source: {source_info}\n"
                                        display_text += f"...{format_document_snippet(src_doc.page_content, 200)}..." # スニペット表示
                                        st.text_area(
                                            f"source_{i}_{idx}",
                                            display_text,
                                            height=100,
                                            disabled=True,
                                            label_visibility="collapsed"
                                        )
                                    else:
                                         st.warning(f"無効なソースドキュメント形式: {type(src_doc)}")
                            else:
                                st.warning("関連情報が見つかりませんでした。")
                    # DataFrame表示やダウンロードボタンはここに実装可能

            elif not st.session_state.generated_questions:
                st.warning("質問が生成されていません。ファイル処理を完了してください。")
            elif not external_vdb: # これは通常発生しないはず（起動時にチェックされるため）
                st.error("外部VectorDBがロードされていません。アプリケーションを再起動してください。")

        elif st.session_state.processing_state == "":
            st.info("サイドバーからチェック対象のファイルをアップロードしてください。")
        elif st.session_state.processing_state == "error":
            st.error("ファイル処理中にエラーが発生しました。ログを確認するか、ファイルを再アップロードしてください。")

    with col2: # 右カラム
        st.subheader("ℹ️ 補足情報")
        if st.session_state.uploaded_file_info:
            st.markdown(f"**ファイル:** `{st.session_state.uploaded_file_info}`")
            store_display = st.session_state.extracted_store_name or "(不明)"
            st.markdown(f"**抽出施工店名:** {store_display}")

            if st.session_state.processing_state == "done":
                 st.markdown(f"**生成質問数:** {len(st.session_state.generated_questions or [])}")

            # 外部DBの情報を表示
            st.markdown("---")
            st.markdown(f"**参照データベース情報**")
            st.markdown(f"**パス:** `{config.persist_directory}`")
            st.markdown(f"**コレクション名:** `{config.collection_name}`")
            if external_vdb:
                try:
                    vdb_count = external_vdb._collection.count()
                    st.markdown(f"**ドキュメント数:** {vdb_count}")
                except Exception as e:
                    logger.warning(f"Could not get external VDB count: {e}")
                    st.markdown("**ドキュメント数:** (取得失敗)")
            else:
                st.markdown("**ステータス:** (未ロード)")
        else:
             st.markdown(f"**参照DBパス:** `{config.persist_directory}`")
             st.markdown(f"**コレクション名:** `{config.collection_name}`")

        # GPU情報は省略

if __name__ == "__main__":
    # アプリ起動時にログ出力
    logger.info(f"Starting RAGapp_check (External DB Verification Mode)... App Config: persist_dir='{config.persist_directory}', collection='{config.collection_name}'")
    main()