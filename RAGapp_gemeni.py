# RAGapp_gemeni.py (Gemini API 統合 + 高度な UI 完成版)

import streamlit as st
import pandas as pd
import os
import datetime
import sys
import logging
import pprint
import json # メタデータフィルター用
import torch # GPU情報表示用
from typing import Optional, Tuple, List, Dict, Any # 型ヒント用に追加

# --- モジュールのインポート ---
try:
    # config_gemeni からインポート
    from config_gemeni import Config, setup_logging
    # rag_query_utils_gemeni からインポート
    from rag_query_utils_gemeni import initialize_pipeline, ask_question_ensemble_stream
    # utils_gemeni からインポート
    from utils_gemeni import format_document_snippet, normalize_str, preprocess_query
    # calculate_semantic_similarity はダミーなのでコメントアウト
    # from utils_gemeni import calculate_semantic_similarity
except ImportError as e:
    print(f"Import Error in RAGapp_gemeni.py: {e}", file=sys.stderr)
    try: st.error(f"Import Error: {e}\n必要なモジュール(config_gemeni, rag_query_utils_gemeni, utils_gemeni)を確認してください。"); st.stop()
    except Exception: sys.exit(f"Import Error: {e}")

# --- グローバル設定 ---
try:
    config = Config()
    # ログファイル名を変更
    logger = setup_logging(config, log_filename="streamlit_app_gemeni.log")
    EVALUATION_FILE = "evaluation_log_gemeni.csv" # 評価ファイル名も変更
except Exception as global_e:
    print(f"Global Setup Error: {global_e}", file=sys.stderr)
    try: st.error(f"Global Setup Error: {global_e}"); st.stop()
    except Exception: sys.exit(f"Global Setup Error: {global_e}")

# --- パイプライン初期化 (キャッシュ付き) ---
@st.cache_resource
def load_pipeline_cached(lora_adapter_path: Optional[str] = None) -> tuple:
    """中間LLMを含むパイプラインコンポーネントをロード（キャッシュ付き）"""
    logger.info(f"Attempting to initialize pipeline with LoRA: {lora_adapter_path}")
    try:
        # initialize_pipeline は rag_query_utils_gemeni からインポートされたもの
        pipeline_components = initialize_pipeline(config, logger, lora_adapter_path=lora_adapter_path)
        if not all(comp is not None for comp in pipeline_components): # いずれかがNoneなら失敗
            logger.error("Pipeline initialization failed within load_pipeline_cached.")
            return (None,) * 4 # Noneを4つ持つタプルを返す
        logger.info("Pipeline components initialized successfully.")
        return pipeline_components # (vectordb, intermediate_llm, tokenizer, embedding) のタプル
    except Exception as e:
        logger.critical(f"Fatal error during pipeline initialization: {e}", exc_info=True)
        return (None,) * 4

# --- 評価記録関数 ---
def record_evaluation(filename, timestamp, query, final_answer, source_docs, answer_evaluation, basis_evaluation, comment):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file_exists = os.path.isfile(filepath); fieldnames = ['timestamp', 'query', 'answer', 'source_docs_summary', 'answer_evaluation', 'basis_evaluation', 'comment']
    try:
        source_summary = "N/A"
        if source_docs: unique_sources = list(set(doc.metadata.get('source', 'N/A') for doc in source_docs)); source_summary = ", ".join(unique_sources)[:200]
        new_eval = pd.DataFrame([{'timestamp': timestamp, 'query': query, 'answer': final_answer, 'source_docs_summary': source_summary,
                                  'answer_evaluation': answer_evaluation, 'basis_evaluation': basis_evaluation, 'comment': comment}])
        new_eval.to_csv(filepath, mode='a', header=not file_exists, index=False, encoding='utf-8-sig', lineterminator='\n', columns=fieldnames)
        logger.info(f"Evaluation recorded to {filename}"); return True
    except Exception as e: logger.error(f"Failed record evaluation: {e}", exc_info=True); return False

# --- シンプルなダークテーマCSS (変更なし) ---
DARK_THEME_CSS = """
<style>
/* --- 基本スタイル --- */
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"; background-color: #0e1117; color: #fafafa; line-height: 1.6; }
.stApp { background-color: #0e1117; }
.main .block-container { max-width: 850px; margin: auto; padding: 1.5rem 2rem 3rem 2rem; }
/* --- 見出し --- */
h1 { color: #fafafa; font-size: 2rem; text-align: center; margin-bottom: 1.5rem; font-weight: 600;}
h3 { color: #3b7cff; font-size: 1.3rem; margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.3rem;}
h4 { color: #ccc; font-size: 1.1rem; margin-top: 1.2rem; margin-bottom: 0.5rem; }
/* --- ボタン --- */
div[data-testid="stButton"] > button { background-color: #3b7cff; color: white; padding: 0.5rem 1.2rem; border: none; border-radius: 5px; font-weight: 500; transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out; cursor: pointer;}
div[data-testid="stButton"] > button:hover { background-color: #5c9bff; transform: translateY(-1px); }
div[data-testid="stButton"] > button:active { background-color: #2a5db0; transform: translateY(0px); }
/* --- テキスト入力 --- */
div[data-testid="stTextArea"] textarea, div[data-testid="stChatInput"] textarea { border: 1px solid #555; background-color: #262730; color: #fafafa; padding: 0.6rem; border-radius: 5px; font-size: 1rem; width: 100%; box-sizing: border-box; }
div[data-testid="stTextArea"] textarea:focus, div[data-testid="stChatInput"] textarea:focus { border-color: #3b7cff; box-shadow: 0 0 0 2px rgba(59, 124, 255, 0.3); outline: none; }
/* --- 情報表示ボックス --- */
div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stWarning"], div[data-testid="stError"] { border-radius: 5px; padding: 1rem 1.2rem; border: none; background-color: #262730; box-shadow: 0 1px 3px rgba(0,0,0,0.2); margin-bottom: 1rem; color: #fafafa; }
div[data-testid="stInfo"] { border-left: 5px solid #3b7cff; }
div[data-testid="stSuccess"] { border-left: 5px solid #3dd56d; }
div[data-testid="stWarning"] { border-left: 5px solid #ffc107; }
div[data-testid="stError"] { border-left: 5px solid #dc3545; }
/* --- チャットメッセージ --- */
div[data-testid="stChatMessage"] { background-color: #262730; border: 1px solid #333; border-radius: 8px; margin-bottom: 1rem; padding: 1rem 1.2rem; }
/* --- Expander --- */
div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 5px; background-color: #1c1e24; margin-top: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
div[data-testid="stExpander"] summary { font-weight: 500; color: #eee; font-size: 1rem; padding: 0.8rem 1rem; cursor: pointer; }
div[data-testid="stExpander"] summary:hover { background-color: #262730; }
div[data-testid="stExpanderDetails"] { padding: 0.5rem 1.5rem 1.5rem 1.5rem; border-top: 1px solid #333; background-color: #262730; }
/* --- コード表示 --- */
div[data-testid="stCodeBlock"] > pre { background-color: #0e1117 !important; border: 1px solid #333 !important; border-radius: 4px !important; padding: 0.8rem !important; color: #eee !important; font-family: 'Courier New', Courier, monospace !important; }
/* --- ラジオボタン --- */
div[data-testid="stRadio"] label { color: #ccc; margin-right: 0.8rem; }
/* --- データフレーム --- */
div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 4px; }
</style>
"""

# --- Streamlit アプリのメイン処理 ---
def main():
    # ページタイトルを変更
    st.set_page_config(page_title="RAG Evaluation (Gemini)", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    st.title("何となくで答える君 (Gemini版)") # タイトル変更

    # --- LoRA パスをセッション状態で管理 ---
    if "lora_adapter_path" not in st.session_state:
        st.session_state.lora_adapter_path = config.lora_adapter_path

    # --- パイプライン初期化（キャッシュを利用）---
    with st.spinner("システムを初期化中... (初回またはLoRA変更時)"):
        pipeline_components = load_pipeline_cached(lora_adapter_path=st.session_state.lora_adapter_path)

    if pipeline_components[0] is None: # vectordb が None かチェック
        st.error("システムの初期化に失敗しました。ログを確認してください。")
        st.stop()
    vectordb, intermediate_llm, tokenizer, embedding_function = pipeline_components
    logger.info("Pipeline components ready.")

    # --- その他のUI用セッション状態初期化 ---
    ui_state_defaults = {
        "query_history": [], "current_query": "", "current_answer_stream": None,
        "current_source_docs": [], "evaluation_recorded_for_last_answer": False,
        "last_full_answer": "", "metadata_filter_str": '{}', "variant_answers": [],
        "variant_settings": [{"k": config.rag_variant_k[i] if i < len(config.rag_variant_k) else 3,
                              # 中間LLMパラメータはGemini APIでは直接使わないがUI要素として残す
                              "temperature": config.temperature,
                              "top_p": config.top_p,
                              "repetition_penalty": config.repetition_penalty}
                             for i in range(getattr(config, 'num_default_variants', 3))] # configからデフォルトVariant数を取得
    }
    for key, default_value in ui_state_defaults.items():
        if key not in st.session_state: st.session_state[key] = default_value

    # --- サイドバー ---
    with st.sidebar:
        st.header("⚙️ 設定")
        st.markdown("##### LoRA 設定 (中間LLM用)") # 注釈追加
        lora_path_input = st.text_input("LoRAアダプタパス", value=st.session_state.lora_adapter_path or "", help="空欄でLoRA無効。変更後は[再初期化]実行。")
        if st.button("再初期化 (LoRA適用/解除)"):
            new_path = lora_path_input.strip() if lora_path_input else None
            if new_path != st.session_state.lora_adapter_path:
                st.session_state.lora_adapter_path = new_path
                load_pipeline_cached.clear()
                st.toast("パイプラインを再初期化します...")
                st.rerun()
            else:
                st.toast("LoRAパスに変更がないため、再初期化はスキップされました。")

        st.divider()
        st.markdown("##### アンサンブル設定")
        st.caption("各Variantの検索パラメータ (k) を調整") # 説明変更
        for i in range(len(st.session_state.variant_settings)):
            with st.expander(f"Variant {i+1} 設定", expanded=(i==0)):
                settings = st.session_state.variant_settings[i]
                settings["k"] = st.number_input(f"検索数 k (V{i+1})", 1, 20, int(settings.get("k", 3)), key=f"k_{i}")
                # 中間生成パラメータ (temperature, top_p, rep_pen) は表示のみ、または削除してもよい
                st.caption(f"(中間LLM Temp: {settings.get('temperature', config.temperature):.2f}, Top_p: {settings.get('top_p', config.top_p):.2f}, RepPen: {settings.get('repetition_penalty', config.repetition_penalty):.2f})")

        st.divider()
        st.markdown("##### 検索フィルター")
        filter_input = st.text_area("メタデータフィルター (JSON)", st.session_state.metadata_filter_str, height=100, help='例: {"type": "仕様"}')
        st.session_state.metadata_filter_str = filter_input

        st.divider()
        if st.button("表示クリア"):
            keys_to_clear = ["query_history", "current_query", "current_answer_stream", "current_source_docs", "evaluation_recorded_for_last_answer", "last_full_answer", "variant_answers"]
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.toast("表示をクリアしました。"); st.rerun()

        st.divider(); st.markdown("##### システム情報")
        try:
            if torch.cuda.is_available():
                idx = torch.cuda.current_device(); name = torch.cuda.get_device_name(idx); alloc = torch.cuda.memory_allocated(idx)/1e9; reserved = torch.cuda.memory_reserved(idx)/1e9
                st.caption(f"GPU: {name}\nMem: A {alloc:.2f} / R {reserved:.2f} GB")
            else: st.caption("Mode: CPU")
        except Exception as gpu_e: st.warning(f"GPU情報取得失敗: {gpu_e}")

    # --- 会話履歴の表示 ---
    st.markdown("### 会話履歴")
    if not st.session_state.query_history: st.info("まだ会話はありません。")
    else:
        for message in st.session_state.query_history:
            with st.chat_message(message["role"]): st.markdown(message["content"])

    # --- 進行中の回答表示エリア ---
    streaming_placeholder = st.empty()

    # --- 質問入力 (`st.chat_input`) ---
    user_query_input = st.chat_input("質問を入力してください...")

    # --- 新しい質問が入力された場合の処理 ---
    if user_query_input:
        # utils_gemeni からインポートした関数を使用
        st.session_state.current_query = preprocess_query(user_query_input)
        st.session_state.evaluation_recorded_for_last_answer = False
        st.session_state.last_full_answer = ""
        st.session_state.variant_answers = []
        st.session_state.current_source_docs = []
        st.session_state.current_answer_stream = None

        # メタデータフィルターのパース
        parsed_metadata_filter = None
        try:
            filter_str = st.session_state.metadata_filter_str.strip()
            if filter_str and filter_str != '{}': parsed_metadata_filter = json.loads(filter_str)
            if parsed_metadata_filter and not isinstance(parsed_metadata_filter, dict): st.warning("フィルターはJSON辞書形式で。", icon="⚠️"); parsed_metadata_filter = None
            elif parsed_metadata_filter: logger.info(f"Applying metadata filter: {parsed_metadata_filter}")
            else: logger.info("No metadata filter applied.")
        except json.JSONDecodeError: st.warning("フィルターJSON形式不正。", icon="⚠️"); parsed_metadata_filter = None

        # 回答生成プロセス開始
        try:
            logger.info(f"Calling ask_question_ensemble_stream with query: {st.session_state.current_query}")
            # rag_query_utils_gemeni からインポートした関数を使用
            response = ask_question_ensemble_stream(
                vectordb=vectordb,
                intermediate_llm=intermediate_llm, # 引数としては渡すが内部では使われない想定
                tokenizer=tokenizer,             # 引数としては渡すが内部では使われない想定
                embedding_function=embedding_function,
                config=config,
                query=st.session_state.current_query,
                logger=logger,
                metadata_filter=parsed_metadata_filter,
                variant_params=st.session_state.variant_settings # サイドバーで設定された k 値など
            )
            st.session_state.current_answer_stream = response.get("result_stream")
            st.session_state.current_source_docs = response.get("source_documents", [])
            st.session_state.variant_answers = response.get("variant_answers", [])
            logger.info("Response object received from ask_question_ensemble_stream.")
            # ユーザー質問を履歴に追加
            st.session_state.query_history.append({"role": "user", "content": st.session_state.current_query})

        except Exception as e:
            logger.error(f"Error during ask_question_ensemble_stream call: {e}", exc_info=True)
            st.error(f"質問処理中にエラーが発生しました: {e}")
            st.session_state.current_answer_stream = iter([f"エラー発生: {e}"])
            st.session_state.current_source_docs = []
            st.session_state.variant_answers = []
            # ユーザー質問履歴追加
            if not st.session_state.query_history or st.session_state.query_history[-1]['content'] != st.session_state.current_query:
                 st.session_state.query_history.append({"role": "user", "content": st.session_state.current_query})

        # ストリーミング表示処理のために再実行
        st.rerun()

    # --- 回答ストリーミング表示 ---
    if st.session_state.current_answer_stream:
        with streaming_placeholder.container():
             with st.chat_message("assistant"):
                 answer_placeholder = st.empty()
                 full_response = ""
                 logger.info("Starting answer streaming...")
                 try:
                     for chunk in st.session_state.current_answer_stream:
                         full_response += chunk
                         answer_placeholder.markdown(full_response + "▌")
                     answer_placeholder.markdown(full_response)
                     st.session_state.last_full_answer = full_response
                     # 履歴追加 (重複やエラーメッセージの上書きを考慮)
                     if not st.session_state.query_history or st.session_state.query_history[-1]['role'] == 'user':
                          st.session_state.query_history.append({"role": "assistant", "content": full_response})
                     elif st.session_state.query_history[-1]['role'] == 'assistant' and "エラー" in st.session_state.query_history[-1]['content']:
                          st.session_state.query_history[-1]['content'] = full_response # エラーメッセージを上書き
                     elif st.session_state.query_history[-1]['content'] != full_response: # 直前の回答と異なる場合のみ追加（リラン対策）
                           st.session_state.query_history.append({"role": "assistant", "content": full_response})

                     logger.info("Streaming finished.")

                 except Exception as stream_e:
                     logger.error(f"Error during streaming display: {stream_e}", exc_info=True)
                     st.error(f"回答表示中にエラーが発生しました: {stream_e}")
                     st.session_state.last_full_answer = f"表示エラー: {stream_e}"
                     error_message = st.session_state.last_full_answer
                     # エラー履歴追加 (重複考慮)
                     if not st.session_state.query_history or st.session_state.query_history[-1]['role'] == 'user':
                         st.session_state.query_history.append({"role": "assistant", "content": error_message})
                     elif st.session_state.query_history[-1]['content'] != error_message: # 直前のメッセージと異なる場合のみ
                          st.session_state.query_history[-1]['content'] = error_message # 直前のassistantメッセージをエラーで上書き

                 finally:
                     st.session_state.current_answer_stream = None
                     st.rerun() # 詳細表示などを更新するためにリラン

    # --- 詳細情報とvariant比較・評価 ---
    if st.session_state.last_full_answer:
        with st.expander("詳細情報 (Variant比較・参照データ・評価)"):

            # variantごとの回答比較
            if st.session_state.variant_answers:
                st.markdown("#### Variant ごとの中間回答 (Geminiによる)") # 説明変更
                for idx, v_ans in enumerate(st.session_state.variant_answers):
                    params = st.session_state.variant_settings[idx]
                    st.markdown(f"**Variant {idx+1}** (k={params['k']})") # kのみ表示
                    st.text_area(f"V{idx+1}_Answer", v_ans, height=100, disabled=True, label_visibility="collapsed", key=f"v_ans_{idx}")
                st.divider()

            # 取得したソースデータの表示
            st.markdown("#### 参照された可能性のあるデータ")
            if st.session_state.current_source_docs:
                st.caption(f"関連データチャンク (ユニーク): {len(st.session_state.current_source_docs)} 件")
                for i, doc in enumerate(st.session_state.current_source_docs):
                    meta_display = []; cols=config.metadata_display_columns; score=doc.metadata.get('rerank_score') # リランキングスコアも表示試行
                    for col_name in cols:
                        if v:=doc.metadata.get(col_name): meta_display.append(f"**{col_name[:4]}**: `{str(v)[:20]}`")
                    if score: meta_display.append(f"**Score**: `{score:.3f}`")
                    st.markdown(f"**CHUNK {i+1}** ({' | '.join(meta_display)})")
                    # utils_gemeni からインポートした関数を使用
                    st.text_area(f"Content_{i+1}", doc.page_content, height=100, disabled=True, label_visibility="collapsed", key=f"src_content_{i}")
            else: st.info("参照されたデータはありません。")
            st.divider()

            # 評価入力セクション
            st.markdown("#### この最終回答を評価")
            if not st.session_state.evaluation_recorded_for_last_answer:
                 # 評価フォームのキーを一意にする
                 eval_key_suffix = f"_{len(st.session_state.query_history)}"
                 with st.form(f"evaluation_form_{eval_key_suffix}"):
                     eval_cols = st.columns(2)
                     with eval_cols[0]: ans_opts = ["未選択", "✅ OK", "❌ NG", "🤔 部分的"]; ans_eval = st.radio("回答?", ans_opts, key=f"eval_ans{eval_key_suffix}", horizontal=True, label_visibility="collapsed")
                     with eval_cols[1]: basis_opts = ["未選択", "✅ 適切", "❌ 不適切", "🤷 不明"]; bas_eval = st.radio("根拠?", basis_opts, key=f"eval_bas{eval_key_suffix}", horizontal=True, label_visibility="collapsed")
                     cmt = st.text_area("コメント", key=f"eval_com{eval_key_suffix}", height=80, placeholder="コメント...")
                     submitted = st.form_submit_button("評価記録")
                     if submitted:
                         # 直前のユーザー質問を取得
                         user_query_index = -1
                         for j in range(len(st.session_state.query_history) -1, -1, -1):
                             if st.session_state.query_history[j]['role'] == 'user':
                                 user_query_index = j
                                 break
                         qry = st.session_state.query_history[user_query_index]['content'] if user_query_index != -1 else "N/A"

                         if ans_eval != "未選択" and bas_eval != "未選択":
                             ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                             ok=record_evaluation(EVALUATION_FILE, ts, qry, st.session_state.last_full_answer, st.session_state.current_source_docs, ans_eval, bas_eval, cmt)
                             if ok: st.toast("✔️記録完了"); st.session_state.evaluation_recorded_for_last_answer = True; st.rerun()
                             else: st.error("記録失敗")
                         else: st.warning("両方の評価を選択")
            else: st.success("✔️この回答は評価済みです。")
            st.divider()

            # 評価ログの表示
            st.markdown("#### 評価ログ（最新10件）")
            eval_file_path = os.path.join(os.path.dirname(__file__), EVALUATION_FILE)
            @st.cache_data
            def load_evaluation_data(fp):
                if os.path.exists(fp):
                    try: return pd.read_csv(fp)
                    except pd.errors.EmptyDataError: return pd.DataFrame()
                    except Exception as e: logger.error(f"Log display error: {e}"); return None
                else: return None
            df_eval = load_evaluation_data(eval_file_path)
            if df_eval is not None:
                if not df_eval.empty:
                    cols = ['timestamp', 'query', 'answer_evaluation', 'basis_evaluation', 'comment']; st.dataframe(df_eval[cols].tail(10).reset_index(drop=True), use_container_width=True)
                    @st.cache_data
                    def get_csv_data(df): return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    csv_d = get_csv_data(df_eval); st.download_button(label="全ログDL", data=csv_d, file_name=EVALUATION_FILE, mime='text/csv')
                else: st.info("評価ログなし")
            else: st.info(f"ログファイル({EVALUATION_FILE})なし")


# --- スクリプト実行 ---
if __name__ == "__main__":
    main()

# --- ここまで RAGapp_gemeni.py ---