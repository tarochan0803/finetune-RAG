# RAGapp.py - 高性能 RAG システム with Streamlit UI

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
    from config import Config, setup_logging
    # ask_question_ensemble_stream をインポート
    from rag_query_utils import initialize_pipeline, ask_question_ensemble_stream
    # utils から必要な関数をインポート
    from utils import format_document_snippet, normalize_str, preprocess_query
    # 工務店マスタ機能をインポート
    from company_master import CompanyMaster
    # 自動入力機能をインポート  
    from form_auto_fill import display_auto_fill_section
    # calculate_semantic_similarity はダミーなのでコメントアウト
    # from utils import calculate_semantic_similarity
except ImportError as e:
    print(f"Import Error in RAGapp.py: {e}", file=sys.stderr)
    try: st.error(f"Import Error: {e}\n必要なモジュールを確認してください。"); st.stop()
    except Exception: sys.exit(f"Import Error: {e}")

# --- グローバル設定 ---
try:
    config = Config()
    logger = setup_logging(config, log_filename="streamlit_app_hybrid.log") # ログファイル名変更
    EVALUATION_FILE = "evaluation_log.csv"
except Exception as global_e:
    print(f"Global Setup Error: {global_e}", file=sys.stderr)
    try: st.error(f"Global Setup Error: {global_e}"); st.stop()
    except Exception: sys.exit(f"Global Setup Error: {global_e}")

# --- パイプライン初期化 (キャッシュ付き) ---
# 中間LLM(ELYZA 7B), Tokenizer, VectorDB, Embedding をキャッシュ
@st.cache_resource
def load_pipeline_cached(lora_adapter_path: Optional[str] = None) -> tuple:
    """中間LLMを含むパイプラインコンポーネントをロード（キャッシュ付き）"""
    logger.info(f"Attempting to initialize pipeline with LoRA: {lora_adapter_path}")
    try:
        # initialize_pipeline は vectordb, intermediate_llm, tokenizer, embedding を返す
        pipeline_components = initialize_pipeline(config, logger, lora_adapter_path=lora_adapter_path)
        if not all(comp is not None for comp in pipeline_components): # いずれかがNoneなら失敗
            logger.error("Pipeline initialization failed within load_pipeline_cached.")
            # Noneを4つ持つタプルを返す
            return (None,) * 4
        logger.info("Pipeline components initialized successfully.")
        return pipeline_components
    except Exception as e:
        logger.critical(f"Fatal error during pipeline initialization: {e}", exc_info=True)
        return (None,) * 4

def record_evaluation(filename: str, timestamp: str, query: str, final_answer: str, 
                     source_docs: List, answer_evaluation: str, basis_evaluation: str, comment: str) -> bool:
    """評価データをCSVファイルに記録する"""
    try:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        file_exists = os.path.isfile(filepath)
        fieldnames = ['timestamp', 'query', 'answer', 'source_docs_summary', 
                     'answer_evaluation', 'basis_evaluation', 'comment']
        
        source_summary = "N/A"
        if source_docs:
            unique_sources = list(set(doc.metadata.get('source', 'N/A') for doc in source_docs))
            source_summary = ", ".join(unique_sources)[:200]
        
        new_eval = pd.DataFrame([{
            'timestamp': timestamp,
            'query': query,
            'answer': final_answer,
            'source_docs_summary': source_summary,
            'answer_evaluation': answer_evaluation,
            'basis_evaluation': basis_evaluation,
            'comment': comment
        }])
        
        new_eval.to_csv(filepath, mode='a', header=not file_exists, index=False, 
                       encoding='utf-8-sig', lineterminator='\n', columns=fieldnames)
        logger.info(f"Evaluation recorded to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to record evaluation: {e}", exc_info=True)
        return False

# --- シンプルなダークテーマCSS ---
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
    st.set_page_config(page_title="RAG Evaluation (Hybrid)", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    st.title("何となくで答える君")

    # --- LoRA パスをセッション状態で管理 ---
    # アプリ起動時に config.py から初期値を読み込む
    if "lora_adapter_path" not in st.session_state:
        st.session_state.lora_adapter_path = config.lora_adapter_path

    # --- パイプライン初期化（キャッシュを利用）---
    # `load_pipeline_cached` を呼び出してコンポーネントを取得
    # LoRAパスが変更されるとキャッシュがクリアされ再実行される
    with st.spinner("🚀 システムを初期化中... (初回またはLoRA変更時)"):
        pipeline_components = load_pipeline_cached(lora_adapter_path=st.session_state.lora_adapter_path)

    # パイプラインコンポーネントの確認
    if pipeline_components[0] is None:
        st.error("⚠️ システムの初期化に失敗しました。\n🔍 ログを確認し、config.py や環境変数を確認してください。")
        st.stop()
    
    vectordb, intermediate_llm, tokenizer, embedding_function = pipeline_components
    logger.info("Pipeline components ready.")
    
    # 工務店マスタの初期化
    if "company_master" not in st.session_state:
        with st.spinner("🏢 工務店マスタデータを初期化中..."):
            st.session_state.company_master = CompanyMaster(config)
    company_master = st.session_state.company_master

    # セッション状態の初期化
    ui_state_defaults = {
        "query_history": [],
        "current_query": "",
        "current_answer_stream": None,
        "current_source_docs": [],
        "evaluation_recorded_for_last_answer": False,
        "last_full_answer": "",
        "metadata_filter_str": '{}',
        "variant_answers": [],
        "variant_settings": [
            {
                "k": config.rag_variant_k[i] if i < len(config.rag_variant_k) else 3,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "repetition_penalty": config.repetition_penalty
            } for i in range(3)
        ]
    }
    
    for key, default_value in ui_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- サイドバー ---
    with st.sidebar:
        st.header("⚙️ 設定")
        st.markdown("##### LoRA 設定")
        lora_path_input = st.text_input(
            "LoRAアダプタパス", 
            value=st.session_state.lora_adapter_path or "",
            help="空欄でLoRA無効。変更後は[再初期化]実行。"
        )
        
        if st.button("🔄 再初期化 (LoRA適用/解除)"):
            new_path = lora_path_input.strip() if lora_path_input else None
            if new_path != st.session_state.lora_adapter_path:
                st.session_state.lora_adapter_path = new_path
                load_pipeline_cached.clear()
                st.success("パイプラインを再初期化します...")
                st.rerun()
            else:
                st.info("LoRAパスに変更がないため、再初期化はスキップされました。")

        st.divider()
        st.markdown("##### アンサンブル設定")
        st.caption("各Variantの中間生成パラメータを調整")
        # Variantごとの設定
        for i in range(len(st.session_state.variant_settings)):
            with st.expander(f"Variant {i+1} 設定", expanded=(i==0)):
                settings = st.session_state.variant_settings[i]
                settings["k"] = st.number_input(f"検索数 k (V{i+1})", 1, 20, int(settings.get("k", 3)), key=f"k_{i}")
                # 中間生成用のパラメータとして設定 (API用とは別)
                settings["temperature"] = st.slider(f"Temperature (V{i+1})", 0.0, 1.0, float(settings.get("temperature", config.temperature)), 0.05, key=f"temp_{i}")
                settings["top_p"] = st.slider(f"Top_p (V{i+1})", 0.1, 1.0, float(settings.get("top_p", config.top_p)), 0.05, key=f"top_p_{i}")
                settings["repetition_penalty"] = st.slider(f"Rep Penalty (V{i+1})", 1.0, 2.0, float(settings.get("repetition_penalty", config.repetition_penalty)), 0.05, key=f"rep_pen_{i}")

        st.divider()
        st.markdown("##### 検索フィルター")
        filter_input = st.text_area("メタデータフィルター (JSON)", st.session_state.metadata_filter_str, height=100, help='例: {"type": "仕様"}')
        st.session_state.metadata_filter_str = filter_input # 変更を即時反映 (rerunは不要)

        st.divider()
        if st.button("🗑️ 表示クリア"):
            keys_to_clear = [
                "query_history", "current_query", "current_answer_stream",
                "current_source_docs", "evaluation_recorded_for_last_answer",
                "last_full_answer", "variant_answers"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("表示をクリアしました。")
            st.rerun()

        st.divider()
        st.markdown("##### 💻 システム情報")
        try:
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                alloc = torch.cuda.memory_allocated(idx) / 1e9
                reserved = torch.cuda.memory_reserved(idx) / 1e9
                st.success(f"🎯 GPU: {name}\n📊 Memory: {alloc:.2f}GB / {reserved:.2f}GB")
            else:
                st.info("💻 Mode: CPU")
        except Exception as gpu_e:
            st.warning(f"⚠️ GPU情報取得失敗: {gpu_e}")

    # 新規データ入力フォーム
    st.markdown("### 📝 工務店情報入力フォーム")
    
    # フォーム用のセッション状態初期化
    form_defaults = {
        "form_company_name": "",
        "form_meeting_company": "",
        "form_meeting_person": "",
        "form_tel": "",
        "form_mobile": "",
        "form_email": "",
        "form_outer_wall_strong": "",
        "form_inner_wall_strong": "",
        "form_non_strong_wall": "",
        "form_wall_type": "大壁",
        "form_sub_materials": "有",
        "form_temporary_brace": "無",
        "form_foundation_packing": "無",
        "form_airtight_packing": "無",
        "form_airtight_range": "",
        "form_steel_post": "無",
        "form_hardware_install": "無",
        "form_other_notes": "",
        "company_candidates": [],
        "selected_company": "",
        "show_airtight_range": False,
        "auto_fill_requested": False
    }
    
    for key, default_value in form_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    with st.form("construction_company_form"):
        st.markdown("#### 🏢 基本情報")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            company_input = st.text_input(
                "工務店名 *",
                value=st.session_state.form_company_name,
                placeholder="例: 株式会社○○建設",
                help="工務店名を入力してください。類似候補が自動表示されます。"
            )
        
        with col2:
            if st.form_submit_button("🔍 工務店検索", use_container_width=True):
                if company_input.strip():
                    # 工務店名表記揺れ対策機能の実行
                    st.session_state.form_company_name = company_input
                    search_results = company_master.search_companies(company_input, limit=5)
                    
                    if search_results:
                        candidates = []
                        for result in search_results:
                            company_name = result['company']['original_name']
                            candidates.append(f"{company_name}")
                        
                        st.session_state.company_candidates = candidates
                        st.success(f"🎯 {len(candidates)}件の候補が見つかりました")
                    else:
                        st.session_state.company_candidates = []
                        st.warning("❓ 該当する工務店が見つかりませんでした")
        
        # 工務店候補表示エリア
        if st.session_state.company_candidates:
            st.markdown("**🎯 候補選択:**")
            selected = st.radio(
                "該当する工務店を選択してください",
                options=st.session_state.company_candidates + ["該当なし（新規登録）"],
                key="company_selection"
            )
            if selected != "該当なし（新規登録）":
                st.session_state.selected_company = selected
                
                st.session_state.selected_company = selected
        
        # 連絡先情報
        col1, col2 = st.columns(2)
        with col1:
            meeting_company = st.text_input(
                "打ち合わせ担当会社",
                value=st.session_state.form_meeting_company
            )
            tel = st.text_input(
                "TEL",
                value=st.session_state.form_tel,
                placeholder="例: 03-1234-5678"
            )
            email = st.text_input(
                "メール",
                value=st.session_state.form_email,
                placeholder="例: contact@example.com"
            )
        
        with col2:
            meeting_person = st.text_input(
                "打ち合わせ担当者",
                value=st.session_state.form_meeting_person
            )
            mobile = st.text_input(
                "携帯",
                value=st.session_state.form_mobile,
                placeholder="例: 090-1234-5678"
            )
        
        st.divider()
        st.markdown("#### 🧱 壁面材仕様")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            outer_wall = st.text_input(
                "耐力壁(外部)",
                value=st.session_state.form_outer_wall_strong,
                placeholder="例: 構造用合板 9mm"
            )
        with col2:
            inner_wall = st.text_input(
                "耐力壁(内部)",
                value=st.session_state.form_inner_wall_strong,
                placeholder="例: 石膏ボード 12.5mm"
            )
        with col3:
            non_strong_wall = st.text_input(
                "非耐力壁",
                value=st.session_state.form_non_strong_wall,
                placeholder="例: 石膏ボード 9.5mm"
            )
        
        st.divider()
        st.markdown("#### 🏠 外壁仕様")
        
        wall_type = st.radio(
            "外壁仕様",
            options=["大壁", "真壁"],
            index=0 if st.session_state.form_wall_type == "大壁" else 1,
            horizontal=True
        )
        
        st.divider()
        st.markdown("#### 🔧 副資材")
        
        sub_materials = st.radio(
            "副資材の供給",
            options=["有", "無"],
            index=0 if st.session_state.form_sub_materials == "有" else 1,
            horizontal=True
        )
        
        if sub_materials == "有":
            col1, col2 = st.columns(2)
            with col1:
                temporary_brace = st.radio(
                    "仮筋交",
                    options=["有", "無"],
                    index=0 if st.session_state.form_temporary_brace == "有" else 1,
                    horizontal=True
                )
                foundation_packing = st.radio(
                    "基礎パッキン",
                    options=["有", "無"],
                    index=0 if st.session_state.form_foundation_packing == "有" else 1,
                    horizontal=True
                )
            
            with col2:
                airtight_packing = st.radio(
                    "気密パッキン",
                    options=["有", "無"],
                    index=0 if st.session_state.form_airtight_packing == "有" else 1,
                    horizontal=True
                )
                steel_post = st.radio(
                    "鋼製束",
                    options=["有", "無"],
                    index=0 if st.session_state.form_steel_post == "有" else 1,
                    horizontal=True
                )
            
            # 気密パッキンが「有」の場合、範囲入力欄を表示
            if airtight_packing == "有":
                airtight_range = st.text_input(
                    "気密パッキン範囲",
                    value=st.session_state.form_airtight_range,
                    placeholder="適用範囲を記入してください"
                )
        
        st.divider()
        st.markdown("#### ⚙️ 金物")
        
        hardware_install = st.radio(
            "金物取付",
            options=["有", "無"],
            index=0 if st.session_state.form_hardware_install == "有" else 1,
            horizontal=True
        )
        
        st.divider()
        st.markdown("#### 📋 その他")
        
        other_notes = st.text_area(
            "その他記載事項",
            value=st.session_state.form_other_notes,
            height=100,
            placeholder="特記事項や補足情報を入力してください..."
        )
        
        st.divider()
        
        # フォーム送信ボタン
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "💾 データを保存",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            # フォームデータの保存処理
            form_data = {
                "工務店名": company_input,
                "打ち合わせ担当会社": meeting_company,
                "打ち合わせ担当者": meeting_person,
                "TEL": tel,
                "携帯": mobile,
                "メール": email,
                "耐力壁(外部)": outer_wall,
                "耐力壁(内部)": inner_wall,
                "非耐力壁": non_strong_wall,
                "外壁仕様": wall_type,
                "副資材の供給": sub_materials,
                "仮筋交": temporary_brace if sub_materials == "有" else "-",
                "基礎パッキン": foundation_packing if sub_materials == "有" else "-",
                "気密パッキン": airtight_packing if sub_materials == "有" else "-",
                "気密パッキン範囲": airtight_range if sub_materials == "有" and airtight_packing == "有" else "-",
                "鋼製束": steel_post if sub_materials == "有" else "-",
                "金物取付": hardware_install,
                "その他記載事項": other_notes
            }
            
            # セッション状態の更新
            for key, value in {
                "form_company_name": company_input,
                "form_meeting_company": meeting_company,
                "form_meeting_person": meeting_person,
                "form_tel": tel,
                "form_mobile": mobile,
                "form_email": email,
                "form_outer_wall_strong": outer_wall,
                "form_inner_wall_strong": inner_wall,
                "form_non_strong_wall": non_strong_wall,
                "form_wall_type": wall_type,
                "form_sub_materials": sub_materials,
                "form_temporary_brace": temporary_brace if sub_materials == "有" else "無",
                "form_foundation_packing": foundation_packing if sub_materials == "有" else "無",
                "form_airtight_packing": airtight_packing if sub_materials == "有" else "無",
                "form_airtight_range": airtight_range if sub_materials == "有" and airtight_packing == "有" else "",
                "form_steel_post": steel_post if sub_materials == "有" else "無",
                "form_hardware_install": hardware_install,
                "form_other_notes": other_notes
            }.items():
                st.session_state[key] = value
            
            st.success("✅ データが保存されました！")
            
            # 保存されたデータの表示
            with st.expander("📋 保存されたデータを確認", expanded=True):
                for key, value in form_data.items():
                    if value and value != "-":
                        st.write(f"**{key}**: {value}")
            
            # CSVエクスポート機能
            import pandas as pd
            import io
            
            df_export = pd.DataFrame([form_data])
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📄 CSVでダウンロード",
                data=csv_data.encode('utf-8-sig'),
                file_name=f"construction_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # フォーム外の自動入力セクション
    display_auto_fill_section(
        company_master, vectordb, intermediate_llm, tokenizer, embedding_function, config
    )
    
    st.divider()
    
    # 会話履歴の表示
    st.markdown("### 💬 会話履歴")
    if not st.session_state.query_history:
        st.info("👋 質問を入力して会話を開始してください。")
    else:
        for message in st.session_state.query_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 進行中の回答表示エリア
    streaming_placeholder = st.empty()

    # 質問入力
    user_query_input = st.chat_input("💭 質問を入力してください...")

    # --- 新しい質問が入力された場合の処理 ---
    if user_query_input:
        st.session_state.current_query = preprocess_query(user_query_input)
        st.session_state.evaluation_recorded_for_last_answer = False
        st.session_state.last_full_answer = ""
        st.session_state.variant_answers = []
        st.session_state.current_source_docs = []
        st.session_state.current_answer_stream = None # ストリームをクリア

        # 履歴に追加して表示 (重複表示を防ぐため、一旦コメントアウト。ストリーム完了後に追加)
        # st.session_state.query_history.append({"role": "user", "content": st.session_state.current_query})
        # with st.chat_message("user"): st.markdown(st.session_state.current_query)

        # メタデータフィルターの解析
        parsed_metadata_filter = None
        try:
            filter_str = st.session_state.metadata_filter_str.strip()
            if filter_str and filter_str != '{}':
                parsed_metadata_filter = json.loads(filter_str)
                if not isinstance(parsed_metadata_filter, dict):
                    st.warning("🔍 フィルターはJSON辞書形式で入力してください。", icon="⚠️")
                    parsed_metadata_filter = None
                else:
                    logger.info(f"Applying metadata filter: {parsed_metadata_filter}")
            else:
                logger.info("No metadata filter applied.")
        except json.JSONDecodeError:
            st.warning("🔍 フィルターのJSON形式が不正です。", icon="⚠️")
            parsed_metadata_filter = None

        # 回答生成プロセス開始
        try:
            logger.info(f"Calling ask_question_ensemble_stream with query: {st.session_state.current_query}")
            # <<< CORRECTED: ローカル変数からパイプラインコンポーネントを渡す >>>
            response = ask_question_ensemble_stream(
                vectordb=vectordb,                 # ローカル変数から
                intermediate_llm=intermediate_llm, # ローカル変数から
                tokenizer=tokenizer,               # ローカル変数から
                embedding_function=embedding_function, # ローカル変数から
                config=config,
                query=st.session_state.current_query,
                logger=logger,
                metadata_filter=parsed_metadata_filter,
                variant_params=st.session_state.variant_settings # サイドバーで設定された値
            )
            # <<< END CORRECTED >>>
            st.session_state.current_answer_stream = response.get("result_stream")
            st.session_state.current_source_docs = response.get("source_documents", [])
            st.session_state.variant_answers = response.get("variant_answers", [])
            logger.info("Response object received from ask_question_ensemble_stream.")
            # 質問を履歴に追加
            st.session_state.query_history.append({
                "role": "user", 
                "content": st.session_state.current_query
            })

        except Exception as e:
            logger.error(f"Error during ask_question_ensemble_stream call: {e}", exc_info=True)
            error_msg = f"質問処理中にエラーが発生しました: {str(e)[:100]}...\n設定や入力内容を確認してください。"
            st.error(error_msg)
            st.session_state.current_answer_stream = iter([f"エラーが発生しました: {str(e)[:100]}..."])
            st.session_state.current_source_docs = []
            st.session_state.variant_answers = []
            
            if (not st.session_state.query_history or 
                st.session_state.query_history[-1]['content'] != st.session_state.current_query):
                st.session_state.query_history.append({
                    "role": "user", 
                    "content": st.session_state.current_query
                })

        # ストリーミング表示処理のために再実行
        st.rerun()

    # --- 回答ストリーミング表示 ---
    if st.session_state.current_answer_stream:
        with streaming_placeholder.container():
             with st.chat_message("assistant"):
                 answer_placeholder = st.empty() # ここにチャンクを追記していく
                 full_response = ""
                 logger.info("Starting answer streaming...")
                 try:
                     for chunk in st.session_state.current_answer_stream:
                          full_response += chunk
                          answer_placeholder.markdown(full_response + "▌") # カーソル風アニメーション
                     answer_placeholder.markdown(full_response) # カーソルを消して最終結果表示
                     st.session_state.last_full_answer = full_response # 完全な回答を保存

                     # 回答のコピー機能
                     if full_response:
                         with st.expander("📋 回答をコピー", expanded=False):
                             st.code(full_response, language="markdown")
                             st.download_button(
                                 label="💾 ファイルとしてダウンロード",
                                 data=full_response,
                                 file_name=f"rag_answer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                 mime="text/plain",
                                 key="copy_answer_button",
                                 use_container_width=True
                             )

                     # 履歴への追加
                     if (not st.session_state.query_history or 
                         st.session_state.query_history[-1]['content'] != full_response):
                         
                         if (st.session_state.query_history and 
                             st.session_state.query_history[-1]['role'] == 'user'):
                             st.session_state.query_history.append({
                                 "role": "assistant", 
                                 "content": full_response
                             })
                         elif (st.session_state.query_history and 
                               "エラー" in st.session_state.query_history[-1]['content']):
                             st.session_state.query_history[-1]['content'] = full_response
                     logger.info("Streaming finished.")

                 except Exception as stream_e:
                     logger.error(f"Error during streaming display: {stream_e}", exc_info=True)
                     error_msg = f"回答表示中にエラーが発生しました: {str(stream_e)[:100]}...\nブラウザをリロードするか、再度質問を試してください。"
                     st.error(error_msg)
                     st.session_state.last_full_answer = f"表示エラー: {str(stream_e)[:100]}..."
                     error_message = st.session_state.last_full_answer
                     
                     if (not st.session_state.query_history or 
                         st.session_state.query_history[-1].get("content") != error_message):
                         if st.session_state.query_history and st.session_state.query_history[-1]['role'] == 'user':
                             st.session_state.query_history.append({
                                 "role": "assistant", 
                                 "content": error_message
                             })
                         elif st.session_state.query_history:
                             st.session_state.query_history[-1]['content'] = error_message

                 finally:
                     st.session_state.current_answer_stream = None
                     st.rerun()

    # --- 詳細情報とvariant比較・評価 ---
    # last_full_answer が確定したら表示
    if st.session_state.last_full_answer:
        with st.expander("詳細情報 (Variant比較・参照データ・評価)"):

            # Variant比較
            if st.session_state.variant_answers:
                st.markdown("#### 🔄 Variant 比較")
                for idx, v_ans in enumerate(st.session_state.variant_answers):
                    params = st.session_state.variant_settings[idx]
                    with st.expander(f"🎯 Variant {idx+1} (k={params['k']}, temp={params['temperature']:.2f})", expanded=False):
                        st.text_area(
                            "Variant Response", 
                            v_ans, 
                            height=150, 
                            disabled=True, 
                            label_visibility="collapsed", 
                            key=f"v_ans_{idx}"
                        )
                st.divider()

            # 参照データの表示
            st.markdown("#### 📚 参照データ")
            if st.session_state.current_source_docs:
                st.info(f"🔍 関連データチャンク: {len(st.session_state.current_source_docs)} 件")
                
                for i, doc in enumerate(st.session_state.current_source_docs):
                    meta_display = []
                    cols = config.metadata_display_columns
                    score = doc.metadata.get('rerank_score')
                    
                    for col_name in cols:
                        if value := doc.metadata.get(col_name):
                            meta_display.append(f"**{col_name[:4]}**: `{str(value)[:20]}`")
                    
                    if score:
                        meta_display.append(f"**Score**: `{score:.3f}`")
                    
                    with st.expander(f"📄 Chunk {i+1} - {' | '.join(meta_display)}", expanded=False):
                        st.text_area(
                            "Document Content",
                            doc.page_content,
                            height=150,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"src_content_{i}",
                            help="参照ドキュメントの内容"
                        )
            else:
                st.info("🔍 参照されたデータはありません。")
            st.divider()

            # 評価入力セクション
            st.markdown("#### ⭐ 回答の評価")
            if not st.session_state.evaluation_recorded_for_last_answer:
                eval_key_suffix = f"_{len(st.session_state.query_history)}"
                
                with st.form(f"evaluation_form_{eval_key_suffix}"):
                    st.markdown("🔍 **回答の質を評価してください**")
                    
                    eval_cols = st.columns(2)
                    with eval_cols[0]:
                        ans_opts = ["未選択", "✅ 優秀", "👍 良い", "🤔 部分的", "❌ 不十分"]
                        ans_eval = st.radio(
                            "💬 回答の品質",
                            ans_opts,
                            key=f"eval_ans{eval_key_suffix}",
                            horizontal=False
                        )
                    
                    with eval_cols[1]:
                        basis_opts = ["未選択", "✅ 適切", "⚠️ 部分的", "❌ 不適切", "🤷 不明"]
                        bas_eval = st.radio(
                            "📄 根拠の適切さ",
                            basis_opts,
                            key=f"eval_bas{eval_key_suffix}",
                            horizontal=False
                        )
                    
                    cmt = st.text_area(
                        "📝 コメント（任意）",
                        key=f"eval_com{eval_key_suffix}",
                        height=80,
                        placeholder="改善点やフィードバックを入力..."
                    )
                    
                    submitted = st.form_submit_button("💾 評価を記録", use_container_width=True)
                    
                    if submitted:
                        if ans_eval != "未選択" and bas_eval != "未選択":
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            query = (st.session_state.query_history[-2]['content'] 
                                   if len(st.session_state.query_history) >= 2 else "N/A")
                            
                            success = record_evaluation(
                                EVALUATION_FILE, timestamp, query,
                                st.session_state.last_full_answer,
                                st.session_state.current_source_docs,
                                ans_eval, bas_eval, cmt
                            )
                            
                            if success:
                                st.success("✔️ 評価を記録しました！")
                                st.session_state.evaluation_recorded_for_last_answer = True
                                st.rerun()
                            else:
                                st.error("⚠️ 評価の記録に失敗しました。")
                        else:
                            st.warning("⚠️ 回答と根拠の両方を評価してください。")
            else:
                st.success("✔️ この回答は既に評価済みです。")
            st.divider()

            # 評価ログの表示
            st.markdown("#### 📈 評価ログ")
            eval_file_path = os.path.join(os.path.dirname(__file__), EVALUATION_FILE)
            
            @st.cache_data
            def load_evaluation_data(filepath: str) -> Optional[pd.DataFrame]:
                if os.path.exists(filepath):
                    try:
                        return pd.read_csv(filepath)
                    except pd.errors.EmptyDataError:
                        return pd.DataFrame()
                    except Exception as e:
                        logger.error(f"Error loading evaluation log: {e}")
                        return None
                return None
            
            df_eval = load_evaluation_data(eval_file_path)
            
            if df_eval is not None and not df_eval.empty:
                with st.expander(f"📋 最新の評価ログ ({len(df_eval)}件)", expanded=False):
                    display_cols = ['timestamp', 'query', 'answer_evaluation', 'basis_evaluation', 'comment']
                    st.dataframe(
                        df_eval[display_cols].tail(10).reset_index(drop=True),
                        use_container_width=True
                    )
                    
                    @st.cache_data
                    def get_csv_data(df: pd.DataFrame) -> bytes:
                        return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    
                    csv_data = get_csv_data(df_eval)
                    st.download_button(
                        label="💾 全評価ログをダウンロード",
                        data=csv_data,
                        file_name=f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
            else:
                st.info("📈 まだ評価ログはありません。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        st.error(f"アプリケーションエラー: {str(e)[:100]}...")
        st.info("ブラウザをリロードして再度お試しください。")