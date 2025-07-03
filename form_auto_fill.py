# form_auto_fill.py - フォーム外での自動入力処理

import streamlit as st
import logging
from auto_fill_utils import auto_fill_form_data

logger = logging.getLogger("RAGApp")

def display_auto_fill_section(company_master, vectordb, intermediate_llm, tokenizer, embedding_function, config):
    """フォーム外での自動入力セクションを表示（Gemini API優先）"""
    
    # 工務店が選択されている場合のみ表示
    if st.session_state.selected_company:
        st.markdown("#### 🤖 自動入力機能")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"選択中の工務店: **{st.session_state.selected_company}**")
        
        with col2:
            # API設定状況の表示
            if hasattr(config, 'gemini_api_key') and config.gemini_api_key:
                st.success("🚀 Gemini 2.0 Flash Exp Ready")
                button_label = "⚡ AI自動入力 (Gemini)"
            else:
                st.warning("🔧 ローカル処理モード")
                button_label = "🛠️ ローカル自動入力"
                
            if st.button(button_label, use_container_width=True, type="primary"):
                execute_auto_fill(
                    st.session_state.selected_company,
                    vectordb, intermediate_llm, tokenizer, embedding_function, config
                )

def execute_auto_fill(company_name, vectordb, intermediate_llm, tokenizer, embedding_function, config):
    """自動入力を実行（Gemini API優先）"""
    
    # 処理モードの表示
    if hasattr(config, 'gemini_api_key') and config.gemini_api_key:
        spinner_text = "🤖 Gemini 2.0で工務店仕様を分析中..."
    else:
        spinner_text = "🔍 ローカル処理で工務店仕様を検索中..."
        
    with st.spinner(spinner_text):
        try:
            auto_fill_results = auto_fill_form_data(
                company_name, 
                vectordb, intermediate_llm, tokenizer, embedding_function, config, logger
            )
            
            if auto_fill_results:
                # セッション状態に結果を保存
                for field, value in auto_fill_results.items():
                    st.session_state[field] = value
                
                # 使用したAPIの表示
                if hasattr(config, 'gemini_api_key') and config.gemini_api_key:
                    st.success(f"🚀 Gemini 2.0で {len(auto_fill_results)}項目の自動入力が完了しました！")
                else:
                    st.success(f"🛠️ ローカル処理で {len(auto_fill_results)}項目の自動入力が完了しました！")
                
                # 自動入力された内容を表示
                with st.expander("📋 自動入力された内容", expanded=True):
                    for field, value in auto_fill_results.items():
                        field_name = get_field_display_name(field)
                        st.write(f"**{field_name}**: {value}")
                
                st.info("💡 AI生成結果を確認・修正して保存してください。")
                st.rerun()
                
            else:
                if hasattr(config, 'gemini_api_key') and config.gemini_api_key:
                    st.warning("⚠️ Gemini APIでこの工務店の詳細仕様データは見つかりませんでした。")
                else:
                    st.warning("⚠️ ローカル処理でこの工務店の詳細仕様データは見つかりませんでした。\n💡 GEMINI_API_KEYを設定するとより高精度な検索が可能です。")
                
        except Exception as e:
            logger.error(f"Auto-fill execution error: {e}", exc_info=True)
            st.error(f"❌ 自動入力中にエラーが発生しました: {str(e)[:100]}...")

def get_field_display_name(field_name):
    """フィールド名を表示用名前に変換"""
    field_names = {
        "form_outer_wall_strong": "耐力壁(外部)",
        "form_inner_wall_strong": "耐力壁(内部)", 
        "form_non_strong_wall": "非耐力壁",
        "form_wall_type": "外壁仕様",
        "form_sub_materials": "副資材の供給",
        "form_temporary_brace": "仮筋交",
        "form_foundation_packing": "基礎パッキン",
        "form_airtight_packing": "気密パッキン",
        "form_steel_post": "鋼製束",
        "form_hardware_install": "金物取付"
    }
    return field_names.get(field_name, field_name)