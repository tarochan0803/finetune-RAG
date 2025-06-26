# RAGapp_optimized.py - 最適化完了版RAGアプリ
import streamlit as st
import pandas as pd
import os
import datetime
import sys
import logging
import json
import torch
from typing import Optional, Tuple, List, Dict, Any

# 最適化モジュールのインポート
try:
    from config_optimized import Config, setup_logging
    from rag_query_utils import initialize_pipeline, ask_question_ensemble_stream
    from utils import format_document_snippet, normalize_str, preprocess_query
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# グローバル設定
config = Config()
logger = setup_logging(config, log_filename="rag_optimized_app.log")

@st.cache_resource
def load_optimized_pipeline():
    """最適化パイプライン初期化"""
    logger.info("最適化パイプライン初期化中...")
    try:
        pipeline_components = initialize_pipeline(config, logger)
        if not all(comp is not None for comp in pipeline_components):
            logger.error("パイプライン初期化失敗")
            return (None,) * 4
        logger.info("最適化パイプライン初期化完了")
        return pipeline_components
    except Exception as e:
        logger.critical(f"パイプライン初期化エラー: {e}", exc_info=True)
        return (None,) * 4

def main():
    st.set_page_config(page_title="最適化RAG", layout="centered")
    st.title("🚀 最適化RAGシステム")
    st.caption("高品質データ + ファインチューニング + 推論最適化")
    
    # パイプライン初期化
    with st.spinner("最適化システム初期化中..."):
        vectordb, intermediate_llm, tokenizer, embedding_function = load_optimized_pipeline()
    
    if vectordb is None:
        st.error("システム初期化に失敗しました")
        st.stop()
    
    # セッション状態初期化
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_answer_stream" not in st.session_state:
        st.session_state.current_answer_stream = None
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 最適化設定")
        st.success("✅ 高品質データ（60,403件）")
        st.success("✅ ファインチューニング済み")
        st.success("✅ 推論最適化")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.info(f"GPU: {gpu_name}")
    
    # 会話履歴表示
    for message in st.session_state.query_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 質問入力
    if user_input := st.chat_input("質問を入力してください..."):
        # ユーザーメッセージ表示
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.query_history.append({"role": "user", "content": user_input})
        
        # AI応答生成
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                # 最適化クエリ処理
                response = ask_question_ensemble_stream(
                    vectordb=vectordb,
                    intermediate_llm=intermediate_llm,
                    tokenizer=tokenizer,
                    embedding_function=embedding_function,
                    config=config,
                    query=preprocess_query(user_input),
                    logger=logger
                )
                
                # ストリーミング表示
                full_response = ""
                for chunk in response.get("result_stream", []):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.query_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"エラーが発生しました: {e}"
                response_placeholder.error(error_msg)
                st.session_state.query_history.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
