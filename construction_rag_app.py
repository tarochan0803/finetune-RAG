#!/usr/bin/env python3
"""
建設業界特化RAGシステム - Streamlitアプリケーション
"""

import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
sys.path.append('/home/ncnadmin/my_rag_project')

from config import Config
from construction_rag_system import ConstructionRAGSystem, RAGResponse

# ページ設定
st.set_page_config(
    page_title="建設業界RAGシステム",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2e8b57;
        margin: 1rem 0;
    }
    .search-result {
        background-color: #fff8dc;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
    }
    .confidence-high { color: #2e8b57; font-weight: bold; }
    .confidence-medium { color: #ff8c00; font-weight: bold; }
    .confidence-low { color: #dc143c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """RAGシステムを初期化（キャッシュ）"""
    config = Config()
    rag_system = ConstructionRAGSystem(config)
    
    with st.spinner("RAGシステムを初期化中..."):
        if rag_system.initialize():
            return rag_system, config
        else:
            st.error("RAGシステムの初期化に失敗しました")
            return None, None

def format_confidence_score(score: float) -> str:
    """信頼度スコアをフォーマット"""
    if score >= 0.7:
        return f'<span class="confidence-high">高 ({score:.2f})</span>'
    elif score >= 0.4:
        return f'<span class="confidence-medium">中 ({score:.2f})</span>'
    else:
        return f'<span class="confidence-low">低 ({score:.2f})</span>'

def display_search_results(search_results, title: str):
    """検索結果を表示"""
    with st.expander(f"🔍 {title}", expanded=False):
        for i, results in enumerate(search_results):
            st.subheader(f"検索戦略 {i+1}")
            if results:
                for j, result in enumerate(results[:3]):  # 上位3件のみ表示
                    st.markdown(f"""
                    <div class="search-result">
                        <strong>#{j+1} | 会社:</strong> {result.company}<br>
                        <strong>スコア:</strong> {result.score:.3f}<br>
                        <strong>内容:</strong> {result.document[:200]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("検索結果がありませんでした")

def create_performance_chart(response: RAGResponse):
    """パフォーマンスチャートを作成"""
    # 処理時間の可視化
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['処理時間'],
        y=[response.processing_time],
        name='秒',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="処理時間",
        yaxis_title="秒",
        showlegend=False,
        height=300
    )
    
    return fig

def create_confidence_gauge(confidence: float):
    """信頼度ゲージを作成"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "信頼度 (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """メイン処理"""
    
    # ヘッダー
    st.markdown('<h1 class="main-header">🏗️ 建設業界RAGシステム</h1>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # システム情報
        st.subheader("システム情報")
        st.info("""
        **建設業界特化RAGシステム**
        - データ数: 31,199件
        - 対象会社: 186社
        - モデル: Qwen1.5-1.8B + LoRA
        - API: Gemini 1.5 Flash
        """)
        
        # デバッグモード
        debug_mode = st.checkbox("デバッグモード", value=False)
        
        # 検索設定
        st.subheader("検索設定")
        search_variants = st.slider("検索バリアント数", 1, 3, 3)
        
        # システム初期化
        if st.button("🔄 システム再初期化"):
            st.cache_resource.clear()
            st.rerun()
    
    # RAGシステム初期化
    rag_system, config = initialize_rag_system()
    
    if rag_system is None:
        st.error("システムを初期化できませんでした。設定を確認してください。")
        return
    
    # メインコンテンツ
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 質問入力")
        
        # 質問例
        example_queries = [
            "株式会社平成建設の壁面材仕様は何ですか？",
            "大壁仕様の会社を教えてください",
            "羽柄材の供給がある会社はありますか？",
            "鋼製束が有の会社一覧を教えてください",
            "真壁仕様を採用している会社は？"
        ]
        
        selected_example = st.selectbox(
            "質問例を選択（または下に直接入力）",
            [""] + example_queries
        )
        
        # 質問入力
        user_query = st.text_area(
            "質問を入力してください",
            value=selected_example,
            height=100,
            placeholder="例: 株式会社○○の仕様について教えてください"
        )
        
        # 検索実行
        if st.button("🔍 検索実行", type="primary") and user_query.strip():
            
            with st.spinner("処理中..."):
                start_time = time.time()
                
                # クエリ処理
                response = rag_system.process_query(user_query.strip())
                
                end_time = time.time()
            
            # 結果表示
            st.markdown(f'<div class="query-box"><strong>🔍 質問:</strong> {user_query}</div>', 
                       unsafe_allow_html=True)
            
            st.markdown(f'<div class="answer-box"><strong>📝 回答:</strong><br>{response.final_answer}</div>', 
                       unsafe_allow_html=True)
            
            # 統計情報
            col1_stat, col2_stat, col3_stat = st.columns(3)
            with col1_stat:
                st.metric("処理時間", f"{response.processing_time:.2f}秒")
            with col2_stat:
                confidence_html = format_confidence_score(response.confidence_score)
                st.markdown(f"**信頼度:** {confidence_html}", unsafe_allow_html=True)
            with col3_stat:
                total_results = sum(len(results) for results in response.search_results)
                st.metric("検索結果数", total_results)
            
            # デバッグ情報
            if debug_mode:
                st.subheader("🔧 デバッグ情報")
                
                # タブで詳細情報を表示
                tab1, tab2, tab3 = st.tabs(["中間回答", "検索結果", "パフォーマンス"])
                
                with tab1:
                    for i, answer in enumerate(response.intermediate_answers):
                        st.text_area(f"中間回答 {i+1}", answer, height=100)
                
                with tab2:
                    display_search_results(response.search_results, "詳細検索結果")
                
                with tab3:
                    col1_perf, col2_perf = st.columns(2)
                    with col1_perf:
                        fig_time = create_performance_chart(response)
                        st.plotly_chart(fig_time, use_container_width=True)
                    with col2_perf:
                        fig_conf = create_confidence_gauge(response.confidence_score)
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # JSON出力
                    if st.checkbox("JSON出力を表示"):
                        response_dict = {
                            "query": response.query,
                            "final_answer": response.final_answer,
                            "processing_time": response.processing_time,
                            "confidence_score": response.confidence_score,
                            "intermediate_answers": response.intermediate_answers
                        }
                        st.json(response_dict)
    
    with col2:
        st.subheader("📊 システム状態")
        
        # システム稼働時間
        if 'start_time' not in st.session_state:
            st.session_state.start_time = datetime.now()
        
        uptime = datetime.now() - st.session_state.start_time
        st.metric("稼働時間", f"{uptime.seconds // 3600}時間{(uptime.seconds % 3600) // 60}分")
        
        # セッション統計
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        if user_query and st.button("🔍 検索実行", type="primary", key="search_btn_2"):
            st.session_state.query_count += 1
        
        st.metric("検索回数", st.session_state.query_count)
        
        # リソース使用状況（簡易版）
        st.subheader("💾 リソース情報")
        st.info(f"""
        **データベース:** ChromaDB
        **コレクション:** {config.collection_name if config else 'N/A'}
        **埋め込みモデル:** multilingual-e5-base
        **LLMモデル:** Qwen1.5-1.8B + LoRA
        **API:** Gemini 1.5 Flash
        """)
        
        # ヘルプセクション
        st.subheader("❓ 使い方")
        with st.expander("詳細ガイド"):
            st.markdown("""
            ### 質問の仕方
            
            **良い質問例:**
            - 「株式会社○○の壁面材仕様は？」
            - 「大壁仕様を採用している会社は？」
            - 「羽柄材供給がある会社を教えて」
            
            **コツ:**
            - 具体的な会社名や仕様名を含める
            - 「は」「を」「について」などの助詞を使う
            - 複数の条件がある場合は分けて質問
            
            ### 機能説明
            - **アンサンブル検索**: 複数の検索戦略を組み合わせ
            - **ファインチューニング**: 建設業界特化モデル
            - **信頼度**: 回答の確実性を数値化
            """)

if __name__ == "__main__":
    main()