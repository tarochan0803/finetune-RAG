# simple_rag_app.py - シンプル即座RAGアプリ
# 既存のRAGインフラ活用で即座に動作

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# PATHに追加
sys.path.append('/home/ncnadmin/.local/bin')
os.environ['PATH'] = '/home/ncnadmin/.local/bin:' + os.environ.get('PATH', '')

import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any

# 既存のRAGモジュールをインポート
try:
    from config import Config
    from utils import preprocess_query, format_document_snippet
    # 簡易版なので既存のRAGユーティリティを使用
    print("✅ 既存RAGモジュール読み込み成功")
except ImportError as e:
    st.error(f"モジュール読み込みエラー: {e}")
    st.stop()

class SimpleRAGInterface:
    """シンプルRAGインターフェース"""
    
    def __init__(self):
        self.config = Config()
        self.premium_data = self.load_premium_data()
        
    def load_premium_data(self) -> List[Dict]:
        """高品質データ読み込み"""
        try:
            import json
            data = []
            with open("/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            return []
    
    def simple_search(self, query: str, limit: int = 5) -> List[Dict]:
        """シンプル検索（キーワードマッチング）"""
        query_lower = query.lower()
        matches = []
        
        for item in self.premium_data:
            input_text = item.get('input', '').lower()
            output_text = item.get('output', '').lower()
            
            # キーワードマッチング
            score = 0
            keywords = query_lower.split()
            
            for keyword in keywords:
                if keyword in input_text:
                    score += 2
                if keyword in output_text:
                    score += 1
            
            if score > 0:
                matches.append({
                    **item,
                    'score': score
                })
        
        # スコア順でソート
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:limit]
    
    def generate_simple_answer(self, query: str, matches: List[Dict]) -> str:
        """シンプル回答生成（ルールベース）"""
        if not matches:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。"
        
        # 最高スコアの回答を基本とする
        best_match = matches[0]
        answer = best_match.get('output', '')
        
        # 複数の関連情報がある場合は統合
        if len(matches) > 1:
            additional_info = []
            for match in matches[1:3]:  # 上位3件まで
                info = match.get('output', '')
                if info and info != answer and len(info) > 10:
                    additional_info.append(info)
            
            if additional_info:
                answer += f"\n\n補足情報:\n" + "\n".join(f"• {info}" for info in additional_info)
        
        return answer
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """クエリ実行"""
        start_time = time.time()
        
        # 前処理
        processed_query = preprocess_query(user_query)
        
        # 検索
        matches = self.simple_search(processed_query)
        
        # 回答生成
        answer = self.generate_simple_answer(processed_query, matches)
        
        end_time = time.time()
        
        return {
            'answer': answer,
            'matches': matches,
            'processing_time': end_time - start_time,
            'match_count': len(matches)
        }

def main():
    st.set_page_config(page_title="シンプルRAG", layout="centered")
    st.title("⚡ シンプルRAGシステム")
    st.caption("60,403件の高品質データを活用 - 即座に動作")
    
    # システム初期化
    @st.cache_resource
    def load_rag_interface():
        return SimpleRAGInterface()
    
    rag_interface = load_rag_interface()
    
    if not rag_interface.premium_data:
        st.error("❌ データの読み込みに失敗しました")
        st.stop()
    
    st.success(f"✅ {len(rag_interface.premium_data):,}件のデータを読み込み完了")
    
    # サイドバー
    with st.sidebar:
        st.header("📊 システム情報")
        st.metric("データ件数", f"{len(rag_interface.premium_data):,}")
        st.metric("データ品質", "高品質拡張済み")
        st.info("💡 キーワードマッチング検索")
        st.info("🔍 ルールベース回答生成")
        st.info("⚡ 超高速動作")
        
        # サンプル質問
        st.header("💬 サンプル質問")
        sample_queries = [
            "株式会社三建の壁仕様について",
            "仮筋交の標準仕様は？",
            "鋼製束の使用基準",
            "デザオ建設の壁面材仕様",
            "基礎の立上高さについて"
        ]
        
        for sample in sample_queries:
            if st.button(sample, key=f"sample_{sample}"):
                st.session_state['sample_query'] = sample
    
    # メインチャット
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # サンプル質問が選択された場合
    if 'sample_query' in st.session_state:
        user_input = st.session_state['sample_query']
        del st.session_state['sample_query']
    else:
        user_input = None
    
    # 会話履歴表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 質問入力
    if not user_input:
        user_input = st.chat_input("質問を入力してください...")
    
    if user_input:
        # ユーザーメッセージ表示
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI応答
        with st.chat_message("assistant"):
            with st.spinner("🔍 検索中..."):
                result = rag_interface.query(user_input)
            
            # 回答表示
            st.markdown(result["answer"])
            
            # 詳細情報
            with st.expander("📊 検索詳細"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("処理時間", f"{result['processing_time']:.3f}秒")
                with col2:
                    st.metric("マッチ件数", result['match_count'])
                
                if result['matches']:
                    st.subheader("🔍 マッチした情報")
                    for i, match in enumerate(result['matches'][:3]):
                        with st.expander(f"マッチ {i+1} (スコア: {match['score']})"):
                            st.text_area("入力", match.get('input', ''), height=60, disabled=True)
                            st.text_area("出力", match.get('output', ''), height=60, disabled=True)
        
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

if __name__ == "__main__":
    main()