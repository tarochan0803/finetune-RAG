# final_rag_app.py - 最終版RAGアプリケーション
# 高品質データ + 既存RAGインフラ + 最適化 = 完全なローカルRAGシステム

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# PATH設定
sys.path.append('/home/ncnadmin/.local/bin')
os.environ['PATH'] = '/home/ncnadmin/.local/bin:' + os.environ.get('PATH', '')

import streamlit as st
import pandas as pd
import time
import json
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime

class FinalRAGSystem:
    """最終版RAGシステム - 高品質データ活用"""
    
    def __init__(self):
        self.premium_data = self.load_premium_data()
        self.company_data = self.organize_by_company()
        self.spec_categories = self.categorize_specs()
        
    def load_premium_data(self) -> List[Dict]:
        """高品質拡張データ読み込み"""
        try:
            data = []
            with open("/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item)
            return data
        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            return []
    
    def organize_by_company(self) -> Dict[str, List[Dict]]:
        """工務店別データ整理"""
        companies = {}
        for item in self.premium_data:
            input_text = item.get('input', '')
            if '株式会社' in input_text:
                company = input_text.split('代表条件表')[0].strip()
                if company not in companies:
                    companies[company] = []
                companies[company].append(item)
        return companies
    
    def categorize_specs(self) -> Dict[str, List[Dict]]:
        """仕様カテゴリ別整理"""
        categories = {
            '壁仕様': [],
            '耐力壁': [],
            '構造材': [],
            '金物': [],
            '基礎': [],
            '副資材': [],
            'その他': []
        }
        
        for item in self.premium_data:
            input_text = item.get('input', '').lower()
            
            if '壁面材仕様' in input_text or '壁仕様' in input_text:
                categories['壁仕様'].append(item)
            elif '耐力壁' in input_text:
                categories['耐力壁'].append(item)
            elif '土台' in input_text or '柱' in input_text or '横架材' in input_text:
                categories['構造材'].append(item)
            elif '金物' in input_text or 'スクリュー' in input_text:
                categories['金物'].append(item)
            elif '基礎' in input_text:
                categories['基礎'].append(item)
            elif '鋼製束' in input_text or '仮筋交' in input_text:
                categories['副資材'].append(item)
            else:
                categories['その他'].append(item)
        
        return categories
    
    def smart_search(self, query: str, search_type: str = "general") -> List[Dict]:
        """スマート検索"""
        query_lower = query.lower()
        matches = []
        
        # 検索対象データを決定
        if search_type == "company" and '株式会社' in query:
            company_name = None
            for company in self.company_data.keys():
                if company in query:
                    company_name = company
                    break
            
            if company_name:
                search_data = self.company_data[company_name]
            else:
                search_data = self.premium_data
        elif search_type == "category":
            # カテゴリ別検索
            category_data = []
            for category, items in self.spec_categories.items():
                if category.lower() in query_lower:
                    category_data.extend(items)
            search_data = category_data if category_data else self.premium_data
        else:
            search_data = self.premium_data
        
        # スコアリング検索
        for item in search_data:
            input_text = item.get('input', '').lower()
            output_text = item.get('output', '').lower()
            instruction = item.get('instruction', '').lower()
            
            score = 0
            keywords = query_lower.split()
            
            # キーワードマッチングスコア
            for keyword in keywords:
                if keyword in input_text:
                    score += 3  # 入力テキストマッチは高スコア
                if keyword in output_text:
                    score += 2  # 出力テキストマッチ
                if keyword in instruction:
                    score += 1  # 指示テキストマッチ
            
            # 完全一致ボーナス
            if query_lower in input_text or query_lower in output_text:
                score += 5
            
            if score > 0:
                matches.append({
                    **item,
                    'search_score': score
                })
        
        # スコア順ソート
        matches.sort(key=lambda x: x['search_score'], reverse=True)
        return matches[:10]  # 上位10件
    
    def generate_enhanced_answer(self, query: str, matches: List[Dict]) -> str:
        """高品質回答生成"""
        if not matches:
            return "申し訳ございませんが、ご質問に関する情報が見つかりませんでした。\n\n別のキーワードでお試しください。"
        
        # 最高スコアの回答をベースに
        primary_answer = matches[0].get('output', '')
        
        # 工務店名の特定
        company_mentioned = None
        for company in self.company_data.keys():
            if company in query or company in matches[0].get('input', ''):
                company_mentioned = company
                break
        
        # 回答構築
        answer_parts = []
        
        if company_mentioned:
            answer_parts.append(f"**{company_mentioned}** について:")
        
        answer_parts.append(primary_answer)
        
        # 関連情報の追加
        if len(matches) > 1:
            related_info = []
            for match in matches[1:4]:  # 関連情報3件まで
                output = match.get('output', '').strip()
                if output and output != primary_answer and len(output) > 10:
                    # 重複チェック
                    if not any(output in existing for existing in related_info):
                        related_info.append(output)
            
            if related_info:
                answer_parts.append("\n**関連情報:**")
                for info in related_info:
                    answer_parts.append(f"• {info}")
        
        return "\n".join(answer_parts)
    
    def query_rag(self, user_query: str) -> Dict[str, Any]:
        """RAGクエリ実行"""
        start_time = time.time()
        
        # 検索タイプ判定
        search_type = "general"
        if '株式会社' in user_query:
            search_type = "company"
        elif any(cat in user_query.lower() for cat in ['壁', '構造', '金物', '基礎']):
            search_type = "category"
        
        # スマート検索実行
        matches = self.smart_search(user_query, search_type)
        
        # 高品質回答生成
        answer = self.generate_enhanced_answer(user_query, matches)
        
        end_time = time.time()
        
        return {
            'answer': answer,
            'matches': matches,
            'search_type': search_type,
            'processing_time': end_time - start_time,
            'data_source': 'Premium Training Dataset (60,403 items)'
        }

def create_streamlit_app():
    """Streamlitアプリ作成"""
    st.set_page_config(
        page_title="最終版RAG", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 最終版RAGシステム")
    st.caption("60,403件の高品質データ + 最適化検索 = 完全なローカルRAG")
    
    # システム初期化
    @st.cache_resource
    def load_final_rag():
        return FinalRAGSystem()
    
    rag_system = load_final_rag()
    
    if not rag_system.premium_data:
        st.error("❌ システム初期化に失敗しました")
        st.stop()
    
    # メインレイアウト
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # サイドバー情報
        st.header("📊 システム情報")
        
        # 統計情報
        total_data = len(rag_system.premium_data)
        companies = len(rag_system.company_data)
        
        st.metric("総データ数", f"{total_data:,}件")
        st.metric("対応工務店数", f"{companies}社")
        
        # カテゴリ別データ数
        st.subheader("📋 カテゴリ別データ")
        for category, items in rag_system.spec_categories.items():
            if items:
                st.write(f"• {category}: {len(items)}件")
        
        # 工務店リスト
        st.subheader("🏢 対応工務店")
        for company in list(rag_system.company_data.keys())[:10]:
            count = len(rag_system.company_data[company])
            st.write(f"• {company}: {count}件")
        
        if len(rag_system.company_data) > 10:
            st.write(f"... 他{len(rag_system.company_data) - 10}社")
        
        # システム特徴
        st.subheader("✨ システム特徴")
        st.success("✅ 60,403件の高品質データ")
        st.success("✅ スマート検索アルゴリズム")
        st.success("✅ 工務店別特化検索")
        st.success("✅ カテゴリ別整理")
        st.success("✅ 完全ローカル動作")
        st.success("✅ APIコストゼロ")
        
        # パフォーマンス情報
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                st.info(f"🎮 GPU: {gpu_name}")
            except:
                st.info("💻 CPU動作")
        else:
            st.info("💻 CPU動作")
    
    with col1:
        # メインチャット
        st.header("💬 チャット")
        
        # サンプル質問
        st.subheader("💡 サンプル質問")
        sample_cols = st.columns(3)
        
        sample_queries = [
            "株式会社三建の壁仕様について",
            "株式会社デザオ建設の壁面材仕様は？",
            "仮筋交の標準仕様について教えて",
            "鋼製束の使用基準は？",
            "耐力壁仕様の1級9㎜について",
            "基礎の立上高さの規定は？"
        ]
        
        for i, sample in enumerate(sample_queries):
            col = sample_cols[i % 3]
            if col.button(sample, key=f"sample_{i}"):
                st.session_state['sample_query'] = sample
        
        # チャット履歴
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # サンプル質問処理
        if 'sample_query' in st.session_state:
            user_input = st.session_state['sample_query']
            del st.session_state['sample_query']
        else:
            user_input = None
        
        # 会話履歴表示
        chat_container = st.container()
        with chat_container:
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
                with st.spinner("🔍 検索・分析中..."):
                    result = rag_system.query_rag(user_input)
                
                # 回答表示
                st.markdown(result["answer"])
                
                # 詳細情報
                with st.expander("📊 検索詳細情報"):
                    detail_cols = st.columns(4)
                    
                    with detail_cols[0]:
                        st.metric("処理時間", f"{result['processing_time']:.3f}秒")
                    with detail_cols[1]:
                        st.metric("マッチ数", len(result['matches']))
                    with detail_cols[2]:
                        st.metric("検索タイプ", result['search_type'])
                    with detail_cols[3]:
                        st.metric("データソース", "Premium")
                    
                    # マッチした情報の表示
                    if result['matches']:
                        st.subheader("🎯 マッチした情報 (上位5件)")
                        for i, match in enumerate(result['matches'][:5]):
                            with st.expander(f"マッチ {i+1} (スコア: {match.get('search_score', 0)})"):
                                st.text_area("質問/入力", match.get('input', ''), height=80, disabled=True)
                                st.text_area("回答/出力", match.get('output', ''), height=80, disabled=True)
                                if match.get('instruction'):
                                    st.text_area("指示", match.get('instruction', ''), height=60, disabled=True)
            
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        
        # 統計表示
        if st.session_state.messages:
            st.subheader("📈 セッション統計")
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            st.metric("質問数", len(user_messages))
            st.metric("データ活用率", f"{total_data:,}件活用中")

def main():
    create_streamlit_app()

if __name__ == "__main__":
    main()