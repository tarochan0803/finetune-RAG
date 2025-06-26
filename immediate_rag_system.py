# immediate_rag_system.py - 即座に動作するRAGシステム
# ファインチューニング不要、既存モデル+高品質データで即座に動作

import os
import sys
import logging
import streamlit as st
import pandas as pd
import torch
from typing import List, Dict, Any, Optional
import time

# 必要モジュールのインポート
try:
    from langchain_chroma import Chroma
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError as e:
    print(f"Import Error: {e}")
    print("必要なパッケージをインストールしています...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain", "langchain-chroma", "langchain-huggingface"])

class ImmediateRAGSystem:
    """即座に動作するRAGシステム"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.embedding_model = None
        self.vectordb = None
        self.llm_pipeline = None
        self.initialized = False
        
    def setup_logging(self):
        """ロギング設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize_components(self):
        """コンポーネント初期化"""
        self.logger.info("🚀 即座RAGシステム初期化開始")
        
        try:
            # 1. 埋め込みモデル（軽量）
            self.logger.info("📊 埋め込みモデル初期化...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-base",
                model_kwargs={'device': 'cpu'}  # CPU使用で安定性確保
            )
            
            # 2. VectorDB（既存のchroma_dbを使用）
            self.logger.info("🗄️ VectorDB接続...")
            db_path = "./chroma_db"
            if os.path.exists(db_path):
                self.vectordb = Chroma(
                    collection_name="my_collection",
                    persist_directory=db_path,
                    embedding_function=self.embedding_model
                )
                doc_count = self.vectordb._collection.count()
                self.logger.info(f"✅ VectorDB接続完了 ({doc_count}件)")
            else:
                self.logger.warning("❌ VectorDBが見つかりません")
                return False
            
            # 3. LLMパイプライン（軽量モデル使用）
            self.logger.info("🤖 LLMパイプライン初期化...")
            
            # GPU使用可能かチェック
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"使用デバイス: {device}")
            
            if device == "cpu":
                # CPU用軽量設定
                model_name = "rinna/japanese-gpt2-medium"  # より軽量なモデル
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                # GPU用設定（CUDA互換性問題対応）
                model_name = "rinna/japanese-gpt2-medium"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                device=0 if device == "cuda" else -1
            )
            
            self.logger.info("✅ LLMパイプライン初期化完了")
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初期化エラー: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """ドキュメント検索"""
        if not self.vectordb:
            return []
        
        try:
            docs = self.vectordb.similarity_search(query, k=k)
            self.logger.info(f"🔍 検索完了: {len(docs)}件取得")
            return docs
        except Exception as e:
            self.logger.error(f"検索エラー: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """回答生成"""
        if not self.llm_pipeline:
            return "LLMが初期化されていません"
        
        try:
            # コンテキスト作成
            context = "\n".join([doc.page_content[:200] for doc in context_docs[:3]])
            
            # プロンプト作成
            prompt = f"""以下の情報を参考に、質問に答えてください。

参考情報:
{context}

質問: {query}

回答:"""
            
            # 回答生成
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )
            
            # 回答抽出
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "申し訳ございませんが、適切な回答を生成できませんでした。"
            
        except Exception as e:
            self.logger.error(f"回答生成エラー: {e}")
            return f"回答生成中にエラーが発生しました: {e}"
    
    def query_rag(self, query: str) -> Dict[str, Any]:
        """RAGクエリ実行"""
        if not self.initialized:
            return {"answer": "システムが初期化されていません", "sources": []}
        
        start_time = time.time()
        
        # 1. ドキュメント検索
        docs = self.search_documents(query)
        
        # 2. 回答生成
        answer = self.generate_answer(query, docs)
        
        end_time = time.time()
        
        return {
            "answer": answer,
            "sources": docs,
            "processing_time": end_time - start_time,
            "source_count": len(docs)
        }

def create_streamlit_app():
    """Streamlitアプリ作成"""
    st.set_page_config(page_title="即座RAG", layout="centered")
    st.title("⚡ 即座RAGシステム")
    st.caption("ファインチューニング不要 - 即座に動作する高品質RAG")
    
    # システム初期化
    @st.cache_resource
    def load_rag_system():
        system = ImmediateRAGSystem()
        if system.initialize_components():
            return system
        else:
            return None
    
    rag_system = load_rag_system()
    
    if not rag_system:
        st.error("🚫 RAGシステムの初期化に失敗しました")
        st.info("以下を確認してください:")
        st.info("1. chroma_dbディレクトリが存在するか")
        st.info("2. 必要なパッケージがインストールされているか")
        st.stop()
    
    st.success("✅ RAGシステム初期化完了")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ システム情報")
        st.info("💡 軽量モデル使用")
        st.info("🔍 既存VectorDB活用")
        st.info("⚡ 即座に動作")
        
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                st.success(f"🎮 GPU: {gpu_name}")
            except:
                st.info("💻 CPU使用")
        else:
            st.info("💻 CPU使用")
    
    # メインチャット
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 会話履歴表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 質問入力
    if user_input := st.chat_input("質問を入力してください..."):
        # ユーザーメッセージ表示
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI応答
        with st.chat_message("assistant"):
            with st.spinner("🤔 考え中..."):
                result = rag_system.query_rag(user_input)
            
            # 回答表示
            st.markdown(result["answer"])
            
            # 詳細情報
            with st.expander("📊 詳細情報"):
                st.write(f"⏱️ 処理時間: {result['processing_time']:.2f}秒")
                st.write(f"📄 参考文書数: {result['source_count']}件")
                
                if result["sources"]:
                    st.write("📖 参考文書:")
                    for i, doc in enumerate(result["sources"][:3]):
                        st.text_area(
                            f"文書 {i+1}",
                            doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            height=100,
                            disabled=True
                        )
        
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

def main():
    """メイン実行"""
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        create_streamlit_app()
    else:
        # CLI版テスト
        print("🚀 即座RAGシステム - CLI版テスト")
        
        system = ImmediateRAGSystem()
        if not system.initialize_components():
            print("❌ 初期化失敗")
            return
        
        print("✅ 初期化完了")
        
        # テストクエリ
        test_queries = [
            "株式会社三建の壁仕様について教えてください",
            "仮筋交の標準仕様は何ですか",
            "鋼製束の使用基準について"
        ]
        
        for query in test_queries:
            print(f"\n❓ 質問: {query}")
            result = system.query_rag(query)
            print(f"💬 回答: {result['answer']}")
            print(f"⏱️ 処理時間: {result['processing_time']:.2f}秒")
            print("-" * 50)

if __name__ == "__main__":
    main()