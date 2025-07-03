# full_spec_RAG.py - フルスペック版RAGシステム
# ファインチューニング済みモデル + 最高性能RAG = 完全なエンタープライズRAGシステム

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
import gc
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Transformers関連
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    BitsAndBytesConfig, GenerationConfig
)
from peft import PeftModel

# RAG関連
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import chromadb

# 設定読み込み
from config import Config, setup_logging

class FullSpecRAGSystem:
    """フルスペック版RAGシステム - ファインチューニング済みモデル統合"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config, "full_spec_rag.log")
        
        # システム状態
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None
        
        # パフォーマンス追跡
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'cache_hits': 0,
            'model_load_time': 0.0
        }
        
        # 初期化
        self.logger.info("🚀 フルスペック版RAGシステム初期化開始")
        self._initialize_system()
    
    def _initialize_system(self):
        """システム初期化"""
        try:
            # GPU情報確認
            self._check_gpu_status()
            
            # コンポーネント初期化
            self._setup_text_splitter()
            self._setup_embeddings()
            self._setup_vectorstore()
            self._load_finetuned_model()
            
            self.logger.info("✅ フルスペック版RAGシステム初期化完了")
            
        except Exception as e:
            self.logger.error(f"❌ システム初期化エラー: {e}")
            raise
    
    def _check_gpu_status(self):
        """GPU状態確認"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            total_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
            
            self.logger.info(f"🎮 GPU利用可能: {gpu_name}")
            self.logger.info(f"   GPU数: {gpu_count}")
            self.logger.info(f"   VRAM: {total_memory:.1f}GB")
            
            # メモリクリア
            torch.cuda.empty_cache()
            gc.collect()
            
        else:
            self.logger.warning("⚠️ GPU利用不可 - CPU動作")
    
    def _setup_text_splitter(self):
        """テキスト分割器設定"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        self.logger.info(f"📄 テキスト分割器設定完了 (chunk_size: {self.config.chunk_size})")
    
    def _setup_embeddings(self):
        """埋め込みモデル設定"""
        try:
            # 埋め込みモデルはCPU使用（安定性優先）
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            self.logger.info(f"🔤 埋め込みモデル読み込み完了: {self.config.embeddings_model}")
            
        except Exception as e:
            self.logger.error(f"❌ 埋め込みモデル読み込みエラー: {e}")
            raise
    
    def _setup_vectorstore(self):
        """ベクトルストア設定"""
        try:
            # 既存ベクトルストア確認
            if os.path.exists(self.config.persist_directory):
                self.logger.info(f"📁 既存ベクトルストア発見: {self.config.persist_directory}")
                
                # Chromaベクトルストア読み込み
                self.vectorstore = Chroma(
                    persist_directory=self.config.persist_directory,
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings
                )
                
                # ドキュメント数確認
                try:
                    collection = self.vectorstore._collection
                    document_count = collection.count()
                    self.logger.info(f"✅ ベクトルストア読み込み完了: {document_count}件のドキュメント")
                except:
                    self.logger.info("✅ ベクトルストア読み込み完了")
                    
            else:
                self.logger.warning("⚠️ ベクトルストアが見つかりません")
                self.logger.info("💡 新しいベクトルストアを作成します")
                # 新しいベクトルストアを作成
                self.vectorstore = Chroma.from_documents(
                    documents=[], # 初期ドキュメントは空
                    embedding=self.embeddings,
                    collection_name=self.config.collection_name,
                    persist_directory=self.config.persist_directory
                )
                self.vectorstore.persist()
                self.logger.info("✅ 新しいベクトルストアが作成されました")
            
        except Exception as e:
            self.logger.error(f"❌ ベクトルストア設定エラー: {e}")
            self.vectorstore = None
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """ドキュメント検索"""
        try:
            if not self.vectorstore:
                self.logger.warning("⚠️ ベクトルストアが利用できません")
                return []
            
            # ベクトル検索
            docs = self.vectorstore.similarity_search(query, k=k)
            
            self.logger.debug(f"🔍 検索実行: クエリ='{query}', 結果={len(docs)}件")
            
            return docs
            
        except Exception as e:
            self.logger.error(f"❌ ドキュメント検索エラー: {e}")
            return []
    
    def _load_finetuned_model(self):
        """ファインチューニング済みモデル読み込み"""
        try:
            start_time = time.time()
            self.logger.info(f"🤖 ファインチューニング済みモデル読み込み開始")
            self.logger.info(f"   ベースモデル: {self.config.base_model_name}")
            self.logger.info(f"   LoRAアダプター: {self.config.lora_adapter_path}")
            
            # メモリクリア
            torch.cuda.empty_cache()
            gc.collect()
            
            # 量子化設定
            quantization_config = None
            if self.config.use_4bit_quant:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.config.quant_compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                self.logger.info("⚙️ 4bit量子化を有効化")
            
            # Flash Attention 2 設定
            attn_implementation = None
            if self.config.use_flash_attention_2:
                attn_implementation = "flash_attention_2"
                self.logger.info("⚡ Flash Attention 2 を有効化")
            
            device_map = "auto"
            self.logger.info("🎮 GPU使用モード")
            
            # トークナイザー読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # ベースモデル読み込み（シンプル設定）
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=self.config.model_load_dtype,
                trust_remote_code=True,
                device_map=device_map,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation
            )
            
            # LoRAアダプター適用
            if os.path.exists(self.config.lora_adapter_path):
                try:
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        self.config.lora_adapter_path,
                        torch_dtype=self.config.model_load_dtype
                    )
                    self.logger.info("✅ LoRAアダプター適用完了")
                except RuntimeError as e:
                    self.logger.error(f"❌ LoRAアダプター適用エラー: {e}. ベースモデルのみ使用します。")
                    self.model = base_model
            else:
                self.logger.warning(f"⚠️ LoRAアダプターが見つかりません: {self.config.lora_adapter_path}")
                self.logger.info("📝 ベースモデルのみ使用")
                self.model = base_model
            
            # 推論モード設定
            self.model.eval()
            
            load_time = time.time() - start_time
            self.performance_stats['model_load_time'] = load_time
            
            self.logger.info(f"🎉 モデル読み込み完了 ({load_time:.2f}秒)")
            
            # モデル情報表示
            if hasattr(self.model, 'num_parameters'):
                try:
                    total_params = self.model.num_parameters()
                    trainable_params = self.model.num_parameters(only_trainable=True) if hasattr(self.model, 'num_parameters') else 0
                    self.logger.info(f"   総パラメータ数: {total_params:,}")
                    self.logger.info(f"   学習可能パラメータ数: {trainable_params:,}")
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"❌ モデル読み込みエラー: {e}")
            raise
    
    
    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """ファインチューニング済みモデルで回答生成"""
        try:
            if not context_docs:
                # ベクトルストアがない場合はファインチューニング済みモデルのみで回答
                context = "関連資料は見つかりませんでした。"
            else:
                # コンテキスト構築
                context_parts = []
                for i, doc in enumerate(context_docs[:3], 1):  # 上位3件
                    content = doc.page_content.strip()
                    source_info = ""
                    if doc.metadata:
                        company = doc.metadata.get('company', '')
                        category = doc.metadata.get('category', '')
                        if company and category:
                            source_info = f"[{company} - {category}]"
                    
                    context_parts.append(f"参考情報{i}: {source_info}\n{content}")
                
                context = "\n\n".join(context_parts)
            
            # プロンプト構築
            prompt = self.config.intermediate_prompt_template.format(
                context=context,
                question=query
            )
            
            # トークナイズ
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            
            # デバイス設定
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # 生成設定
            generation_config_params = {
                "max_new_tokens": self.config.max_new_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": self.config.do_sample,
                "num_beams": self.config.num_beams,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }

            if self.config.do_sample:
                generation_config_params["temperature"] = self.config.temperature
                generation_config_params["top_p"] = self.config.top_p

            generation_config = GenerationConfig(**generation_config_params)
            
            # 回答生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # デコード
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 後処理
            response = response.strip()
            if not response:
                response = "申し訳ございませんが、適切な回答を生成できませんでした。"
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ 回答生成エラー: {e}")
            return f"エラーが発生しました: {str(e)}"
    
    def query_rag(self, user_query: str, k: int = 5) -> Dict[str, Any]:
        """RAGクエリ実行（メイン関数）"""
        start_time = time.time()
        
        try:
            self.logger.info(f"📝 RAGクエリ開始: {user_query}")
            
            # ドキュメント検索
            docs = self.retrieve_documents(user_query, k=k)
            
            # 回答生成
            response = self.generate_response(user_query, docs)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 統計更新
            self.performance_stats['total_queries'] += 1
            current_avg = self.performance_stats['avg_response_time']
            total_queries = self.performance_stats['total_queries']
            self.performance_stats['avg_response_time'] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )
            
            self.logger.info(f"✅ RAGクエリ完了 ({processing_time:.3f}秒)")
            
            return {
                'answer': response,
                'retrieved_docs': docs,
                'processing_time': processing_time,
                'query': user_query,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'FineTuned + LoRA',
                'retrieval_count': len(docs)
            }
            
        except Exception as e:
            self.logger.error(f"❌ RAGクエリエラー: {e}")
            return {
                'answer': f"エラーが発生しました: {str(e)}",
                'retrieved_docs': [],
                'processing_time': time.time() - start_time,
                'query': user_query,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'Error',
                'retrieval_count': 0
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        status = {
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'vectorstore_ready': self.vectorstore is not None,
            'embeddings_ready': self.embeddings is not None,
            'performance_stats': self.performance_stats.copy()
        }
        
        # GPU情報
        if torch.cuda.is_available():
            status['gpu_available'] = True
            status['gpu_name'] = torch.cuda.get_device_name(0)
            status['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            status['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9
        else:
            status['gpu_available'] = False
        
        # ベクトルストア情報
        if self.vectorstore:
            try:
                collection = self.vectorstore._collection
                status['document_count'] = collection.count()
            except:
                status['document_count'] = 'Unknown'
        
        return status

def create_streamlit_app():
    """Streamlitアプリケーション作成"""
    st.set_page_config(
        page_title="フルスペック版RAG",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 フルスペック版RAGシステム")
    st.caption("ファインチューニング済みモデル + 最高性能RAG = エンタープライズグレード")
    
    # 設定読み込み
    config = Config()
    
    # システム初期化
    @st.cache_resource
    def load_rag_system():
        return FullSpecRAGSystem(config)
    
    try:
        rag_system = load_rag_system()
        system_status = rag_system.get_system_status()
        
    except Exception as e:
        st.error(f"❌ システム初期化エラー: {e}")
        st.stop()
    
    # レイアウト構成
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # システム情報サイドバー
        st.header("🔧 システム状態")
        
        # 基本情報
        if system_status['model_loaded']:
            st.success("✅ モデル: 読み込み完了")
        else:
            st.error("❌ モデル: 読み込み失敗")
        
        if system_status['vectorstore_ready']:
            st.success("✅ ベクトルストア: 準備完了")
            if 'document_count' in system_status:
                st.metric("📄 ドキュメント数", f"{system_status['document_count']:,}")
        else:
            st.error("❌ ベクトルストア: 準備未完了")
        
        # パフォーマンス統計
        st.subheader("📊 パフォーマンス")
        perf = system_status['performance_stats']
        st.metric("⚡ 処理済みクエリ", perf['total_queries'])
        st.metric("⏱ 平均応答時間", f"{perf['avg_response_time']:.3f}秒")
        st.metric("🚀 モデル読み込み時間", f"{perf['model_load_time']:.2f}秒")
        
        # GPU情報
        if system_status['gpu_available']:
            st.subheader("🎮 GPU情報")
            st.info(f"GPU: {system_status['gpu_name']}")
            st.metric("VRAM Total", f"{system_status['gpu_memory_total']:.1f}GB")
            st.metric("VRAM使用中", f"{system_status['gpu_memory_allocated']:.2f}GB")
        else:
            st.warning("💻 CPU動作")
        
        # モデル情報
        st.subheader("🤖 モデル情報")
        st.info(f"**ベース**: {config.base_model_name}")
        st.info(f"**LoRA**: {os.path.basename(config.lora_adapter_path)}")
        st.info(f"**埋め込み**: {config.embeddings_model}")
        
        # 最適化設定
        st.subheader("⚙️ 最適化設定")
        if config.use_4bit_quant:
            st.success("✅ 4bit量子化")
        if config.use_flash_attention_2:
            st.success("✅ Flash Attention 2")
        
        # システム特徴
        st.subheader("✨ システム特徴")
        features = [
            "ファインチューニング済みモデル",
            "LoRAアダプター統合",
            "4bit量子化最適化",
            "Flash Attention 2対応",
            "マルチGPU対応",
            "リアルタイム検索",
            "エンタープライズグレード"
        ]
        
        for feature in features:
            st.success(f"✅ {feature}")
    
    with col1:
        # メインチャットインターフェース
        st.header("💬 RAGチャット")
        
        # サンプル質問
        st.subheader("💡 サンプル質問")
        sample_cols = st.columns(2)
        
        sample_queries = [
            "建築基準法の壁面材仕様について教えて",
            "耐力壁の基準は何ですか？",
            "基礎の立上高さの規定は？",
            "鋼製束の使用基準について",
            "構造材の品質基準は？",
            "金物の取り付け方法は？"
        ]
        
        for i, sample in enumerate(sample_queries):
            col = sample_cols[i % 2]
            if col.button(sample, key=f"sample_{i}"):
                st.session_state['sample_query'] = sample
        
        # チャット履歴初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # サンプル質問処理
        user_input = None
        if 'sample_query' in st.session_state:
            user_input = st.session_state['sample_query']
            del st.session_state['sample_query']
        
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
            
            # AI応答生成
            with st.chat_message("assistant"):
                with st.spinner("🔍 ファインチューニング済みモデルで分析中..."):
                    result = rag_system.query_rag(user_input)
                
                # 回答表示
                st.markdown(result["answer"])
                
                # 詳細情報展開
                with st.expander("📊 処理詳細"):
                    detail_cols = st.columns(4)
                    
                    with detail_cols[0]:
                        st.metric("処理時間", f"{result['processing_time']:.3f}秒")
                    with detail_cols[1]:
                        st.metric("検索結果数", result['retrieval_count'])
                    with detail_cols[2]:
                        st.metric("使用モデル", result['model_used'])
                    with detail_cols[3]:
                        st.metric("処理時刻", result['timestamp'].split('T')[1][:8])
                    
                    # 検索されたドキュメント表示
                    if result['retrieved_docs']:
                        st.subheader("🎯 検索されたドキュメント")
                        for i, doc in enumerate(result['retrieved_docs']):
                            with st.expander(f"ドキュメント {i+1}"):
                                st.text_area("内容", doc.page_content, height=100, disabled=True)
                                if doc.metadata:
                                    st.json(doc.metadata)
            
            # 回答をセッションに追加
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        
        # セッション統計
        if st.session_state.messages:
            st.subheader("📈 セッション統計")
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
            
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("質問数", len(user_messages))
            with stat_cols[1]:
                st.metric("回答数", len(assistant_messages))
            with stat_cols[2]:
                total_chars = sum(len(m["content"]) for m in assistant_messages)
                st.metric("生成文字数", f"{total_chars:,}")

def main():
    """メイン実行関数"""
    try:
        create_streamlit_app()
    except KeyboardInterrupt:
        print("\n🛑 アプリケーション終了")
    except Exception as e:
        print(f"❌ アプリケーションエラー: {e}")

if __name__ == "__main__":
    main()