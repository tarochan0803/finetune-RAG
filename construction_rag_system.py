#!/usr/bin/env python3
"""
建設業界特化RAGシステム - メインクラス
アンサンブル検索、ファインチューニング済みモデル、Gemini API統合
"""

import logging
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TextStreamer
)
from peft import PeftModel
import google.generativeai as genai

from config import Config, setup_logging

@dataclass
class SearchResult:
    """検索結果を格納するデータクラス"""
    document: str
    metadata: Dict[str, Any]
    score: float
    company: str
    source: str

@dataclass
class RAGResponse:
    """RAG応答を格納するデータクラス"""
    query: str
    final_answer: str
    intermediate_answers: List[str]
    search_results: List[List[SearchResult]]
    processing_time: float
    confidence_score: float

class ConstructionRAGSystem:
    """建設業界特化RAGシステムのメインクラス"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config, "construction_rag.log")
        
        # 初期化フラグ
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None
        self._tokenizer = None
        self._base_model = None
        self._fine_tuned_model = None
        
        # Gemini API初期化
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(config.synthesizer_model_name)
            self.logger.info("Gemini API初期化完了")
        else:
            self.gemini_model = None
            self.logger.warning("Gemini APIキーが設定されていません")
    
    def initialize(self) -> bool:
        """システム全体を初期化"""
        try:
            self.logger.info("建設業界RAGシステムを初期化中...")
            
            # 埋め込みモデルの初期化
            self._init_embedding_model()
            
            # ChromaDBの初期化
            self._init_chroma_db()
            
            # ファインチューニング済みモデルの初期化
            self._init_fine_tuned_model()
            
            self.logger.info("✅ RAGシステムの初期化が完了しました")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RAGシステムの初期化に失敗: {e}")
            return False
    
    def _init_embedding_model(self):
        """埋め込みモデルを初期化"""
        self.logger.info(f"埋め込みモデルを読み込み中: {self.config.embeddings_model}")
        self._embedding_model = SentenceTransformer(self.config.embeddings_model)
        self.logger.info("埋め込みモデル初期化完了")
    
    def _init_chroma_db(self):
        """ChromaDBを初期化"""
        self.logger.info("ChromaDBクライアントを初期化中...")
        self._chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        try:
            self._collection = self._chroma_client.get_collection(self.config.collection_name)
            count = self._collection.count()
            self.logger.info(f"既存のコレクションを読み込み: {count} ドキュメント")
        except Exception as e:
            self.logger.error(f"コレクションの読み込みに失敗: {e}")
            raise
    
    def _init_fine_tuned_model(self):
        """ファインチューニング済みモデルを初期化"""
        self.logger.info("ファインチューニング済みモデルを読み込み中...")
        
        try:
            # トークナイザーの読み込み
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True
            )
            
            # ベースモデルの読み込み
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=self.config.model_load_dtype,
                device_map="cpu" if self.config.force_cpu else "auto",
                trust_remote_code=True
            )
            
            # LoRAアダプターの読み込み
            if os.path.exists(self.config.lora_adapter_path):
                self._fine_tuned_model = PeftModel.from_pretrained(
                    self._base_model,
                    self.config.lora_adapter_path
                )
                self.logger.info("LoRAアダプター読み込み完了")
            else:
                self._fine_tuned_model = self._base_model
                self.logger.warning("LoRAアダプターが見つかりません。ベースモデルを使用します")
            
            # パッドトークンの設定
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self.logger.info("ファインチューニング済みモデル初期化完了")
            
        except Exception as e:
            self.logger.error(f"ファインチューニング済みモデルの初期化に失敗: {e}")
            raise
    
    def ensemble_search(self, query: str, num_variants: int = 3) -> List[List[SearchResult]]:
        """アンサンブル検索を実行"""
        self.logger.info(f"アンサンブル検索を実行: '{query}'")
        
        all_results = []
        
        # 異なる検索戦略を実行
        strategies = [
            ("基本検索", lambda q: self._basic_search(q, k=self.config.rag_variant_k[0])),
            ("拡張検索", lambda q: self._expanded_search(q, k=self.config.rag_variant_k[1])),
            ("会社特化検索", lambda q: self._company_focused_search(q, k=self.config.rag_variant_k[2]))
        ]
        
        for strategy_name, search_func in strategies[:num_variants]:
            try:
                self.logger.info(f"{strategy_name}を実行中...")
                results = search_func(query)
                all_results.append(results)
                self.logger.info(f"{strategy_name}完了: {len(results)} 件の結果")
            except Exception as e:
                self.logger.error(f"{strategy_name}でエラー: {e}")
                all_results.append([])
        
        return all_results
    
    def _basic_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """基本的なセマンティック検索"""
        results = self._collection.query(
            query_texts=[query],
            n_results=k
        )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append(SearchResult(
                document=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1 - results['distances'][0][i],  # 距離を類似度に変換
                company=results['metadatas'][0][i].get('company', '不明'),
                source='basic_search'
            ))
        
        return search_results
    
    def _expanded_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """拡張クエリによる検索"""
        # クエリ拡張
        expanded_queries = [query]
        
        # 簡単なクエリ拡張（キーワード追加）
        if "会社" in query or "企業" in query:
            expanded_queries.append(query + " 建設")
        if "仕様" in query:
            expanded_queries.append(query + " 条件")
        
        all_results = []
        for exp_query in expanded_queries:
            results = self._collection.query(
                query_texts=[exp_query],
                n_results=k//len(expanded_queries) + 1
            )
            
            for i in range(len(results['documents'][0])):
                all_results.append(SearchResult(
                    document=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1 - results['distances'][0][i],
                    company=results['metadatas'][0][i].get('company', '不明'),
                    source='expanded_search'
                ))
        
        # 重複削除とスコア順ソート
        unique_results = {}
        for result in all_results:
            doc_id = result.metadata.get('line_number', result.document[:50])
            if doc_id not in unique_results or result.score > unique_results[doc_id].score:
                unique_results[doc_id] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]
    
    def _company_focused_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """会社特化検索"""
        # 会社名が含まれているかチェック
        potential_company_keywords = ["株式会社", "有限会社", "建設", "工務店", "ハウス"]
        
        company_filter = {}
        for keyword in potential_company_keywords:
            if keyword in query:
                # 会社関連の検索として処理
                results = self._collection.query(
                    query_texts=[query],
                    n_results=k*2,  # より多く取得して後でフィルタ
                    where={"type": {"$ne": "invalid"}}  # 基本的なフィルタ
                )
                break
        else:
            # 通常の検索
            results = self._collection.query(
                query_texts=[query],
                n_results=k
            )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append(SearchResult(
                document=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1 - results['distances'][0][i],
                company=results['metadatas'][0][i].get('company', '不明'),
                source='company_focused_search'
            ))
        
        return search_results[:k]
    
    def generate_intermediate_answer(self, query: str, context: str) -> str:
        """ファインチューニング済みモデルで中間回答を生成"""
        prompt = self.config.intermediate_prompt_template.format(
            context=context,
            question=query
        )
        
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            with torch.no_grad():
                outputs = self._fine_tuned_model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    num_beams=self.config.num_beams,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )
            
            response = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"中間回答生成でエラー: {e}")
            return "提供された情報からは判断できません。"
    
    def synthesize_final_answer(self, query: str, intermediate_answers: List[str], 
                              contexts: List[str]) -> str:
        """Gemini APIで最終回答を統合"""
        if not self.gemini_model:
            # Gemini APIが利用できない場合のフォールバック
            valid_answers = [ans for ans in intermediate_answers if "判断できません" not in ans]
            if valid_answers:
                return valid_answers[0]
            else:
                return "提供された情報からは判断できません。"
        
        try:
            # コンテキストスニペットを作成
            context_snippets = []
            for ctx in contexts:
                snippet = ctx[:100] + "..." if len(ctx) > 100 else ctx
                context_snippets.append(snippet)
            
            synthesis_prompt = self.config.synthesis_prompt_template.format(
                original_question=query,
                answer_1=intermediate_answers[0] if len(intermediate_answers) > 0 else "回答なし",
                answer_2=intermediate_answers[1] if len(intermediate_answers) > 1 else "回答なし",
                answer_3=intermediate_answers[2] if len(intermediate_answers) > 2 else "回答なし",
                context_1_snippet=context_snippets[0] if len(context_snippets) > 0 else "コンテキストなし",
                context_2_snippet=context_snippets[1] if len(context_snippets) > 1 else "コンテキストなし",
                context_3_snippet=context_snippets[2] if len(context_snippets) > 2 else "コンテキストなし"
            )
            
            response = self.gemini_model.generate_content(synthesis_prompt)
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"最終回答統合でエラー: {e}")
            # フォールバック処理
            valid_answers = [ans for ans in intermediate_answers if "判断できません" not in ans]
            if valid_answers:
                return valid_answers[0]
            else:
                return "提供された情報からは判断できません。"
    
    def process_query(self, query: str) -> RAGResponse:
        """クエリを処理してRAG応答を生成"""
        start_time = time.time()
        self.logger.info(f"クエリ処理開始: '{query}'")
        
        try:
            # 1. アンサンブル検索
            search_results = self.ensemble_search(query)
            
            # 2. 中間回答生成
            intermediate_answers = []
            contexts = []
            
            for i, results in enumerate(search_results):
                if results:
                    # 上位結果からコンテキストを構築
                    context = "\n".join([f"- {r.document}" for r in results[:3]])
                    contexts.append(context)
                    
                    # 中間回答を生成
                    answer = self.generate_intermediate_answer(query, context)
                    intermediate_answers.append(answer)
                else:
                    contexts.append("")
                    intermediate_answers.append("関連情報が見つかりませんでした。")
            
            # 3. 最終回答統合
            final_answer = self.synthesize_final_answer(query, intermediate_answers, contexts)
            
            processing_time = time.time() - start_time
            
            # 信頼度スコア計算（簡易版）
            confidence_score = self._calculate_confidence(intermediate_answers, search_results)
            
            response = RAGResponse(
                query=query,
                final_answer=final_answer,
                intermediate_answers=intermediate_answers,
                search_results=search_results,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"クエリ処理完了: {processing_time:.2f}秒, 信頼度: {confidence_score:.2f}")
            return response
            
        except Exception as e:
            self.logger.error(f"クエリ処理でエラー: {e}")
            return RAGResponse(
                query=query,
                final_answer=f"処理中にエラーが発生しました: {str(e)}",
                intermediate_answers=[],
                search_results=[],
                processing_time=time.time() - start_time,
                confidence_score=0.0
            )
    
    def _calculate_confidence(self, intermediate_answers: List[str], 
                            search_results: List[List[SearchResult]]) -> float:
        """信頼度スコアを計算"""
        # 簡易的な信頼度計算
        valid_answers = len([ans for ans in intermediate_answers if "判断できません" not in ans])
        total_answers = len(intermediate_answers)
        
        # 検索結果の品質
        avg_search_score = 0.0
        total_results = 0
        for results in search_results:
            if results:
                avg_search_score += sum(r.score for r in results[:3])
                total_results += len(results[:3])
        
        if total_results > 0:
            avg_search_score /= total_results
        
        # 総合信頼度
        answer_confidence = valid_answers / total_answers if total_answers > 0 else 0.0
        search_confidence = avg_search_score
        
        return (answer_confidence * 0.6 + search_confidence * 0.4)

def main():
    """メイン処理（テスト用）"""
    config = Config()
    rag_system = ConstructionRAGSystem(config)
    
    if rag_system.initialize():
        # テストクエリ
        test_queries = [
            "株式会社平成建設の壁面材仕様は何ですか？",
            "大壁仕様の会社を教えてください",
            "羽柄材の供給がある会社はありますか？"
        ]
        
        for query in test_queries:
            print(f"\n🔍 クエリ: {query}")
            response = rag_system.process_query(query)
            print(f"📝 回答: {response.final_answer}")
            print(f"⏱️  処理時間: {response.processing_time:.2f}秒")
            print(f"📊 信頼度: {response.confidence_score:.2f}")

if __name__ == "__main__":
    main()