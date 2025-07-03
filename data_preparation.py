#!/usr/bin/env python3
"""
建設業界RAGシステム - データ準備モジュール
統合データセットからChromaDBベクトルデータベースを構築
"""

import json
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from config import Config, setup_logging

class DataPreparation:
    """統合データセットからベクトルDBを構築するクラス"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config, "data_preparation.log")
        
        # 埋め込みモデルの初期化
        self.logger.info(f"埋め込みモデルを読み込み中: {config.embeddings_model}")
        self.embedding_model = SentenceTransformer(config.embeddings_model)
        
        # ChromaDBクライアントの初期化
        self.logger.info(f"ChromaDBクライアントを初期化中: {config.persist_directory}")
        self.chroma_client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
    def load_integrated_dataset(self) -> List[Dict[str, Any]]:
        """統合データセットをJSONLファイルから読み込む"""
        dataset_path = "/home/ncnadmin/my_rag_project/tourokuten_prediction_finetune.jsonl"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"データセットファイルが見つかりません: {dataset_path}")
        
        self.logger.info(f"データセットを読み込み中: {dataset_path}")
        data = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if 'prompt' in item and 'response' in item:
                        data.append({
                            'id': f"doc_{line_num}",
                            'text': item['prompt'],
                            'company': item['response'],
                            'metadata': {
                                'source': 'integrated_dataset',
                                'line_number': line_num,
                                'type': 'prompt_response'
                            }
                        })
                except json.JSONDecodeError as e:
                    self.logger.warning(f"行 {line_num} でJSONパースエラー: {e}")
                    continue
                    
                if line_num % 5000 == 0:
                    self.logger.info(f"読み込み進捗: {line_num} 行")
        
        self.logger.info(f"データセット読み込み完了: {len(data)} 件")
        return data
    
    def create_vector_database(self, data: List[Dict[str, Any]]) -> None:
        """ベクトルデータベースを作成"""
        
        # 既存のコレクションがあれば削除
        try:
            existing_collection = self.chroma_client.get_collection(self.config.collection_name)
            self.chroma_client.delete_collection(self.config.collection_name)
            self.logger.info(f"既存のコレクション '{self.config.collection_name}' を削除しました")
        except Exception:
            pass  # コレクションが存在しない場合は無視
        
        # 新しいコレクションを作成
        collection = self.chroma_client.create_collection(
            name=self.config.collection_name,
            metadata={"description": "建設業界登録店情報ベクトルデータベース"}
        )
        
        self.logger.info(f"新しいコレクション '{self.config.collection_name}' を作成しました")
        
        # バッチ処理でベクトル化と保存
        batch_size = 100
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.logger.info(f"バッチ {batch_num}/{total_batches} を処理中...")
            
            # テキストを埋め込みベクトルに変換
            texts = [item['text'] for item in batch]
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False).tolist()
            
            # ChromaDBに保存
            ids = [item['id'] for item in batch]
            documents = texts
            metadatas = []
            
            for item in batch:
                metadata = item['metadata'].copy()
                metadata['company'] = item['company']
                metadatas.append(metadata)
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.info(f"バッチ {batch_num}/{total_batches} 完了")
        
        # 統計情報を取得
        count = collection.count()
        self.logger.info(f"ベクトルデータベース構築完了: {count} 件のドキュメントを保存")
        
        return collection
    
    def create_enhanced_dataset(self) -> List[Dict[str, Any]]:
        """拡張データセットを作成（検索精度向上のため）"""
        base_data = self.load_integrated_dataset()
        enhanced_data = []
        
        # 元のデータをそのまま追加
        enhanced_data.extend(base_data)
        
        # 会社名ベースの追加データを作成
        company_groups = {}
        for item in base_data:
            company = item['company']
            if company not in company_groups:
                company_groups[company] = []
            company_groups[company].append(item['text'])
        
        # 会社ごとのサマリーを作成
        for company, prompts in company_groups.items():
            if len(prompts) > 5:  # 十分なデータがある会社のみ
                # 会社の特徴をまとめたテキストを作成
                summary_text = f"{company}の主な特徴: " + ", ".join(prompts[:5])
                enhanced_data.append({
                    'id': f"summary_{len(enhanced_data)}",
                    'text': summary_text,
                    'company': company,
                    'metadata': {
                        'source': 'enhanced_summary',
                        'type': 'company_summary',
                        'original_count': len(prompts)
                    }
                })
        
        self.logger.info(f"拡張データセット作成完了: {len(enhanced_data)} 件（元データ: {len(base_data)} 件）")
        return enhanced_data
    
    def verify_database(self) -> bool:
        """データベースの整合性を確認"""
        try:
            collection = self.chroma_client.get_collection(self.config.collection_name)
            count = collection.count()
            
            # サンプル検索テスト
            test_results = collection.query(
                query_texts=["建設会社"],
                n_results=5
            )
            
            self.logger.info(f"データベース検証完了:")
            self.logger.info(f"  - 総ドキュメント数: {count}")
            self.logger.info(f"  - テスト検索結果: {len(test_results['documents'][0])} 件")
            
            return count > 0 and len(test_results['documents'][0]) > 0
            
        except Exception as e:
            self.logger.error(f"データベース検証失敗: {e}")
            return False

def main():
    """メイン処理"""
    config = Config()
    prep = DataPreparation(config)
    
    try:
        # 拡張データセットを作成
        enhanced_data = prep.create_enhanced_dataset()
        
        # ベクトルデータベースを構築
        collection = prep.create_vector_database(enhanced_data)
        
        # 検証
        if prep.verify_database():
            print("✅ ベクトルデータベースの構築が正常に完了しました")
        else:
            print("❌ ベクトルデータベースの検証に失敗しました")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        logging.error(f"データ準備でエラー: {e}")

if __name__ == "__main__":
    main()