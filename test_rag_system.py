#!/usr/bin/env python3
"""
建設業界RAGシステム - テストスクリプト
"""

import sys
import os
import time
import logging

# プロジェクトルートをパスに追加
sys.path.append('/home/ncnadmin/my_rag_project')

from config import Config
from construction_rag_system import ConstructionRAGSystem

def test_basic_functionality():
    """基本機能テスト"""
    print("🧪 基本機能テストを開始...")
    
    config = Config()
    rag_system = ConstructionRAGSystem(config)
    
    # 初期化テスト
    print("  - システム初期化テスト...")
    init_success = rag_system.initialize()
    if init_success:
        print("    ✅ 初期化成功")
    else:
        print("    ❌ 初期化失敗")
        return False
    
    return True

def test_database_connectivity():
    """データベース接続テスト"""
    print("🔍 データベース接続テスト...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # ChromaDBクライアント接続テスト
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # コレクション取得テスト
        collection = client.get_collection("my_collection")
        count = collection.count()
        
        print(f"    ✅ データベース接続成功: {count} ドキュメント")
        return True
        
    except Exception as e:
        print(f"    ❌ データベース接続失敗: {e}")
        return False

def test_embedding_model():
    """埋め込みモデルテスト"""
    print("🤖 埋め込みモデルテスト...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        test_text = "テスト用のテキストです"
        
        embedding = model.encode([test_text])
        print(f"    ✅ 埋め込み生成成功: 次元数 {len(embedding[0])}")
        return True
        
    except Exception as e:
        print(f"    ❌ 埋め込みモデル失敗: {e}")
        return False

def test_search_functionality(rag_system):
    """検索機能テスト"""
    print("🔍 検索機能テスト...")
    
    test_queries = [
        "建設会社",
        "大壁仕様",
        "株式会社"
    ]
    
    for query in test_queries:
        try:
            print(f"    - クエリ: '{query}'")
            search_results = rag_system.ensemble_search(query, num_variants=1)
            
            if search_results and search_results[0]:
                print(f"      ✅ 検索成功: {len(search_results[0])} 件")
            else:
                print(f"      ⚠️  検索結果なし")
                
        except Exception as e:
            print(f"      ❌ 検索失敗: {e}")
            return False
    
    return True

def test_model_loading():
    """モデル読み込みテスト"""
    print("📦 モデル読み込みテスト...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        config = Config()
        
        # トークナイザーテスト
        print("    - トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            trust_remote_code=True
        )
        print("    ✅ トークナイザー読み込み成功")
        
        # ベースモデルテスト（軽量化）
        print("    - ベースモデル読み込み中...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        print("    ✅ ベースモデル読み込み成功")
        
        return True
        
    except Exception as e:
        print(f"    ❌ モデル読み込み失敗: {e}")
        return False

def test_end_to_end():
    """エンドツーエンドテスト"""
    print("🎯 エンドツーエンドテスト...")
    
    config = Config()
    rag_system = ConstructionRAGSystem(config)
    
    if not rag_system.initialize():
        print("    ❌ システム初期化失敗")
        return False
    
    test_query = "建設会社について教えてください"
    
    try:
        print(f"    - クエリ処理: '{test_query}'")
        start_time = time.time()
        
        response = rag_system.process_query(test_query)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"    ✅ 処理成功:")
        print(f"      - 処理時間: {processing_time:.2f}秒")
        print(f"      - 回答: {response.final_answer[:100]}...")
        print(f"      - 信頼度: {response.confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ エンドツーエンドテスト失敗: {e}")
        return False

def run_performance_test():
    """パフォーマンステスト"""
    print("⚡ パフォーマンステスト...")
    
    config = Config()
    rag_system = ConstructionRAGSystem(config)
    
    if not rag_system.initialize():
        print("    ❌ システム初期化失敗")
        return False
    
    test_queries = [
        "大壁仕様の会社は？",
        "羽柄材供給について",
        "鋼製束の情報を教えて"
    ]
    
    total_time = 0
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"    - テスト {i}/{len(test_queries)}: '{query}'")
            start_time = time.time()
            
            response = rag_system.process_query(query)
            
            end_time = time.time()
            query_time = end_time - start_time
            total_time += query_time
            success_count += 1
            
            print(f"      時間: {query_time:.2f}秒, 信頼度: {response.confidence_score:.2f}")
            
        except Exception as e:
            print(f"      ❌ 失敗: {e}")
    
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"    📊 平均処理時間: {avg_time:.2f}秒")
        print(f"    📊 成功率: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
    
    return success_count == len(test_queries)

def main():
    """メインテスト実行"""
    print("🚀 建設業界RAGシステム テスト開始\n")
    
    tests = [
        ("データベース接続", test_database_connectivity),
        ("埋め込みモデル", test_embedding_model),
        ("モデル読み込み", test_model_loading),
        ("基本機能", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"テスト: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} 成功\n")
            else:
                print(f"❌ {test_name} 失敗\n")
                
        except Exception as e:
            print(f"❌ {test_name} エラー: {e}\n")
            results.append((test_name, False))
    
    # 基本テストが成功した場合のみ高度なテストを実行
    basic_success = all(result for name, result in results)
    
    if basic_success:
        print(f"\n{'='*50}")
        print("高度なテスト")
        print(f"{'='*50}")
        
        advanced_tests = [
            ("エンドツーエンド", test_end_to_end),
            ("パフォーマンス", run_performance_test),
        ]
        
        for test_name, test_func in advanced_tests:
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    print(f"✅ {test_name} 成功\n")
                else:
                    print(f"❌ {test_name} 失敗\n")
                    
            except Exception as e:
                print(f"❌ {test_name} エラー: {e}\n")
                results.append((test_name, False))
    
    # 最終結果
    print(f"\n{'='*50}")
    print("テスト結果サマリー")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n総合結果: {passed}/{total} ({passed/total*100:.1f}%) 成功")
    
    if passed == total:
        print("🎉 全テスト成功！システムは正常に動作しています。")
    else:
        print("⚠️  一部テストが失敗しました。ログを確認してください。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)