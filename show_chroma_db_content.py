

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config

def show_chroma_db_content():
    """
    ChromaDBのコレクションからすべてのドキュメントを取得して表示します。
    """
    try:
        # 設定の読み込み
        config = Config()
        persist_directory = config.persist_directory
        collection_name = config.collection_name
        embeddings_model = config.embeddings_model

        # ChromaDBクライアントの初期化
        client = chromadb.PersistentClient(path=persist_directory)

        # コレクションの取得
        collection = client.get_collection(name=collection_name)

        # すべてのドキュメントを取得
        results = collection.get(include=["metadatas", "documents"])

        # 結果の表示
        if not results or not results["ids"]:
            print("コレクションにドキュメントが見つかりませんでした。")
            return

        print(f"コレクション '{collection_name}' の内容:")
        print("-" * 50)
        for i, doc_id in enumerate(results["ids"]):
            print(f"ドキュメントID: {doc_id}")
            print(f"  メタデータ: {results['metadatas'][i]}")
            print(f"  内容: \n{results['documents'][i]}")
            print("-" * 50)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    show_chroma_db_content()

