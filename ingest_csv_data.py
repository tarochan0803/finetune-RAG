import os
import pandas as pd
import logging
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# config_gemeniから設定をインポート
from config_gemeni import Config

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_documents_from_csv(config: Config) -> list[Document]:
    """
    設定ファイルで指定されたCSVを読み込み、Documentオブジェクトのリストを作成する。
    """
    documents = []
    filepath = config.csv_file
    required_columns = config.required_columns
    
    logging.info(f"Processing CSV file: {filepath}")

    try:
        # CSVファイルをPandasで読み込み
        df = pd.read_csv(filepath)
        
        # 必須カラムの存在チェック
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"CSV file '{filepath}' is missing required columns: {missing}")
            return []

        # データフレームの各行を処理
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Reading {os.path.basename(filepath)}"):
            # メタデータを作成 (必須カラムの値を格納)
            metadata = {col: row[col] for col in required_columns if pd.notna(row[col])}
            metadata["source"] = os.path.basename(filepath) # ファイル名をsourceとして追加
            
            # page_contentを作成 (textカラムをメインとし、他のメタデータも結合して検索性を向上)
            text_content = row.get("text", "")
            prefix_info = ", ".join([f"{col}: {val}" for col, val in metadata.items() if col not in ["text", "source"]])
            full_content = f"{prefix_info}\n\n{text_content}"

            if full_content.strip():
                doc = Document(page_content=full_content, metadata=metadata)
                documents.append(doc)

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the CSV: {e}", exc_info=True)
        
    logging.info(f"Finished processing {filepath}. Found {len(documents)} documents.")
    return documents

def main():
    """
    CSVファイルを処理し、既存のChromaDBにデータを追加する。
    """
    config = Config()
    
    csv_documents = create_documents_from_csv(config)

    if not csv_documents:
        logging.warning("No documents were created from the CSV. Exiting.")
        return

    logging.info(f"Total documents to be added from CSV: {len(csv_documents)}")

    try:
        # 埋め込みモデルの初期化
        logging.info(f"Initializing embedding model: {config.embeddings_model}")
        embedding_function = HuggingFaceEmbeddings(model_name=config.embeddings_model)

        # 既存のChromaDBに接続
        logging.info(f"Connecting to existing ChromaDB...")
        logging.info(f"  - Directory: {config.persist_directory}")
        logging.info(f"  - Collection: {config.collection_name}")
        
        vectordb = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embedding_function
        )
        
        # 更新前の件数を確認
        try:
            db_count_before = vectordb._collection.count()
            logging.info(f"Total documents in collection '{config.collection_name}' before update: {db_count_before}")
        except Exception as e:
            logging.warning(f"Could not verify document count before update: {e}")

        # ドキュメントをバッチで追加
        batch_size = 512
        logging.info(f"Adding documents to ChromaDB in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(csv_documents), batch_size), desc="Adding CSV data to DB"):
            batch = csv_documents[i:i+batch_size]
            vectordb.add_documents(batch)

        logging.info("Successfully added all documents from CSV to ChromaDB.")
        
        # 更新後の件数を確認
        try:
            db_count_after = vectordb._collection.count()
            logging.info(f"Total documents in collection '{config.collection_name}' after update: {db_count_after}")
        except Exception as e:
            logging.warning(f"Could not verify document count after update: {e}")

    except Exception as e:
        logging.critical(f"A critical error occurred during the database operation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
