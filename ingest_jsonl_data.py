

import os
import json
import logging
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# --- 設定 ---
# 処理対象のJSONLファイル
JSONL_FILES = [
    "/home/ncnadmin/my_rag_project/fine_tuning_dynamic_instruction_dataset.jsonl",
    "/home/ncnadmin/my_rag_project/tourokurensiyou_prompt_response.jsonl",
    "/home/ncnadmin/my_rag_project/tourokurensiyou_converted.jsonl",
    "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl"
]

# ChromaDB設定 (config_gemeni.pyから引用)
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "my_collection"
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-base"

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_jsonl_file(filepath: str) -> list[Document]:
    """
    単一のJSONLファイルを処理し、Documentオブジェクトのリストを返す。
    """
    documents = []
    filename = os.path.basename(filepath)
    logging.info(f"Processing file: {filename}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {filename}"):
                try:
                    data = json.loads(line)
                    text_content = ""
                    # ファイルごとにキーを判定してテキストを結合
                    if "input" in data and "output" in data:
                        # inputとoutputを結合
                        text_content = f"Q: {data.get('input', '')}\nA: {data.get('output', '')}"
                    elif "prompt" in data and "response" in data:
                        # promptとresponseを結合
                        text_content = f"Q: {data.get('prompt', '')}\nA: {data.get('response', '')}"
                    else:
                        # 想定外の形式の場合はスキップ
                        logging.warning(f"Skipping line in {filename} due to unknown format: {data.keys()}")
                        continue

                    if text_content:
                        # メタデータにファイル名を追加
                        metadata = {"source": filename, "type": "jsonl_import"}
                        doc = Document(page_content=text_content, metadata=metadata)
                        documents.append(doc)

                except json.JSONDecodeError:
                    logging.error(f"JSON decode error in {filename} on line: {line.strip()}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred in {filename}: {e}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    
    logging.info(f"Finished processing {filename}. Found {len(documents)} documents.")
    return documents

def main():
    """
    すべてのJSONLファイルを処理し、ChromaDBにデータを追加する。
    """
    all_documents = []
    for file_path in JSONL_FILES:
        all_documents.extend(process_jsonl_file(file_path))

    if not all_documents:
        logging.warning("No documents were created. Exiting.")
        return

    logging.info(f"Total documents to be added: {len(all_documents)}")

    try:
        # 埋め込みモデルの初期化
        logging.info(f"Initializing embedding model: {EMBEDDINGS_MODEL}")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

        # ChromaDBの初期化
        logging.info(f"Initializing ChromaDB...")
        logging.info(f"  - Directory: {PERSIST_DIRECTORY}")
        logging.info(f"  - Collection: {COLLECTION_NAME}")
        
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function
        )

        # ドキュメントをバッチで追加 (多すぎる場合に備えて)
        batch_size = 512
        logging.info(f"Adding documents to ChromaDB in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding to DB"):
            batch = all_documents[i:i+batch_size]
            vectordb.add_documents(batch)

        logging.info("Successfully added all documents to ChromaDB.")
        
        # DBの永続化（Chromaのadd_documentsは自動で永続化するが、明示的に呼び出すことも可能）
        # vectordb.persist() 

        # 登録後の件数を確認
        try:
            db_count = vectordb._collection.count()
            logging.info(f"Total documents in collection '{COLLECTION_NAME}' after update: {db_count}")
        except Exception as e:
            logging.warning(f"Could not verify document count after update: {e}")

    except Exception as e:
        logging.critical(f"A critical error occurred during the database operation: {e}", exc_info=True)

if __name__ == "__main__":
    main()

