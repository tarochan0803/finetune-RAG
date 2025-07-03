import os
import logging
import json
import sys
import re
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# config.py から Config クラスをインポート
try:
    from config import Config, setup_logging
except ImportError as e:
    print(f"Import Error: {e}\nconfig.py が見つからないか、必要なモジュールが不足しています。", file=sys.stderr)
    sys.exit(1)

def ingest_data_to_chroma(config: Config, logger: logging.Logger, jsonl_file_path: str):
    """
    JSONLファイルからデータを読み込み、ChromaDBにインデックス化する。
    各行はJSONオブジェクトで、'text'フィールドが必須。他のフィールドはメタデータとして扱われる。
    """
    logger.info(f"Starting data ingestion from {jsonl_file_path} to ChromaDB...")

    if not os.path.exists(jsonl_file_path):
        logger.error(f"Error: JSONL file not found at {jsonl_file_path}")
        return

    try:
        # Embeddingsの初期化
        logger.info(f"Loading embeddings model: {config.embeddings_model}")
        embedding_function = HuggingFaceEmbeddings(model_name=config.embeddings_model)
        logger.info("Embeddings model loaded.")

        # ChromaDBの初期化
        logger.info(f"Initializing ChromaDB at {config.persist_directory} with collection {config.collection_name}")
        vectordb = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embedding_function
        )
        logger.info(f"Current document count in DB: {vectordb._collection.count()}")

        documents_to_add = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    page_content = data.get('response')
                    if not page_content:
                        logger.warning(f"Line {line_num + 1} in {jsonl_file_path} has no 'response' field. Skipping.")
                        continue

                    metadata = {k: v for k, v in data.items() if k not in ['response', 'prompt']}
                    if 'prompt' in data: metadata['prompt'] = data['prompt']

                    # responseから会社名を抽出してmetadata['company']に追加
                    company_name_match = re.match(r"^(.*?)(?:の標準仕様:.*)?$", page_content)
                    if company_name_match:
                        extracted_company_name = company_name_match.group(1).strip()
                        # 会社名として妥当か簡単なチェック
                        if any(suffix in extracted_company_name for suffix in ["株式会社", "有限会社", "合同会社", "協同組合"]):
                            metadata['company'] = extracted_company_name
                        else:
                            # responseが「株式会社三建」のような会社名のみの場合
                            if any(suffix in page_content for suffix in ["株式会社", "有限会社", "合同会社", "協同組合"]):
                                metadata['company'] = page_content
                            else:
                                logger.warning(f"Could not extract company name from response: {page_content[:50]}...")
                    else:
                        logger.warning(f"Could not extract company name from response: {page_content[:50]}...")
                    # メタデータにsourceを追加 (任意)
                    if 'source' not in metadata:
                        metadata['source'] = os.path.basename(jsonl_file_path)

                    documents_to_add.append(Document(page_content=page_content, metadata=metadata))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON on line {line_num + 1} in {jsonl_file_path}: {e}. Skipping line.")
                except Exception as e:
                    logger.error(f"Unexpected error processing line {line_num + 1} in {jsonl_file_path}: {e}. Skipping line.")

        if not documents_to_add:
            logger.info("No valid documents found to add from the JSONL file.")
            return

        logger.info(f"Adding {len(documents_to_add)} documents to ChromaDB in batches...")
        batch_size = 1000  # ChromaDBのバッチサイズ制限を考慮して調整
        for i in range(0, len(documents_to_add), batch_size):
            batch = documents_to_add[i:i + batch_size]
            logger.info(f"Adding batch {i // batch_size + 1}/{(len(documents_to_add) + batch_size - 1) // batch_size} ({len(batch)} documents)...")
            vectordb.add_documents(batch)
            # vectordb.persist() # 各バッチでpersistすると遅くなるため、最後にまとめて行う

        vectordb.persist() # 変更をディスクに保存
        logger.info(f"Successfully added documents. New document count in DB: {vectordb._collection.count()}")

    except Exception as e:
        logger.critical(f"Fatal error during data ingestion: {e}", exc_info=True)

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(config, log_filename="data_ingestion.log")
    jsonl_path = "/home/ncnadmin/my_rag_project/tourokuten_prediction_finetune.jsonl"
    ingest_data_to_chroma(config, logger, jsonl_path)
    logger.info("Data ingestion process finished.")