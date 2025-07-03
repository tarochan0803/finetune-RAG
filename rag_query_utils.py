# rag_query_utils.py (最適化設定・並列処理・ストリーミング最適化関数定義 含む完全版)

import sys
import traceback
import logging
import os
import torch
import re
import time # ストリーミング最適化と時間計測用
import asyncio # 並列処理用
import threading # Lock用
from concurrent.futures import ThreadPoolExecutor # 並列処理用

from config import Config # config.py から Config クラスをインポート
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models.llms import BaseLLM
from langchain_core.vectorstores import VectorStoreRetriever # 型ヒント用
from langchain_core.documents import Document # 型ヒント用
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel # LoRA用
from typing import Tuple, Dict, Any, Optional, List, Generator # 型ヒント用

# utils から必要な関数をインポート (パスが通っている前提)
try:
    from utils import format_document_snippet, normalize_str, expand_query, rerank_documents
    # キャッシュ関数もインポート (ステップ3で定義した場合)
    # from utils import cached_embedding, cached_retrieval, generate_retriever_hash
except ImportError as e:
    # ログ出力とプログラム終了
    init_logger = logging.getLogger("RAGApp_Init") # 初期化用ロガー
    init_logger.error(f"Failed to import from utils.py: {e}. Ensure utils.py is in the Python path.", exc_info=True)
    sys.exit(f"Critical dependency missing: utils.py ({e})")


# 外部 API 用ライブラリのインポート試行 (Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    # この時点では logger が初期化されていない可能性があるため print で警告
    print("Warning: google-generativeai not found. Gemini API synthesizer will not be available.", file=sys.stderr)


# --- GPU使用状況表示関数 ---
def print_gpu_usage(logger: logging.Logger, message: str = "Current GPU Usage"):
    """GPUメモリ使用状況をログに出力する"""
    if not logger: return # ロガーがない場合は何もしない
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            alloc = torch.cuda.memory_allocated(idx) / 1e9
            reserved = torch.cuda.memory_reserved(idx) / 1e9
            logger.info(f"[{message}] CUDA Device [{idx}]: {name}, Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Could not get CUDA device info: {e}")
    else:
        logger.info(f"[{message}] GPU is not available.")

# --- パイプライン初期化関数 (ステップ1: 最適化設定反映) ---
def initialize_pipeline(config: Config, logger: logging.Logger, lora_adapter_path: Optional[str] = None) -> Tuple[Optional[Chroma], Optional[BaseLLM], Optional[AutoTokenizer], Optional[HuggingFaceEmbeddings]]:
    """
    RAGに必要なコンポーネント（VectorDB, 中間LLM, Tokenizer, Embedding）を初期化。
    量子化、Flash Attention、データ型、バッチサイズなどの最適化設定を反映。
    """
    if not logger:
         print("Error: Logger not provided to initialize_pipeline", file=sys.stderr)
         return None, None, None, None
    logger.info("Initializing RAG components (Optimized)...")
    print_gpu_usage(logger, "Pipeline Init Start")
    vectordb = None
    llm = None
    tokenizer = None
    embedding = None

    try:
        # 1. Embeddings
        logger.info(f"Loading embeddings ON CPU: {config.embeddings_model}")
        embedding = HuggingFaceEmbeddings(
            model_name=config.embeddings_model,
            # model_kwargs={'device': 'cuda'} # VRAM余裕あればGPU化
        )
        logger.info("Embeddings loaded.")

        # 2. Vector DB
        logger.info(f"Loading vector DB from: {config.persist_directory}")
        if not os.path.isdir(config.persist_directory):
            raise FileNotFoundError(f"Chroma DB directory not found: {config.persist_directory}")
        vectordb = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embedding,
            client_settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        db_count = vectordb._collection.count()
        logger.info(f"Vector DB loaded with {db_count} documents.")

        # 3. 中間LLMロード
        logger.info(f"Loading Intermediate LLM: {config.base_model_name} (Optimized Config)")
        quantization_config = None
        if config.use_4bit_quant:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.quant_compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"4-bit quantization enabled (Compute dtype: {config.quant_compute_dtype}).")

        attn_implementation = "flash_attention_2" if config.use_flash_attention_2 and torch.cuda.is_available() else None
        if attn_implementation:
             # Flash Attentionが実際に利用可能かどうかのチェックは内部で行われる
             logger.info("Attempting to enable Flash Attention 2.")
        else:
             logger.info("Flash Attention 2 disabled or not applicable.")

        model_kwargs = {
            "quantization_config": quantization_config,
            #"device_map": "auto",
            "torch_dtype": config.model_load_dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
            # "token": config.hf_token,
        }

        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name, **model_kwargs)
        print_gpu_usage(logger, "Base model loaded (Optimized)")

        model = base_model
        tokenizer_load_path = config.base_model_name

        # LoRAアダプター適用
        effective_lora_path = lora_adapter_path if lora_adapter_path is not None else config.lora_adapter_path
        if effective_lora_path and isinstance(effective_lora_path, str) and effective_lora_path.strip():
            if os.path.isdir(effective_lora_path):
                logger.info(f"Applying LoRA adapter: {effective_lora_path}")
                try:
                    model = PeftModel.from_pretrained(base_model, effective_lora_path)
                    tokenizer_load_path = effective_lora_path
                    print_gpu_usage(logger, "LoRA adapter loaded (Optimized)")
                    logger.info("LoRA adapter applied successfully.")
                except Exception as e:
                    logger.error(f"Failed to apply LoRA from {effective_lora_path}: {e}", exc_info=True)
                    logger.warning("Using base model without LoRA due to error.")
                    model = base_model
                    tokenizer_load_path = config.base_model_name
            else:
                logger.warning(f"LoRA path '{effective_lora_path}' not found. Using base model.")
        else:
            logger.info("No LoRA path provided or specified. Using base model.")

        # Tokenizerロード
        logger.info(f"Loading tokenizer from: {tokenizer_load_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True, use_fast=True)

        # PADトークン設定
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token ('{tokenizer.eos_token}').")
            else:
                new_pad_token = '[PAD]'
                logger.warning(f"Tokenizer missing PAD/EOS token. Adding '{new_pad_token}' as pad_token.")
                tokenizer.add_special_tokens({'pad_token': new_pad_token})
                model.resize_token_embeddings(len(tokenizer))

        model.eval()
        logger.info("Model set to evaluation mode.")

        # 4. HuggingFacePipeline 作成
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
             if tokenizer.eos_token_id is not None:
                 logger.warning("PAD token ID is None, using EOS token ID as pad_token_id for pipeline.")
                 pad_token_id = tokenizer.eos_token_id
             else:
                 logger.error("Cannot determine pad_token_id. Using 0, but this might cause issues.")
                 pad_token_id = 0

        generate_params = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample,
            "num_beams": config.num_beams,
            "pad_token_id": pad_token_id,
        }
        logger.info(f"Intermediate LLM pipeline generation params: {generate_params}")

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=getattr(config, 'pipeline_batch_size', 2), # configからバッチサイズ取得、デフォルト2
            **generate_params
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        logger.info("Intermediate HuggingFacePipeline created successfully.")

        logger.info("RAG components initialization complete (Optimized).")
        print_gpu_usage(logger, "Pipeline Init End (Optimized)")
        return vectordb, llm, tokenizer, embedding

    except Exception as e:
        logger.critical(f"Fatal error during optimized pipeline initialization: {e}", exc_info=True)
        return None, None, None, None

# --- 並列処理関連 (ステップ2) ---

# 中間LLMの設定変更と推論実行をスレッドセーフにするためのロック
llm_lock = threading.Lock()

# １つのVariantを処理する同期関数
def process_single_variant(args: tuple) -> tuple:
    """１つのRAGバリアント（検索、リランキング、中間LLM推論）を実行する"""
    variant_index, query, variant_params, config, vectordb, intermediate_llm, embedding_function, metadata_filter, logger = args
    variant_id = variant_index + 1
    log_prefix = f"[Variant {variant_id}]"
    logger.info(f"{log_prefix} Processing start.")

    params = variant_params
    variant_k = params.get("k", 3)
    retrieved_docs_set: List[Document] = []
    context_snippet: str = "N/A"
    top_docs: List[Document] = []

    try:
        # 1. Retriever準備
        retriever_variant = vectordb.as_retriever(
            search_kwargs={"k": variant_k, "filter": metadata_filter}
        )

        # 2. 検索実行
        expanded_queries = [query]
        if config.enable_query_expansion:
             expanded_queries = expand_query(query, config.query_expansion_dict)
             logger.info(f"{log_prefix} Expanded queries: {expanded_queries}")

        seen_ids = set()
        current_retrieved_docs = []
        for q in expanded_queries:
             # === キャッシュ利用する場合 (ステップ3) ===
             # try:
             #     retriever_hash = generate_retriever_hash(retriever_variant) # utils.pyの関数
             #     docs = cached_retrieval(q, variant_k, retriever_hash, retriever_variant) # utils.pyの関数
             # except NameError: # generate_retriever_hashやcached_retrievalがない場合
             #     logger.warning("Cache functions not found, using direct retrieval.")
             #     docs = retriever_variant.invoke(q)
             # === キャッシュ利用しない場合 ===
             docs = retriever_variant.invoke(q)
             current_retrieved_docs.extend(docs)

        # 重複排除
        for doc in current_retrieved_docs:
            if doc.page_content is not None:
                doc_id = (doc.metadata.get('source', 'unknown'), doc.page_content)
                if doc_id not in seen_ids:
                    retrieved_docs_set.append(doc)
                    seen_ids.add(doc_id)

        logger.info(f"{log_prefix} Retrieved {len(retrieved_docs_set)} unique docs.")
        if not retrieved_docs_set:
            logger.warning(f"{log_prefix} No docs retrieved.")
            return "提供された情報からは判断できません。", "N/A", []

    except Exception as search_e:
         logger.error(f"{log_prefix} Error during search: {search_e}", exc_info=True)
         return f"エラー({variant_id}-Search)", "エラー", []

    # 3. リランキング (オプション)
    reranked_docs = retrieved_docs_set
    try:
        if config.enable_reranking:
            if embedding_function:
                 # === キャッシュ利用する場合 (ステップ3) ===
                 # query_embedding = cached_embedding(query, embedding_function) # utils.pyの関数
                 reranked_docs = rerank_documents(query, retrieved_docs_set, embedding_function, logger=logger, k=variant_k, config=config) # utils.pyの関数
                 logger.info(f"{log_prefix} Reranked to {len(reranked_docs)} docs.")
            else:
                 logger.warning(f"{log_prefix} Reranking enabled but no embedding function.")
    except Exception as rerank_e:
         logger.error(f"{log_prefix} Error during reranking: {rerank_e}", exc_info=True)
         return f"エラー({variant_id}-Rerank)", "エラー", retrieved_docs_set

    # 4. コンテキスト作成 & スニペット生成
    top_docs = reranked_docs[:variant_k]
    context = "\n\n".join([doc.page_content for doc in top_docs if doc.page_content])
    context_snippet = format_document_snippet(context, 150)
    logger.debug(f"{log_prefix} Context length: {len(context)}, Snippet: '{context_snippet}'")

    # 5. 中間プロンプト作成
    formatted_intermediate_prompt: str = ""
    try:
        intermediate_prompt_template = PromptTemplate(template=config.intermediate_prompt_template, input_variables=["context", "question"])
        formatted_intermediate_prompt = intermediate_prompt_template.format(context=context, question=query)
    except Exception as prompt_e:
        logger.error(f"{log_prefix} Error formatting prompt: {prompt_e}", exc_info=True)
        return f"エラー({variant_id}-Prompt)", context_snippet, top_docs

    # 6. 中間回答生成 (LLM推論 - ロック付き)
    intermediate_answer = f"エラー({variant_id}-LLM)"
    try:
        logger.info(f"{log_prefix} Generating intermediate answer (LLM inference)...")
        with llm_lock: # LLM設定変更と実行をロック
            logger.debug(f"{log_prefix} Acquired LLM lock.")
            original_gen_config = intermediate_llm.pipeline.model.generation_config.to_dict()
            temp_gen_params = original_gen_config.copy()
            variant_llm_params = { # Variant固有のパラメータを適用
                "temperature": params.get("temperature", config.temperature),
                "top_p": params.get("top_p", config.top_p),
                "repetition_penalty": params.get("repetition_penalty", config.repetition_penalty),
                "do_sample": params.get("temperature", config.temperature) > 0,
            }
            temp_gen_params.update(variant_llm_params)
            logger.debug(f"{log_prefix} Temp LLM params: {variant_llm_params}")

            for key, value in temp_gen_params.items():
                if hasattr(intermediate_llm.pipeline.model.generation_config, key):
                    setattr(intermediate_llm.pipeline.model.generation_config, key, value)

            start_time = time.time()
            intermediate_result_text = intermediate_llm.invoke(formatted_intermediate_prompt)
            end_time = time.time()
            logger.info(f"{log_prefix} LLM inference took {end_time - start_time:.2f} seconds.")

            for key, value in original_gen_config.items(): # 設定を戻す
                if hasattr(intermediate_llm.pipeline.model.generation_config, key):
                     setattr(intermediate_llm.pipeline.model.generation_config, key, value)
            logger.debug(f"{log_prefix} Released LLM lock.")

        intermediate_answer = intermediate_result_text.strip()
        logger.info(f"{log_prefix} Intermediate answer generated successfully.")
        logger.debug(f"{log_prefix} Answer snippet: {intermediate_answer[:100]}...")

    except Exception as llm_e:
        logger.error(f"{log_prefix} Error during LLM inference: {llm_e}", exc_info=True)
    finally:
        print_gpu_usage(logger, f"{log_prefix} GPU usage after LLM inference")

    logger.info(f"{log_prefix} Processing finished.")
    return intermediate_answer, context_snippet, top_docs


# 非同期ラッパー関数
async def process_variant_async(args: tuple) -> tuple:
    """同期関数 process_single_variant をスレッドプールで非同期実行する"""
    loop = asyncio.get_event_loop()
    config = args[3]
    logger = args[-1]
    variant_id = args[0] + 1
    log_prefix = f"[Variant {variant_id}-Async]"
    max_workers = getattr(config, 'max_parallel_variants', 2) # configから並列数を取得

    # スレッドプールで実行
    logger.debug(f"{log_prefix} Submitting to ThreadPoolExecutor (max_workers={max_workers})...")
    try:
        # このExecutorは使い捨て。効率を求めるなら外で管理。
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # loop.run_in_executor で スレッドプール + 同期関数 を非同期実行
            result = await loop.run_in_executor(executor, process_single_variant, args)
        logger.info(f"{log_prefix} Execution finished successfully.")
        return result
    except Exception as async_e:
        logger.error(f"{log_prefix} Error in async execution wrapper: {async_e}", exc_info=True)
        return f"エラー({variant_id}-Async)", "エラー", []


# 並列タスクを実行するためのヘルパー関数
async def run_all_variants_concurrently(args_list: list) -> list:
    """複数のVariant処理タスクを並列実行し、結果をリストで返す"""
    if not args_list: return []
    logger = args_list[0][-1]
    num_tasks = len(args_list)
    logger.info(f"Running {num_tasks} variants concurrently using asyncio.gather...")
    # 各引数に対して非同期タスクを作成
    tasks = [process_variant_async(args) for args in args_list]
    # 全てのタスクが完了するのを待つ
    results = await asyncio.gather(*tasks, return_exceptions=True) # 例外も取得するように変更
    logger.info("All variant tasks completed (or failed).")

    # 結果の確認 (例外が発生した場合の処理)
    processed_results = []
    for i, res in enumerate(results):
        variant_id = i + 1
        if isinstance(res, Exception):
            logger.error(f"[Variant {variant_id}-Async] Task failed with exception: {res}", exc_info=res)
            # エラー時のデフォルト値を設定
            processed_results.append((f"エラー({variant_id}-TaskFail)", "エラー", []))
        else:
            # 正常終了した結果を追加
            processed_results.append(res)
            # 正常終了でも内部でエラーが返る場合があるため、それもログに残す
            if isinstance(res[0], str) and res[0].startswith("エラー"):
                 logger.warning(f"[Variant {variant_id}] Task completed but returned an error state: {res[0]}")

    return processed_results


# --- ストリーミング最適化関数 (ステップ5) ---
def optimized_gemini_stream_generator(response_stream, buffer_size: int = 3, delay: float = 0.01):
    """APIからのストリームをバッファリングし、まとめてyieldするジェネレータ"""
    buffer = []
    logger = logging.getLogger("RAGApp") # 必要に応じてロガー取得
    chunk_count = 0
    yielded_count = 0
    try:
        for chunk in response_stream:
            chunk_count += 1
            text_content = getattr(chunk, 'text', None)
            if text_content:
                buffer.append(text_content)
                # logger.debug(f"Stream chunk {chunk_count} received, buffer size: {len(buffer)}") # 詳細すぎる場合コメントアウト
                if len(buffer) >= buffer_size:
                    yielded_text = ''.join(buffer)
                    yield yielded_text
                    yielded_count += 1
                    # logger.debug(f"Yielded buffered block {yielded_count} (size {len(buffer)})")
                    buffer = []
            # else: logger.debug(f"Stream chunk {chunk_count} received without text content.")

            if delay > 0: time.sleep(delay)

        if buffer: # 残りを出力
            yielded_text = ''.join(buffer)
            yield yielded_text
            yielded_count += 1
            logger.debug(f"Yielded final buffered block {yielded_count} (size {len(buffer)})")
        logger.info(f"Optimized stream processing finished. Total chunks: {chunk_count}, Yielded blocks: {yielded_count}.")

    except Exception as e:
        logger.error(f"Error during optimized API stream processing: {e}", exc_info=True)
        yield f"\n\n[ストリーミング処理エラー: {e}]"


# --- アンサンブル質問応答関数 (ステップ2: 並列処理反映) ---
def ask_question_ensemble_stream(
    vectordb: Optional[Chroma],
    intermediate_llm: Optional[BaseLLM],
    tokenizer: Optional[AutoTokenizer],
    embedding_function: Optional[HuggingFaceEmbeddings],
    config: Config,
    query: str,
    logger: logging.Logger,
    metadata_filter: Optional[dict] = None,
    variant_params: Optional[list] = None
) -> dict:
    """複数Variantの検索・中間回答生成を並列処理し、外部APIで最終統合（ストリーミング）"""
    start_time_total = time.time()
    if not all([vectordb, intermediate_llm, tokenizer, embedding_function, config, logger]):
        # loggerがNoneの場合でもエラーを出力試行
        err_msg = "System Error: RAG components or config/logger missing."
        print(err_msg, file=sys.stderr)
        if logger: logger.error(err_msg)
        return {"result_stream": iter([err_msg]), "source_documents": [], "variant_answers": []}

    if not isinstance(query, str) or not query.strip():
        logger.warning("Invalid query received.")
        return {"result_stream": iter(["質問を入力してください。"]), "source_documents": [], "variant_answers": []}

    if config.synthesizer_api == "gemini" and (not GEMINI_AVAILABLE or not config.gemini_api_key):
        logger.error("Gemini API not configured or library not available.")
        return {"result_stream": iter(["システムエラー: Gemini API設定不備。"]), "source_documents": [], "variant_answers": []}

    logger.info(f"Processing query (Ensemble Parallel): '{query[:100]}...'")
    print_gpu_usage(logger, "Start Ensemble RAG Query")

    # --- (無効化) ファインチューニングモデルによる直接予測 ---
    # company情報が取得できない問題を修正するため、この機能を無効化し、
    # すべてのクエリが通常のRAGフローを通過するようにします。

    # --- 通常のRAGフロー (ファインチューニングモデルによる直接予測がトリガーされない場合) ---
    # Variantパラメータ設定
    if not variant_params:
        num_default_variants = getattr(config, 'num_default_variants', 3)
        variant_params = [
            {"k": config.rag_variant_k[i] if i < len(config.rag_variant_k) else 3,
             "temperature": config.temperature, "top_p": config.top_p,
             "repetition_penalty": config.repetition_penalty}
             for i in range(num_default_variants)
        ]
    num_variants = len(variant_params)
    logger.info(f"Number of variants to process in parallel: {num_variants}")

    intermediate_answers: List[str] = [""] * num_variants
    context_snippets: List[str] = ["N/A"] * num_variants
    all_source_documents_nested: List[List[Document]] = [[] for _ in range(num_variants)]
    results = [] # asyncioの結果格納用

    try:
        # 各Variant処理関数に渡す引数をタプルのリストとして準備
        args_list = [
            (i, query, variant_params[i], config, vectordb, intermediate_llm, embedding_function, metadata_filter, logger)
            for i in range(num_variants)
        ]

        # asyncioイベントループを開始し、並列処理を実行
        logger.info("Starting asyncio event loop for parallel processing...")
        # 同期関数内から呼び出す場合は asyncio.run() を使う
        # 注意: Streamlit環境など、既にイベントループが動いている場合は別の方法が必要な場合がある
        # (例: nest_asyncio を使う、または Streamlit 自体を非同期対応させる)
        # ここでは基本的な asyncio.run() を使用
        results = asyncio.run(run_all_variants_concurrently(args_list))
        logger.info("Asyncio event loop finished.")

        # 結果を整理 (run_all_variants_concurrently で例外処理済み)
        for i, (res_ans, res_snippet, res_docs) in enumerate(results):
            intermediate_answers[i] = res_ans
            context_snippets[i] = res_snippet
            all_source_documents_nested[i] = res_docs if isinstance(res_docs, list) else [] # エラー時は[]を保証

    except RuntimeError as rt_err:
         # asyncio.run() がネストされた場合などに発生する可能性
         logger.error(f"RuntimeError during asyncio execution (maybe nested loop?): {rt_err}", exc_info=True)
         # フォールバック: 逐次実行 (オプション)
         logger.warning("Falling back to sequential execution due to asyncio error.")
         try:
             results_seq = [process_single_variant(args) for args in args_list]
             for i, (res_ans, res_snippet, res_docs) in enumerate(results_seq):
                 intermediate_answers[i] = res_ans
                 context_snippets[i] = res_snippet
                 all_source_documents_nested[i] = res_docs if isinstance(res_docs, list) else []
         except Exception as seq_e:
              logger.error(f"Sequential fallback also failed: {seq_e}", exc_info=True)
              return {"result_stream": iter([f"並列・逐次処理エラー: {seq_e}"]),
                      "source_documents": [],
                      "variant_answers": [f"エラーV{i+1}" for i in range(num_variants)]}

    except Exception as parallel_e:
         logger.error(f"General error during parallel variant processing setup/execution: {parallel_e}", exc_info=True)
         return {"result_stream": iter([f"並列処理エラー: {parallel_e}"]),
                 "source_documents": [],
                 "variant_answers": [f"エラーV{i+1}" for i in range(num_variants)]}

    # --- 中間回答生成完了 ---
    time_after_variants = time.time()
    logger.info(f"Intermediate answer generation complete. Took {time_after_variants - start_time_total:.2f} sec.")
    logger.debug(f"Intermediate Answers: {intermediate_answers}")

    # 参照ドキュメントリストをフラット化し、重複排除
    all_retrieved_docs = [doc for doc_list in all_source_documents_nested for doc in doc_list]
    unique_source_documents: List[Document] = []
    seen_content_hashes_final = set()
    for doc in all_retrieved_docs:
        if doc.page_content:
             doc_hash = hash(f"{doc.metadata.get('source', 'N/A')}|{doc.page_content}")
             if doc_hash not in seen_content_hashes_final:
                 unique_source_documents.append(doc)
                 seen_content_hashes_final.add(doc_hash)
    logger.info(f"Total retrieved docs (raw): {len(all_retrieved_docs)}, Unique source docs: {len(unique_source_documents)}")

    # --- 最終統合ステップ (外部API) ---
    final_answer_stream = None
    valid_intermediate_answers = [
        ans for ans in intermediate_answers
        if ans and not ans.startswith("エラー") and ans != "提供された情報からは判断できません。"
    ]

    if not valid_intermediate_answers:
        logger.warning("No valid intermediate answers found for synthesis.")
        final_answer_text = "関連する情報が見つかりませんでした。"
        if any(ans.startswith("エラー") for ans in intermediate_answers):
             final_answer_text = "回答生成中に一部エラーが発生しました。"
        final_answer_stream = iter([final_answer_text])
    else:
        logger.info(f"Synthesizing final answer using API: {config.synthesizer_api} ({config.synthesizer_model_name})...")
        try:
            synthesis_prompt_template = PromptTemplate(
                template=config.synthesis_prompt_template,
                input_variables=["original_question", "answer_1", "context_1_snippet", "answer_2", "context_2_snippet", "answer_3", "context_3_snippet"]
            )
            synthesis_input_dict = {"original_question": query}
            for j in range(3): # 最大3つのVariant結果を使う
                synthesis_input_dict[f"answer_{j+1}"] = intermediate_answers[j] if j < len(intermediate_answers) else "N/A"
                synthesis_input_dict[f"context_{j+1}_snippet"] = context_snippets[j] if j < len(context_snippets) else "N/A"

            formatted_synthesis_prompt = synthesis_prompt_template.format(**synthesis_input_dict)
            logger.debug(f"Synthesis prompt for API (first 300 chars): {formatted_synthesis_prompt[:300]}...")

            if config.synthesizer_api == "gemini":
                genai.configure(api_key=config.gemini_api_key)
                generation_config_api = genai.types.GenerationConfig(
                    candidate_count=1, temperature=config.synthesizer_temperature,
                    top_p=config.synthesizer_top_p, max_output_tokens=config.synthesizer_max_new_tokens
                )
                safety_settings=[ {"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                model_api = genai.GenerativeModel(config.synthesizer_model_name)

                response_stream = model_api.generate_content(
                    formatted_synthesis_prompt, generation_config=generation_config_api,
                    safety_settings=safety_settings, stream=True
                )

                # 最適化されたストリームジェネレータを使用
                final_answer_stream = optimized_gemini_stream_generator(
                    response_stream,
                    buffer_size=getattr(config, 'stream_buffer_size', 3), # configからバッファサイズ取得
                    delay=getattr(config, 'stream_delay', 0.01)        # configから遅延取得
                )
                logger.info("Gemini API stream initiated for synthesis.")
            else:
                 logger.error(f"Unsupported API synthesizer: {config.synthesizer_api}")
                 final_answer_stream = iter([f"エラー: 未対応の統合API ({config.synthesizer_api})"])

        except Exception as api_e:
            logger.error(f"Error during API call for synthesis: {api_e}", exc_info=True)
            final_answer_stream = iter([f"統合API呼び出しエラー: {api_e}"])

    # --- 最終的な戻り値 ---
    time_end_total = time.time()
    logger.info(f"Total query processing time: {time_end_total - start_time_total:.2f} sec.")
    print_gpu_usage(logger, "End Ensemble RAG Query")

    return {
        "result_stream": final_answer_stream,
        "source_documents": unique_source_documents,
        "variant_answers": intermediate_answers
    }

# --- ここまで rag_query_utils.py ---