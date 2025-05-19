# rag_query_utils_gemeni.py (案1+案2 反映版 - 中間回答2ステップ生成)

import sys
import traceback
import logging
import os
import torch
import re
import time            # ストリーミング最適化と時間計測用
import asyncio         # 並列処理用
import threading       # Lock用
from concurrent.futures import ThreadPoolExecutor    # 並列処理用

# config_gemeni から Config クラスをインポート
try:
    from config_gemeni import Config
except ImportError as e:
    init_logger = logging.getLogger("RAGapp_gemeni_Init")
    init_logger.critical(f"Failed to import Config from config_gemeni.py: {e}. Ensure config_gemeni.py exists and is configured.", exc_info=True)
    sys.exit("Critical dependency missing: config_gemeni.py")


from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models.llms import BaseLLM
from langchain_core.vectorstores import VectorStoreRetriever    # 型ヒント用
from langchain_core.documents import Document                   # 型ヒント用
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel  # LoRA用
from typing import Tuple, Dict, Any, Optional, List, Generator

# utils_gemeni から必要な関数をインポート (パスが通っている前提)
try:
    from utils_gemeni import format_document_snippet, normalize_str, expand_query, rerank_documents
    # キャッシュ関数もインポート (定義した場合)
    # from utils_gemeni import cached_embedding, cached_retrieval, generate_retriever_hash
except ImportError as e:
    init_logger = logging.getLogger("RAGapp_gemeni_Init")  # 初期化用ロガー名を変更
    init_logger.error(f"Failed to import from utils_gemeni.py: {e}. Ensure utils_gemeni.py is in the Python path.", exc_info=True)
    sys.exit(f"Critical dependency missing: utils_gemeni.py ({e})")

# 外部 API 用ライブラリのインポート試行 (Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not found. Gemini API functions will not be available.", file=sys.stderr)

# --- ロガー取得 ---
# このモジュールレベルでロガーを取得
logger = logging.getLogger("RAGapp_gemeni")

# --- GPU使用状況表示関数 ---
def print_gpu_usage(logger: logging.Logger, message: str = "Current GPU Usage"):
    """GPUメモリ使用状況をログに出力する"""
    if not logger:
        return
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

# --- パイプライン初期化関数 (変更なし) ---
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
            # model_kwargs={'device': 'cuda'}  # VRAM余裕あればGPU化
        )
        logger.info("Embeddings loaded.")

        # 2. Vector DB
        logger.info(f"Loading vector DB from: {config.persist_directory}")
        if not os.path.isdir(config.persist_directory):
            raise FileNotFoundError(f"Chroma DB directory not found: {config.persist_directory}")
        vectordb = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embedding
        )
        # DBが空でないかチェック
        try:
             db_count = vectordb._collection.count()
             if db_count == 0:
                  logger.warning(f"Vector DB loaded successfully from {config.persist_directory}, but it contains 0 documents in collection '{config.collection_name}'. Search will likely yield no results.")
             else:
                  logger.info(f"Vector DB loaded with {db_count} documents.")
        except Exception as db_err:
             logger.error(f"Failed to get document count from Chroma DB: {db_err}", exc_info=True)
             # DB読み込み自体は成功したとみなし、処理は続行するが警告は出す
             logger.warning("Could not verify document count in the database.")


        # 3. 中間LLMロード (中間回答生成を Gemini API で行うため、本来は不要かもしれないが、
        #    将来的な切り替えや他の目的で使う可能性を考慮し、ロード処理は残す)
        logger.info(f"Loading Intermediate LLM (optional, as Gemini is used): {config.base_model_name}")
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
             logger.info("Attempting to enable Flash Attention 2.")
        else:
             logger.info("Flash Attention 2 disabled or not applicable.")

        model_kwargs = {
            "quantization_config": quantization_config,
            # "device_map": "auto",
            "torch_dtype": config.model_load_dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
            # "token": config.hf_token,
        }
        # ローカルLLMのロード (エラーハンドリング強化)
        try:
             base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name, **model_kwargs)
             print_gpu_usage(logger, "Base model loaded (Optional)")
             model = base_model
             tokenizer_load_path = config.base_model_name
        except Exception as model_load_err:
             logger.error(f"Failed to load base model '{config.base_model_name}': {model_load_err}", exc_info=True)
             logger.warning("Proceeding without local intermediate LLM. Ensure Gemini API is configured.")
             model = None # モデルロード失敗
             tokenizer_load_path = config.base_model_name # Tokenizerはベース名から試行

        # LoRAアダプター適用 (modelがロードできた場合のみ)
        if model:
             effective_lora_path = lora_adapter_path if lora_adapter_path is not None else config.lora_adapter_path
             if effective_lora_path and isinstance(effective_lora_path, str) and effective_lora_path.strip():
                 if os.path.isdir(effective_lora_path):
                     logger.info(f"Applying LoRA adapter: {effective_lora_path}")
                     try:
                         model = PeftModel.from_pretrained(model, effective_lora_path) # base_model ではなく model を使う
                         tokenizer_load_path = effective_lora_path
                         print_gpu_usage(logger, "LoRA adapter loaded (Optional)")
                         logger.info("LoRA adapter applied successfully.")
                     except Exception as e:
                         logger.error(f"Failed to apply LoRA from {effective_lora_path}: {e}", exc_info=True)
                         logger.warning("Using base model without LoRA due to error.")
                         # model は base_model のまま
                         tokenizer_load_path = config.base_model_name
                 else:
                     logger.warning(f"LoRA path '{effective_lora_path}' not found. Using base model.")
             else:
                 logger.info("No LoRA path provided or specified. Using base model (if loaded).")

        # Tokenizerロード (モデルロード成否に関わらず試行)
        try:
             logger.info(f"Loading tokenizer from: {tokenizer_load_path}")
             tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True, use_fast=True)
             # PADトークン設定 (tokenizer がロードできた場合のみ)
             if tokenizer.pad_token is None:
                 if tokenizer.eos_token:
                     tokenizer.pad_token = tokenizer.eos_token
                     logger.info(f"Set pad_token to eos_token ('{tokenizer.eos_token}').")
                 else:
                     # モデルが存在する場合のみトークン追加＆リサイズ
                     if model:
                          new_pad_token = '[PAD]'
                          logger.warning(f"Tokenizer missing PAD/EOS token. Adding '{new_pad_token}' as pad_token.")
                          tokenizer.add_special_tokens({'pad_token': new_pad_token})
                          model.resize_token_embeddings(len(tokenizer))
                     else:
                          logger.warning("Tokenizer missing PAD/EOS token, but model not loaded. Cannot add new token.")
        except Exception as tokenizer_err:
             logger.error(f"Failed to load tokenizer from '{tokenizer_load_path}': {tokenizer_err}", exc_info=True)
             tokenizer = None # Tokenizerロード失敗

        # モデルとTokenizerがロードできた場合のみ後続処理
        if model and tokenizer:
             model.eval()
             logger.info("Model set to evaluation mode.")

             # 4. HuggingFacePipeline 作成 (ローカルLLMを使う場合のみ)
             pad_token_id = tokenizer.pad_token_id
             if pad_token_id is None:
                  if tokenizer.eos_token_id is not None:
                      logger.warning("PAD token ID is None, using EOS token ID as pad_token_id for pipeline.")
                      pad_token_id = tokenizer.eos_token_id
                  else:
                      logger.error("Cannot determine pad_token_id for pipeline. Using 0, but this might cause issues.")
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
             logger.info(f"Intermediate LLM pipeline generation params (optional): {generate_params}")

             hf_pipeline = pipeline(
                 "text-generation",
                 model=model,
                 tokenizer=tokenizer,
                 batch_size=getattr(config, 'pipeline_batch_size', 2),
                 **generate_params
             )
             llm = HuggingFacePipeline(pipeline=hf_pipeline)
             logger.info("Intermediate HuggingFacePipeline created (optional).")
        else:
             logger.warning("Skipping HuggingFacePipeline creation as model or tokenizer failed to load.")
             llm = None

        logger.info("RAG components initialization attempt finished.")
        print_gpu_usage(logger, "Pipeline Init End")
        # llm は None の可能性があるので注意
        return vectordb, llm, tokenizer, embedding

    except FileNotFoundError as fnf_err:
         logger.critical(f"Chroma DB directory not found: {fnf_err}", exc_info=True)
         return None, None, None, None # DBが見つからない場合は致命的
    except Exception as e:
        logger.critical(f"Fatal error during pipeline initialization: {e}", exc_info=True)
        return None, None, None, None

# --- 並列処理関連 ---
llm_lock = threading.Lock()  # ローカルLLMを使う場合のスレッドセーフ用 (今回は未使用)

# --- NEW: ステップA - 情報抽出用 Gemini呼び出し関数 ---
def extract_relevant_info_with_gemini(context: str, question: str, config: Config) -> str:
    """
    Gemini APIを使い、コンテキストから質問に直接関連する情報を抽出する。
    推論は抑制し、事実ベースの情報を抜き出すことを目指す。
    """
    if not context: # コンテキストが空の場合はAPIを呼ばずに返す
        logger.warning("Context is empty for info extraction.")
        return "関連情報なし"

    prompt = f"""
以下の「参考情報」を読み、ユーザーの「質問」に直接関連する可能性のある箇所を**そのまま抜き出して**ください。
もし関連する情報が全く見つからない場合は、必ず「関連情報なし」とだけ回答してください。
あなたの意見や推測、要約は含めないでください。

# 参考情報:
{context}

# 質問:
{question}

# 抽出結果:
"""
    logger.debug(f"Extracting info with prompt:\n{prompt[:300]}...")
    try:
        if not GEMINI_AVAILABLE or not genai:
             logger.error("Gemini library not available for info extraction.")
             return "[情報抽出エラー: Geminiライブラリ未検出]"
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.synthesizer_model_name) # 抽出用モデル
        generation_config = genai.types.GenerationConfig(temperature=0.1) # 低温で事実抽出
        response = model.generate_content(prompt, generation_config=generation_config)
        extracted_text = response.text.strip()
        logger.debug(f"Extracted info: {extracted_text[:200]}...")
        return extracted_text
    except Exception as e:
        logger.error(f"Error calling Gemini API for info extraction: {e}", exc_info=True)
        # エラーの詳細を含めることも検討
        return f"[情報抽出エラー: {type(e).__name__}]"

# --- NEW: ステップB - 推論・回答生成用 Gemini呼び出し関数 (案1の考え方を取り込む) ---
def generate_inferred_answer_with_gemini(extracted_info: str, question: str, config: Config, original_context: str = "") -> str:
    """
    抽出された情報と質問に基づき、推論を交えて最終的な中間回答を生成する。
    抽出情報が不十分な場合は元のコンテキストも考慮するオプションを追加。
    """
    # コンテキストソースを決定
    context_source = extracted_info
    prompt_context_label = "# 抽出された関連情報:"

    if extracted_info.startswith("[情報抽出エラー") or extracted_info == "関連情報なし":
         # 抽出失敗または情報なしの場合、元のコンテキストを使うか、情報なしとするか
         if original_context:
              logger.warning(f"Info extraction failed or yielded no results ('{extracted_info}'). Falling back to using original context for inference.")
              context_source = original_context
              prompt_context_label = "# 元の参考情報 (抽出失敗/情報なしのため):"
         else:
              logger.warning(f"Info extraction failed or yielded no results ('{extracted_info}'), and no original context provided. Cannot generate inferred answer.")
              return "提供された情報からは判断できません。" # またはエラーを示す文字列

    prompt = f"""
以下の「{prompt_context_label[2:]}」とユーザーの「質問」を考慮し、質問に対する最も適切で役立つ回答を生成してください。

# 指示:
1. まず、「{prompt_context_label[2:]}」に質問に対する直接的な答えが含まれているか確認してください。あれば、それを基に明確に回答してください。
2. 直接的な答えがない場合でも、「{prompt_context_label[2:]}」の内容から**合理的に推論**できることを含めて回答を組み立ててください。
3. 推測に基づいて回答する場合は、「{prompt_context_label[2:]}からは〇〇と推測されます。」のように、推測であることを示唆する表現を任意で含めても良いです。
4. 関連情報が質問に答える上で全く不十分な場合は、「提供された情報からは判断できません。」と回答してください。
5. 回答は、質問の意図を汲み取り、具体的で分かりやすい言葉で記述してください。

{prompt_context_label}
{context_source}

# 質問:
{question}

# 回答:
"""
    logger.debug(f"Generating inferred answer with prompt:\n{prompt[:300]}...")
    try:
        if not GEMINI_AVAILABLE or not genai:
             logger.error("Gemini library not available for inferred answer generation.")
             return "[推論回答生成エラー: Geminiライブラリ未検出]"
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.synthesizer_model_name) # 推論用モデル
        generation_config = genai.types.GenerationConfig(
            temperature=config.synthesizer_temperature, # 設定ファイルから取得
            # top_p=config.synthesizer_top_p # 必要なら設定
            # max_output_tokens=config.synthesizer_max_new_tokens # 必要なら設定
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        final_answer = response.text.strip()
        logger.debug(f"Generated inferred answer: {final_answer[:200]}...")
        return final_answer
    except Exception as e:
        logger.error(f"Error calling Gemini API for inferred answer generation: {e}", exc_info=True)
        return f"[推論回答生成エラー: {type(e).__name__}]"

# --- process_single_variant 関数の修正 ---
def process_single_variant(args: tuple) -> tuple:
    """
    １つのRAGバリアントを実行。中間回答生成を2ステップに変更。
    ステップA: 情報抽出, ステップB: 推論・回答生成
    """
    variant_index, query, variant_params, config, vectordb, intermediate_llm, embedding_function, metadata_filter, logger = args
    variant_id = variant_index + 1
    log_prefix = f"[Variant {variant_id}]"
    logger.info(f"{log_prefix} Processing start for query: '{query[:50]}...'")

    params = variant_params
    variant_k = params.get("k", config.rag_variant_k[0] if config.rag_variant_k else 3) # デフォルトk値
    retrieved_docs_set: List[Document] = []
    context_snippet: str = "N/A"
    top_docs: List[Document] = []
    context: str = "" # tryブロック外で初期化

    try:
        # 1. Retriever準備
        if not vectordb:
             logger.error(f"{log_prefix} VectorDB is not initialized. Cannot perform search.")
             return f"エラー({variant_id}-NoDB)", "エラー", []
        retriever_variant = vectordb.as_retriever(
            search_kwargs={"k": variant_k, "filter": metadata_filter}
        )
        logger.debug(f"{log_prefix} Retriever created with k={variant_k}, filter={metadata_filter}")

        # 2. 検索実行
        expanded_queries = [query]
        if config.enable_query_expansion:
            try:
                 expanded_queries = expand_query(query, config.query_expansion_dict)
                 logger.info(f"{log_prefix} Expanded queries: {expanded_queries}")
            except Exception as exp_e:
                 logger.warning(f"{log_prefix} Query expansion failed: {exp_e}. Using original query.")
                 expanded_queries = [query]

        seen_ids = set()
        current_retrieved_docs = []
        for q_idx, q in enumerate(expanded_queries):
            try:
                 docs = retriever_variant.invoke(q)
                 logger.debug(f"{log_prefix} Retrieved {len(docs)} docs for sub-query {q_idx+1}: '{q[:50]}...'")
                 current_retrieved_docs.extend(docs)
            except Exception as search_q_e:
                 logger.error(f"{log_prefix} Error during search for sub-query '{q[:50]}...': {search_q_e}", exc_info=True)
                 # このサブクエリは失敗しても、他のサブクエリの結果で続行する

        # 重複排除
        for doc in current_retrieved_docs:
            if doc.page_content is not None:
                doc_id = (doc.metadata.get('source', 'unknown'), doc.page_content)
                if doc_id not in seen_ids:
                    retrieved_docs_set.append(doc)
                    seen_ids.add(doc_id)

        logger.info(f"{log_prefix} Retrieved {len(retrieved_docs_set)} unique documents total.")
        # 検索結果0件でも後続処理へ

    except Exception as search_e:
        logger.error(f"{log_prefix} Error during search setup/execution: {search_e}", exc_info=True)
        return f"エラー({variant_id}-Search)", "エラー", []

    # 3. リランキング (オプション)
    reranked_docs = retrieved_docs_set
    if config.enable_reranking and retrieved_docs_set: # ドキュメントがある場合のみ実行
        try:
            if embedding_function:
                reranked_docs = rerank_documents(query, retrieved_docs_set, embedding_function, logger=logger, k=variant_k)
                logger.info(f"{log_prefix} Reranked to {len(reranked_docs)} docs.")
            else:
                logger.warning(f"{log_prefix} Reranking enabled but no embedding function available.")
        except Exception as rerank_e:
            logger.error(f"{log_prefix} Error during reranking: {rerank_e}", exc_info=True)
            # リランキング失敗でも元の検索結果で続行
            reranked_docs = retrieved_docs_set

    # 4. コンテキスト作成 & スニペット生成
    top_docs = reranked_docs[:variant_k] # リランキング結果または元の検索結果の上位k件
    context = "\n\n".join([doc.page_content for doc in top_docs if doc.page_content])
    try:
        context_snippet = format_document_snippet(context, 150)
    except Exception as fmt_e:
        logger.warning(f"{log_prefix} Failed to format context snippet: {fmt_e}")
        context_snippet = context[:150] + "..." if context else "N/A"
    logger.debug(f"{log_prefix} Context length: {len(context)}, Snippet: '{context_snippet}'")

    # 5. (中間プロンプト作成はスキップ)

    # 6. 中間回答生成 (2ステップ)
    intermediate_answer = f"エラー({variant_id}-IntermediateGen)" # デフォルトエラー

    # ステップ 6A: 情報抽出
    logger.info(f"{log_prefix} Step 6A: Extracting relevant info using Gemini API...")
    extracted_info = extract_relevant_info_with_gemini(context, query, config)

    # ステップ 6B: 推論・回答生成
    logger.info(f"{log_prefix} Step 6B: Generating inferred answer using Gemini API...")
    # 抽出情報に加え、元のコンテキストも渡して推論させる
    intermediate_answer = generate_inferred_answer_with_gemini(extracted_info, query, config, original_context=context)

    logger.info(f"{log_prefix} Intermediate answer generation finished.")
    logger.debug(f"{log_prefix} Final intermediate answer snippet: {intermediate_answer[:100]}...")

    logger.info(f"{log_prefix} Processing finished.")
    return intermediate_answer, context_snippet, top_docs


# --- 非同期ラッパー関数 (変更なし) ---
async def process_variant_async(args: tuple) -> tuple:
    """同期関数 process_single_variant をスレッドプールで非同期実行する"""
    loop = asyncio.get_event_loop()
    config = args[3]
    logger = args[-1]
    variant_id = args[0] + 1
    log_prefix = f"[Variant {variant_id}-Async]"
    # configから並列数を取得、なければデフォルト2
    max_workers = getattr(config, 'max_parallel_variants', 2)

    logger.debug(f"{log_prefix} Submitting to ThreadPoolExecutor (max_workers={max_workers})...")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            result = await loop.run_in_executor(executor, process_single_variant, args)
        logger.info(f"{log_prefix} Execution finished successfully.")
        return result
    except Exception as async_e:
        logger.error(f"{log_prefix} Error in async execution wrapper: {async_e}", exc_info=True)
        return f"エラー({variant_id}-Async)", "エラー", []


# --- 並列タスクを実行するためのヘルパー関数 (変更なし) ---
async def run_all_variants_concurrently(args_list: list) -> list:
    """複数のVariant処理タスクを並列実行し、結果をリストで返す"""
    if not args_list:
        return []
    logger = args_list[0][-1] # リストの最初の要素からロガーを取得
    num_tasks = len(args_list)
    logger.info(f"Running {num_tasks} variants concurrently using asyncio.gather...")
    tasks = [process_variant_async(args) for args in args_list]
    # 全てのタスクが完了するのを待つ (例外もキャッチ)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All variant tasks completed (or failed).")

    processed_results = []
    for i, res in enumerate(results):
        variant_id = i + 1
        if isinstance(res, Exception):
            logger.error(f"[Variant {variant_id}-Async] Task failed with exception: {res}", exc_info=res)
            processed_results.append((f"エラー({variant_id}-TaskFail)", "エラー", []))
        # 正常終了の場合でもタプルの形式をチェック
        elif isinstance(res, tuple) and len(res) == 3:
             processed_results.append(res)
             # 内部でエラーが返されている場合も警告ログ
             if isinstance(res[0], str) and res[0].startswith("エラー"):
                  logger.warning(f"[Variant {variant_id}] Task completed but returned an error state: {res[0]}")
        else:
            # 予期しない結果フォーマットの場合
             logger.error(f"[Variant {variant_id}-Async] Task returned unexpected result format: {res}")
             processed_results.append((f"エラー({variant_id}-ResultFormat)", "エラー", []))

    return processed_results


# --- ストリーミング最適化関数 (変更なし) ---
def optimized_gemini_stream_generator(response_stream, buffer_size: int = 3, delay: float = 0.01):
    """APIからのストリームをバッファリングし、まとめてyieldするジェネレータ"""
    buffer = []
    # logger = logging.getLogger("RAGapp_gemeni") # グローバルloggerを使用
    chunk_count = 0
    yielded_count = 0
    try:
        for chunk in response_stream:
            chunk_count += 1
            # APIからのレスポンス形式に合わせて text 属性などを取得
            # 例: chunk.text, chunk['candidates'][0]['content']['parts'][0]['text'] など
            text_content = None
            try:
                 # Gemini API の GenerateContentResponse ストリームを想定
                 if hasattr(chunk, 'text'):
                      text_content = chunk.text
                 elif hasattr(chunk, 'parts') and chunk.parts: # Safety feedback など text を持たない場合がある
                      text_content = "".join(part.text for part in chunk.parts if hasattr(part, 'text'))
                 # 他のAPI形式の場合はここを調整
            except Exception as parse_err:
                 logger.warning(f"Could not parse text from stream chunk: {parse_err}. Chunk: {chunk}", exc_info=True)

            if text_content:
                buffer.append(text_content)
                if len(buffer) >= buffer_size:
                    yielded_text = ''.join(buffer)
                    yield yielded_text
                    yielded_count += 1
                    buffer = []
            # else: logger.debug(f"Stream chunk {chunk_count} received without yieldable text content.")

            if delay > 0:
                time.sleep(delay)

        if buffer: # 残りを出力
            yielded_text = ''.join(buffer)
            yield yielded_text
            yielded_count += 1
            logger.debug(f"Yielded final buffered block {yielded_count} (size {len(buffer)})")
        logger.info(f"Optimized stream processing finished. Total chunks received: {chunk_count}, Yielded blocks: {yielded_count}.")

    except Exception as e:
        logger.error(f"Error during optimized API stream processing: {e}", exc_info=True)
        yield f"\n\n[ストリーミング処理エラー: {e}]"


# --- アンサンブル質問応答関数 (変更なし) ---
def ask_question_ensemble_stream(
    vectordb: Optional[Chroma],
    intermediate_llm: Optional[BaseLLM], # 引数としては残すが内部では使用しない
    tokenizer: Optional[AutoTokenizer], # 引数としては残すが内部では使用しない
    embedding_function: Optional[HuggingFaceEmbeddings],
    config: Config,
    query: str,
    logger: logging.Logger, # 引数 logger を受け取る
    metadata_filter: Optional[dict] = None,
    variant_params: Optional[list] = None
) -> dict:
    """複数Variantの検索・中間回答生成を並列処理し、外部APIで最終統合（ストリーミング）"""
    start_time_total = time.time()
    # llm, tokenizer のチェックを緩和 (Geminiのみ使う場合)
    if not all([vectordb, embedding_function, config, logger]):
        err_msg = "System Error: RAG components (VDB, Embeddings) or config/logger missing."
        print(err_msg, file=sys.stderr)
        if logger: logger.error(err_msg)
        return {"result_stream": iter([err_msg]), "source_documents": [], "variant_answers": []}

    if not isinstance(query, str) or not query.strip():
        logger.warning("Invalid query received.")
        return {"result_stream": iter(["質問を入力してください。"]), "source_documents": [], "variant_answers": []}

    # Gemini API 利用可否チェック
    if config.synthesizer_api == "gemini":
         if not GEMINI_AVAILABLE:
              logger.error("Gemini API selected, but google-generativeai library is not available.")
              return {"result_stream": iter(["システムエラー: Geminiライブラリ未検出。"]), "source_documents": [], "variant_answers": []}
         if not config.gemini_api_key:
              logger.error("Gemini API key is not configured.")
              return {"result_stream": iter(["システムエラー: Gemini APIキー未設定。"]), "source_documents": [], "variant_answers": []}
    # 他のAPIを使う場合のチェックもここに追加可能

    logger.info(f"Processing query (Ensemble Parallel): '{query[:100]}...'")
    print_gpu_usage(logger, "Start Ensemble RAG Query")

    # Variantパラメータ設定
    if not variant_params:
        num_default_variants = getattr(config, 'num_default_variants', 3)
        variant_params = [
            {"k": config.rag_variant_k[i] if i < len(config.rag_variant_k) else 3,
             # 中間LLMパラメータは現状使わないが、将来のために残すことも可能
             "temperature": config.temperature, "top_p": config.top_p,
             "repetition_penalty": config.repetition_penalty}
             for i in range(num_default_variants)
        ]
    num_variants = len(variant_params)
    logger.info(f"Number of variants to process: {num_variants}")

    # --- 並列処理の実行 ---
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
        logger.info("Starting parallel variant processing...")
        # Streamlit等の既存ループがある環境を考慮し、エラーハンドリング付きで実行
        try:
            loop = asyncio.get_running_loop()
             # 既にループが動いている場合は run_in_executor を使う (ThreadPoolExecutorの管理は関数内)
             # logger.debug("Event loop already running. Using run_in_executor within gather.")
             # この方法は run_all_variants_concurrently の中で run_in_executor を使うため不要
             # results = await run_all_variants_concurrently(args_list) # await は async def の中でないと使えない
             # 同期関数内からは asyncio.run() を使うのが基本だが、Streamlitでは問題が起きる
             # そのため、ここでは逐次実行フォールバックを前提とするか、
             # Streamlitの非同期対応 (st.experimental_rerunなど) を利用する設計にする。
             # 今回はRuntimeErrorをキャッチして逐次実行する方式を採用。
            results = asyncio.run(run_all_variants_concurrently(args_list))

        except RuntimeError as rt_err:
            # "cannot run event loop while another loop is running" 等
            logger.error(f"RuntimeError during asyncio.run: {rt_err}. Falling back to sequential execution.")
            try:
                 results = [process_single_variant(args) for args in args_list]
                 logger.info("Sequential fallback execution completed.")
            except Exception as seq_e:
                 logger.error(f"Sequential fallback also failed: {seq_e}", exc_info=True)
                 return {"result_stream": iter([f"並列・逐次処理エラー: {seq_e}"]),
                         "source_documents": [],
                         "variant_answers": [f"エラーV{i+1}" for i in range(num_variants)]}

        logger.info("Variant processing finished (async or sequential).")

        # 結果を整理 (run_all_variants_concurrently で例外処理済みと想定)
        for i, res in enumerate(results):
             if isinstance(res, tuple) and len(res) == 3:
                  intermediate_answers[i] = res[0]
                  context_snippets[i] = res[1]
                  all_source_documents_nested[i] = res[2] if isinstance(res[2], list) else []
             else: # run_all_variants_concurrently がエラータプルを返した場合など
                  logger.error(f"Received unexpected result for Variant {i+1}: {res}")
                  intermediate_answers[i] = res[0] if isinstance(res, tuple) and len(res)>0 else f"エラー({i+1}-Result)"
                  context_snippets[i] = res[1] if isinstance(res, tuple) and len(res)>1 else "エラー"
                  all_source_documents_nested[i] = res[2] if isinstance(res, tuple) and len(res)>2 and isinstance(res[2], list) else []


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
    logger.info(f"Total unique source docs after combining variants: {len(unique_source_documents)}")

    # --- 最終統合ステップ (外部API) ---
    final_answer_stream = None
    # 有効な中間回答をフィルタリング (エラーメッセージなどを除外)
    valid_intermediate_answers = [
        ans for ans in intermediate_answers
        if ans and not ans.startswith("エラー") and not ans.startswith("[") # エラーを示す文字列を除外
            and "判断できません" not in ans # 「判断できません」系の回答も除外 (推論を期待するため)
            # and ans != "関連情報なし" # 必要に応じて追加
    ]
    logger.info(f"Found {len(valid_intermediate_answers)} valid intermediate answers for synthesis.")

    if not valid_intermediate_answers:
        logger.warning("No valid intermediate answers found for synthesis. Using fallback message.")
        # 有効な回答がない場合、最終的なフォールバックメッセージを決定
        final_answer_text = "関連する情報が見つかりませんでした。"
        # エラーが一つでも含まれていたかチェック
        if any(ans.startswith("エラー") or ans.startswith("[") for ans in intermediate_answers):
             final_answer_text = "回答生成中に一部エラーが発生しました。関連情報が見つからないか、処理中に問題が発生しました。"
        # DBは読めたが情報が見つからなかった場合
        elif any("提供された情報からは判断できません" in ans for ans in intermediate_answers):
             final_answer_text = "関連情報は検索されましたが、質問に明確に答えるには不十分でした。"

        final_answer_stream = iter([final_answer_text])
    else:
        logger.info(f"Synthesizing final answer using API: {config.synthesizer_api} ({config.synthesizer_model_name})...")
        try:
            # 最終統合用プロンプトテンプレートを取得
            # synthesis_prompt_template_str = getattr(config, 'synthesis_prompt_template', "...") # configから取得
            # input_variables を動的に設定 (config内のテンプレートが f-string ベースの場合)
            # required_vars = ["original_question"] + [f"answer_{j+1}" for j in range(num_variants)] + [f"context_{j+1}_snippet" for j in range(num_variants)]
            # synthesis_prompt = PromptTemplate(template=synthesis_prompt_template_str, input_variables=required_vars)
            # ※ config_gemeni.py でプロンプトを動的に生成する実装に変更済みなので、そちらを使う

            synthesis_input_dict = {"original_question": query}
            for j in range(num_variants): # configで定義されたvariant数分のプレースホルダーを埋める
                ans_key = f"answer_{j+1}"
                ctx_key = f"context_{j+1}_snippet"
                synthesis_input_dict[ans_key] = intermediate_answers[j] if j < len(intermediate_answers) else "N/A"
                synthesis_input_dict[ctx_key] = context_snippets[j] if j < len(context_snippets) else "N/A"

            # configから動的に生成されたテンプレート文字列を取得
            final_synthesis_prompt_template = config.synthesis_prompt_template

            # format_map を使って、存在するキーのみでフォーマット
            formatted_synthesis_prompt = final_synthesis_prompt_template.format_map(synthesis_input_dict)

            logger.debug(f"Synthesis prompt for API (first 300 chars): {formatted_synthesis_prompt[:300]}...")

            if config.synthesizer_api == "gemini":
                genai.configure(api_key=config.gemini_api_key)
                generation_config_api = genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=config.synthesizer_temperature,
                    top_p=config.synthesizer_top_p,
                    max_output_tokens=config.synthesizer_max_new_tokens
                )
                # 安全性設定 (必要に応じて調整)
                safety_settings=[
                    {"category": c, "threshold": "BLOCK_NONE"} for c in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT"
                    ]
                ]
                model_api = genai.GenerativeModel(config.synthesizer_model_name)

                response_stream = model_api.generate_content(
                    formatted_synthesis_prompt,
                    generation_config=generation_config_api,
                    safety_settings=safety_settings,
                    stream=True
                )

                final_answer_stream = optimized_gemini_stream_generator(
                    response_stream,
                    buffer_size=getattr(config, 'stream_buffer_size', 3),
                    delay=getattr(config, 'stream_delay', 0.01)
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
        "variant_answers": intermediate_answers # 各Variantの最終中間回答(ステップBの結果)
    }

# --- ここまで rag_query_utils_gemeni.py ---