# auto_fill_utils.py - RAGによるフォーム自動入力機能 (ワンショットJSON抽出方式)

import logging
import json
import traceback
from typing import Dict, Any, Optional

# Gemini API import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger("RAGApp")

# --- JSON抽出用のフィールド定義 ---
# LLMにこのキーでJSONを生成するように依頼する
FORM_FIELDS_FOR_EXTRACTION = {
    "form_meeting_company": "打ち合わせ担当会社",
    "form_meeting_person": "打ち合わせ担当者",
    "form_tel": "電話番号 (例: 03-1234-5678)",
    "form_mobile": "携帯電話番号 (例: 090-1234-5678)",
    "form_email": "メールアドレス (例: contact@example.com)",
    "form_outer_wall_strong": "耐力壁(外部)の仕様 (例: 構造用合板 9mm)",
    "form_inner_wall_strong": "耐力壁(内部)の仕様 (例: 石膏ボード 12.5mm)",
    "form_non_strong_wall": "非耐力壁の仕様 (例: 石膏ボード 9.5mm)",
    "form_wall_type": "外壁仕様 (「大壁」または「真壁」)",
    "form_sub_materials": "副資材の供給の有無 (「有」または「無」)",
    "form_temporary_brace": "仮筋交の有無 (「有」または「無」)",
    "form_foundation_packing": "基礎パッキンの有無 (「有」または「無」)",
    "form_airtight_packing": "気密パッキンの有無 (「有」または「無」)",
    "form_airtight_range": "気密パッキンの適用範囲",
    "form_steel_post": "鋼製束の有無 (「有」または「無」)",
    "form_hardware_install": "金物取付の有無 (「有」または「無」)",
    "form_other_notes": "その他記載事項"
}

def auto_fill_form_data(company_name: str, vectordb, intermediate_llm, tokenizer, embedding_function, config, logger) -> Dict[str, str]:
    """
    指定された工務店名に基づき、RAGとLLMによるワンショットJSON抽出でフォームデータを自動入力する。
    """
    logger.info(f"Starting one-shot auto-fill for company: {company_name}")

    try:
        # 1. 検索は1回だけ: 工務店名で関連ドキュメントをまとめて取得
        logger.debug(f"Searching ChromaDB with company name: '{company_name}', k=15")
        # 情報を網羅するため、取得ドキュメント数を増やす (k=15)
        docs = vectordb.similarity_search(company_name, k=15)

        if not docs:
            logger.warning(f"No documents found via semantic search for company: {company_name}")
            return {}

        logger.debug(f"ChromaDB search returned {len(docs)} documents.")
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 2. 質問も1回だけ: ワンショットJSON抽出プロンプトを作成
        # Gemini APIが利用可能かチェック
        use_gemini = (GEMINI_AVAILABLE and hasattr(config, 'auto_fill_use_gemini') and 
                      config.auto_fill_use_gemini and hasattr(config, 'gemini_api_key') and config.gemini_api_key)

        if use_gemini:
            json_string = execute_gemini_json_extraction(context, company_name, config, logger)
        else:
            # ローカルLLM用の実装
            logger.info("Using local LLM for JSON extraction.")
            json_string = execute_local_llm_json_extraction(context, company_name, intermediate_llm, tokenizer, logger)

        if not json_string:
            logger.error("Failed to get JSON response from LLM.")
            return {}

        # 3. 解析して入力: JSONをパースして結果を整形
        logger.debug(f"LLM returned JSON string: {json_string}")
        extracted_data = json.loads(json_string)
        
        auto_fill_results = {}
        for field_name, value in extracted_data.items():
            if field_name in FORM_FIELDS_FOR_EXTRACTION and value not in [None, "", "情報なし", "不明"]:
                processed_value = process_rag_result(field_name, str(value))
                if processed_value is not None: # Changed condition to allow empty strings if returned
                    auto_fill_results[field_name] = processed_value
                    logger.info(f"Auto-filled {field_name}: {processed_value}")
                else:
                    logger.debug(f"Skipping auto-fill for {field_name} as processed_value is None (original: {value})")

        logger.info(f"Auto-fill completed. Filled {len(auto_fill_results)} fields.")
        return auto_fill_results

    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to parse JSON response from LLM: {json_err}", exc_info=True)
        logger.error(f"Invalid JSON string was: {json_string}")
        return {}
    except Exception as e:
        logger.error(f"Auto-fill process failed: {e}", exc_info=True)
        return {}

def execute_gemini_json_extraction(context: str, company_name: str, config, logger) -> Optional[str]:
    """Gemini APIにワンショットでJSON抽出を依頼する"""
    try:
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.auto_fill_model)

        # JSONスキーマを文字列としてプロンプトに埋め込む
        json_schema_str = json.dumps(FORM_FIELDS_FOR_EXTRACTION, ensure_ascii=False, indent=2)

        prompt = f"""# 指示
以下の「参考情報」を基に、「{company_name}」に関する情報を抽出し、指定されたJSON形式で回答してください。

# 参考情報
{context}

# 抽出ルール
- 必ず「参考情報」に書かれている内容のみを根拠としてください。
- 各項目について、情報が見つからない場合は `null` または `"情報なし"` としてください。
- 回答はJSONオブジェクトのみとし、前後に説明文や```json ```などを付けないでください。

# JSONスキーマ
{json_schema_str}

# 回答 (JSON)
"""

        logger.debug(f"Sending one-shot JSON extraction prompt to Gemini for {company_name}")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json" # JSONモードを有効化
            )
        )

        if response and response.text:
            result = response.text.strip()
            logger.info(f"Gemini API (JSON mode) response for {company_name} received.")
            return result
        else:
            logger.warning("Gemini API returned empty response in JSON mode.")
            return None

    except Exception as e:
        logger.error(f"Gemini API JSON extraction failed: {e}", exc_info=True)
        return None

def execute_local_llm_json_extraction(context: str, company_name: str, intermediate_llm, tokenizer, logger) -> Optional[str]:
    """
    ローカルLLMにワンショットでJSON抽出を依頼する。
    """
    try:
        json_schema_str = json.dumps(FORM_FIELDS_FOR_EXTRACTION, ensure_ascii=False, indent=2)

        prompt = f"""# 指示
以下の「参考情報」を基に、「{company_name}」に関する情報を抽出し、指定されたJSON形式で回答してください。

# 参考情報
{context}

# 抽出ルール
- 必ず「参考情報」に書かれている内容のみを根拠としてください。
- 各項目について、情報が見つからない場合は `null` または `"情報なし"` としてください。
- 回答はJSONオブジェクトのみとし、前後に説明文や```json ```などを付けないでください。

# JSONスキーマ
{json_schema_str}

# 回答 (JSON)
"""
        logger.debug(f"Sending one-shot JSON extraction prompt to local LLM for {company_name}")
        
        # ローカルLLMで回答生成
        # HuggingFacePipelineのinvokeメソッドは文字列を返す
        raw_response = intermediate_llm.invoke(prompt)
        
        # LLMの出力からJSON部分を抽出するロジック
        response_marker = "# 回答 (JSON)\n"
        parts = raw_response.split(response_marker)

        if len(parts) > 1:
            # Take the last part, which should contain the answer JSON
            json_candidate = parts[-1].strip()
            
            # Now, find the first '{' and last '}' within this candidate string
            json_start = json_candidate.find("{")
            json_end = json_candidate.rfind("}")

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = json_candidate[json_start : json_end + 1]
                logger.info(f"Local LLM JSON response for {company_name} extracted after splitting.")
                return json_string
            else:
                logger.warning(f"Local LLM returned malformed JSON after marker: {json_candidate[:200]}...")
                return None
        else:
            logger.warning(f"Local LLM response did not contain the expected JSON marker: {raw_response[:200]}...")
            return None

    except Exception as e:
        logger.error(f"Local LLM JSON extraction failed: {e}", exc_info=True)
        return None

def process_rag_result(field_name: str, rag_result: str) -> Optional[str]:
    """
    LLMが生成した値を各フィールドに適した形式に変換する。
    主に「有」「無」などのキーワードを正規化する。
    """
    if not rag_result:
        return None

    result_lower = rag_result.lower().strip()

    # ラジオボタン項目のキーワードマッチング
    radio_keywords = {
        "有": ["有", "あり", "供給", "提供", "行う", "true"],
        "無": ["無", "なし", "供給しない", "提供しない", "行わない", "false"],
        "大壁": ["大壁", "おおかべ"],
        "真壁": ["真壁", "しんかべ"]
    }

    # フィールドがラジオボタン系か判定
    is_radio_field = any(field_name.endswith(s) for s in [
        '_type', '_materials', '_brace', '_packing', '_post', '_install'
    ])

    if is_radio_field:
        for canonical_value, keywords in radio_keywords.items():
            for keyword in keywords:
                if keyword in result_lower:
                    return canonical_value
        # マッチしない場合は元の値を返さず、Noneとする
        return None 

    # テキストフィールドはそのまま返す
    return rag_result