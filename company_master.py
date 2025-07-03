# company_master.py - 工務店マスタデータの管理と検索機能

import json
import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from functools import lru_cache

try:
    import jaconv
    JACONV_AVAILABLE = True
except ImportError:
    JACONV_AVAILABLE = False
    print("Warning: jaconv not available. Installing with: pip install jaconv")

try:
    from sentence_transformers import SentenceTransformer
    from langchain_chroma import Chroma
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    print("Warning: Vector search dependencies not available")

logger = logging.getLogger("RAGApp")

class CompanyMaster:
    """工務店マスタデータの管理と表記揺れ対策機能"""
    
    def __init__(self, config=None):
        self.config = config
        self.companies = []
        self.embedding_model = None
        self.vector_db = None
        self.load_companies()
        
        if VECTOR_SEARCH_AVAILABLE and config:
            self.setup_vector_search()
    
    def load_companies(self):
        """工務店データをロード"""
        # データセットから抽出した工務店一覧
        company_list = [
            "BESS柏（株式会社imayama）", "ＥＣＯ ＨＯＵＳＥ株式会社", "ＥＣＯＨＯＵＳＥ株式会社",
            "ＦＤＭ株式会社", "ＨＯＭＥＣＲＡＦＴ株式会社", "ｏｕ２株式会社", "Ｒ.クラフト株式会社",
            "ＴＯＡＨＯＭＥ株式会社", "アサヒハウジング有限会社", "オーガニック・スタジオ株式会社",
            "オーガニックスタジオ株式会社", "クウェスト株式会社", "さくら工房株式会社",
            "サンキホーム株式会社", "ジェイホームズ株式会社", "センコー産業株式会社",
            "ダイクス建設株式会社", "ダンハウス株式会社", "ニケンハウジング株式会社",
            "はだしの家．株式会社", "フクダ・ロングライフデザイン株式会社", "フクダロングライフデザイン株式会社",
            "フジ住宅株式会社", "ホクシンハウス株式会社", "三陽建設株式会社", "中島建設株式会社",
            "五大ディ・シー・エム株式会社", "五大ディシーエム株式会社", "円徳建工株式会社",
            "匠建設株式会社", "古川製材株式会社", "夢工房だいあん株式会社",
            "大和ハウス工業株式会社", "大輪建設株式会社", "太豊建設株式会社",
            "宏州建設株式会社（ＩＮＯＳ）", "宮部建設株式会社", "山栄ホーム株式会社",
            "幸和ハウジング株式会社", "志馬建設株式会社", "拡運建設株式会社",
            "日本住宅ツーバイ株式会社", "有限会社インターコラボデザイン", "有限会社コバヤシホーム",
            "有限会社コバヤシ建築", "有限会社たけひろ建築工房", "有限会社伊東工務店",
            "有限会社鳥越住建", "松代建設工業株式会社", "栃井建設工業株式会社",
            "株式会社Base Prosper", "株式会社BaseProsper", "株式会社ＥＭＳＳ",
            "株式会社ＪＥＤ", "株式会社ｋｏｔｏｒｉ", "株式会社ＫＵＲＡＳＵ",
            "株式会社marukan", "株式会社ＭＩＫＩホーム", "株式会社PASSIVE DESIGN COME HOME",
            "株式会社PASSIVEDESIGNCOMEHOME", "株式会社ＳＶＤ建築工房", "株式会社ＷＨＡＬＥＨＯＵＳＥ",
            "株式会社アーキテックシンワ", "株式会社アーキ・モーダ", "株式会社アーキモーダ",
            "株式会社アースティック", "株式会社アートホーム北見", "株式会社アートホーム千歳",
            "株式会社アートホーム札幌", "株式会社アールティーシーマネージメント",
            "株式会社アール・ティー・シーマネージメント", "株式会社アールプランナー",
            "株式会社アイシークリエーション一級建築士事務所", "株式会社アイナデベロップメント",
            "株式会社アイビック", "株式会社アキヤマ", "株式会社アスカハウジング",
            "株式会社インフィルプラス", "株式会社エーティーエム建築", "株式会社エスケーホーム",
            "株式会社エスコト社", "株式会社エナミホームズ", "株式会社エヌ・シー・エヌ",
            "株式会社エヌテック", "株式会社オーワークス", "株式会社キリガヤ",
            "株式会社クレストホームズ", "株式会社コージーライフ", "株式会社サクラ工研",
            "株式会社サンキ建設", "株式会社サンハウス", "株式会社じょぶ",
            "株式会社スタジオカーサ", "株式会社タイコーアーキテクト", "株式会社タイコーハウジングコア",
            "株式会社タマック", "株式会社チェックハウス", "株式会社デザオ建設",
            "株式会社テラジマアーキテクツ", "株式会社テリオス", "株式会社トーシンホーム",
            "株式会社トータテハウジング", "株式会社ナカタホーム岡山", "株式会社にいむの杜",
            "株式会社ネクサスアーキテクト", "株式会社ノーブルホーム", "株式会社ハウステックス",
            "株式会社バウムスタイルアーキテクト", "株式会社ハヤシ工務店", "株式会社ビーエムシー",
            "株式会社ビー・ツー", "株式会社ビーツー", "株式会社ひかり工務店",
            "株式会社ビルド・ワークス", "株式会社ビルドワークス", "株式会社フリーダムデザイン",
            "株式会社ホープス", "株式会社ホーム・テック", "株式会社ホームテック",
            "株式会社ホームランディック", "株式会社マイ工務店", "株式会社マサキ工務店",
            "株式会社マルナカホーム", "株式会社ミューズの家", "株式会社ヤガワ",
            "株式会社ヤマダホームズ", "株式会社ユニテ", "株式会社リモルデザイン",
            "株式会社三和建設", "株式会社三建", "株式会社中藏", "株式会社丸尾建築",
            "株式会社亀岡工務店", "株式会社二幸住建", "株式会社伊田工務店",
            "株式会社伊藤工務店", "株式会社住まい設計工房", "株式会社住創館",
            "株式会社創建", "株式会社印南建設", "株式会社参創ハウテック",
            "株式会社和工務店", "株式会社大兼工務店", "株式会社大熊工業",
            "株式会社大興ネクスタ", "株式会社大雄", "株式会社富士建設",
            "株式会社小田急ハウジング", "株式会社平成建設", "株式会社新名工務店",
            "株式会社新宅工務店", "株式会社日伸建設", "株式会社明治ホームズ",
            "株式会社星野建築事務所", "株式会社東産業", "株式会社梅原建設",
            "株式会社楠亀工務店", "株式会社永博", "株式会社洞口",
            "株式会社浅井良工務店", "株式会社渋谷", "株式会社滝石建設",
            "株式会社福屋工務店", "株式会社空間建築工房", "株式会社素箱",
            "株式会社興和アークビルド", "株式会社藤井工務店", "株式会社藪崎工務店",
            "株式会社西峰工務店", "株式会社近藤組", "株式会社鈴木組",
            "株式会社関工務所", "株式会社高砂建設", "株式会社黒木建設",
            "渋沢テクノ建設株式会社", "皐工務店株式会社", "磯田建設株式会社",
            "福島工務店株式会社", "立松建設株式会社", "立石産業株式会社",
            "笹沢建設株式会社", "米屋建設株式会社", "自由宅工房株式会社",
            "近藤建設工業株式会社", "阿部建設株式会社"
        ]
        
        # 正規化済みデータとして格納
        for i, company in enumerate(company_list):
            normalized = self.normalize_company_name(company)
            self.companies.append({
                "id": f"company_{i:04d}",
                "original_name": company,
                "normalized_name": normalized,
                "search_keywords": self.extract_search_keywords(company)
            })
        
        logger.info(f"Loaded {len(self.companies)} companies into master data")
    
    def normalize_company_name(self, company_name: str) -> str:
        """工務店名の正規化処理"""
        if not isinstance(company_name, str):
            return ""
        
        try:
            # 1. Unicode正規化
            normalized = unicodedata.normalize('NFKC', company_name)
            
            # 2. jaconvによる変換（利用可能な場合）
            if JACONV_AVAILABLE:
                # 全角英数を半角に
                normalized = jaconv.z2h(normalized, digit=True, ascii=True)
                # 半角カナを全角に
                normalized = jaconv.h2z(normalized, kana=True)
            
            # 3. 会社形態の統一
            company_replacements = [
                (r'㈱', '株式会社'),
                (r'㈲', '有限会社'),
                (r'(株)', '株式会社'),
                (r'(有)', '有限会社'),
                (r'合同会社', '合同会社'),
                (r'協同組合', '協同組合'),
            ]
            
            for pattern, replacement in company_replacements:
                normalized = re.sub(pattern, replacement, normalized)
            
            # 4. 空白の正規化
            normalized = re.sub(r'[\s　]+', '', normalized)  # 全角半角スペース削除
            
            # 5. 小文字化（英字部分のみ）
            normalized = re.sub(r'[A-Z]', lambda m: m.group().lower(), normalized)
            
            return normalized.strip()
            
        except Exception as e:
            logger.warning(f"Company name normalization error for '{company_name}': {e}")
            return company_name
    
    def extract_search_keywords(self, company_name: str) -> List[str]:
        """検索用キーワードを抽出"""
        keywords = []
        
        # 会社形態を除いた部分を抽出
        base_patterns = [
            r'株式会社(.+)',
            r'(.+)株式会社',
            r'有限会社(.+)',
            r'(.+)有限会社',
            r'(.+)工務店',
            r'(.+)建設',
            r'(.+)ホーム',
            r'(.+)ハウス',
            r'(.+)建築'
        ]
        
        for pattern in base_patterns:
            match = re.search(pattern, company_name)
            if match:
                keyword = match.group(1).strip()
                if keyword:
                    keywords.append(keyword)
        
        # 英字部分の抽出
        english_parts = re.findall(r'[A-Za-z]+', company_name)
        keywords.extend(english_parts)
        
        return list(set(keywords))  # 重複除去
    
    def setup_vector_search(self):
        """ベクトル検索の初期化"""
        try:
            if not self.config or not hasattr(self.config, 'embeddings_model'):
                logger.warning("Config or embeddings_model not available for vector search")
                return
            
            # 埋め込みモデルの初期化
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model,
                model_kwargs={'device': 'cpu' if self.config.force_cpu else 'auto'}
            )
            
            logger.info("Vector search initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup vector search: {e}")
            self.embedding_model = None
    
    @lru_cache(maxsize=512)
    def search_companies(self, query: str, limit: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """工務店名の検索（表記揺れ対策）"""
        if not query or len(query.strip()) < 2:
            return []
        
        normalized_query = self.normalize_company_name(query)
        results = []
        
        # 1. 完全一致検索
        for company in self.companies:
            if normalized_query == company['normalized_name']:
                results.append({
                    'company': company,
                    'score': 1.0,
                    'match_type': 'exact'
                })
        
        if results:
            return results[:limit]
        
        # 2. 部分一致検索
        for company in self.companies:
            if normalized_query in company['normalized_name'] or company['normalized_name'] in normalized_query:
                # 類似度計算（簡易版）
                score = self.calculate_similarity(normalized_query, company['normalized_name'])
                if score >= threshold:
                    results.append({
                        'company': company,
                        'score': score,
                        'match_type': 'partial'
                    })
        
        # 3. キーワード一致検索
        query_keywords = self.extract_search_keywords(query)
        for company in self.companies:
            for keyword in query_keywords:
                if any(keyword in comp_keyword for comp_keyword in company['search_keywords']):
                    score = 0.7  # キーワード一致スコア
                    if not any(r['company']['id'] == company['id'] for r in results):
                        results.append({
                            'company': company,
                            'score': score,
                            'match_type': 'keyword'
                        })
        
        # 4. ベクトル検索（利用可能な場合）
        if self.embedding_model and len(results) < limit:
            vector_results = self.vector_search(query, limit - len(results))
            results.extend(vector_results)
        
        # スコアでソートして上位を返す
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """文字列類似度の簡易計算"""
        if not str1 or not str2:
            return 0.0
        
        # レーベンシュタイン距離ベースの類似度
        len1, len2 = len(str1), len(str2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # 簡易版: 共通部分の割合
        common_chars = 0
        for char in str1:
            if char in str2:
                common_chars += 1
        
        similarity = common_chars / max(len1, len2)
        return similarity
    
    def vector_search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """ベクトル検索による類似工務店名の検索"""
        if not self.embedding_model:
            return []
        
        try:
            # クエリのベクトル化
            query_vector = self.embedding_model.embed_query(query)
            
            # 全工務店名とのコサイン類似度計算
            results = []
            for company in self.companies:
                company_vector = self.embedding_model.embed_query(company['original_name'])
                
                # コサイン類似度計算
                similarity = self.cosine_similarity(query_vector, company_vector)
                
                if similarity > 0.5:  # 閾値
                    results.append({
                        'company': company,
                        'score': similarity,
                        'match_type': 'vector'
                    })
            
            # スコアでソート
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """コサイン類似度の計算"""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation error: {e}")
            return 0.0
    
    def get_company_suggestions(self, input_text: str) -> List[str]:
        """入力テキストに基づく工務店名候補の取得"""
        search_results = self.search_companies(input_text, limit=5)
        
        suggestions = []
        for result in search_results:
            company_name = result['company']['original_name']
            score = result['score']
            match_type = result['match_type']
            
            suggestion = f"{company_name} (類似度: {score:.2f}, {match_type})"
            suggestions.append(suggestion)
        
        return suggestions