# advanced_data_augmenter.py - 高度なデータ拡張システム
# より多様で実用的な質問パターンを生成

import json
import random
import re
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter

class AdvancedDataAugmenter:
    def __init__(self, input_jsonl_path: str):
        self.input_path = input_jsonl_path
        self.original_data = self.load_data()
        self.company_specs = self.analyze_companies()
        self.spec_categories = self.analyze_spec_categories()
        
    def load_data(self) -> List[Dict]:
        """JSONLデータを読み込み"""
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def analyze_companies(self) -> Dict[str, List[Dict]]:
        """工務店別にデータを分析"""
        companies = defaultdict(list)
        for item in self.original_data:
            input_text = item.get('input', '')
            if '株式会社' in input_text:
                company = input_text.split('代表条件表')[0].strip()
                companies[company].append(item)
        return dict(companies)
    
    def analyze_spec_categories(self) -> Dict[str, Set[str]]:
        """仕様カテゴリを分析"""
        categories = defaultdict(set)
        for item in self.original_data:
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            # カテゴリ抽出
            if '壁面材仕様' in input_text:
                categories['壁仕様'].add(output_text)
            elif '耐力壁仕様' in input_text:
                categories['耐力壁'].add(output_text)
            elif '仮筋交' in input_text:
                categories['仮筋交'].add(output_text)
            elif '鋼製束' in input_text:
                categories['鋼製束'].add(output_text)
                
        return {k: list(v) for k, v in categories.items()}
    
    def generate_comparison_questions(self) -> List[Dict]:
        """比較質問を生成"""
        augmented = []
        
        # 工務店間比較
        companies = list(self.company_specs.keys())
        for i in range(min(len(companies), 10)):  # 上位10社
            for j in range(i + 1, min(len(companies), 10)):
                company_a = companies[i]
                company_b = companies[j]
                
                # 壁仕様比較
                augmented.append({
                    "instruction": f"{company_a}と{company_b}の壁仕様の違いを教えてください。",
                    "input": f"比較対象: {company_a} vs {company_b} | 項目: 壁面材仕様",
                    "output": f"{company_a}と{company_b}の壁仕様を比較するには、それぞれの代表条件表を確認する必要があります。"
                })
                
        return augmented
    
    def generate_conditional_questions(self) -> List[Dict]:
        """条件付き質問を生成"""
        augmented = []
        
        conditions = [
            "狭小地の場合",
            "特殊な敷地条件の場合", 
            "コストを抑えたい場合",
            "高品質仕様にしたい場合",
            "施工期間を短縮したい場合"
        ]
        
        for company, specs in list(self.company_specs.items())[:5]:  # 上位5社
            for condition in conditions:
                augmented.append({
                    "instruction": f"{condition}、{company}ではどのような仕様になりますか？",
                    "input": f"{company}の条件付き仕様 | 条件: {condition}",
                    "output": f"{condition}の{company}の仕様については、個別の条件に応じて標準仕様から調整される可能性があります。詳細は条件表を確認してください。"
                })
        
        return augmented
    
    def generate_negative_questions(self) -> List[Dict]:
        """否定形・確認質問を生成"""
        augmented = []
        
        # 仕様確認の否定形
        negative_patterns = [
            ("大壁仕様ではありませんか？", "真壁", "いいえ、真壁仕様です。"),
            ("真壁仕様ではありませんか？", "大壁", "いいえ、大壁仕様です。"),
            ("仮筋交は不要ですか？", "仮筋交は有", "いいえ、仮筋交は必要です。"),
            ("鋼製束は使用しませんか？", "鋼製束", "使用状況については条件表で確認してください。")
        ]
        
        for company, specs in list(self.company_specs.items())[:3]:
            for question, check_word, answer in negative_patterns:
                augmented.append({
                    "instruction": f"{company}は{question}",
                    "input": f"{company}の仕様確認",
                    "output": answer
                })
        
        return augmented
    
    def generate_technical_questions(self) -> List[Dict]:
        """技術的詳細質問を生成"""
        augmented = []
        
        technical_aspects = [
            ("構造計算", "構造計算の方法や基準について"),
            ("基礎設計", "基礎の立上高さや配筋について"), 
            ("金物仕様", "使用する金物の種類や規格について"),
            ("材料等級", "使用する木材の等級や品質について"),
            ("施工方法", "特殊な施工方法や注意点について")
        ]
        
        for company in list(self.company_specs.keys())[:3]:
            for aspect, description in technical_aspects:
                augmented.append({
                    "instruction": f"{company}の{aspect}について詳しく教えてください。",
                    "input": f"{company}代表条件表 | 技術項目: {aspect}",
                    "output": f"{company}の{description}は代表条件表の詳細項目で確認できます。"
                })
        
        return augmented
    
    def generate_process_questions(self) -> List[Dict]:
        """プロセス・手順質問を生成"""
        augmented = []
        
        process_questions = [
            "見積もり時の注意点は何ですか？",
            "設計時に確認すべき項目は何ですか？", 
            "生産管理での重要ポイントは何ですか？",
            "積算時の標準単価はどこで確認できますか？",
            "特殊加工が必要な場合の対応方法は？"
        ]
        
        for company in list(self.company_specs.keys())[:3]:
            for question in process_questions:
                augmented.append({
                    "instruction": f"{company}で{question}",
                    "input": f"{company}の業務プロセス確認",
                    "output": f"{company}の該当項目については代表条件表の関連セクションで確認してください。"
                })
        
        return augmented
    
    def generate_error_detection_questions(self) -> List[Dict]:
        """エラー検出・検証質問を生成"""
        augmented = []
        
        error_scenarios = [
            "仕様が条件表と異なる場合の対応",
            "設計ミスを発見した場合の手順",
            "積算エラーの確認方法",
            "施工不具合の予防策",
            "品質管理のチェックポイント"
        ]
        
        for company in list(self.company_specs.keys())[:2]:
            for scenario in error_scenarios:
                augmented.append({
                    "instruction": f"{company}で{scenario}について教えてください。",
                    "input": f"{company}のエラー対応・品質管理",
                    "output": f"{scenario}については、{company}の標準手順に従って対応してください。詳細は条件表や社内規定を確認してください。"
                })
        
        return augmented
    
    def generate_contextual_questions(self) -> List[Dict]:
        """文脈理解質問を生成"""
        augmented = []
        
        # 実際のデータから文脈を抽出
        for item in self.original_data[:50]:  # 最初の50件を使用
            original_input = item.get('input', '')
            original_output = item.get('output', '')
            
            if '株式会社' in original_input and len(original_output) > 5:
                # パラフレーズ版
                augmented.append({
                    "instruction": "この工務店の仕様について説明してください。",
                    "input": original_input,
                    "output": f"この工務店では、{original_output}となっています。"
                })
                
                # 簡潔版
                augmented.append({
                    "instruction": "要点を教えてください。",
                    "input": original_input,
                    "output": original_output
                })
        
        return augmented
    
    def save_augmented_data(self, output_path: str):
        """拡張データを保存"""
        print("🔄 データ拡張実行中...")
        
        # 各種拡張パターンを生成
        comparison_data = self.generate_comparison_questions()
        conditional_data = self.generate_conditional_questions()
        negative_data = self.generate_negative_questions()
        technical_data = self.generate_technical_questions()
        process_data = self.generate_process_questions()
        error_data = self.generate_error_detection_questions()
        contextual_data = self.generate_contextual_questions()
        
        # 統計情報
        augmentation_stats = {
            "元データ": len(self.original_data),
            "比較質問": len(comparison_data),
            "条件付き質問": len(conditional_data),
            "否定形質問": len(negative_data),
            "技術詳細質問": len(technical_data),
            "プロセス質問": len(process_data),
            "エラー検出質問": len(error_data),
            "文脈理解質問": len(contextual_data)
        }
        
        # 全データを結合
        all_data = (
            self.original_data + 
            comparison_data + 
            conditional_data + 
            negative_data + 
            technical_data + 
            process_data + 
            error_data + 
            contextual_data
        )
        
        # シャッフル
        random.shuffle(all_data)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 統計表示
        print("📊 データ拡張完了！")
        for category, count in augmentation_stats.items():
            print(f"   {category}: {count:,}件")
        print(f"   合計: {len(all_data):,}件")
        print(f"   保存先: {output_path}")
        
        return len(all_data)

def main():
    augmenter = AdvancedDataAugmenter("/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl")
    total_samples = augmenter.save_augmented_data("/home/ncnadmin/my_rag_project/premium_training_dataset.jsonl")
    
    print(f"\n🎉 プレミアム学習データセット生成完了！")
    print(f"📈 データ品質向上: より多様な{total_samples:,}件の質問パターン")

if __name__ == "__main__":
    main()