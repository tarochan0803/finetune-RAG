# enhanced_training_data_generator.py
# 高品質な学習データ生成スクリプト

import json
# import pandas as pd  # 不要なので削除
from typing import List, Dict, Any

class TrainingDataEnhancer:
    def __init__(self, existing_jsonl_path: str):
        self.existing_data = self.load_existing_data(existing_jsonl_path)
        
    def load_existing_data(self, path: str) -> List[Dict]:
        """既存のJSONLデータを読み込み"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def generate_enhanced_samples(self) -> List[Dict]:
        """既存データから高品質サンプルを生成"""
        enhanced_samples = []
        
        # 1. 質問バリエーション生成
        for sample in self.existing_data:
            input_text = sample['input']
            output_text = sample['output']
            
            # 会社名抽出
            if '代表条件表' in input_text:
                company = input_text.split('代表条件表')[0].strip()
                
                # バリエーション1: 直接質問形式
                enhanced_samples.append({
                    "instruction": f"{company}の標準仕様について教えてください。",
                    "input": input_text.split('|')[-1].strip(),  # 最後の項目のみ
                    "output": output_text
                })
                
                # バリエーション2: 条件付き質問
                enhanced_samples.append({
                    "instruction": f"{company}で施工する場合の仕様を確認してください。",
                    "input": input_text,
                    "output": f"{company}の標準仕様では、{output_text}"
                })
                
                # バリエーション3: 比較形式
                enhanced_samples.append({
                    "instruction": "工務店の標準仕様を確認してください。",
                    "input": f"工務店: {company}\n確認項目: {input_text.split('|')[-1].strip()}",
                    "output": output_text
                })
        
        # 2. 否定形質問の追加
        for sample in self.existing_data:
            if '壁面材仕様' in sample['input']:
                if '大壁' in sample['output']:
                    enhanced_samples.append({
                        "instruction": "この工務店は真壁仕様ですか？",
                        "input": sample['input'],
                        "output": "いいえ、大壁仕様です。"
                    })
                elif '真壁' in sample['output']:
                    enhanced_samples.append({
                        "instruction": "この工務店は大壁仕様ですか？",
                        "input": sample['input'],
                        "output": "いいえ、真壁仕様です。"
                    })
        
        # 3. 複合質問の追加
        company_specs = {}
        for sample in self.existing_data:
            if '代表条件表' in sample['input']:
                company = sample['input'].split('代表条件表')[0].strip()
                if company not in company_specs:
                    company_specs[company] = []
                company_specs[company].append(sample)
        
        for company, specs in company_specs.items():
            if len(specs) >= 2:
                enhanced_samples.append({
                    "instruction": f"{company}の主要な標準仕様をまとめて教えてください。",
                    "input": f"{company}の標準仕様一覧",
                    "output": f"{company}の標準仕様: " + "、".join([spec['output'] for spec in specs[:3]])
                })
        
        return enhanced_samples
    
    def save_enhanced_data(self, output_path: str):
        """拡張データを保存"""
        enhanced_data = self.generate_enhanced_samples()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 既存データ
            for sample in self.existing_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # 拡張データ
            for sample in enhanced_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"元データ: {len(self.existing_data)}件")
        print(f"拡張データ: {len(enhanced_data)}件") 
        print(f"合計: {len(self.existing_data) + len(enhanced_data)}件")
        print(f"保存先: {output_path}")

if __name__ == "__main__":
    enhancer = TrainingDataEnhancer("/home/ncnadmin/my_rag_project/fine_tuning_dynamic_instruction_dataset.jsonl")
    enhancer.save_enhanced_data("/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl")