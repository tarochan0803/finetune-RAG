#!/usr/bin/env python3
import json

def convert_enhanced_to_prompt_response(input_file, output_file):
    """Convert enhanced_training_dataset.jsonl to prompt-response format"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            # Extract company name from input field
            input_text = data.get('input', '')
            output_text = data.get('output', '')
            
            # Extract company name (first part before 代表条件表条件)
            if '代表条件表条件' in input_text:
                company_name = input_text.split('代表条件表条件')[0].strip()
                
                # Create prompt from the output content
                prompt_response = {
                    "prompt": output_text,
                    "response": company_name
                }
                
                outfile.write(json.dumps(prompt_response, ensure_ascii=False) + '\n')

def merge_jsonl_files(file1, file2, file3, output_file):
    """Merge three JSONL files into one"""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Copy original tourokuten_prediction_finetune.jsonl
        with open(file1, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)
        
        # Copy tourokurensiyou_prompt_response.jsonl
        with open(file2, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)
        
        # Convert and copy enhanced_training_dataset.jsonl
        with open(file3, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line.strip())
                
                # Extract company name from input field
                input_text = data.get('input', '')
                output_text = data.get('output', '')
                
                # Extract company name (first part before 代表条件表条件)
                if '代表条件表条件' in input_text:
                    company_name = input_text.split('代表条件表条件')[0].strip()
                    
                    # Create prompt from the output content
                    prompt_response = {
                        "prompt": output_text,
                        "response": company_name
                    }
                    
                    outfile.write(json.dumps(prompt_response, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    print("統合を開始します...")
    
    # ファイルパス
    original_file = "/home/ncnadmin/my_rag_project/tourokuten_prediction_finetune.jsonl"
    tourokurensiyou_file = "/home/ncnadmin/my_rag_project/tourokurensiyou_prompt_response.jsonl"
    enhanced_file = "/home/ncnadmin/my_rag_project/enhanced_training_dataset.jsonl"
    
    # バックアップを作成
    import shutil
    backup_file = "/home/ncnadmin/my_rag_project/tourokuten_prediction_finetune_backup.jsonl"
    shutil.copy(original_file, backup_file)
    print(f"バックアップを作成しました: {backup_file}")
    
    # ファイルを統合
    merge_jsonl_files(original_file, tourokurensiyou_file, enhanced_file, original_file)
    
    print("統合が完了しました!")
    
    # 統計情報を表示
    with open(original_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"統合後の総行数: {total_lines}")