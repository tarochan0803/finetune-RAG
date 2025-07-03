#!/usr/bin/env python3
"""
Convert tourokurensiyou.ini data to JSONL format for fine-tuning.
Creates prompt-response pairs where various company details serve as prompts
and the company name serves as the response.
"""

import json
import csv
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text or text.strip() == "":
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common placeholder values
    if text.lower() in ['null', 'none', 'n/a', '-', '']:
        return ""
    
    return text

def extract_prompts_from_row(row: Dict[str, str], company_name: str) -> List[Dict[str, str]]:
    """Extract meaningful prompt-response pairs from a single row."""
    prompts = []
    
    # Skip if no company name
    if not company_name:
        return prompts
    
    # Contact information prompts
    if row.get('KoumuTantouName'):
        prompts.append({
            "prompt": f"工務担当者は{clean_text(row['KoumuTantouName'])}です",
            "response": company_name
        })
    
    if row.get('KoumuTantouTel'):
        prompts.append({
            "prompt": f"工務担当者の電話番号は{clean_text(row['KoumuTantouTel'])}です",
            "response": company_name
        })
    
    if row.get('KoumuTantouMail'):
        prompts.append({
            "prompt": f"工務担当者のメールアドレスは{clean_text(row['KoumuTantouMail'])}です",
            "response": company_name
        })
    
    if row.get('SekkeiTantouName'):
        prompts.append({
            "prompt": f"設計担当者は{clean_text(row['SekkeiTantouName'])}です",
            "response": company_name
        })
    
    if row.get('SekkeiTantouTel'):
        prompts.append({
            "prompt": f"設計担当者の電話番号は{clean_text(row['SekkeiTantouTel'])}です",
            "response": company_name
        })
    
    if row.get('SekkeiTantouMail'):
        prompts.append({
            "prompt": f"設計担当者のメールアドレスは{clean_text(row['SekkeiTantouMail'])}です",
            "response": company_name
        })
    
    # Registration code
    if row.get('TourokuTenCd'):
        prompts.append({
            "prompt": f"登録店コードは{clean_text(row['TourokuTenCd'])}です",
            "response": company_name
        })
    
    # Technical specifications (sekkei fields)
    sekkei_fields = [
        ('sekkei01_Textarea1', '設計仕様1'),
        ('sekkei02_Textarea1', '設計仕様2'),
        ('sekkei03_Textarea1', '設計仕様3'),
        ('sekkei04_Textarea1', '設計仕様4'),
        ('sekkei05_Textarea1', '設計仕様5'),
        ('sekkei06_Textarea1', '設計仕様6'),
        ('sekkei07_Textarea1', '設計仕様7'),
        ('sekkei08_Textarea1', '設計仕様8'),
        ('sekkei09_Textarea1', '設計仕様9'),
        ('sekkei10_Textarea1', '設計仕様10'),
        ('sekkei11_Textarea1', '設計仕様11'),
        ('sekkei12_Textarea1', '設計仕様12'),
        ('sekkei13_Textarea1', '設計仕様13'),
        ('sekkei14_Textarea1', '設計仕様14'),
        ('sekkei16_Textarea1', '設計仕様16'),
        ('sekkei17_Textarea1', '設計仕様17'),
        ('sekkei18_Textarea1', '設計仕様18'),
        ('sekkei19_Textarea1', '設計仕様19'),
        ('sekkei20_Textarea1', '設計仕様20'),
        ('sekkei21_Textarea1', '設計仕様21'),
        ('sekkei22_Textarea1', '設計仕様22'),
        ('sekkei23_Textarea1', '設計仕様23'),
        ('sekkei24_Textarea1', '設計仕様24'),
    ]
    
    for field, description in sekkei_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Floor specifications (yuka fields)
    yuka_fields = [
        ('yuka01_Textarea1', '床仕様1'),
        ('yuka02_Textarea1', '床仕様2'),
        ('yuka03_Textarea1', '床仕様3'),
        ('yuka04_Textarea1', '床仕様4'),
    ]
    
    for field, description in yuka_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Wall specifications (kabe fields)
    kabe_fields = [
        ('kabe01_Textarea1', '壁仕様1'),
        ('kabe02_Textarea1', '壁仕様2'),
        ('kabe03_Textarea1', '壁仕様3'),
        ('kabe04_Textarea1', '壁仕様4'),
    ]
    
    for field, description in kabe_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Fittings specifications (fuku fields)
    fuku_fields = [
        ('fuku01_Textarea1', '部材仕様1'),
        ('fuku02_Textarea1', '部材仕様2'),
        ('fuku03_Textarea1', '部材仕様3'),
        ('fuku04_Textarea1', '部材仕様4'),
        ('fuku05_Textarea1', '部材仕様5'),
        ('fuku06_Textarea1', '部材仕様6'),
        ('fuku07_Textarea1', '部材仕様7'),
        ('fuku08_Textarea1', '部材仕様8'),
    ]
    
    for field, description in fuku_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Hardware specifications (kana fields)
    kana_fields = [
        ('kana01_Textarea1', '金物仕様1'),
        ('kana02_Textarea1', '金物仕様2'),
        ('kana03_Textarea1', '金物仕様3'),
    ]
    
    for field, description in kana_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Stone specifications (seki fields)
    seki_fields = [
        ('seki01_Textarea1', '石仕様1'),
        ('seki02_Textarea1', '石仕様2'),
        ('seki03_Textarea1', '石仕様3'),
        ('seki04_Textarea1', '石仕様4'),
        ('seki05_Textarea1', '石仕様5'),
        ('seki06_Textarea1', '石仕様6'),
        ('seki07_Textarea1', '石仕様7'),
        ('seki08_Textarea1', '石仕様8'),
        ('seki09_Textarea1', '石仕様9'),
    ]
    
    for field, description in seki_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Rafter specifications (hane fields)
    hane_fields = [
        ('hane01_Textarea1', '羽柄仕様1'),
        ('hane02_Textarea1', '羽柄仕様2'),
        ('hane03_Textarea1', '羽柄仕様3'),
        ('hane04_Textarea1', '羽柄仕様4'),
        ('hane05_Textarea1', '羽柄仕様5'),
        ('hane06_Textarea1', '羽柄仕様6'),
        ('hane07_Textarea1', '羽柄仕様7'),
        ('hane08_Textarea1', '羽柄仕様8'),
        ('hane09_Textarea1', '羽柄仕様9'),
        ('hane10_Textarea1', '羽柄仕様10'),
        ('hane11_Textarea1', '羽柄仕様11'),
        ('hane12_Textarea1', '羽柄仕様12'),
        ('hane13_Textarea1', '羽柄仕様13'),
        ('hane14_Textarea1', '羽柄仕様14'),
    ]
    
    for field, description in hane_fields:
        if row.get(field) and clean_text(row[field]):
            prompts.append({
                "prompt": f"{description}: {clean_text(row[field])}",
                "response": company_name
            })
    
    # Special file reference
    if row.get('HagaraPcSheetFileName') and clean_text(row['HagaraPcSheetFileName']):
        prompts.append({
            "prompt": f"専用シート: {clean_text(row['HagaraPcSheetFileName'])}",
            "response": company_name
        })
    
    return prompts

def main():
    input_file = '/home/ncnadmin/my_rag_project/tourokurensiyou.ini'
    output_file = '/home/ncnadmin/my_rag_project/tourokurensiyou_prompt_response.jsonl'
    
    all_prompts = []
    
    print("Processing tourokurensiyou.ini file...")
    
    # Read the TSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row_num, row in enumerate(reader, 1):
            company_name = clean_text(row.get('登録店名', ''))
            
            if not company_name:
                print(f"Row {row_num}: Skipping - no company name")
                continue
            
            # Extract prompts from this row
            prompts = extract_prompts_from_row(row, company_name)
            all_prompts.extend(prompts)
            
            if row_num % 100 == 0:
                print(f"Processed {row_num} rows, generated {len(all_prompts)} prompts so far")
    
    print(f"\nTotal prompts generated: {len(all_prompts)}")
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt_data in all_prompts:
            f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
    
    print(f"Output written to: {output_file}")
    
    # Show some sample prompts
    print("\nSample prompts:")
    for i, prompt_data in enumerate(all_prompts[:10]):
        print(f"{i+1}. {prompt_data}")

if __name__ == "__main__":
    main()