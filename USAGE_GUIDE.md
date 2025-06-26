# 🚀 ファインチューニング＆RAGシステム 使用ガイド

## 📊 コード選択フローチャート

```
開始
  ↓
初回ファインチューニング？
  ↓ Yes              ↓ No
[A] 初回FT           継続学習？
                      ↓ Yes        ↓ No
                   [B] 継続学習    RAGのみ？
                                    ↓ Yes
                                 [C] RAG実行
```

---

## 🎯 A. 初回ファインチューニング

### A1. 高性能版（推奨）
```bash
python3 full_spec_finetune.py
```
**特徴:**
- ✅ 最高性能・多機能
- ✅ 自動GPU最適化
- ✅ Weights & Biases統合
- ✅ 包括的なログ・評価
- ⚠️ 設定が複雑

### A2. 軽量版
```bash
python3 auto_optimized_finetune.py
```
**特徴:**
- ✅ シンプル・高速
- ✅ 自動最適化
- ✅ 即座に実行可能
- ⚠️ 機能は最小限

### A3. 大規模版（13B+モデル）
```bash
python3 scale_up_finetune.py
```
**特徴:**
- ✅ 13Bモデル対応
- ✅ RTX 5070最適化
- ✅ メモリ効率重視

---

## 🔄 B. 継続学習（2回目以降）

### B1. 基本的な継続学習
```bash
python3 continual_learning_finetune.py \
  --task-name "医療対話" \
  --data-path "medical_data.jsonl" \
  --description "医療専門知識学習"
```

### B2. 評価・レポート
```bash
# 全タスク性能評価
python3 continual_learning_finetune.py --evaluate

# 学習履歴レポート
python3 continual_learning_finetune.py --report
```

---

## 🤖 C. RAGシステム実行

### C1. 高性能RAG（推奨）
```bash
python3 final_rag_app.py
```
**特徴:**
- ✅ 最新技術統合
- ✅ 高精度検索
- ✅ ストリーミング対応
- ✅ Web UI付き

### C2. 軽量RAG
```bash
python3 simple_rag_app.py
```
**特徴:**
- ✅ 軽量・高速
- ✅ 最小限の依存関係
- ⚠️ 基本機能のみ

---

## 🔧 設定ファイル

### config.py の重要設定
```python
class Config:
    # ファインチューニング済みモデルパス
    lora_adapter_path = "./full_spec_rag_model"  # A1の出力先
    # または
    lora_adapter_path = "./optimized_rag_model"  # A2の出力先
    # または  
    lora_adapter_path = "./continual_learning_checkpoints/task_X_final"  # Bの出力先
```

---

## 📋 推奨ワークフロー

### 🥇 初心者向け（シンプル）
```bash
# 1. 軽量ファインチューニング
python3 auto_optimized_finetune.py

# 2. config.py編集
# lora_adapter_path = "./optimized_rag_model"

# 3. RAG実行
python3 simple_rag_app.py
```

### 🥈 標準ユーザー向け
```bash
# 1. 高性能ファインチューニング
python3 full_spec_finetune.py

# 2. config.py編集  
# lora_adapter_path = "./full_spec_rag_model"

# 3. 高性能RAG実行
python3 final_rag_app.py
```

### 🥉 上級ユーザー向け
```bash
# 1. 初回ファインチューニング
python3 full_spec_finetune.py

# 2. 継続学習（複数タスク）
python3 continual_learning_finetune.py --task-name "タスク1" --data-path "data1.jsonl"
python3 continual_learning_finetune.py --task-name "タスク2" --data-path "data2.jsonl"

# 3. 性能評価
python3 continual_learning_finetune.py --evaluate

# 4. config.py編集
# lora_adapter_path = "./continual_learning_checkpoints/task_2_final"

# 5. RAG実行
python3 final_rag_app.py
```

---

## ⚡ クイックスタート

### 最短3ステップ
```bash
# 1. ファインチューニング（約30分〜2時間）
python3 auto_optimized_finetune.py

# 2. 設定更新
sed -i 's|lora_adapter_path = ".*"|lora_adapter_path = "./optimized_rag_model"|' config.py

# 3. RAG起動
python3 simple_rag_app.py
```

---

## 🔍 トラブルシューティング

### Q1. ファインチューニングが失敗する
**A1:** GPU VRAM不足の可能性
```bash
# 軽量版を試す
python3 cpu_optimized_finetune.py
```

### Q2. RAGでモデルが読み込めない
**A2:** config.pyのパスを確認
```python
# ファインチューニング出力先と一致させる
lora_adapter_path = "./actual_output_directory"
```

### Q3. 性能が低い
**A3:** より良いデータ・設定を使用
```bash
# 高品質データでファインチューニング
python3 full_spec_finetune.py

# 継続学習で性能向上
python3 continual_learning_finetune.py --task-name "改善タスク" --data-path "quality_data.jsonl"
```

---

## 📊 性能比較

| コード | 品質 | 速度 | メモリ | 難易度 |
|--------|------|------|--------|--------|
| auto_optimized_finetune.py | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| full_spec_finetune.py | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| continual_learning_finetune.py | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| simple_rag_app.py | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| final_rag_app.py | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 🎯 目的別推奨

### 🚀 速度重視
```bash
python3 auto_optimized_finetune.py
python3 simple_rag_app.py
```

### 🎖️ 品質重視  
```bash
python3 full_spec_finetune.py
python3 final_rag_app.py
```

### 🔄 継続学習重視
```bash
python3 continual_learning_finetune.py
python3 final_rag_app.py
```

### 💻 リソース制約
```bash
python3 cpu_optimized_finetune.py
python3 simple_rag_app.py
```

---

## 📞 サポート

問題が発生した場合：
1. ログファイルを確認
2. GPU/CPU使用量をチェック  
3. 設定ファイルを再確認
4. より軽量なバージョンを試行