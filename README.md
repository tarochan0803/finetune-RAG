# 🚀 Advanced RAG System with Fine-tuning

エンタープライズグレードの高度なRAGシステム & ファインチューニング統合環境

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 特徴

### 🎯 高性能RAGシステム
- **🔥 最新技術統合**: ChromaDB + LoRA + 量子化最適化
- **⚡ 高速検索**: 高度なベクトル検索とハイブリッド検索
- **🎨 美しいUI**: Streamlit Webインターフェース
- **📊 詳細分析**: 検索精度・性能メトリクス
- **💡 自動入力**: RAGを活用したフォームの自動入力機能

### 🧠 先進的ファインチューニング
- **🔄 継続学習**: 知識の忘却を防ぐEWC技術
- **📚 カリキュラム学習**: 段階的な難易度調整
- **🤖 自動最適化**: GPU環境に応じた設定自動調整
- **📈 包括的評価**: Weights & Biases統合

### ⚙️ 柔軟な設定
- **🎛️ 多層構成**: 軽量版から企業版まで対応
- **🔧 プラグイン対応**: MS-Swift + Megatron-Core統合
- **📊 詳細ログ**: TensorBoard + Wandb + 独自メトリクス

---

## 🚀 クイックスタート

### 1️⃣ インストール
```bash
git clone https://github.com/tarochan0803/finetune-RAG.git
cd finetune-RAG
pip install -r requirements.txt
```

### 2️⃣ ファインチューニング（推奨）
```bash
# 高性能版
python3 full_spec_finetune.py

# または軽量版
python3 auto_optimized_finetune.py
```

### 3️⃣ 設定更新
`config.py`を開き、`lora_adapter_path`をファインチューニングの出力先に設定します。
```python
# config.py
lora_adapter_path = "./tourokuten_finetune_model_full"  # 例: ファインチューニング出力先
```

### 4️⃣ RAGシステム起動
```bash
# Web UI付き高性能版
streamlit run RAGapp.py
```

---

## 📁 プロジェクト構成

```
finetune-RAG/
├── 🎯 RAGシステム
│   ├── RAGapp.py                  # メインの高性能RAGアプリ（Streamlit UI）
│   ├── final_rag_app.py           # (旧)高性能RAG
│   ├── simple_rag_app.py          # (旧)軽量RAG
│   └── immediate_rag_system.py    # 即席RAG
├── 🧠 ファインチューニング
│   ├── full_spec_finetune.py      # フルスペック版（推奨）
│   ├── auto_optimized_finetune.py # 自動最適化版
│   ├── continual_learning_finetune.py # 継続学習版
│   ├── scale_up_finetune.py       # 大規模モデル版
│   └── cpu_optimized_finetune.py  # CPU最適化版
├── 🔧 統合システム
│   └── megatron_swift_finetune.py # MS-Swift + Megatron
├── ⚙️ 設定・ユーティリティ
│   ├── config.py                  # メイン設定
│   ├── utils.py                   # ユーティリティ
│   ├── rag_query_utils.py         # クエリ処理
│   ├── form_auto_fill.py          # フォーム自動入力UI
│   ├── auto_fill_utils.py         # フォーム自動入力ロジック
│   └── company_master.py          # 工務店マスタ機能
├── 📊 データ処理
│   ├── enhanced_training_data_generator.py
│   └── advanced_data_augmenter.py
└── 📚 ドキュメント
    ├── README.md                  # このファイル
    ├── USAGE_GUIDE.md             # 詳細使用ガイド
    └── docs/                      # 追加ドキュメント
```

---

## 🎯 使用方法（目的別）

### 🥇 初心者向け（最短経路）
```bash
# 1. 軽量ファインチューニング（30分）
python3 auto_optimized_finetune.py

# 2. 設定更新
sed -i 's|lora_adapter_path = ".*"|lora_adapter_path = "./optimized_rag_model"|' config.py

# 3. RAG起動
streamlit run RAGapp.py
```

### 🥈 標準ユーザー向け（高品質）
```bash
# 1. 高性能ファインチューニング（1-2時間）
python3 full_spec_finetune.py

# 2. 設定更新
sed -i 's|lora_adapter_path = ".*"|lora_adapter_path = "./tourokuten_finetune_model_full"|' config.py

# 3. 高性能RAG起動
streamlit run RAGapp.py
```

### 🥉 上級ユーザー向け（最高性能）
```bash
# 1. 初回ファインチューニング
python3 full_spec_finetune.py

# 2. 継続学習（複数タスク）
python3 continual_learning_finetune.py --task-name "医療対話" --data-path "medical.jsonl"
python3 continual_learning_finetune.py --task-name "技術Q&A" --data-path "tech.jsonl"

# 3. 性能評価
python3 continual_learning_finetune.py --evaluate

# 4. RAG起動
streamlit run RAGapp.py
```

---

## 🔧 設定オプション

### GPU環境別推奨設定

| GPU | VRAM | 推奨ファインチューニング | 推奨RAG |
|-----|------|----------------------|---------|
| RTX 4090 | 24GB | `full_spec_finetune.py` | `RAGapp.py` |
| RTX 4080 | 16GB | `auto_optimized_finetune.py` | `RAGapp.py` |
| RTX 4070 | 12GB | `auto_optimized_finetune.py` | `RAGapp.py` |
| RTX 4060 | 8GB | `cpu_optimized_finetune.py` | `RAGapp.py` |

### モデル選択
```python
# config.py で変更可能
base_model_name = "Qwen/Qwen1.5-1.8B"  # 現在のデフォルトモデル
# base_model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"  # 7B標準
# base_model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"  # 13B高性能
```

---

## 🚀 高度な機能

### 🔄 継続学習
既存のファインチューニング済みモデルに新しい知識を追加：
```bash
python3 continual_learning_finetune.py \
  --task-name "新しいドメイン" \
  --data-path "new_domain.jsonl" \
  --description "専門分野の知識追加"
```

### 📊 性能評価
```bash
# 全タスクの性能評価
python3 continual_learning_finetune.py --evaluate

# 詳細レポート生成
python3 continual_learning_finetune.py --report
```

### 🔧 MS-Swift + Megatron統合
大規模分散学習：
```bash
python3 megatron_swift_finetune.py \
  --model "Qwen/Qwen3-8B-Base" \
  --dataset "kunishou/ApolloCorpus-ja"
```

---

## 📊 性能ベンチマーク

### ファインチューニング比較

| モード | 品質 | 速度 | メモリ効率 | 複雑さ |
|--------|------|------|-----------|--------|
| `auto_optimized` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| `full_spec` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| `continual_learning` | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |

### RAG性能比較

| モード | 精度 | 速度 | リソース | UI品質 |
|--------|------|------|----------|--------|
| `RAGapp.py` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `simple_rag` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 🛠️ カスタマイズ

### データ形式
```jsonl
{"instruction": "質問する", "input": "ユーザー入力", "output": "期待する回答"}
{"instruction": "", "input": "簡単な質問", "output": "回答"}
```

### 設定例
```python
# config.py カスタマイズ例
class Config:
    # モデル設定
    base_model_name = "your-model-path"
    lora_adapter_path = "./your-tuned-model"
    
    # RAG設定
    chunk_size = 500
    chunk_overlap = 50
    rag_variant_k = [5, 7, 10] # 検索するドキュメント数
    
    # UI設定
    app_title = "カスタムRAGシステム"
```

---

## 🔍 トラブルシューティング

### よくある問題

#### 1. GPU VRAM不足
```bash
# より軽量な設定を使用
python3 cpu_optimized_finetune.py
```

#### 2. モデル読み込みエラー
```python
# config.py のパスを確認
lora_adapter_path = "./actual_output_directory"
```

#### 3. 性能が低い場合
```bash
# より良いデータでファインチューニング
python3 full_spec_finetune.py

# 継続学習で改善
python3 continual_learning_finetune.py --task-name "改善" --data-path "quality_data.jsonl"
```

---

## 📚 詳細ドキュメント

- [📖 USAGE_GUIDE.md](USAGE_GUIDE.md) - 詳細な使用方法
- [🔧 設定リファレンス](docs/config_reference.md)
- [🧪 API仕様](docs/api_reference.md)
- [🚀 デプロイガイド](docs/deployment.md)

---

## 🤝 コントリビューション

プルリクエスト・Issue報告歓迎！

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

---

## 🙏 謝辞

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ChromaDB](https://www.trychroma.com/)
- [MS-Swift](https://github.com/modelscope/ms-swift)
- [PEFT](https://github.com/huggingface/peft)

---

## 📞 サポート

- 🐛 バグ報告: [Issues](https://github.com/tarochan0803/finetune-RAG/issues)
- 💬 質問・議論: [Discussions](https://github.com/tarochan0803/finetune-RAG/discussions)
- 📧 Email: taro0803tennis@gmail.com

---

<div align="center">

**⭐ このプロジェクトが役立ったらスターをお願いします！**

Made with ❤️ by [tarochan0803]

</div>
