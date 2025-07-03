# GitHubアップロード用デプロイメントガイド

## 📦 アップロード対象ファイル一覧

以下のファイルを選択してGitHubにアップロードしてください：

### 🔧 メインアプリケーション
```
RAGapp.py                    # メインStreamlitアプリケーション
config.py                    # システム設定ファイル
requirements.txt             # Python依存関係
```

### 🏗️ RAGエンジン
```
rag_query_utils.py          # RAG処理エンジン
utils.py                    # ユーティリティ関数
data_preparation.py         # データベース構築スクリプト
```

### 🏢 工務店フォーム機能（新機能）
```
company_master.py           # 工務店マスタ管理
auto_fill_utils.py          # 自動入力処理
form_auto_fill.py           # フォーム外自動入力UI
```

### 📊 データファイル
```
tourokuten_prediction_finetune.jsonl  # 工務店データセット
```

### 📖 ドキュメント
```
README.md                   # プロジェクト概要
DEPLOYMENT_GUIDE.md         # このファイル
```

### 🤖 ファインチューニングモデル（オプション）
```
tourokuten_finetune_model_full/  # LoRAアダプター
```

## 🚫 アップロード除外ファイル

以下のファイルは**アップロードしないでください**：

### 大容量ファイル
```
chroma_db/                  # ベクトルデータベース（再構築可能）
pytorch/                    # PyTorchソースコード
old/                        # 古いファイル
ai_root/                    # 大容量モデル
cache/                      # キャッシュファイル
```

### ログファイル
```
*.log                       # 各種ログファイル
logs/                       # ログディレクトリ
```

### 一時ファイル
```
checkpoints/                # 学習チェックポイント
.venv/                      # 仮想環境
__pycache__/               # Pythonキャッシュ
```

## 📝 .gitignoreファイルの作成

以下の内容で`.gitignore`ファイルを作成してください：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# Data and models
chroma_db/
cache/
checkpoints/
ai_root/models/
pytorch/
old/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Environment variables
.env.local
.env.development.local
.env.test.local
.env.production.local

# Large files
*.bin
*.safetensors
*.gguf

# Temporary files
*.tmp
*.temp
```

## 🔄 アップロード手順

1. **リポジトリの準備**
   ```bash
   cd /home/ncnadmin/my_rag_project
   git init
   git remote add origin https://github.com/tarochan0803/finetune-RAG.git
   ```

2. **必要ファイルの選択**
   ```bash
   # 主要ファイルを追加
   git add RAGapp.py
   git add config.py
   git add requirements.txt
   git add rag_query_utils.py
   git add utils.py
   git add data_preparation.py
   git add company_master.py
   git add auto_fill_utils.py
   git add form_auto_fill.py
   git add README.md
   git add DEPLOYMENT_GUIDE.md
   git add .gitignore
   ```

3. **データファイル追加（サイズ確認後）**
   ```bash
   # ファイルサイズを確認
   ls -lh tourokuten_prediction_finetune.jsonl
   
   # 100MB未満の場合のみ追加
   git add tourokuten_prediction_finetune.jsonl
   ```

4. **コミット・プッシュ**
   ```bash
   git commit -m "feat: Add advanced RAG system with construction company form

   - ✨ Add construction company information input form  
   - 🔍 Implement company name fuzzy matching system
   - 🤖 Add RAG-powered auto-fill functionality
   - 📊 Support for 134 construction companies
   - 🎨 Enhanced Streamlit UI with dark theme
   - ⚡ Improved search accuracy with company prioritization
   
   🚀 Generated with Claude Code
   
   Co-Authored-By: Claude <noreply@anthropic.com>"
   
   git push -u origin master
   ```

## 📋 必要な環境変数

GitHubでの使用時に必要な環境変数：

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## 🔧 セットアップ手順（ユーザー向け）

GitHubからクローン後の手順：

```bash
# 1. リポジトリクローン
git clone https://github.com/tarochan0803/finetune-RAG.git
cd finetune-RAG

# 2. 仮想環境作成
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 3. 依存関係インストール
pip install -r requirements.txt
pip install jaconv

# 4. 環境変数設定
export GEMINI_API_KEY="your_api_key"

# 5. データベース構築
python data_preparation.py

# 6. アプリケーション起動
streamlit run RAGapp.py
```

## 📊 ファイルサイズ概算

| ファイル | 概算サイズ | 重要度 |
|----------|------------|--------|
| RAGapp.py | ~50KB | 🔴 必須 |
| config.py | ~15KB | 🔴 必須 |
| rag_query_utils.py | ~25KB | 🔴 必須 |
| utils.py | ~15KB | 🔴 必須 |
| company_master.py | ~12KB | 🔴 必須 |
| auto_fill_utils.py | ~8KB | 🔴 必須 |
| form_auto_fill.py | ~4KB | 🔴 必須 |
| tourokuten_prediction_finetune.jsonl | ~50MB | 🟡 データ |
| tourokuten_finetune_model_full/ | ~500MB | 🟡 モデル |

**総容量（必須ファイルのみ）**: 約130KB
**総容量（データ含む）**: 約50MB
**総容量（モデル含む）**: 約550MB

## 🎯 推奨アップロード構成

### Tier 1: 最小構成（~130KB）
- 全ソースコード
- 設定ファイル
- ドキュメント

### Tier 2: データ付き（~50MB）
- Tier 1 + データセット

### Tier 3: フル構成（~550MB）
- Tier 2 + ファインチューニングモデル

GitHub無料プランの制限（100MB/ファイル、1GB/リポジトリ）を考慮し、**Tier 2構成**を推奨します。

## 🔗 関連リンク

- **GitHub Repository**: https://github.com/tarochan0803/finetune-RAG
- **Streamlit Documentation**: https://docs.streamlit.io/
- **LangChain Documentation**: https://docs.langchain.com/
- **ChromaDB Documentation**: https://docs.trychroma.com/

---

*Last Updated: 2025-07-02*