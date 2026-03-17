# MLflow LLMOps デモ

## 概要

Databricks + MLflow で LLM アプリの品質管理サイクル（トレーシング → 評価 → プロンプト最適化 → ジャッジアラインメント → 本番監視）を一気通貫で回すデモ。

## メタデータ

| 項目     | 値                            |
| -------- | ----------------------------- |
| オーナー | @mizoo-snow21                 |

## 前提条件

- Databricks ワークスペース
- MLflow 3.1+
- モデルエンドポイント: `databricks-meta-llama-3-3-70b-instruct`

## セットアップ手順

1. Databricks CLI で認証:
   ```bash
   databricks auth login --profile <your-profile>
   ```

2. ノートブックをワークスペースにアップロード:
   ```bash
   for f in config.py 0*.py; do
     name="${f%.py}"
     databricks workspace import \
       "/Users/<your-email>/llmops_demo/$name" \
       --file "$f" --format SOURCE --language PYTHON --overwrite \
       --profile <your-profile>
   done
   ```

3. サーバーレスジョブとして実行する場合は以下の依存が必要:
   ```
   mlflow>=3.1, openai, litellm, databricks-agents, dspy
   ```

## 使い方

ノートブックを 00 → 06 の順に実行:

```
00 → 01 → 02 → 03 → 04 → 05 → 06
```

| # | ノートブック | 内容 | ストーリー |
|---|------------|------|-----------|
| 00 | `00_setup_and_chatbot.py` | セットアップ & チャットボット | 動くボットを作る。品質は未知 |
| 01 | `01_tracing.py` | MLflow Tracing | 実行の中身を可視化。定量評価がない |
| 02 | `02_evaluation.py` | GenAI Evaluation | 品質を数値化。改善すべき箇所が判明 |
| 03 | `03_prompt_management.py` | Prompt Registry & GEPA | プロンプトを自動最適化。ジャッジ自体の精度は？ |
| 04 | `04_judge_labeling.py` | ベースジャッジ & ラベリング | SME がトレースにフィードバックを付与 |
| 05 | `05_judge_alignment.py` | MemAlign アラインメント | SME 基準に沿った正確なジャッジが完成 |
| 06 | `06_production_monitoring.py` | Production Monitoring | 本番で品質を継続監視 |

## 設定

`config.py` で全ノートブック共通の設定値を管理:

| 変数 | 値 |
|------|---|
| `CATALOG` | `main` |
| `SCHEMA` | `llmops_demo` |
| `MODEL_ENDPOINT` | `databricks-meta-llama-3-3-70b-instruct` |
| `EXPERIMENT_NAME` | `/Shared/llmops-demo` |
| `JUDGE_NAME` | `helpfulness` |

## ローカルテスト

```bash
# ローカル環境セットアップ
uv sync

# 個別テスト（04 のジャッジ・ラベリング）
uv run python test/test_04_local.py

# E2E テスト（04 → ラベリング → 05 align の全フロー）
uv run python test/test_e2e.py
```
