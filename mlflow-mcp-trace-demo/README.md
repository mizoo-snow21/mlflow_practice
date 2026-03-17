# MLflow MCP Server × Claude Code でトレースを調査するデモ

## 概要

MLflow MCP Server を利用し、Claude Code から自然言語で Databricks 管理下の MLflow トレースを調査するデモ。Databricks 上で logfire-mcp に近いトレース調査体験を再現する。

## メタデータ

| 項目     | 値                          |
| -------- | --------------------------- |
| オーナー | @mizoo-snow21               |

## 検証内容

- `mlflow[mcp]` パッケージに同梱された MCP サーバーを Claude Code から利用し、Databricks マネージド MLflow のトレースを検索・分析・評価できるか
- logfire-mcp が OpenTelemetry に対して提供するトレース調査体験を、MLflow MCP で Databricks 上に再現できるか
- エラートレースの発見、遅延分析、フィードバック記録、自動評価のワークフローが自然言語で完結するか

## 仕組み

```
Claude Code  ──MCP──>  mlflow mcp server  ──REST API──>  Databricks Managed MLflow
```

- `mlflow[mcp]` を利用すると、MLflow の MCP サーバーを起動できる
- `.mcp.json` により、Claude Code 起動時に自動接続される
- Databricks ワークスペースの認証には Databricks CLI のプロファイルを利用する

## セットアップ

### 前提条件

- [uv](https://docs.astral.sh/uv/) がインストール済み
- Databricks CLI で認証済み

```bash
databricks auth login --profile <your-profile>
```

### .mcp.json（Claude Code 用 MCP 設定）

プロジェクトルートに配置済み。別のワークスペースを使う場合は `DATABRICKS_CONFIG_PROFILE` を変更する。

```json
{
  "mcpServers": {
    "mlflow-mcp": {
      "command": "uv",
      "args": ["run", "mlflow", "mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "databricks",
        "DATABRICKS_CONFIG_PROFILE": "<your-profile>"
      }
    }
  }
}
```

### サンプルトレースの生成

```bash
uv run python generate_traces.py
```

20 件のサンプルトレースが Databricks ワークスペース上の実験に記録される。クエリ内容に応じて、正常系・エラー系・高遅延系のトレースが含まれる。

### Claude Code を起動

```bash
claude
```

このディレクトリで起動すると `.mcp.json` により MLflow MCP Server が自動起動し、Databricks マネージド MLflow に接続する。

## チュートリアル: 問題のあるトレースを調査する

### Step 1: エラートレースを発見する

```
> エラーになっているトレースを探して
```

Claude Code は内部的に `search_traces` ツールを使い、`status = 'ERROR'` のトレースを一覧表示する。
各トレースのエラー種別（`ConnectionError`, `PermissionError`, `RuntimeError`）とメッセージが確認できる。

### Step 2: 特定のエラーを深掘りする

```
> billing 関連のエラーだけ絞り込んで、原因を分析して
```

Claude Code は `search_traces` で対象を絞り込み、`get_trace` で各トレースのスパンを確認して、根本原因を特定する。

### Step 3: 遅延トレースを調査する

```
> 実行時間が1秒以上のトレースを探して、ボトルネックを教えて
```

Claude Code はトレースのスパンツリーを分析し、どのスパンが遅延の原因になっているかを特定する。例:

- `retrieve_knowledge` スパンが 2.5 秒 → pricing に関するクエリの検索が遅い
- `generate_answer` スパンが 3.0 秒 → 複雑な質問で LLM の推論に時間がかかっている

### Step 4: 問題のあるトレースにフィードバックを残す

```
> エラートレースに "needs-investigation" タグを付けて
```

Claude Code は `set_trace_tag` を使って該当トレースにタグを付与する。
チームメンバーが Databricks UI 上でフィルタできるようになる。

品質スコアも記録できる。

```
> billing エラーのトレースに feedback を残して。名前は "retrieval_quality"、スコアは 0、理由は "Knowledge base connection failure" で
```

### Step 5: 全体の健全性をチェックする

```
> 直近のトレース全体のエラー率と平均レイテンシを教えて
```

Claude Code は `search_traces` で全トレースを取得し、集計結果をサマリーとして返す。

### Step 6: 自動評価を実行する（応用）

```
> 使えるスコアラーを教えて
> 最新の正常トレースを5件評価して
```

Claude Code は `list_scorers` でスコアラー一覧を取得し、`evaluate_traces` で品質評価を実行する。

## MCP で使えるプロンプト例

| やりたいこと | プロンプト例 |
|---|---|
| トレース一覧を見る | 「最新のトレースを10件見せて」 |
| エラートレースを探す | 「エラーになっているトレースを探して」 |
| 特定のトレースを調べる | 「このトレースIDの詳細を見せて: abc123」 |
| 条件で絞り込む | 「1秒以上かかっているトレースを探して」 |
| エラー原因の分析 | 「billing関連のエラーの原因を分析して」 |
| 遅延の調査 | 「一番遅いトレースはどれ？ボトルネックを特定して」 |
| フィードバックを記録 | 「問題のあるトレースにフィードバックを残して」 |
| タグを付ける | 「このトレースに "needs-fix" タグを付けて」 |
| 品質評価を実行 | 「最新のトレースを評価して」 |

## MCP ツール一覧

| ツール名 | 説明 |
|---|---|
| `search_traces` | トレースの検索・フィルタリング |
| `get_trace` | 特定トレースの詳細取得 |
| `evaluate_traces` | トレースの品質評価 |
| `list_scorers` | 利用可能なスコアラー一覧 |
| `log_feedback` | トレースへのフィードバック記録 |
| `log_expectation` | 期待値の記録 |
| `set_trace_tag` / `delete_trace_tag` | トレースタグの管理 |
| `delete_traces` | トレースの削除 |
| `register_llm_judge` | LLM ジャッジの登録 |
| `get_assessment` / `update_assessment` / `delete_assessment` | 評価の管理 |

## logfire-mcp との対比

| 機能 | logfire-mcp (OpenTelemetry) | MLflow MCP (Databricks MLflow) |
|---|---|---|
| トレース検索 | OpenTelemetry トレースをクエリ | MLflow トレースを検索・フィルタ |
| メトリクス | OpenTelemetry メトリクス | トレースの実行時間・ステータス |
| 認証 | ブラウザ認証 / API キー | Databricks CLI プロファイル |
| ホスティング | Pydantic Logfire (SaaS) | Databricks ワークスペース (マネージド) |
| フィードバック | - | `log_feedback` でスコア・理由を記録 |
| 評価 | - | `evaluate_traces` で自動品質評価 |
| タグ管理 | - | `set_trace_tag` / `delete_trace_tag` |

## 結果・わかったこと

- MLflow MCP Server は logfire-mcp のトレース調査機能に加えて、フィードバック・評価・タグ管理といった MLOps 向けの機能を備えている
- Databricks マネージド MLflow との接続は `MLFLOW_TRACKING_URI=databricks` + CLI プロファイルで設定可能
- Claude Code から自然言語でトレースの検索・分析・フィードバックまで一気通貫で行える
