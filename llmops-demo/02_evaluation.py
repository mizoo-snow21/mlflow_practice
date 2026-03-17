# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02: GenAI Evaluation
# MAGIC
# MAGIC - **前章の結果**: トレースで実行の中身が見えるようになった
# MAGIC - **この章のゴール**: 品質スコアを定量的に評価する
# MAGIC - **次章への橋渡し**: スコアが低い箇所が判明 → プロンプト改善が必要

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 設定の読み込み

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import mlflow
import mlflow.genai
from mlflow.genai.scorers import (
    Guidelines,
    ExpectationsGuidelines,
    Correctness,
    RelevanceToQuery,
    Safety,
    scorer,
)
from mlflow.entities import Feedback
from openai import OpenAI

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.openai.autolog()

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

client = OpenAI(
    api_key=token,
    base_url=f"https://{host}/serving-endpoints",
)

def chat(user_message, system_prompt="あなたは親切なアシスタントです。"):
    def _call():
        response = client.chat.completions.create(
            model=MODEL_ENDPOINT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content
    return _call_with_retry(_call)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 評価データセットの作成
# MAGIC
# MAGIC `inputs` / `expectations` のネスト構造で評価データを定義します。

# COMMAND ----------

eval_data = [
    {
        "inputs": {"query": "Databricks とは何ですか？"},
        "expectations": {
            "expected_facts": ["データとAIのプラットフォーム", "Lakehouse"]
        },
    },
    {
        "inputs": {"query": "Delta Lake の主な特徴は？"},
        "expectations": {
            "expected_facts": ["ACID トランザクション", "スキーマ管理", "タイムトラベル"]
        },
    },
    {
        "inputs": {"query": "Unity Catalog の役割を説明してください。"},
        "expectations": {
            "expected_facts": ["データガバナンス", "アクセス制御", "メタデータ管理"]
        },
    },
    {
        "inputs": {"query": "MLflow の主要コンポーネントは何ですか？"},
        "expectations": {
            "expected_facts": ["トラッキング", "モデルレジストリ", "評価"]
        },
    },
    {
        "inputs": {"query": "Apache Spark とは何ですか？"},
        "expectations": {
            "expected_facts": ["分散処理フレームワーク", "大規模データ", "並列処理"]
        },
    },
    {
        "inputs": {"query": "Databricks ワークフローの利点は？"},
        "expectations": {
            "expected_facts": ["ジョブスケジューリング", "オーケストレーション", "モニタリング"]
        },
    },
    {
        "inputs": {"query": "データレイクハウスとは何ですか？"},
        "expectations": {
            "expected_facts": ["データレイク", "データウェアハウス", "統合アーキテクチャ"]
        },
    },
    {
        "inputs": {"query": "Photon エンジンの利点は？"},
        "expectations": {
            "expected_facts": ["高速クエリ", "ネイティブ実行エンジン", "パフォーマンス"]
        },
    },
    {
        "inputs": {"query": "Databricks SQL とは？"},
        "expectations": {
            "expected_facts": ["SQL ウェアハウス", "BI ツール連携", "サーバーレス"]
        },
    },
    {
        "inputs": {"query": "Feature Store の役割は？"},
        "expectations": {
            "expected_facts": ["特徴量管理", "再利用", "一貫性"]
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. predict_fn の定義
# MAGIC
# MAGIC `predict_fn` は `**kwargs` で `inputs` のキーを受け取ります（dict ではない点に注意）。

# COMMAND ----------

def predict_fn(**inputs):
    query = inputs["query"]
    return chat(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Built-in Scorers で評価
# MAGIC
# MAGIC Safety, RelevanceToQuery, Correctness, Guidelines を組み合わせて評価します。

# COMMAND ----------

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[
        Safety(),
        RelevanceToQuery(),
        Correctness(),
        Guidelines(
            name="conciseness",
            guidelines="The response must be concise and under 200 words",
        ),
    ],
)

# COMMAND ----------

print("=== 評価メトリクス ===")
for k, v in results.metrics.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# 詳細結果の確認
print(results.result_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Custom Scorer（ドメイン固有メトリクス）
# MAGIC
# MAGIC `@scorer` デコレータでカスタム評価関数を作成します。

# COMMAND ----------

@scorer
def response_language(inputs, outputs):
    """回答が日本語で書かれているか確認"""
    import re
    has_japanese = bool(
        re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", str(outputs))
    )
    return Feedback(
        name="japanese_response",
        value=has_japanese,
        rationale="日本語が含まれている" if has_japanese else "日本語が含まれていない",
    )

# COMMAND ----------

custom_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[response_language],
)

print("=== カスタムスコアラー結果 ===")
for k, v in custom_results.metrics.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. make_judge()：カスタム LLM ジャッジ
# MAGIC
# MAGIC LLM を使ったカスタム評価ジャッジを作成します。

# COMMAND ----------

from mlflow.genai.judges import make_judge

technical_judge = make_judge(
    name="technical_accuracy",
    instructions=(
        "ユーザーの質問: {{ inputs }}\n"
        "アシスタントの回答: {{ outputs }}\n\n"
        "回答の技術的正確性を評価してください。\n"
        "技術用語の使い方が正しく、情報が最新かを確認し、\n"
        "'yes'（正確）または 'no'（不正確）で回答してください。"
    ),
    model=f"databricks:/{MODEL_ENDPOINT}",
)

judge_results = mlflow.genai.evaluate(
    data=eval_data[:5],
    predict_fn=predict_fn,
    scorers=[technical_judge],
)

print(judge_results.result_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ExpectationsGuidelines（行単位ガイドライン）
# MAGIC
# MAGIC 各行に個別のガイドラインを設定できます。

# COMMAND ----------

per_row_data = [
    {
        "inputs": {"query": "Python のリスト内包表記を説明して"},
        "expectations": {
            "guidelines": ["コード例を含むこと", "初心者にもわかる説明であること"]
        },
    },
    {
        "inputs": {"query": "Spark の shuffle とは何ですか？"},
        "expectations": {
            "guidelines": ["パフォーマンスへの影響を説明すること", "具体例を含むこと"]
        },
    },
    {
        "inputs": {"query": "REST API の設計原則は？"},
        "expectations": {
            "guidelines": ["HTTPメソッドに言及すること", "リソース指向設計を説明すること"]
        },
    },
]

per_row_results = mlflow.genai.evaluate(
    data=per_row_data,
    predict_fn=predict_fn,
    scorers=[ExpectationsGuidelines()],
)

print(per_row_results.result_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MLflow-managed Dataset（UC 永続化）
# MAGIC
# MAGIC 評価データセットを Unity Catalog テーブルに永続化し、再利用可能にします。

# COMMAND ----------

import mlflow.genai.datasets

try:
    eval_dataset = mlflow.genai.datasets.get_dataset(f"{CATALOG}.{SCHEMA}.{EVAL_DATASET_NAME}")
except Exception:
    eval_dataset = mlflow.genai.datasets.create_dataset(
        uc_table_name=f"{CATALOG}.{SCHEMA}.{EVAL_DATASET_NAME}"
    )
eval_dataset.merge_records(eval_data)
print(f"評価データセットを作成しました: {CATALOG}.{SCHEMA}.{EVAL_DATASET_NAME}")

# COMMAND ----------

# 永続化したデータセットを使って再評価
existing = mlflow.genai.datasets.get_dataset(f"{CATALOG}.{SCHEMA}.{EVAL_DATASET_NAME}")

reeval_results = mlflow.genai.evaluate(
    data=existing,
    predict_fn=predict_fn,
    scorers=[Safety(), Correctness()],
)

print("=== UC Dataset からの再評価結果 ===")
for k, v in reeval_results.metrics.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC 品質評価の仕組みを構築しました:
# MAGIC - **Built-in Scorers**: Safety, Correctness, RelevanceToQuery, Guidelines で多角的に評価
# MAGIC - **Custom Scorer**: ドメイン固有のメトリクス（日本語チェック等）を追加可能
# MAGIC - **make_judge()**: LLM を使ったカスタム評価ジャッジ
# MAGIC - **ExpectationsGuidelines**: 行ごとの個別ガイドライン評価
# MAGIC - **UC Dataset**: 評価データの永続化・再利用
# MAGIC
# MAGIC 評価スコアが低い部分が見つかりました。改善するにはプロンプトの最適化が有効です。
# MAGIC
# MAGIC → 次章（03_prompt_management）で、プロンプトの管理と自動最適化を行います。
