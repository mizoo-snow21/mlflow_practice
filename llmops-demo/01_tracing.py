# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01: MLflow Tracing
# MAGIC
# MAGIC - **前章の結果**: 動くチャットボットができた
# MAGIC - **この章のゴール**: 実行の中身をトレースで可視化する
# MAGIC - **次章への橋渡し**: 課題は見えた → 定量的な品質評価が必要

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 設定の読み込み

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import mlflow
from openai import OpenAI

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)

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
# MAGIC ## 1. Auto-instrumentation
# MAGIC
# MAGIC `mlflow.openai.autolog()` を有効化するだけで、OpenAI SDK の呼び出しが自動的にトレースされます。

# COMMAND ----------

mlflow.openai.autolog()

# COMMAND ----------

# 自動的にトレースされる
response = chat("MLflow とは何ですか？")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow の Experiments UI → Traces タブで、トレースが記録されていることを確認してください。
# MAGIC 入力・出力・レイテンシ・トークン使用量が自動で取得されています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 手動トレース（`@mlflow.trace` デコレータ）
# MAGIC
# MAGIC 自動トレースに加え、自分で定義した関数にもトレースを追加できます。

# COMMAND ----------

@mlflow.trace(span_type="CHAIN")
def preprocess(user_message):
    """入力テキストの前処理"""
    return user_message.strip()

@mlflow.trace(span_type="CHAIN")
def postprocess(response):
    """出力テキストの後処理"""
    return response.strip()

@mlflow.trace(span_type="CHAIN")
def enhanced_chat(user_message):
    """前処理 → LLM 呼び出し → 後処理のパイプライン"""
    processed = preprocess(user_message)
    response = chat(processed)
    return postprocess(response)

# COMMAND ----------

result = enhanced_chat("Delta Lake の主な利点を3つ教えてください。")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Traces タブで `enhanced_chat` のトレースを確認すると、以下のスパン構造が見えます:
# MAGIC
# MAGIC ```
# MAGIC enhanced_chat (CHAIN)
# MAGIC ├── preprocess (CHAIN)
# MAGIC ├── chat_completions (CHAT_MODEL) ← 自動トレース
# MAGIC └── postprocess (CHAIN)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. カスタムスパン（`span_type` の指定）
# MAGIC
# MAGIC `mlflow.start_span()` を使うと、より細かい制御が可能です。

# COMMAND ----------

@mlflow.trace(span_type="CHAIN")
def chat_with_context(user_message):
    """システムプロンプトを付与してから LLM に渡す例"""
    system_prompt = "あなたはDatabricksの専門家です。技術的に正確な回答をしてください。"
    result = chat(user_message, system_prompt=system_prompt)
    return result

# COMMAND ----------

result = chat_with_context("Unity Catalog とは何ですか？")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. トレース検索（`mlflow.search_traces()`）
# MAGIC
# MAGIC 記録されたトレースをプログラムでフィルタリング・分析できます。

# COMMAND ----------

# 成功したトレースを検索
traces_df = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"],
    max_results=5,
)
print(f"取得件数: {len(traces_df)}")
print(f"列名: {list(traces_df.columns)}")
print(traces_df.to_string())

# COMMAND ----------

# 実行時間が長いトレースを検索
slow_traces = mlflow.search_traces(
    filter_string="attributes.execution_time_ms > 1000",
    max_results=10,
)
print(f"1秒以上かかったトレース: {len(slow_traces)} 件")
if len(slow_traces) > 0:
    print(slow_traces.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. トークン使用量の確認
# MAGIC
# MAGIC トレースからトークン使用量を確認し、コスト管理に活用できます。

# COMMAND ----------

# 最新のトレースを取得してトークン使用量を確認
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"],
    max_results=5,
    return_type="list",
)

for t in traces:
    usage = t.info.token_usage
    if usage:
        print(f"Trace: {t.info.trace_id[:12]}...")
        if isinstance(usage, dict):
            print(f"  Input tokens:  {usage.get('input_tokens', 'N/A')}")
            print(f"  Output tokens: {usage.get('output_tokens', 'N/A')}")
            print(f"  Total tokens:  {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"  Input tokens:  {usage.input_tokens}")
            print(f"  Output tokens: {usage.output_tokens}")
            print(f"  Total tokens:  {usage.total_tokens}")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC トレーシングにより、チャットボットの内部動作を可視化できました:
# MAGIC - **自動トレース**: OpenAI SDK 呼び出しが自動記録される
# MAGIC - **手動トレース**: カスタム関数にもトレースを追加できる
# MAGIC - **検索・分析**: プログラムでトレースをフィルタリングできる
# MAGIC - **トークン使用量**: コスト管理に活用できる
# MAGIC
# MAGIC しかし、「回答の品質は良いのか？」という問いには答えられていません。
# MAGIC
# MAGIC → 次章（02_evaluation）で、品質スコアを定量化します。
