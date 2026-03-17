# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06: Production Monitoring
# MAGIC
# MAGIC - **前章の結果**: ドメイン専門家にアラインしたジャッジができた
# MAGIC - **この章のゴール**: 本番環境でのトレース収集と継続的品質監視を構築する
# MAGIC - **次章への橋渡し**: -（最終章）

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 設定の読み込み

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import time
import mlflow
import mlflow.genai
from mlflow.entities import UCSchemaLocation
from mlflow.tracing.enablement import set_experiment_trace_location
from mlflow.tracing import set_databricks_monitoring_sql_warehouse_id
from mlflow.genai.scorers import (
    Safety,
    Guidelines,
    ScorerSamplingConfig,
    list_scorers,
    get_scorer,
    delete_scorer,
)
from openai import OpenAI

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.openai.autolog()

# 環境に応じた認証情報取得（ノートブック / サーバーレスジョブ 両対応）
try:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
except Exception:
    token = os.environ.get("DATABRICKS_TOKEN")
if not token:
    raise RuntimeError("API token を取得できません。ノートブックで実行するか DATABRICKS_TOKEN を設定してください。")

try:
    host = spark.conf.get("spark.databricks.workspaceUrl")
except Exception:
    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
if not host:
    raise RuntimeError("Workspace URL を取得できません。")

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

def predict_fn(**inputs):
    query = inputs["query"]
    return chat(query)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. UC Trace Ingestion セットアップ
# MAGIC
# MAGIC トレースを Unity Catalog スキーマに永続化します。
# MAGIC これにより、SQL でトレースを分析したりダッシュボードを作成できます。
# MAGIC
# MAGIC **前提**: SQL Warehouse ID を設定してください。

# COMMAND ----------

# SQL Warehouse ID の取得（ウィジェット → 環境変数 → デフォルト）
try:
    SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
except Exception:
    SQL_WAREHOUSE_ID = os.environ.get("SQL_WAREHOUSE_ID", "")
if not SQL_WAREHOUSE_ID:
    # e2-demo-tokyo の DBAcademy Warehouse
    SQL_WAREHOUSE_ID = "bec52b183a4cfe2a"
    print(f"デフォルトの SQL Warehouse を使用: {SQL_WAREHOUSE_ID}")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

# COMMAND ----------

# UC スキーマにトレースを永続化
# ※ この機能はワークスペースで "OpenTelemetry Collector for Delta Tables" が有効な場合のみ利用可能
UC_TRACE_ENABLED = False
try:
    set_experiment_trace_location(
        location=UCSchemaLocation(catalog_name=CATALOG, schema_name=SCHEMA),
        experiment_id=EXPERIMENT_ID,
    )
    mlflow.tracing.set_destination(
        destination=UCSchemaLocation(catalog_name=CATALOG, schema_name=SCHEMA)
    )
    UC_TRACE_ENABLED = True
    print(f"トレース格納先: {CATALOG}.{SCHEMA}")
    print("以下のテーブルが自動作成されます:")
    print(f"  - {CATALOG}.{SCHEMA}.mlflow_experiment_trace_otel_logs")
    print(f"  - {CATALOG}.{SCHEMA}.mlflow_experiment_trace_otel_metrics")
    print(f"  - {CATALOG}.{SCHEMA}.mlflow_experiment_trace_otel_spans")
except Exception as e:
    print(f"⚠️ UC Trace Ingestion は利用できません: {e}")
    print("  → トレースは MLflow Experiment に記録されます（通常のトレーシング機能は利用可能）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. スコアラーの登録と起動
# MAGIC
# MAGIC Production Monitoring では、スコアラーを `register()` → `start()` の2ステップで有効化します。
# MAGIC `register()` だけでは監視が始まりません。

# COMMAND ----------

# SQL Warehouse をモニタリング用に設定
try:
    set_databricks_monitoring_sql_warehouse_id(
        sql_warehouse_id=SQL_WAREHOUSE_ID,
        experiment_id=EXPERIMENT_ID,
    )
    print(f"モニタリング用 SQL Warehouse を設定しました: {SQL_WAREHOUSE_ID}")
except Exception as e:
    print(f"⚠️ モニタリング SQL Warehouse の設定に失敗: {e}")

# COMMAND ----------

# Safety スコアラー（全トレースを評価）
try:
    safety = Safety().register(name="safety_monitor")
except Exception:
    safety = get_scorer(name="safety_monitor")
safety = safety.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0)
)
print("Safety モニタリング開始（sample_rate=1.0）")

# COMMAND ----------

# Guidelines スコアラー（50%サンプリング）
try:
    tone = Guidelines(
        name="professional_tone",
        guidelines="The response must be professional and helpful",
    ).register(name="tone_monitor")
except Exception:
    tone = get_scorer(name="tone_monitor")
tone = tone.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5)
)
print("Tone モニタリング開始（sample_rate=0.5）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. テストトレースの生成
# MAGIC
# MAGIC モニタリングが動作していることを確認するため、テストトレースを生成します。

# COMMAND ----------

test_queries = [
    "Databricks の料金体系を教えてください。",
    "Delta Lake でタイムトラベルを使う方法は？",
    "MLflow のモデルレジストリとは？",
]

for q in test_queries:
    response = chat(q)
    print(f"Q: {q}")
    print(f"A: {response[:100]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ユーザーフィードバックの収集
# MAGIC
# MAGIC `mlflow.log_feedback()` で、ユーザーからのフィードバックをトレースに紐付けます。

# COMMAND ----------

# 最新のトレースIDを取得
recent_traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"],
    max_results=1,
    return_type="list",
)

if recent_traces:
    trace_id = recent_traces[0].info.trace_id
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_rating",
        value=4.0,
        rationale="回答が正確で分かりやすかった",
    )
    print(f"フィードバックを記録しました（trace_id: {trace_id[:12]}...）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 本番トレース → オフライン再評価ループ
# MAGIC
# MAGIC 本番で収集したトレースを UC Dataset に追加し、定期的にオフライン再評価を行います。
# MAGIC これにより品質劣化を早期に検知できます。

# COMMAND ----------

from mlflow.genai.scorers import Correctness

# 過去24時間のトレースを取得
cutoff = int((time.time() - 86400) * 1000)
prod_traces = mlflow.search_traces(
    filter_string=f"attributes.status = 'OK' AND attributes.timestamp_ms > {cutoff}",
)
print(f"過去24時間のトレース: {len(prod_traces)} 件")

# COMMAND ----------

# UC Dataset に追加
import mlflow.genai.datasets

try:
    try:
        prod_dataset = mlflow.genai.datasets.create_dataset(
            uc_table_name=f"{CATALOG}.{SCHEMA}.prod_eval_dataset"
        )
    except Exception:
        prod_dataset = mlflow.genai.datasets.get_dataset(f"{CATALOG}.{SCHEMA}.prod_eval_dataset")
    if len(prod_traces) > 0:
        prod_dataset.merge_records(prod_traces)
        print(f"{len(prod_traces)} 件のトレースを UC Dataset に追加しました。")
except Exception as e:
    print(f"⚠️ UC Dataset への追加をスキップ: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. スコアラー管理
# MAGIC
# MAGIC 登録済みスコアラーの一覧・更新・停止・削除が可能です。

# COMMAND ----------

# 登録済みスコアラーの一覧
scorers = list_scorers()
print("=== 登録済みスコアラー ===")
for s in scorers:
    rate = s.sampling_config.sample_rate if hasattr(s, "sampling_config") and s.sampling_config else "N/A"
    print(f"  {s.name}: sample_rate={rate}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### サンプルレートの更新例

# COMMAND ----------

# # サンプルレートの更新（必要に応じてコメントアウトを解除）
# safety_scorer = get_scorer(name="safety_monitor")
# safety_scorer = safety_scorer.update(
#     sampling_config=ScorerSamplingConfig(sample_rate=0.8)
# )
# print("Safety スコアラーのサンプルレートを 0.8 に更新しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ### スコアラーの停止・削除例

# COMMAND ----------

# # スコアラーの停止（必要に応じてコメントアウトを解除）
# safety_scorer = get_scorer(name="safety_monitor")
# safety_scorer = safety_scorer.stop()
# print("Safety スコアラーを停止しました。")
#
# # スコアラーの削除
# delete_scorer(name="safety_monitor")
# print("Safety スコアラーを削除しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. UC トレーステーブルへの SQL クエリ（ダッシュボード用）
# MAGIC
# MAGIC UC に格納されたトレースデータに対して SQL で分析できます。

# COMMAND ----------

# UC Trace テーブルが利用可能な場合のみ SQL 分析を実行
if UC_TRACE_ENABLED:
    display(spark.sql(f"""
        SELECT
          DATE(timestamp) AS trace_date,
          COUNT(DISTINCT trace_id) AS trace_count,
          ROUND(
            SUM(CASE WHEN status_code = 'ERROR' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
          ) AS error_pct
        FROM {CATALOG}.{SCHEMA}.mlflow_experiment_trace_otel_spans
        WHERE parent_span_id IS NULL
        GROUP BY DATE(timestamp)
        ORDER BY trace_date DESC
    """))
else:
    print("⚠️ UC Trace Ingestion が未有効のため、SQL 分析をスキップします。")
    print("  有効化後は以下の SQL でトレースを分析できます:")
    print(f"  SELECT * FROM {CATALOG}.{SCHEMA}.mlflow_experiment_trace_otel_spans LIMIT 10")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC 本番モニタリングの仕組みを構築しました:
# MAGIC - **UC Trace Ingestion**: トレースを Unity Catalog に永続化
# MAGIC - **Production Monitoring**: Safety / Guidelines スコアラーで継続的に品質評価
# MAGIC - **ユーザーフィードバック**: `mlflow.log_feedback()` でフィードバックをトレースに紐付け
# MAGIC - **再評価ループ**: 本番トレース → UC Dataset → オフライン再評価で品質劣化を検知
# MAGIC - **SQL 分析**: UC テーブルに対する SQL クエリでダッシュボード構築
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## End-to-End まとめ
# MAGIC
# MAGIC | # | テーマ | 成果 |
# MAGIC |---|--------|------|
# MAGIC | 00 | セットアップ | 動くチャットボット |
# MAGIC | 01 | トレーシング | 実行の可視化 |
# MAGIC | 02 | 評価 | 品質の定量化 |
# MAGIC | 03 | プロンプト最適化 | 自動的な品質改善 |
# MAGIC | 04 | Judge Alignment | ドメイン基準の評価 |
# MAGIC | 05 | モニタリング | 継続的な品質監視 |
# MAGIC
# MAGIC このデモシリーズにより、LLMOps のライフサイクル全体を体験できました。
