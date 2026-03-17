# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 00: セットアップ & 基本チャットボット
# MAGIC
# MAGIC - **前章の結果**: -
# MAGIC - **この章のゴール**: 動くチャットボットを作る
# MAGIC - **次章への橋渡し**: ボットは動くが品質が未知 → 可視化が必要

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 設定の読み込み

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. カタログ・スキーマの作成

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"カタログ: {CATALOG}, スキーマ: {SCHEMA} を作成しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. MLflow 実験の設定

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow 実験を設定しました: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. OpenAI SDK 経由で Foundation Model API に接続

# COMMAND ----------

from openai import OpenAI

# Databricks ノートブック上では dbutils から自動取得
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

client = OpenAI(
    api_key=token,
    base_url=f"https://{host}/serving-endpoints",
)
print(f"OpenAI クライアントを作成しました（エンドポイント: {MODEL_ENDPOINT}）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. シンプルなチャットボット関数

# COMMAND ----------

def chat(user_message, system_prompt="あなたは親切なアシスタントです。"):
    """Foundation Model API を使ったシンプルなチャットボット（リトライ付き）"""
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
# MAGIC ## 5. 基本的な対話テスト

# COMMAND ----------

test_questions = [
    "MLflow とは何ですか？",
    "Databricks の主な機能を3つ教えてください。",
    "Delta Lake と従来のデータレイクの違いは？",
]

for q in test_questions:
    print(f"Q: {q}")
    print(f"A: {chat(q)}")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. エージェントの登録（バージョン管理）
# MAGIC
# MAGIC チャットボットを `ChatAgent` として MLflow に登録し、バージョン管理を開始します。
# MAGIC これにより MLflow Experiment UI の「エージェントのバージョン」でエージェントを追跡できます。

# COMMAND ----------

import tempfile, os

# エージェントコードを一時ファイルに書き出し（code-based logging）
import os as _os
from mlflow.models.resources import DatabricksServingEndpoint

# ログ時の検証で OpenAI クライアントが認証できるよう環境変数を設定
_os.environ["OPENAI_API_KEY"] = token
_os.environ["OPENAI_BASE_URL"] = f"https://{host}/serving-endpoints"

agent_code = '''
import os
import uuid
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from openai import OpenAI

class LLMOpsDemoAgent(ChatAgent):
    """Foundation Model API を使ったチャットエージェント（v1: デフォルトプロンプト）"""

    def __init__(self):
        self.client = OpenAI()
        self.model_endpoint = "databricks-meta-llama-3-3-70b-instruct"

    def predict(self, messages, context=None, custom_inputs=None):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if not any(m["role"] == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": "あなたは親切なアシスタントです。"})
        response = self.client.chat.completions.create(
            model=self.model_endpoint,
            messages=msgs,
        )
        return ChatAgentResponse(
            messages=[ChatAgentMessage(
                role="assistant",
                content=response.choices[0].message.content,
                id=str(uuid.uuid4()),
            )]
        )

mlflow.models.set_model(LLMOpsDemoAgent())
'''

agent_file = _os.path.join(tempfile.mkdtemp(), "agent.py")
with open(agent_file, "w") as f:
    f.write(agent_code)

# COMMAND ----------

# エージェントをログ & Unity Catalog に登録
with mlflow.start_run(run_name="agent_v1_default_prompt"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent_file,
        registered_model_name=AGENT_MODEL_NAME,
        resources=[DatabricksServingEndpoint(endpoint_name=MODEL_ENDPOINT)],
    )

from mlflow import MlflowClient

uc_client = MlflowClient(registry_uri="databricks-uc")
versions = uc_client.search_model_versions(f"name='{AGENT_MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)
uc_client.set_registered_model_alias(AGENT_MODEL_NAME, "production", latest_version)
print(f"エージェント登録完了: {AGENT_MODEL_NAME} (version {latest_version}, alias: production)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC チャットボットが動作することを確認し、エージェントとして登録しました:
# MAGIC - **チャットボット**: Foundation Model API 経由で動作確認
# MAGIC - **エージェント登録**: Unity Catalog にバージョン管理付きで登録
# MAGIC - **production エイリアス**: 現バージョンを production に設定
# MAGIC
# MAGIC しかし、以下の疑問が残ります:
# MAGIC - LLM の内部でどのような処理が行われているのか？
# MAGIC - レスポンスの品質はどの程度か？
# MAGIC
# MAGIC → 次章（01_tracing）で、トレーシングによる可観測性を実現します。
