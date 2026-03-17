# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03: Prompt Registry & Optimization (GEPA)
# MAGIC
# MAGIC - **前章の結果**: 品質スコアを定量化し、改善すべき箇所が判明した
# MAGIC - **この章のゴール**: プロンプトをバージョン管理し、自動最適化で改善する
# MAGIC - **次章への橋渡し**: プロンプトは改善できた → 評価者（ジャッジ）自体の精度は？

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 設定の読み込み

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import mlflow
import mlflow.genai
from mlflow.genai.scorers import Correctness
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
# MAGIC ## 1. Prompt Registry：プロンプトの登録
# MAGIC
# MAGIC `mlflow.genai.register_prompt()` でプロンプトをバージョン管理します。

# COMMAND ----------

prompt = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=(
        "あなたは親切なAIアシスタントです。以下のガイドラインに従ってください:\n"
        "1. 正確な情報を提供する\n"
        "2. 簡潔に回答する\n"
        "3. わからないことは正直に伝える\n\n"
        "ユーザーの質問: {{ query }}"
    ),
)
print(f"プロンプト登録完了: {prompt.name} (version {prompt.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. バージョン管理とエイリアス
# MAGIC
# MAGIC `production` / `staging` エイリアスでバージョンを管理します。

# COMMAND ----------

# production エイリアスを設定
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=1)
print(f"エイリアス 'production' を version 1 に設定しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. プロンプトの読み込みと使用
# MAGIC
# MAGIC `load_prompt()` でレジストリからプロンプトを取得し、テンプレート変数を埋めます。

# COMMAND ----------

PROMPT_URI = f"prompts:/{PROMPT_NAME}@production"

p = mlflow.genai.load_prompt(PROMPT_URI)
formatted = p.format(query="Databricks とは何ですか？")
print("=== フォーマット済みプロンプト ===")
print(formatted)

# COMMAND ----------

# プロンプトを使ってチャット
response = chat("Databricks とは何ですか？", system_prompt=formatted)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. GEPA 最適化：自動プロンプト改善
# MAGIC
# MAGIC `optimize_prompts()` で、評価スコアを最大化するようにプロンプトを自動改善します。
# MAGIC
# MAGIC **注意**: 最適化データには `inputs` と `expectations` の両方が必須です。

# COMMAND ----------

optimization_data = [
    {
        "inputs": {"query": "Spark とは？"},
        "expectations": {
            "expected_response": "Apache Spark は分散処理フレームワークで、大規模データの並列処理を実現する。"
        },
    },
    {
        "inputs": {"query": "Unity Catalog の役割は？"},
        "expectations": {
            "expected_response": "Unity Catalog はデータと AI のガバナンスを統一的に管理するカタログサービスです。"
        },
    },
    {
        "inputs": {"query": "Delta Lake と Parquet の違いは？"},
        "expectations": {
            "expected_response": "Delta Lake は Parquet にACIDトランザクション、スキーマ管理、タイムトラベルなどの機能を追加したオープンソースのストレージレイヤーです。"
        },
    },
    {
        "inputs": {"query": "MLflow Tracking とは？"},
        "expectations": {
            "expected_response": "MLflow Tracking は実験のパラメータ、メトリクス、アーティファクトを記録・比較するためのコンポーネントです。"
        },
    },
    {
        "inputs": {"query": "Databricks Workflows の利点は？"},
        "expectations": {
            "expected_response": "Databricks Workflows はジョブのスケジューリング、オーケストレーション、モニタリングを統合的に管理できるサービスです。"
        },
    },
    {
        "inputs": {"query": "Photon エンジンとは？"},
        "expectations": {
            "expected_response": "Photon は Databricks のネイティブ実行エンジンで、SQL とDataFrameワークロードを高速化します。"
        },
    },
    {
        "inputs": {"query": "サーバーレス SQL ウェアハウスの特徴は？"},
        "expectations": {
            "expected_response": "サーバーレス SQL ウェアハウスはインフラ管理不要で、自動スケーリングにより必要な時だけリソースを使用できます。"
        },
    },
    {
        "inputs": {"query": "Databricks のクラスタとは？"},
        "expectations": {
            "expected_response": "Databricks クラスタはノートブックやジョブを実行するための計算リソースのセットで、自動スケーリングや自動終了をサポートします。"
        },
    },
    {
        "inputs": {"query": "メダリオンアーキテクチャとは？"},
        "expectations": {
            "expected_response": "メダリオンアーキテクチャはBronze（生データ）、Silver（クレンジング済み）、Gold（ビジネスレベル）の3層でデータを整理する設計パターンです。"
        },
    },
    {
        "inputs": {"query": "Databricks の Auto Loader とは？"},
        "expectations": {
            "expected_response": "Auto Loader はクラウドストレージに到着した新しいファイルを自動的にインクリメンタルに取り込む機能です。"
        },
    },
    {
        "inputs": {"query": "Databricks のノートブックの特徴は？"},
        "expectations": {
            "expected_response": "Databricks ノートブックはPython、SQL、R、Scalaをサポートし、コラボレーション機能やバージョン管理を備えた対話型開発環境です。"
        },
    },
    {
        "inputs": {"query": "Vector Search とは？"},
        "expectations": {
            "expected_response": "Databricks Vector Search はベクトル埋め込みを使って類似データを高速に検索するサービスで、RAGアプリケーションなどに活用されます。"
        },
    },
    {
        "inputs": {"query": "Databricks Apps とは？"},
        "expectations": {
            "expected_response": "Databricks Apps はデータやAIモデルを活用したWebアプリケーションをDatabricks上で構築・デプロイできるサービスです。"
        },
    },
    {
        "inputs": {"query": "Lakeflow Connect とは？"},
        "expectations": {
            "expected_response": "Lakeflow Connect は外部データソースからDatabricksへのデータ取り込みをノーコードで設定できるインジェストサービスです。"
        },
    },
    {
        "inputs": {"query": "Model Serving とは？"},
        "expectations": {
            "expected_response": "Databricks Model Serving はMLflowモデルやAIエージェントをREST APIエンドポイントとしてデプロイ・提供するサービスです。"
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### predict_fn と aggregation 関数の定義

# COMMAND ----------

def predict_fn_for_optimization(**inputs):
    query = inputs["query"]
    p = mlflow.genai.load_prompt(PROMPT_URI)
    system_content = p.format(query=query)
    return chat(query, system_prompt=system_content)


def objective_function(scores):
    """Correctness スコアを 0-1 に正規化する aggregation 関数"""
    feedback = scores.get("correctness")
    if feedback and hasattr(feedback, "value"):
        return 1.0 if feedback.value == "yes" else 0.0
    return 0.5

# COMMAND ----------

# MAGIC %md
# MAGIC ### GEPA 最適化の実行

# COMMAND ----------

from mlflow.genai.optimize import GepaPromptOptimizer

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn_for_optimization,
    train_data=optimization_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=f"databricks:/{MODEL_ENDPOINT}",
        max_metric_calls=50,
        display_progress_bar=True,
    ),
    scorers=[Correctness()],
    aggregation=objective_function,
)

# COMMAND ----------

print(f"初期スコア: {result.initial_eval_score}")
print(f"最終スコア: {result.final_eval_score}")
print(f"\n=== 最適化後プロンプト（先頭500文字）===")
print(result.optimized_prompts[0].template[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 最適化結果の条件付きプロモーション
# MAGIC
# MAGIC スコアが改善した場合のみ、最適化後のプロンプトを `production` に昇格します。

# COMMAND ----------

if result.final_eval_score > result.initial_eval_score:
    new_version = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=result.optimized_prompts[0].template,
    )
    mlflow.genai.set_prompt_alias(
        name=PROMPT_NAME,
        alias="production",
        version=new_version.version,
    )
    print(
        f"Version {new_version.version} を production に昇格しました "
        f"({result.initial_eval_score:.3f} → {result.final_eval_score:.3f})"
    )
else:
    print(
        f"スコアが改善しなかったため、production エイリアスは変更しません "
        f"({result.initial_eval_score:.3f} → {result.final_eval_score:.3f})"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 最適化済みエージェントの再登録
# MAGIC
# MAGIC プロンプト最適化の結果を反映したエージェントを新バージョンとして登録します。
# MAGIC これにより「エージェントのバージョン」で最適化前後を比較できます。

# COMMAND ----------

import tempfile
import os as _os
from mlflow.models.resources import DatabricksServingEndpoint

# ログ時の検証で OpenAI クライアントが認証できるよう環境変数を設定
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_host = spark.conf.get("spark.databricks.workspaceUrl")
_os.environ["OPENAI_API_KEY"] = _token
_os.environ["OPENAI_BASE_URL"] = f"https://{_host}/serving-endpoints"

optimized_agent_code = f'''
import os
import uuid
import mlflow
import mlflow.genai
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from openai import OpenAI

PROMPT_NAME = "{PROMPT_NAME}"

class LLMOpsDemoAgentOptimized(ChatAgent):
    """Foundation Model API を使ったチャットエージェント（Prompt Registry 連携）"""

    def __init__(self):
        self.client = OpenAI()
        self.model_endpoint = "databricks-meta-llama-3-3-70b-instruct"

    def predict(self, messages, context=None, custom_inputs=None):
        msgs = [{{"role": m.role, "content": m.content}} for m in messages]
        user_msg = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        prompt = mlflow.genai.load_prompt(f"prompts:/{{PROMPT_NAME}}@production")
        system_content = prompt.format(query=user_msg)
        msgs = [m for m in msgs if m["role"] != "system"]
        msgs.insert(0, {{"role": "system", "content": system_content}})
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

mlflow.models.set_model(LLMOpsDemoAgentOptimized())
'''

agent_file = _os.path.join(tempfile.mkdtemp(), "agent_optimized.py")
with open(agent_file, "w") as f:
    f.write(optimized_agent_code)

# COMMAND ----------

# 最適化済みエージェントを新バージョンとして登録
run_name = "agent_v2_optimized_prompt"
if result.final_eval_score > result.initial_eval_score:
    run_name = f"agent_v2_optimized_{result.final_eval_score:.3f}"

with mlflow.start_run(run_name=run_name):
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
print(
    f"最適化済みエージェント登録完了: {AGENT_MODEL_NAME} "
    f"(version {latest_version}, alias: production)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC プロンプト管理と最適化の仕組みを構築しました:
# MAGIC - **Prompt Registry**: プロンプトのバージョン管理と `production` / `staging` エイリアス
# MAGIC - **GEPA 最適化**: 評価スコアを最大化するようにプロンプトを自動改善
# MAGIC - **条件付きプロモーション**: スコアが改善した場合のみ production に昇格
# MAGIC - **エージェント再登録**: 最適化済みプロンプトを使うエージェントを新バージョンとして登録
# MAGIC
# MAGIC しかし、ここで使っている評価ジャッジ（Correctness 等）は汎用的なものです。
# MAGIC ドメイン専門家の判断基準に合致しているかは不明です。
# MAGIC
# MAGIC → 次章（04_judge_labeling）で、SME レビュー用ラベリングを準備します。
