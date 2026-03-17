# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04: ベースジャッジ作成 & ラベリングセッション
# MAGIC
# MAGIC - **前章の結果**: GEPA でプロンプトを最適化できた
# MAGIC - **この章のゴール**: ベースジャッジで評価し、SME レビュー用のラベリングセッションを作成する
# MAGIC - **次のステップ**: SME がレビュー完了後、05 でアラインメントを実行

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 設定の読み込み

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import mlflow
import mlflow.genai
from mlflow.genai.judges import make_judge
from mlflow.genai import evaluate, create_labeling_session, label_schemas
from mlflow.genai.scorers import get_scorer, ScorerSamplingConfig
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

# 実験IDを取得
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment が見つかりません: {EXPERIMENT_NAME}")
EXPERIMENT_ID = experiment.experiment_id

# 現在のユーザー名を取得
try:
    CURRENT_USER = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except Exception:
    CURRENT_USER = os.environ.get("DATABRICKS_USER", "unknown@example.com")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ベースジャッジの作成と登録
# MAGIC
# MAGIC `make_judge()` でカスタム LLM ジャッジを定義します。
# MAGIC
# MAGIC **重要**: ジャッジ名はラベルスキーマ名と完全一致させる必要があります（`align()` がペアリングに使用）。

# COMMAND ----------

base_judge = make_judge(
    name=JUDGE_NAME,
    instructions=(
        "ユーザーの質問: {{ inputs }}\n"
        "アシスタントの回答: {{ outputs }}\n\n"
        "回答の有用性を1-5のスケールで評価してください。\n"
        " 1: 全く役に立たない  2: ほとんど役に立たない\n"
        " 3: やや役に立つ  4: 役に立つ  5: 非常に役に立つ"
    ),
    feedback_value_type=float,
    model=f"databricks:/{MODEL_ENDPOINT}",
)

try:
    registered_judge = base_judge.register(experiment_id=EXPERIMENT_ID)
    print(f"ベースジャッジを登録しました: {registered_judge.name}")
except Exception:
    registered_judge = base_judge.update(
        experiment_id=EXPERIMENT_ID,
        sampling_config=ScorerSamplingConfig(sample_rate=0.0),
    )
    print(f"ベースジャッジを更新しました: {registered_judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 評価実行とトレースタグ付け

# COMMAND ----------

eval_data = [
    {"inputs": {"query": "Databricks とは？"}},
    {"inputs": {"query": "Delta Lake の利点は？"}},
    {"inputs": {"query": "Unity Catalog の役割は？"}},
    {"inputs": {"query": "MLflow でできることは？"}},
    {"inputs": {"query": "Apache Spark の特徴は？"}},
    {"inputs": {"query": "Photon エンジンとは？"}},
    {"inputs": {"query": "サーバーレスコンピュートの利点は？"}},
    {"inputs": {"query": "メダリオンアーキテクチャとは？"}},
    {"inputs": {"query": "Databricks SQL の特徴は？"}},
    {"inputs": {"query": "Auto Loader の仕組みは？"}},
    {"inputs": {"query": "Feature Store の役割は？"}},
    {"inputs": {"query": "Model Serving とは？"}},
]

results = evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[base_judge],
)

# COMMAND ----------

# 成功したトレースにタグを付与
ok_trace_ids = results.result_df.loc[
    results.result_df["state"] == "OK", "trace_id"
].tolist()
for trace_id in ok_trace_ids:
    try:
        mlflow.set_trace_tag(trace_id=trace_id, key="eval", value="complete")
    except Exception as e:
        print(f"WARN: trace tag 付与失敗 {trace_id}: {e}")

print(f"{len(ok_trace_ids)} 件のトレースにタグを付与しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ラベリングセッションの作成
# MAGIC
# MAGIC **重要**: ラベルスキーマの `name` はジャッジの `name` と完全一致させてください。

# COMMAND ----------

# ラベルスキーマの作成（名前 = ジャッジ名）
try:
    feedback_schema = label_schemas.create_label_schema(
        name=JUDGE_NAME,
        type="feedback",
        title=JUDGE_NAME,
        input=label_schemas.InputNumeric(min_value=1.0, max_value=5.0),
        instruction=(
            "回答の有用性を1-5で評価してください。\n"
            " 1: 全く役に立たない\n"
            " 2: ほとんど役に立たない\n"
            " 3: やや役に立つ\n"
            " 4: 役に立つ\n"
            " 5: 非常に役に立つ"
        ),
        enable_comment=True,
        overwrite=True,
    )
    print(f"ラベルスキーマを作成しました: {feedback_schema.name}")
except Exception:
    print(f"ラベルスキーマは既存のため再利用します: {JUDGE_NAME}")

# COMMAND ----------

# ラベリングセッションの作成
import uuid
labeling_session = create_labeling_session(
    name=f"llmops_demo_{JUDGE_NAME}_{uuid.uuid4().hex[:8]}",
    assigned_users=[CURRENT_USER],
    label_schemas=[JUDGE_NAME],
)

# 評価済みトレースをセッションに追加（Review App に表示される）
tagged_traces = mlflow.search_traces(
    locations=[EXPERIMENT_ID],
    filter_string="tag.eval = 'complete'",
    return_type="list",
)
labeling_session = labeling_session.add_traces(tagged_traces)

print(f"ラベリングセッションを作成しました。（{len(tagged_traces)} 件のトレース）")
print(f"以下の URL を SME に共有してください:")
print(f"  {labeling_session.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次のステップ
# MAGIC
# MAGIC 1. 上の URL を SME に共有してください
# MAGIC 2. SME がトレースをレビューし、有用性スコア (1-5) を付与します
# MAGIC 3. レビュー完了後、**05_judge_alignment** を実行してアラインメントを行います
# MAGIC
# MAGIC （デモの場合は、MLflow Experiment UI のトレース一覧から数件のトレースを手動で評価してください）
# MAGIC
# MAGIC ### プログラムでラベルを付与する場合（デモ用）
# MAGIC
# MAGIC 本番では Review App で SME が手動レビューしますが、
# MAGIC デモや自動テストでは以下のセルでプログラムラベリングが可能です。

# COMMAND ----------

# --- プログラムラベリング（デモ用）---
# 本番では Review App を使用。デモ・テスト時はこのセルを実行して HUMAN ラベルを自動付与。
# 再実行しても重複ラベルは付与しない（既存ラベルをスキップ）。
from mlflow.entities import AssessmentSource

labeled_traces = mlflow.search_traces(
    locations=[EXPERIMENT_ID],
    filter_string="tag.eval = 'complete'",
    return_type="list",
)
print(f"ラベル対象トレース: {len(labeled_traces)} 件")

label_count = 0
skip_count = 0
for t in labeled_traces:
    # 同一ユーザーが同名 HUMAN ラベルを既に付与済みなら再付与しない。
    # 値の更新ではなく、再実行安全性を優先するデモ実装。
    already_labeled = False
    # search_traces は assessments を含まない場合があるため get_trace() で個別取得
    full_trace = mlflow.get_trace(t.info.trace_id)
    assessments = (
        getattr(full_trace.info, "assessments", None)
        or getattr(full_trace, "assessments", None)
        or []
    )
    for a in assessments:
        name = getattr(a, "name", None) or getattr(a, "assessment_name", None)
        src = getattr(a, "source", None)
        src_type = getattr(src, "source_type", None) if src else None
        src_id = getattr(src, "source_id", None) if src else None
        if name == JUDGE_NAME and src_type == "HUMAN" and src_id == CURRENT_USER:
            already_labeled = True
            break

    if already_labeled:
        skip_count += 1
        continue

    mlflow.log_feedback(
        trace_id=t.info.trace_id,
        name=JUDGE_NAME,
        value=4.0,
        source=AssessmentSource(
            source_type="HUMAN",
            source_id=CURRENT_USER,
        ),
    )
    label_count += 1

print(f"新規ラベル付与: {label_count} 件 / スキップ: {skip_count} 件")
print("→ 05_judge_alignment を実行してアラインメントに進んでください。")
