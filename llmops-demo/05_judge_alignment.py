# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05: MemAlign によるジャッジアラインメント
# MAGIC
# MAGIC - **前提**: 04 でベースジャッジの評価とラベリングセッション作成が完了していること
# MAGIC - **前提**: SME がトレースにフィードバック（helpfulness スコア）を付与済みであること
# MAGIC - **この章のゴール**: SME フィードバックでジャッジを自動改善し、再評価する

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
from mlflow.genai import evaluate
from mlflow.genai.scorers import get_scorer, ScorerSamplingConfig
from openai import OpenAI

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.openai.autolog()

try:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
except Exception:
    token = os.environ.get("DATABRICKS_TOKEN")
if not token:
    raise RuntimeError("API token を取得できません。")

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
    return chat(inputs["query"])

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SME ラベルの確認

# COMMAND ----------

# タグ付きトレースを取得
traces = mlflow.search_traces(
    locations=[EXPERIMENT_ID],
    filter_string="tag.eval = 'complete'",
    return_type="list",
)
print(f"アラインメント対象トレース: {len(traces)} 件")

if not traces:
    raise RuntimeError("対象トレースがありません。先に 04 を実行してください。")

# SME ラベル付きトレースの件数を確認
# search_traces は assessments を含まない場合があるため get_trace() で個別取得
labeled_count = 0
for t in traces:
    full_trace = mlflow.get_trace(t.info.trace_id)
    assessments = getattr(full_trace.info, "assessments", None) or getattr(full_trace, "assessments", None) or []
    for a in assessments:
        name = getattr(a, "name", None) or getattr(a, "assessment_name", None)
        src = getattr(a, "source", None)
        src_type = getattr(src, "source_type", None) if src else None
        if name == JUDGE_NAME and src_type == "HUMAN":
            labeled_count += 1
            break
print(f"SME ラベル付きトレース: {labeled_count} 件")

if labeled_count == 0:
    raise RuntimeError(
        "SME ラベルがありません。04 のラベリングセッション URL でレビューを完了してから再実行してください。"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. MemAlign によるアラインメント
# MAGIC
# MAGIC SME のフィードバックを基に、ジャッジの指示を自動改善します。

# COMMAND ----------

from mlflow.genai.judges.optimizers import MemAlignOptimizer

optimizer = MemAlignOptimizer(
    reflection_lm=f"databricks:/{MODEL_ENDPOINT}",
    retrieval_k=5,
    embedding_model="databricks:/databricks-gte-large-en",
)

base_judge = get_scorer(name=JUDGE_NAME)
aligned_judge = base_judge.align(traces=traces, optimizer=optimizer)

print("アラインメント完了！")
print(f"\n=== アラインド・ジャッジの指示（先頭500文字）===")
print(aligned_judge.instructions[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. アラインド・ジャッジの登録と再評価

# COMMAND ----------

# アラインド・ジャッジを更新（既存レコードを上書き）
aligned_judge_registered = aligned_judge.update(
    experiment_id=EXPERIMENT_ID,
    sampling_config=ScorerSamplingConfig(sample_rate=0.0),
)
print(f"アラインド・ジャッジを登録しました: {aligned_judge_registered.name}")

# COMMAND ----------

# 再評価（04 と同じ 12 件で確認）
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

aligned_results = evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[aligned_judge],
)

print("=== アラインド・ジャッジの評価結果 ===")
# make_judge はフィードバック型のため metrics dict は空になる場合がある
# スコアは result_df の helpfulness/value カラムに格納される
if aligned_results.metrics:
    for k, v in aligned_results.metrics.items():
        print(f"  {k}: {v}")
else:
    df = aligned_results.result_df
    score_cols = [c for c in df.columns if c.endswith("/value") and JUDGE_NAME in c]
    for col in score_cols:
        values = df[col].dropna().tolist()
        if values:
            avg = sum(float(v) for v in values) / len(values)
            print(f"  {col}: 平均 {avg:.2f} ({len(values)} 件)")
    print(f"  評価件数: {len(df)} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ### スコア変化の解釈
# MAGIC
# MAGIC アラインド・ジャッジのスコアがベースジャッジより**低くなる**ことがあります。
# MAGIC これは**正常であり、精度向上の証拠**です。
# MAGIC
# MAGIC - ベースジャッジ: 汎用的な基準で甘めに評価 → 高スコア
# MAGIC - アラインド・ジャッジ: ドメイン専門家の基準で厳密に評価 → 適切なスコア

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC ジャッジアラインメントが完了しました:
# MAGIC - **ベースジャッジ** → SME フィードバック → **MemAlign** → **アラインド・ジャッジ**
# MAGIC - ドメイン基準に沿った正確な評価が可能に
# MAGIC
# MAGIC → 次章（06_production_monitoring）で、本番モニタリングを構築します。
