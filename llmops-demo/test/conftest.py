"""
テスト共通基盤: 認証・クライアント設定・テストハーネスを共通化。

全テストファイルから build_test_context() を呼び出して初期化する。
import 時に副作用（認証・MLflow 初期化）を起こさない設計。
"""
import os
import sys
import traceback
from dataclasses import dataclass

import mlflow
import mlflow.genai
from databricks.sdk import WorkspaceClient
from openai import OpenAI

# --- 設定定数 (config.py と同等) ---
CATALOG = "main"
SCHEMA = "llmops_demo"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EXPERIMENT_NAME = "/Shared/llmops-demo"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.llmops_demo_system_prompt"
JUDGE_NAME = "helpfulness"
EVAL_DATASET_NAME = "llmops_demo_eval_v1"


@dataclass
class TestContext:
    """テスト実行に必要なリソースを保持するコンテキスト。"""
    workspace_client: WorkspaceClient
    openai_client: OpenAI
    host: str
    token: str
    experiment_id: str
    current_user: str


def build_test_context() -> TestContext:
    """Databricks 認証・MLflow 初期化を行い TestContext を返す。"""
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    # MLflow は DATABRICKS_CONFIG_PROFILE 環境変数経由でプロファイルを参照する
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    workspace_client = WorkspaceClient(profile=profile)
    host = workspace_client.config.host
    token = workspace_client.config.token
    if not host or not token:
        raise RuntimeError("Databricks 認証情報を取得できませんでした。")

    openai_client = OpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints",
    )

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.openai.autolog()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment が見つかりません: {EXPERIMENT_NAME}")

    current_user = workspace_client.current_user.me().user_name

    return TestContext(
        workspace_client=workspace_client,
        openai_client=openai_client,
        host=host,
        token=token,
        experiment_id=experiment.experiment_id,
        current_user=current_user,
    )


def chat(ctx: TestContext, user_message, system_prompt="あなたは親切なアシスタントです。"):
    """LLM エンドポイントにチャットリクエストを送信。"""
    response = ctx.openai_client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def make_predict_fn(ctx: TestContext):
    """evaluate() 用の predict 関数を生成。"""
    def predict_fn(**inputs):
        return chat(ctx, inputs["query"])
    return predict_fn


# --- テストハーネス ---
class TestRunner:
    """シンプルなテストハーネス。各テストファイルでインスタンス化して利用。"""

    def __init__(self, suite_name):
        self.suite_name = suite_name
        self.passed = 0
        self.failed = 0

    def run(self, name, fn):
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"  PASS")
            self.passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"{self.suite_name}: {self.passed} passed, {self.failed} failed, {total} total")
        print(f"{'='*60}")
        return self.failed

    def exit(self):
        sys.exit(1 if self.summary() > 0 else 0)
