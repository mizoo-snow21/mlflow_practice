"""
Local verification for notebooks 04/05 (Judge Alignment).
Tests make_judge, evaluate, set_trace_tag, create_label_schema, create_labeling_session.
Skips align() — E2E テスト (test_e2e.py) でカバー。

Usage:
  uv run python test/test_04_local.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import mlflow
from conftest import (
    JUDGE_NAME,
    MODEL_ENDPOINT,
    TestRunner,
    build_test_context,
    chat,
    make_predict_fn,
)
from mlflow.genai.judges import make_judge
from mlflow.genai import evaluate, create_labeling_session, label_schemas
from mlflow.genai.scorers import ScorerSamplingConfig

ctx = None
predict_fn = None
runner = TestRunner("Notebook 04 Local Verification")

# 共有状態
base_judge = None
registered_judge = None
eval_results = None


def setup():
    """テスト実行前の初期化。main() から 1 回だけ呼ぶ。"""
    global ctx, predict_fn
    ctx = build_test_context()
    predict_fn = make_predict_fn(ctx)


# =============================================
# Test 1: make_judge
# =============================================
def test_01_make_judge():
    global base_judge
    base_judge = make_judge(
        name=JUDGE_NAME,
        instructions=(
            "User question: {{ inputs }}\n"
            "Assistant response: {{ outputs }}\n\n"
            "Rate the helpfulness of the response on a scale of 1-5.\n"
            " 1: Not helpful at all  2: Barely helpful\n"
            " 3: Somewhat helpful  4: Helpful  5: Very helpful"
        ),
        feedback_value_type=float,
        model=f"databricks:/{MODEL_ENDPOINT}",
    )
    assert base_judge is not None, "make_judge returned None"
    assert base_judge.name == JUDGE_NAME
    print(f"  Judge created: name={base_judge.name}")


# =============================================
# Test 2: Register judge to experiment
# =============================================
def test_02_register_judge():
    global registered_judge
    assert base_judge is not None, "base_judge not created yet"
    try:
        registered_judge = base_judge.register(experiment_id=ctx.experiment_id)
        print(f"  Registered judge (new): {registered_judge.name}")
    except ValueError as e:
        if "already been registered" in str(e):
            registered_judge = base_judge.update(
                experiment_id=ctx.experiment_id,
                sampling_config=ScorerSamplingConfig(sample_rate=0.0),
            )
            print(f"  Updated judge: {registered_judge.name}")
        else:
            raise
    assert registered_judge is not None


# =============================================
# Test 3: evaluate() with judge on 3 samples
# =============================================
def test_03_evaluate():
    global eval_results
    assert base_judge is not None, "base_judge not created yet"

    eval_data = [
        {"inputs": {"query": "Databricks とは？"}},
        {"inputs": {"query": "Delta Lake の利点は？"}},
        {"inputs": {"query": "Unity Catalog の役割は？"}},
    ]

    eval_results = evaluate(
        data=eval_data,
        predict_fn=predict_fn,
        scorers=[base_judge],
    )
    assert eval_results is not None
    assert isinstance(eval_results.metrics, dict)
    print(f"  Metrics: {eval_results.metrics}")
    print(f"  Result DF shape: {eval_results.result_df.shape}")


# =============================================
# Test 4: set_trace_tag on successful traces
# =============================================
def test_04_set_trace_tag():
    assert eval_results is not None, "eval_results not available"

    ok_trace_ids = eval_results.result_df.loc[
        eval_results.result_df["state"] == "OK", "trace_id"
    ]
    assert len(ok_trace_ids) > 0, "No OK traces found"

    tagged_count = 0
    for trace_id in ok_trace_ids:
        mlflow.set_trace_tag(trace_id=trace_id, key="eval_04_test", value="complete")
        tagged_count += 1

    print(f"  Tagged {tagged_count} traces with eval_04_test=complete")

    traces = mlflow.search_traces(
        filter_string="tag.eval_04_test = 'complete'",
        max_results=5,
    )
    assert len(traces) > 0, "No traces found with tag"
    print(f"  Verified: found {len(traces)} tagged traces")


# =============================================
# Test 5: create_label_schema (name must match judge name)
# =============================================
feedback_schema = None


def test_05_create_label_schema():
    global feedback_schema
    try:
        feedback_schema = label_schemas.create_label_schema(
            name=JUDGE_NAME,
            type="feedback",
            title=JUDGE_NAME,
            input=label_schemas.InputNumeric(min_value=1.0, max_value=5.0),
            instruction=(
                "Rate helpfulness 1-5.\n"
                " 1: Not helpful at all\n"
                " 2: Barely helpful\n"
                " 3: Somewhat helpful\n"
                " 4: Helpful\n"
                " 5: Very helpful"
            ),
            enable_comment=True,
            overwrite=True,
        )
        print(f"  Label schema created: name={feedback_schema.name}")
    except Exception as e:
        if "already exists" in str(e) or "Cannot rename or remove" in str(e):
            print(f"  Label schema '{JUDGE_NAME}' already exists (referenced by session)")
            feedback_schema = type("Schema", (), {"name": JUDGE_NAME})()
        else:
            raise
    assert feedback_schema is not None
    assert feedback_schema.name == JUDGE_NAME


# =============================================
# Test 6: create_labeling_session
# =============================================
def test_06_create_labeling_session():
    print(f"  Using current user: {ctx.current_user}")
    labeling_session = create_labeling_session(
        name="helpfulness_labeling_04_test",
        assigned_users=[ctx.current_user],
        label_schemas=[JUDGE_NAME],
    )
    assert labeling_session is not None
    print(f"  Labeling session created")
    if hasattr(labeling_session, "url") and labeling_session.url:
        print(f"  URL: {labeling_session.url}")


# =============================================
# Test 7: MemAlignOptimizer import (no execution)
# =============================================
def test_07_memalign_import():
    from mlflow.genai.judges.optimizers import MemAlignOptimizer

    optimizer = MemAlignOptimizer(
        reflection_lm=f"databricks:/{MODEL_ENDPOINT}",
        retrieval_k=5,
        embedding_model="databricks:/databricks-gte-large-en",
    )
    assert optimizer is not None
    print(f"  MemAlignOptimizer created (align() は test_e2e.py でテスト)")


# =============================================
# Test 8: search_traces with return_type="list"
# =============================================
def test_08_search_traces_list():
    traces = mlflow.search_traces(
        locations=[ctx.experiment_id],
        filter_string="tag.eval_04_test = 'complete'",
        return_type="list",
    )
    assert isinstance(traces, list)
    assert len(traces) > 0, "No traces found"
    print(f"  search_traces(return_type='list') returned {len(traces)} traces")
    t = traces[0]
    print(f"  Trace type: {type(t).__name__}")
    if hasattr(t, "info"):
        print(f"  Trace ID: {t.info.trace_id}")


# =============================================
# Run all tests
# =============================================
def main():
    print("=" * 60)
    print("LLMOps Demo - Notebook 04 Local Verification")
    print("=" * 60)

    setup()

    runner.run("01: make_judge", test_01_make_judge)
    runner.run("02: Register judge to experiment", test_02_register_judge)
    runner.run("03: evaluate() with judge (3 samples)", test_03_evaluate)
    runner.run("04: set_trace_tag on OK traces", test_04_set_trace_tag)
    runner.run("05: create_label_schema (name=judge name)", test_05_create_label_schema)
    runner.run("06: create_labeling_session", test_06_create_labeling_session)
    runner.run("07: MemAlignOptimizer import", test_07_memalign_import)
    runner.run("08: search_traces return_type=list", test_08_search_traces_list)

    runner.exit()


if __name__ == "__main__":
    main()
