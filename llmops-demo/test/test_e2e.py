"""
E2E テスト: 04 → プログラムラベリング → 05 (align) の全フローを検証。

これまで align() はテストスキップされていたが、このテストで初めてカバーする。
プログラムラベリングで HUMAN ラベルを自動付与し、MemAlign を実行する。

Usage:
  uv run python test/test_e2e.py
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
    make_predict_fn,
)
from mlflow.genai.judges import make_judge
from mlflow.genai import evaluate
from mlflow.genai.scorers import ScorerSamplingConfig
from mlflow.entities import AssessmentSource

ctx = None
predict_fn = None
runner = TestRunner("E2E: 04 → ラベリング → 05 align")

# 共有状態
base_judge = None
aligned_judge = None
eval_results = None


def setup():
    """テスト実行前の初期化。main() から 1 回だけ呼ぶ。"""
    global ctx, predict_fn
    ctx = build_test_context()
    predict_fn = make_predict_fn(ctx)


# =============================================
# Phase 1: ベースジャッジ作成 (04 相当)
# =============================================
def test_01_create_base_judge():
    global base_judge
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
    assert base_judge is not None
    assert base_judge.name == JUDGE_NAME
    print(f"  ベースジャッジ作成: {base_judge.name}")

    try:
        base_judge.register(experiment_id=ctx.experiment_id)
        print(f"  登録完了")
    except ValueError as e:
        if "already been registered" in str(e):
            base_judge.update(
                experiment_id=ctx.experiment_id,
                sampling_config=ScorerSamplingConfig(sample_rate=0.0),
            )
            print(f"  更新完了")
        else:
            raise


# =============================================
# Phase 2: 評価実行 + トレースタグ付け (04 相当)
# =============================================
def test_02_evaluate_and_tag():
    global eval_results
    assert base_judge is not None, "base_judge が未作成"

    eval_data = [
        {"inputs": {"query": "Databricks とは？"}},
        {"inputs": {"query": "Delta Lake の利点は？"}},
        {"inputs": {"query": "MLflow でできることは？"}},
    ]

    eval_results = evaluate(
        data=eval_data,
        predict_fn=predict_fn,
        scorers=[base_judge],
    )
    assert eval_results is not None
    print(f"  評価メトリクス: {eval_results.metrics}")

    ok_ids = eval_results.result_df.loc[
        eval_results.result_df["state"] == "OK", "trace_id"
    ].tolist()
    for tid in ok_ids:
        mlflow.set_trace_tag(trace_id=tid, key="eval", value="complete")
    print(f"  タグ付けトレース: {len(ok_ids)} 件")
    assert len(ok_ids) > 0, "OK トレースが 0 件"


# =============================================
# Phase 3: プログラムラベリング (04 新セル相当 — 重複防止付き)
# =============================================
def test_03_programmatic_labeling():
    """HUMAN ラベルをプログラムで付与（Review App の代替）。既存ラベルはスキップ。"""
    traces = mlflow.search_traces(
        locations=[ctx.experiment_id],
        filter_string="tag.eval = 'complete'",
        return_type="list",
    )
    assert len(traces) > 0, "ラベル対象トレースが 0 件"

    label_count = 0
    skip_count = 0
    for t in traces:
        already_labeled = False
        for a in (getattr(t, "assessments", None) or []):
            name = getattr(a, "name", None) or getattr(a, "assessment_name", None)
            src = getattr(a, "source", None)
            src_type = getattr(src, "source_type", None) if src else None
            src_id = getattr(src, "source_id", None) if src else None
            if name == JUDGE_NAME and src_type == "HUMAN" and src_id == ctx.current_user:
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
                source_id=ctx.current_user,
            ),
        )
        label_count += 1

    print(f"  新規ラベル: {label_count} 件 / スキップ: {skip_count} 件")
    assert label_count + skip_count > 0


# =============================================
# Phase 4: SME ラベル確認 (05 相当 — getattr で安全にアクセス)
# =============================================
def test_04_verify_labels():
    """SME ラベルが付与されていることを確認（get_trace で個別取得）。"""
    # search_traces は log_feedback 直後にアセスメントを反映しない場合がある
    # get_trace() で個別にフェッチすることで最新状態を確認
    trace_ids = eval_results.result_df.loc[
        eval_results.result_df["state"] == "OK", "trace_id"
    ].tolist()

    labeled_count = 0
    for tid in trace_ids:
        t = mlflow.get_trace(tid)
        for a in (getattr(t.info, "assessments", None) or getattr(t, "assessments", None) or []):
            name = getattr(a, "name", None) or getattr(a, "assessment_name", None)
            src = getattr(a, "source", None)
            src_type = getattr(src, "source_type", None) if src else None
            if name == JUDGE_NAME and src_type == "HUMAN":
                labeled_count += 1
                break

    print(f"  SME ラベル付きトレース: {labeled_count}/{len(trace_ids)} 件")
    assert labeled_count > 0, "HUMAN ラベルが見つかりません"


# =============================================
# Phase 5: MemAlign アラインメント (05 相当)
# =============================================
def test_05_align():
    """MemAlign でジャッジをアラインメント — これまでスキップされていたテスト。"""
    global aligned_judge
    from mlflow.genai.judges.optimizers import MemAlignOptimizer

    traces = mlflow.search_traces(
        locations=[ctx.experiment_id],
        filter_string="tag.eval = 'complete'",
        return_type="list",
    )
    assert len(traces) > 0

    optimizer = MemAlignOptimizer(
        reflection_lm=f"databricks:/{MODEL_ENDPOINT}",
        retrieval_k=5,
        embedding_model="databricks:/databricks-gte-large-en",
    )

    aligned_judge = base_judge.align(traces=traces, optimizer=optimizer)

    assert aligned_judge is not None, "align() が None を返しました"
    assert aligned_judge.instructions, "アラインド・ジャッジの instructions が空"
    print(f"  アラインメント完了!")
    print(f"  instructions (先頭200文字): {aligned_judge.instructions[:200]}")


# =============================================
# Phase 6: アラインド・ジャッジで再評価 (05 相当 — 12 件)
# =============================================
def test_06_re_evaluate_with_aligned():
    """アラインド・ジャッジで再評価（04/05 と同じ 12 件）。"""
    assert aligned_judge is not None, "aligned_judge が未作成"

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
        scorers=[aligned_judge],
    )
    assert results is not None
    # make_judge はフィードバック型スコアラーのため metrics dict は空になる
    # スコアは result_df に格納される
    df = results.result_df
    assert len(df) == len(eval_data), f"結果行数が不一致: {len(df)} != {len(eval_data)}"
    print(f"  再評価完了: {len(df)} 件")
    print(f"  result_df columns: {list(df.columns)}")
    if results.metrics:
        for k, v in results.metrics.items():
            print(f"    {k}: {v}")


# =============================================
# 実行
# =============================================
def main():
    print("=" * 60)
    print("E2E Test: 04 → プログラムラベリング → 05 align")
    print("=" * 60)

    setup()

    runner.run("01: ベースジャッジ作成", test_01_create_base_judge)
    runner.run("02: 評価実行 + トレースタグ付け", test_02_evaluate_and_tag)
    runner.run("03: プログラムラベリング", test_03_programmatic_labeling)
    runner.run("04: SME ラベル確認 (getattr)", test_04_verify_labels)
    runner.run("05: MemAlign アラインメント", test_05_align)
    runner.run("06: アラインド・ジャッジで再評価 (12件)", test_06_re_evaluate_with_aligned)

    runner.exit()


if __name__ == "__main__":
    main()
