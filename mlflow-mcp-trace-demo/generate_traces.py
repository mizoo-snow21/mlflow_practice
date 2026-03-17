"""
MLflow MCP Server デモ用 - 問題のあるトレースを含むサンプルデータを生成

生成されるトレース:
- 正常なトレース（高速応答）
- エラートレース（retrieval失敗、LLM timeout等）
- 遅延トレース（1秒以上）
"""

import os
import random
import time

import mlflow


DEFAULT_PROFILE = "e2-demo-field-eng"
DEFAULT_EXPERIMENT = "/Users/yukihiro.mizoguchi@databricks.com/customer-support-agent"


def configure_mlflow() -> str:
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE", DEFAULT_PROFILE)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)

    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name)
    return experiment_name


@mlflow.trace
def customer_support_agent(query: str, user_id: str) -> str:
    """カスタマーサポートエージェントのメイン処理"""
    mlflow.update_current_trace(
        tags={"user_id": user_id, "agent_version": "v2.1"},
        metadata={"source": "web_chat"},
    )
    context = retrieve_knowledge(query)
    return generate_answer(query, context)


@mlflow.trace(span_type="RETRIEVER")
def retrieve_knowledge(query: str) -> str:
    """ナレッジベースから関連情報を検索"""
    normalized = query.lower()

    if "billing" in normalized and random.random() < 0.5:
        time.sleep(0.3)
        raise ConnectionError("Knowledge base connection timeout - billing index unavailable")

    if "internal" in normalized:
        time.sleep(0.2)
        raise PermissionError("Access denied: internal documents require elevated permissions")

    delay = random.uniform(0.1, 0.5)
    if "pricing" in normalized:
        delay = random.uniform(1.5, 3.0)
    time.sleep(delay)

    return f"Retrieved context for: {query}"


@mlflow.trace(span_type="LLM")
def generate_answer(query: str, context: str) -> str:
    """LLM で回答を生成"""
    normalized = query.lower()

    delay = random.uniform(0.2, 0.8)
    if "complex" in normalized:
        delay = random.uniform(2.0, 4.0)

    time.sleep(delay)

    if "rate limit" in normalized:
        raise RuntimeError("OpenAI API rate limit exceeded. Retry after 60s.")

    return f"Here's the answer to your question about {query}: Based on {context}, ..."


def main() -> None:
    random.seed(42)
    experiment_name = configure_mlflow()

    test_cases = [
        # 正常系
        ("How do I reset my password?", "user_001"),
        ("What are your business hours?", "user_002"),
        ("How to update my profile?", "user_003"),
        ("Where can I find the user guide?", "user_004"),
        ("What payment methods do you accept?", "user_005"),
        # エラー系 - billing index問題
        ("I have a billing issue with my last invoice", "user_006"),
        ("Can you check my billing history?", "user_007"),
        ("Billing department contact info", "user_008"),
        # エラー系 - 権限不足
        ("Show me internal documentation", "user_009"),
        ("Access internal admin panel", "user_010"),
        # 遅延系 - pricing クエリ
        ("What is the pricing for enterprise plan?", "user_011"),
        ("Compare pricing tiers", "user_012"),
        ("Pricing for 100 users", "user_013"),
        # 遅延系 - 複雑なクエリ
        ("Complex migration from competitor to our platform", "user_014"),
        ("Complex integration with SAP and Salesforce", "user_015"),
        # エラー系 - rate limit
        ("rate limit test query", "user_016"),
        # 追加の正常系
        ("Thank you for your help!", "user_017"),
        ("How to cancel my subscription?", "user_018"),
        ("Do you offer a free trial?", "user_019"),
        ("What's new in the latest release?", "user_020"),
    ]

    print("🚀 トレース生成開始...")
    print(f"   MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   実験名: {experiment_name}")
    print(f"   テストケース数: {len(test_cases)}")
    print()

    success = 0
    errors = 0

    for query, user_id in test_cases:
        try:
            customer_support_agent(query, user_id)
            print(f"  ✅ [{user_id}] {query[:50]}")
            success += 1
        except Exception as exc:
            print(f"  ❌ [{user_id}] {query[:50]} -> {type(exc).__name__}: {exc}")
            errors += 1

    print()
    print(f"📊 結果: {success} 成功 / {errors} エラー / {len(test_cases)} 合計")


if __name__ == "__main__":
    main()
