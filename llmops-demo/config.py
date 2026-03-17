# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 共有設定 (config)
# MAGIC
# MAGIC 全ノートブックで共通利用する設定値を定義します。

# COMMAND ----------

CATALOG = "main"
SCHEMA = "llmops_demo"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EXPERIMENT_NAME = "/Shared/llmops-demo"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.llmops_demo_system_prompt"
JUDGE_NAME = "helpfulness"
EVAL_DATASET_NAME = "llmops_demo_eval_v1"
AGENT_MODEL_NAME = f"{CATALOG}.{SCHEMA}.llmops_demo_agent"

# COMMAND ----------

# レートリミット対策: リトライ付き LLM 呼び出しヘルパー
import time as _time

def _call_with_retry(fn, max_retries=5, base_wait=10):
    """429 レートリミットエラー時に指数バックオフでリトライ"""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "RATE_LIMIT" in str(e).upper() or "REQUEST_LIMIT" in str(e).upper():
                wait = base_wait * (2 ** attempt)
                print(f"  Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                _time.sleep(wait)
            else:
                raise
    return fn()
