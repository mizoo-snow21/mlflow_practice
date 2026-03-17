"""
Local verification for Notebook 06: Production Monitoring.

Tests the core APIs used in 06_production_monitoring.py without requiring
dbutils, spark, or a SQL Warehouse ID.

Usage:
  uv run python test_05_local.py
"""
import os
import sys
import time
import traceback

# --- Settings (mirror config.py) ---
CATALOG = "main"
SCHEMA = "llmops_demo"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EXPERIMENT_NAME = "/Shared/llmops-demo"

# Databricks auth
os.environ["DATABRICKS_CONFIG_PROFILE"] = "e2-demo-tokyo"

import mlflow
from openai import OpenAI
from databricks.sdk import WorkspaceClient

w = WorkspaceClient(profile="e2-demo-tokyo")
host = w.config.host
token = w.config.token

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.openai.autolog()

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id


def chat(user_message, system_prompt="You are a helpful assistant."):
    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


# --- Test harness ---
passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        failed += 1


# =============================================
# Test 1: Imports
# =============================================
def test_imports():
    """Verify all production monitoring imports resolve."""
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

    print(f"  UCSchemaLocation:                      {UCSchemaLocation}")
    print(f"  set_experiment_trace_location:          {set_experiment_trace_location}")
    print(f"  set_databricks_monitoring_sql_warehouse_id: {set_databricks_monitoring_sql_warehouse_id}")
    print(f"  Safety:                                 {Safety}")
    print(f"  Guidelines:                             {Guidelines}")
    print(f"  ScorerSamplingConfig:                   {ScorerSamplingConfig}")
    print(f"  list_scorers:                           {list_scorers}")
    print(f"  get_scorer:                             {get_scorer}")
    print(f"  delete_scorer:                          {delete_scorer}")

    # Verify they are callable / classes
    assert callable(set_experiment_trace_location)
    assert callable(set_databricks_monitoring_sql_warehouse_id)
    assert callable(list_scorers)
    assert callable(get_scorer)
    assert callable(delete_scorer)


# =============================================
# Test 2: Generate a traced chat call
# =============================================
def test_generate_trace():
    """Generate a trace via autolog so subsequent tests have data."""
    response = chat("What is MLflow?")
    assert response and len(response) > 10, f"Response too short: {response}"
    print(f"  Response: {response[:100]}...")
    # Small pause to let trace propagate
    time.sleep(2)


# =============================================
# Test 3: mlflow.log_feedback() on an existing trace
# =============================================
def test_log_feedback():
    """Find a recent trace and attach feedback."""
    traces = mlflow.search_traces(
        filter_string="attributes.status = 'OK'",
        order_by=["attributes.timestamp_ms DESC"],
        max_results=1,
        return_type="list",
    )
    assert len(traces) > 0, "No OK traces found to attach feedback to"

    trace_id = traces[0].info.trace_id
    print(f"  Target trace_id: {trace_id}")

    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_rating",
        value=4.0,
        rationale="Accurate and clear response (test_05_local)",
    )
    print(f"  Feedback logged successfully for trace {trace_id[:16]}...")


# =============================================
# Test 4: search_traces with time-based filter
# =============================================
def test_search_traces_time_filter():
    """Search traces from the past 24 hours using timestamp filter."""
    cutoff = int((time.time() - 86400) * 1000)
    traces_df = mlflow.search_traces(
        filter_string=f"attributes.status = 'OK' AND attributes.timestamp_ms > {cutoff}",
        max_results=10,
    )
    print(f"  Traces in past 24h: {len(traces_df)}")
    if len(traces_df) > 0:
        print(f"  Columns: {list(traces_df.columns)}")
        print(f"  First trace_id: {traces_df.iloc[0]['trace_id'] if 'trace_id' in traces_df.columns else 'N/A'}")
    # We just generated one, so there should be at least 1
    assert len(traces_df) >= 1, f"Expected >= 1 trace in past 24h, got {len(traces_df)}"


# =============================================
# Test 5: list_scorers()
# =============================================
def test_list_scorers():
    """Call list_scorers() and display registered scorers."""
    from mlflow.genai.scorers import list_scorers

    scorers = list_scorers()
    print(f"  Registered scorers: {len(scorers)}")
    for s in scorers:
        rate = "N/A"
        if hasattr(s, "sampling_config") and s.sampling_config:
            rate = s.sampling_config.sample_rate
        name = s.name if hasattr(s, "name") else type(s).__name__
        print(f"    - {name}: sample_rate={rate}, type={type(s).__name__}")
    # list_scorers should return a list (possibly empty)
    assert isinstance(scorers, list), f"Expected list, got {type(scorers)}"


# =============================================
# Test 6: Scorer instantiation (no register/start - just verify objects)
# =============================================
def test_scorer_instantiation():
    """Verify Safety, Guidelines, ScorerSamplingConfig can be instantiated."""
    from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig

    safety = Safety()
    print(f"  Safety instance: {safety}")

    guidelines = Guidelines(
        name="professional_tone",
        guidelines="The response must be professional and helpful",
    )
    print(f"  Guidelines instance: {guidelines}")

    config = ScorerSamplingConfig(sample_rate=0.5)
    print(f"  ScorerSamplingConfig(sample_rate=0.5): {config}")
    assert config.sample_rate == 0.5, f"Expected 0.5, got {config.sample_rate}"


# =============================================
# Test 7: Model endpoint format verification
# =============================================
def test_model_endpoint_format():
    """Verify the model format used in production monitoring."""
    model_uri = f"databricks:/{MODEL_ENDPOINT}"
    print(f"  Model URI format: {model_uri}")
    assert model_uri.startswith("databricks:/"), "Model URI must start with databricks:/"
    assert MODEL_ENDPOINT in model_uri


# =============================================
# Run all tests
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Notebook 05: Production Monitoring - Local Verification")
    print("=" * 60)
    print(f"  Profile:    e2-demo-tokyo")
    print(f"  Host:       {host}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Experiment ID: {EXPERIMENT_ID}")
    print(f"  Model:      {MODEL_ENDPOINT}")

    run_test("05-1: Imports", test_imports)
    run_test("05-2: Generate traced chat", test_generate_trace)
    run_test("05-3: mlflow.log_feedback()", test_log_feedback)
    run_test("05-4: search_traces (time filter)", test_search_traces_time_filter)
    run_test("05-5: list_scorers()", test_list_scorers)
    run_test("05-6: Scorer instantiation", test_scorer_instantiation)
    run_test("05-7: Model endpoint format", test_model_endpoint_format)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    if failed > 0:
        print("\nNote: UC trace ingestion setup was skipped (requires SQL Warehouse ID).")

    sys.exit(1 if failed > 0 else 0)
