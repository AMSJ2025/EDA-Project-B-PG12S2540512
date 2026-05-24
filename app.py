import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(
    page_title="Mini Project B — Time-Series Forecasting Starter",
    layout="wide",
)


def get_openrouter_api_key():
    """Read key from Streamlit Secrets, environment, or UI password input."""
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass

    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    return st.text_input(
        "OpenRouter API key",
        type="password",
        help="Used only when you click the AI grader button.",
    )


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return audit


def clean_time_series(df: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, target_col])
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    return work


def resample_time_series(df: pd.DataFrame, timestamp_col: str, target_col: str, rule: str | None) -> pd.DataFrame:
    if not rule or rule == "No resampling":
        return df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    resampled = (
        df.set_index(timestamp_col)[numeric_cols]
        .resample(rule)
        .mean()
        .dropna(subset=[target_col])
        .reset_index()
    )
    return resampled


def make_baseline_feature_table(df: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    features = df[[timestamp_col, target_col]].copy()
    features["lag_1"] = features[target_col].shift(1)
    features["lag_24"] = features[target_col].shift(24)
    features["rolling_mean_24"] = features[target_col].shift(1).rolling(24).mean()
    features["hour"] = features[timestamp_col].dt.hour
    features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    features["month"] = features[timestamp_col].dt.month
    features["y_target"] = features[target_col].shift(-horizon)
    return features.dropna().reset_index(drop=True)


def build_submission_json(
    student_name: str,
    student_id: str,
    deployed_url: str,
    repo_url: str,
    project_title: str,
    project_goal: str,
    timestamp_col: str,
    target_col: str,
    horizon: int,
    resampling_rule: str,
    raw_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    results_df,
    insights_text: str,
    missing_discussion: str,
    outlier_discussion: str,
    resampling_discussion: str,
):
    results_table = [] if results_df is None else results_df.to_dict(orient="records")

    evidence = {
        "student": {
            "name": student_name,
            "id": student_id,
            "deployed_url": deployed_url,
            "repo_url": repo_url,
        },
        "project": {
            "title": project_title,
            "goal": project_goal,
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "forecast_horizon": int(horizon),
            "resampling_rule": resampling_rule,
        },
        "data_evidence": {
            "raw_rows": int(len(raw_df)),
            "clean_rows": int(len(ts_df)),
            "feature_rows": int(len(feature_df)),
            "timestamp_min": str(ts_df[timestamp_col].min()) if len(ts_df) else "",
            "timestamp_max": str(ts_df[timestamp_col].max()) if len(ts_df) else "",
            "missing_discussion": missing_discussion,
            "outlier_discussion": outlier_discussion,
            "resampling_discussion": resampling_discussion,
        },
        "feature_engineering": {
            "baseline_features_present": ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"],
            "student_added_features": [],
        },
        "modeling_evaluation": {
            "has_metrics_table": isinstance(results_df, pd.DataFrame),
            "results_table": results_table,
            "time_based_split_evidence": "",
            "models_used": [],
        },
        "dashboard": {
            "has_extra_dashboard_plots": False,
            "dashboard_notes": "",
        },
        "presentation": {
            "insights": insights_text,
        },
    }
    return evidence


def build_project_card(evidence: dict) -> str:
    project = evidence["project"]
    student = evidence["student"]
    data = evidence["data_evidence"]
    return f"""# Project Card — {project['title']}

## Student
- Name: {student['name']}
- ID: {student['id']}

## Goal
{project['goal']}

## Dataset
- Timestamp column: {project['timestamp_column']}
- Target column: {project['target_column']}
- Raw rows: {data['raw_rows']}
- Clean rows: {data['clean_rows']}
- Time coverage: {data['timestamp_min']} to {data['timestamp_max']}

## Forecast setup
- Horizon: {project['forecast_horizon']}
- Resampling: {project['resampling_rule']}

## Required discussions
### Missing timestamps / missing values
{data['missing_discussion']}

### Outliers
{data['outlier_discussion']}

### Resampling
{data['resampling_discussion']}

## Student insights
{evidence['presentation']['insights']}

## Links
- Deployed app: {student['deployed_url']}
- Repository: {student['repo_url']}
"""


def call_ai_grader(api_key: str, evidence_json: str):
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", evidence_json)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Mini Project B AI Grader",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_ai_response(text: str):
    try:
        return json.loads(text), None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except Exception as exc:
            return None, f"Found JSON-like text but could not parse it: {exc}"

    return None, "Could not parse JSON from the AI response."


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("This starter prepares data and evidence only. Students add models, metrics, and dashboard improvements.")

with st.sidebar:
    st.header("Student info")
    student_name = st.text_input("Student name", value="Ahmed Al Omairi")
    student_id = st.text_input("Student ID", value="PG12S2540512")
    deployed_url = st.text_input("Streamlit deployed URL")
    repo_url = st.text_input("GitHub repo URL")
    project_title = st.text_input("Project title", value="Electricity Demand Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast future electricity demand using timestamp-based features and student-added models.",
    )
    dataset_path = st.text_input("Dataset path", value="data/dataset_sample.csv")

st.header("1. Load dataset + preview + audit")
try:
    raw_df = load_dataset(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.subheader("First 10 rows")
st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("Columns, dtypes, and missing values")
audit = audit_dataframe(raw_df)
st.dataframe(audit, use_container_width=True)

st.header("2. Timestamp and target selection")
cols = list(raw_df.columns)

default_timestamp_index = cols.index("timestamp") if "timestamp" in cols else 0
numeric_like = []
for col in cols:
    converted = pd.to_numeric(raw_df[col], errors="coerce")
    if converted.notna().mean() >= 0.5:
        numeric_like.append(col)

default_target = "electricity_demand_mw" if "electricity_demand_mw" in cols else (numeric_like[0] if numeric_like else cols[0])

timestamp_col = st.selectbox("Timestamp column", cols, index=default_timestamp_index)
target_col = st.selectbox("Target column", cols, index=cols.index(default_target))

ts_df = clean_time_series(raw_df, timestamp_col, target_col)

col1, col2, col3 = st.columns(3)
col1.metric("Clean rows", f"{len(ts_df):,}")
col2.metric("Start", str(ts_df[timestamp_col].min()) if len(ts_df) else "N/A")
col3.metric("End", str(ts_df[timestamp_col].max()) if len(ts_df) else "N/A")

if ts_df.empty:
    st.error("No usable rows after parsing timestamp and target.")
    st.stop()

st.header("3. Optional resampling + forecast horizon")
resampling_label_to_rule = {
    "No resampling": None,
    "Hourly mean": "h",
    "Daily mean": "D",
    "Weekly mean": "W",
    "Monthly mean": "MS",
}
resampling_choice = st.selectbox("Optional resampling", list(resampling_label_to_rule.keys()))
horizon = st.number_input("Forecast horizon, in rows after resampling", min_value=1, max_value=168, value=24, step=1)

prepared_df = resample_time_series(ts_df, timestamp_col, target_col, resampling_label_to_rule[resampling_choice])

st.subheader("Prepared time-series preview")
st.dataframe(prepared_df.head(10), use_container_width=True)

fig, ax = plt.subplots()
plot_df = prepared_df.tail(min(500, len(prepared_df)))
ax.plot(plot_df[timestamp_col], plot_df[target_col])
ax.set_title("Target over time")
ax.set_xlabel(timestamp_col)
ax.set_ylabel(target_col)
plt.xticks(rotation=30)
st.pyplot(fig)

st.header("4. Baseline feature table creation")
feature_df = make_baseline_feature_table(prepared_df, timestamp_col, target_col, int(horizon))

baseline_features = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
X = feature_df[baseline_features]
y = feature_df["y_target"]

st.write(f"Feature table rows: {len(feature_df):,}")
st.dataframe(feature_df.head(20), use_container_width=True)

st.info(
    "X and y are prepared from baseline features only. "
    "Students must add modeling, metrics, and extra visuals in the sections below."
)

st.header("5. STUDENT ADDITIONS — MODELING")
st.markdown("Add your model training, time-based split, predictions, and metrics table here.")
st.code(
    """
# Paste your modeling code below this marker.
# Required outcome for grading:
# results_df = pd.DataFrame([...]) with model names and metrics.
# Example columns: model, MAE, RMSE, MAPE
results_df = None
""",
    language="python",
)

# STUDENT ADDITIONS — MODELING START
results_df = None
# STUDENT ADDITIONS — MODELING END

st.header("6. STUDENT ADDITIONS — DASHBOARD")
st.markdown("Add extra plots, KPIs, interpretation, and dashboard improvements here.")
st.code(
    """
# Paste your dashboard code below this marker.
# Add visuals that explain your model results and forecast behavior.
""",
    language="python",
)

# STUDENT ADDITIONS — DASHBOARD START
# STUDENT ADDITIONS — DASHBOARD END

st.header("7. Export submission files")
missing_discussion = st.text_area(
    "Discuss missing timestamps / missing values",
    value="Dataset was parsed by timestamp and target; invalid timestamp/target rows were removed. Add more detail after checking gaps.",
)
outlier_discussion = st.text_area(
    "Discuss outliers",
    value="Initial target values were reviewed visually. Add your outlier method and findings here.",
)
resampling_discussion = st.text_area(
    "Discuss resampling",
    value=f"Selected resampling option: {resampling_choice}. Explain why this is appropriate for your forecast horizon.",
)
insights_text = st.text_area(
    "Insights and interpretation",
    value="Add final insights after adding models, metrics, and dashboard plots.",
)

submission = build_submission_json(
    student_name=student_name,
    student_id=student_id,
    deployed_url=deployed_url,
    repo_url=repo_url,
    project_title=project_title,
    project_goal=project_goal,
    timestamp_col=timestamp_col,
    target_col=target_col,
    horizon=int(horizon),
    resampling_rule=resampling_choice,
    raw_df=raw_df,
    ts_df=ts_df,
    feature_df=feature_df,
    results_df=results_df,
    insights_text=insights_text,
    missing_discussion=missing_discussion,
    outlier_discussion=outlier_discussion,
    resampling_discussion=resampling_discussion,
)

submission_json = json.dumps(submission, indent=2)
project_card_md = build_project_card(submission)

st.download_button(
    "Download submission.json",
    data=submission_json,
    file_name="submission.json",
    mime="application/json",
)

st.download_button(
    "Download project_card.md",
    data=project_card_md,
    file_name="project_card.md",
    mime="text/markdown",
)

with st.expander("Preview submission.json"):
    st.json(submission)

with st.expander("Preview project_card.md"):
    st.markdown(project_card_md)

st.header("8. AI grader (/80)")
st.warning(
    "Run the grader after adding your model, metrics table, dashboard visuals, and insights. "
    "The starter alone will receive a low score because results_df is None by default."
)

api_key = get_openrouter_api_key()
if st.button("Run AI grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key.")
    else:
        with st.spinner("Calling AI grader..."):
            try:
                raw_output = call_ai_grader(api_key, submission_json)
                parsed, parse_error = parse_ai_response(raw_output)
                if parsed is not None:
                    st.subheader("Parsed AI grade JSON")
                    st.json(parsed)
                else:
                    st.subheader("Raw AI grader output")
                    st.code(raw_output)
                    st.error(parse_error)
            except Exception as exc:
                st.error(f"AI grader failed: {exc}")
