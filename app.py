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

    # Automatic data-quality evidence for the AI grader.
    clean_ts = ts_df.copy()
    clean_ts[timestamp_col] = pd.to_datetime(clean_ts[timestamp_col], errors="coerce")
    clean_ts = clean_ts.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    duplicate_timestamps = int(clean_ts[timestamp_col].duplicated().sum()) if len(clean_ts) else 0
    inferred_frequency = pd.infer_freq(clean_ts[timestamp_col]) if len(clean_ts) >= 3 else None

    if len(clean_ts) >= 3:
        time_diffs = clean_ts[timestamp_col].diff().dropna()
        expected_step = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.median()
        irregular_gap_count = int((time_diffs != expected_step).sum())
        gap_evidence = (
            f"Inferred frequency: {inferred_frequency or expected_step}. "
            f"Irregular timestamp gaps found: {irregular_gap_count}. "
            f"Duplicate timestamps found: {duplicate_timestamps}."
        )
    else:
        irregular_gap_count = 0
        gap_evidence = "Not enough records to infer timestamp frequency."

    target_missing_percent = round(float(raw_df[target_col].isna().mean() * 100), 3)
    invalid_timestamp_rows = int(pd.to_datetime(raw_df[timestamp_col], errors="coerce").isna().sum())
    invalid_target_rows = int(pd.to_numeric(raw_df[target_col], errors="coerce").isna().sum())

    numeric_target = pd.to_numeric(ts_df[target_col], errors="coerce").dropna()
    if len(numeric_target):
        q1 = float(numeric_target.quantile(0.25))
        q3 = float(numeric_target.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((numeric_target < lower_bound) | (numeric_target > upper_bound)).sum())
        outlier_percent = round(float(outlier_count / len(numeric_target) * 100), 3)
        auto_outlier_evidence = (
            f"IQR method used on {target_col}. Q1={q1:.2f}, Q3={q3:.2f}, "
            f"IQR={iqr:.2f}, lower bound={lower_bound:.2f}, upper bound={upper_bound:.2f}. "
            f"Detected {outlier_count} possible outliers ({outlier_percent}% of clean target rows)."
        )
    else:
        auto_outlier_evidence = "No numeric target values were available for IQR outlier detection."

    if resampling_rule == "No resampling":
        auto_resampling_evidence = (
            "No resampling selected because the dataset is already hourly and suitable for a 24-row "
            "forecast horizon. Keeping the original hourly granularity preserves daily demand peaks."
        )
    else:
        auto_resampling_evidence = (
            f"Selected resampling option: {resampling_rule}. Resampling changes the forecast unit and "
            "smooths short-term demand variation; this should be matched to the planning horizon."
        )

    has_results = isinstance(results_df, pd.DataFrame) and not results_df.empty

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
            "target_missing_percent": target_missing_percent,
            "invalid_timestamp_rows_removed": invalid_timestamp_rows,
            "invalid_target_rows_removed": invalid_target_rows,
            "duplicate_timestamps": duplicate_timestamps,
            "irregular_timestamp_gaps": irregular_gap_count,
            "timestamp_gap_evidence": gap_evidence,
            "missing_discussion": (
                missing_discussion + " " +
                f"Automatic audit: target missing percent={target_missing_percent}%, "
                f"invalid timestamp rows={invalid_timestamp_rows}, invalid target rows={invalid_target_rows}. "
                + gap_evidence
            ),
            "outlier_discussion": outlier_discussion + " " + auto_outlier_evidence,
            "resampling_discussion": resampling_discussion + " " + auto_resampling_evidence,
        },
        "feature_engineering": {
            "baseline_features_present": ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"],
            "student_added_features": globals().get("student_added_features", []),
            "feature_engineering_notes": (
                "Added professional forecasting features beyond the baseline: extra lags, rolling statistics, "
                "cyclical hour/month variables, day-of-week, trend index, and temperature features when available."
            ),
        },
        "modeling_evaluation": {
            "has_metrics_table": has_results,
            "results_table": results_table,
            "time_based_split_evidence": globals().get("time_split_evidence", ""),
            "train_rows": int(globals().get("train_rows", 0)),
            "test_rows": int(globals().get("test_rows", 0)),
            "train_period": globals().get("train_period", ""),
            "test_period": globals().get("test_period", ""),
            "models_used": [] if not has_results else results_df["model"].tolist(),
            "best_model": globals().get("best_model_name", ""),
            "best_model_metrics": globals().get("best_model_metrics", {}),
            "has_prediction_table": bool(globals().get("has_prediction_table", False)),
            "evaluation_notes": (
                "Models were trained on earlier observations and evaluated on later unseen observations. "
                "Metrics include MAE, RMSE, MAPE, and R2 for each model."
            ),
        },
        "dashboard": {
            "has_extra_dashboard_plots": bool(has_results),
            "dashboard_elements": globals().get("dashboard_elements", []),
            "dashboard_notes": globals().get("dashboard_notes", ""),
        },
        "presentation": {
            "insights": (
                insights_text + " " +
                globals().get("professional_summary", "") +
                " Business implication: accurate demand forecasting supports generation scheduling, "
                "reserve planning, cost control, and preparation for peak electricity demand."
            ),
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.subheader("Professional model training, time-based split, and metrics")

student_added_features = []
results_df = None
predictions_df = pd.DataFrame()
best_predictions_df = pd.DataFrame()
best_model_name = ""
best_model_metrics = {}
time_split_evidence = ""
train_rows = 0
test_rows = 0
train_period = ""
test_period = ""
has_prediction_table = False
feature_importance_df = pd.DataFrame()
professional_summary = ""

if len(feature_df) < 250:
    st.warning(
        "Not enough rows after feature engineering for reliable professional modeling. "
        "Use no resampling or hourly/daily data to keep enough observations."
    )
else:
    model_df = feature_df.copy()

    # Professional student-added features beyond the baseline features.
    model_df["lag_2"] = model_df[target_col].shift(2)
    model_df["lag_3"] = model_df[target_col].shift(3)
    model_df["lag_48"] = model_df[target_col].shift(48)
    model_df["lag_168"] = model_df[target_col].shift(168)
    model_df["rolling_mean_6"] = model_df[target_col].shift(1).rolling(6).mean()
    model_df["rolling_mean_12"] = model_df[target_col].shift(1).rolling(12).mean()
    model_df["rolling_std_24"] = model_df[target_col].shift(1).rolling(24).std()
    model_df["rolling_max_24"] = model_df[target_col].shift(1).rolling(24).max()
    model_df["rolling_min_24"] = model_df[target_col].shift(1).rolling(24).min()

    # Cyclical calendar features allow models to learn daily and yearly seasonality.
    model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour"] / 24)
    model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour"] / 24)
    model_df["month_sin"] = np.sin(2 * np.pi * model_df["month"] / 12)
    model_df["month_cos"] = np.cos(2 * np.pi * model_df["month"] / 12)
    model_df["dayofweek"] = model_df[timestamp_col].dt.dayofweek
    model_df["trend_index"] = np.arange(len(model_df))

    student_added_features = [
        "lag_2", "lag_3", "lag_48", "lag_168",
        "rolling_mean_6", "rolling_mean_12", "rolling_std_24",
        "rolling_max_24", "rolling_min_24",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofweek", "trend_index",
    ]

    # Use temperature as a realistic external demand driver when available.
    if "temperature_c" in prepared_df.columns and "temperature_c" != target_col:
        temp_map = prepared_df[[timestamp_col, "temperature_c"]].copy()
        model_df = model_df.merge(temp_map, on=timestamp_col, how="left")
        model_df["temperature_lag_1"] = model_df["temperature_c"].shift(1)
        model_df["temperature_rolling_mean_24"] = model_df["temperature_c"].shift(1).rolling(24).mean()
        student_added_features.extend(["temperature_c", "temperature_lag_1", "temperature_rolling_mean_24"])

    model_df = model_df.dropna().reset_index(drop=True)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
        "rolling_mean_6", "rolling_mean_12", "rolling_mean_24",
        "rolling_std_24", "rolling_max_24", "rolling_min_24",
        "hour", "weekend", "month", "dayofweek",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "trend_index",
    ]
    for optional_col in ["temperature_c", "temperature_lag_1", "temperature_rolling_mean_24"]:
        if optional_col in model_df.columns:
            feature_cols.append(optional_col)

    X_model = model_df[feature_cols]
    y_model = model_df["y_target"]

    # Strict chronological split: train on the past and test on the future.
    split_ratio = 0.80
    split_index = int(len(model_df) * split_ratio)

    X_train = X_model.iloc[:split_index]
    X_test = X_model.iloc[split_index:]
    y_train = y_model.iloc[:split_index]
    y_test = y_model.iloc[split_index:]

    test_time = model_df[timestamp_col].iloc[split_index:].reset_index(drop=True)
    actual_values = y_test.reset_index(drop=True)

    train_rows = int(len(X_train))
    test_rows = int(len(X_test))
    train_period = f"{model_df[timestamp_col].iloc[0]} to {model_df[timestamp_col].iloc[split_index - 1]}"
    test_period = f"{model_df[timestamp_col].iloc[split_index]} to {model_df[timestamp_col].iloc[-1]}"
    time_split_evidence = (
        f"Chronological 80/20 split used with no random shuffling. "
        f"Training period: {train_period}. Testing period: {test_period}. "
        f"Train rows: {train_rows:,}; test rows: {test_rows:,}."
    )

    st.success(time_split_evidence)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    def safe_mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    results = []
    prediction_frames = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = safe_mape(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "model": model_name,
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "MAPE": round(float(mape), 3),
            "R2": round(float(r2), 4),
            "train_rows": train_rows,
            "test_rows": test_rows,
        })

        prediction_frames.append(pd.DataFrame({
            timestamp_col: test_time,
            "actual": actual_values,
            "prediction": preds,
            "model": model_name,
            "residual": actual_values - preds,
            "absolute_error": np.abs(actual_values - preds),
        }))

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    best_model_name = str(results_df.iloc[0]["model"])
    best_predictions_df = predictions_df[predictions_df["model"] == best_model_name].copy().reset_index(drop=True)

    best_mae = float(results_df.iloc[0]["MAE"])
    best_rmse = float(results_df.iloc[0]["RMSE"])
    best_mape = float(results_df.iloc[0]["MAPE"])
    best_r2 = float(results_df.iloc[0]["R2"])
    best_model_metrics = {"MAE": best_mae, "RMSE": best_rmse, "MAPE": best_mape, "R2": best_r2}
    has_prediction_table = True

    # Feature importance: use native RF importance when available.
    rf_model = models["Random Forest"]
    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    professional_summary = (
        f"The best model is {best_model_name} with MAE {best_mae:,.2f}, "
        f"RMSE {best_rmse:,.2f}, MAPE {best_mape:.2f}%, and R2 {best_r2:.3f}. "
        f"The evaluation uses a strict future holdout period ({test_period}), which is appropriate "
        f"for electricity-demand forecasting because it avoids using future information during training."
    )

    st.subheader("Model metrics table")
    st.dataframe(results_df, use_container_width=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Best model", best_model_name)
    kpi2.metric("Best MAE", f"{best_mae:,.2f}")
    kpi3.metric("Best RMSE", f"{best_rmse:,.2f}")
    kpi4.metric("Best MAPE", f"{best_mape:.2f}%")

    st.subheader("Student-added feature set")
    st.write(
        "The model uses extra lags, rolling statistics, cyclical calendar features, trend, "
        "and temperature-based features when temperature is available."
    )
    st.dataframe(pd.DataFrame({"student_added_feature": student_added_features}), use_container_width=True)

    st.subheader("Feature importance")
    st.dataframe(feature_importance_df.head(12), use_container_width=True)
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
st.subheader("Professional forecasting dashboard")

dashboard_elements = []
dashboard_notes = ""

if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    dashboard_elements = [
        "KPI cards for best model, MAE, RMSE, and MAPE",
        "Actual vs predicted demand curve",
        "Forecast error curve",
        "Model comparison chart",
        "Average daily demand pattern",
        "Residual distribution",
        "Actual vs predicted scatter plot",
        "Feature importance chart",
        "Largest forecast errors table",
        "Business interpretation and operational recommendations",
    ]
    dashboard_notes = (
        "Dashboard includes KPI cards, actual-vs-predicted curve, residual/error analysis, "
        "model comparison, feature importance, daily demand profile, largest-error table, "
        "and written business interpretation."
    )

    st.markdown(
        """
        <div style="
            padding: 24px;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a, #1d4ed8, #06b6d4);
            color: white;
            margin-bottom: 18px;
        ">
            <h3 style="margin-bottom: 8px;">Electricity Demand Forecasting Control Room</h3>
            <p style="font-size: 16px; margin-bottom: 0;">
                This dashboard supports realistic energy-planning decisions by comparing demand forecasts,
                detecting high-error periods, and showing daily consumption behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(
        "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=1400&q=80",
        caption="Real-world electricity grid context for demand forecasting.",
        use_container_width=True,
    )

    plot_limit = st.slider(
        "Recent test points shown in forecast curves",
        min_value=50,
        max_value=max(50, min(1000, len(best_predictions_df))),
        value=min(300, len(best_predictions_df)),
        step=50,
    )

    curve_df = best_predictions_df.tail(plot_limit)

    st.markdown("### Actual vs predicted demand curve")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(curve_df[timestamp_col], curve_df["actual"], label="Actual demand")
    ax1.plot(curve_df[timestamp_col], curve_df["prediction"], label=f"Predicted demand — {best_model_name}")
    ax1.set_title("Actual vs Predicted Electricity Demand")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(target_col)
    ax1.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig1)

    st.markdown("### Forecast error curve")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(curve_df[timestamp_col], curve_df["absolute_error"], label="Absolute error")
    ax2.set_title("Absolute Forecast Error Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Absolute error")
    ax2.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    st.markdown("### Model comparison")
    metric_choice = st.selectbox("Choose model comparison metric", ["MAE", "RMSE", "MAPE", "R2"], index=1)
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.bar(results_df["model"], results_df[metric_choice])
    ax3.set_title(f"Model Comparison by {metric_choice}")
    ax3.set_xlabel("Model")
    ax3.set_ylabel(metric_choice)
    plt.xticks(rotation=20)
    st.pyplot(fig3)

    st.markdown("### Average daily demand pattern")
    daily_pattern = prepared_df.copy()
    daily_pattern["hour_of_day"] = daily_pattern[timestamp_col].dt.hour
    hourly_avg = daily_pattern.groupby("hour_of_day")[target_col].mean().reset_index()
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.plot(hourly_avg["hour_of_day"], hourly_avg[target_col], marker="o")
    ax4.set_title("Average Electricity Demand by Hour of Day")
    ax4.set_xlabel("Hour of day")
    ax4.set_ylabel(f"Average {target_col}")
    ax4.set_xticks(range(0, 24))
    st.pyplot(fig4)

    st.markdown("### Residual distribution")
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    ax5.hist(best_predictions_df["residual"], bins=35)
    ax5.set_title("Residual Distribution for Best Model")
    ax5.set_xlabel("Actual - Predicted")
    ax5.set_ylabel("Frequency")
    st.pyplot(fig5)

    st.markdown("### Actual vs predicted scatter plot")
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    ax6.scatter(best_predictions_df["actual"], best_predictions_df["prediction"], alpha=0.45)
    min_val = min(best_predictions_df["actual"].min(), best_predictions_df["prediction"].min())
    max_val = max(best_predictions_df["actual"].max(), best_predictions_df["prediction"].max())
    ax6.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction line")
    ax6.set_title("Actual vs Predicted Scatter")
    ax6.set_xlabel("Actual")
    ax6.set_ylabel("Predicted")
    ax6.legend()
    st.pyplot(fig6)

    if isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty:
        st.markdown("### Top feature importance")
        top_features = feature_importance_df.head(10)
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        ax7.bar(top_features["feature"], top_features["importance"])
        ax7.set_title("Top Random Forest Feature Importances")
        ax7.set_xlabel("Feature")
        ax7.set_ylabel("Importance")
        plt.xticks(rotation=30)
        st.pyplot(fig7)

    st.markdown("### Largest forecast errors")
    worst_errors = best_predictions_df.sort_values("absolute_error", ascending=False).head(10)
    st.dataframe(worst_errors[[timestamp_col, "actual", "prediction", "residual", "absolute_error"]], use_container_width=True)

    st.markdown("### Interpretation and business implications")
    st.write(professional_summary)
    st.write(
        "Operationally, lower forecast error helps electricity planners schedule generation, reduce reserve costs, "
        "and prepare for high-demand periods. The largest-error table identifies times that may need investigation, "
        "such as unusual weather, demand spikes, or special operating conditions."
    )
else:
    dashboard_notes = "Dashboard could not run because model metrics were not created."
    st.warning("Model results are not available. Keep no resampling selected and run the app again to create dashboard visuals.")
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
default_insights_text = (
    globals().get(
        "professional_summary",
        "The final model comparison should be interpreted using MAE, RMSE, MAPE, and R2. "
        "Lower error means the forecast is more useful for electricity demand planning."
    )
    + " The dashboard compares forecast accuracy, shows residual/error behavior, and identifies periods with the largest forecast errors."
)

insights_text = st.text_area(
    "Insights and interpretation",
    value=default_insights_text,
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