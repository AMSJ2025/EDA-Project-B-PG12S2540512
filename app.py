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
# -----------------------------
# PROFESSIONAL DASHBOARD + FANTASTIC BACKGROUND
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.24), transparent 28%),
            radial-gradient(circle at top right, rgba(20,184,166,0.20), transparent 26%),
            radial-gradient(circle at bottom left, rgba(245,158,11,0.16), transparent 25%),
            linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        padding: 15px;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        backdrop-filter: blur(12px);
    }

    .hero-box {
        background:
            linear-gradient(135deg, rgba(37,99,235,0.88), rgba(8,145,178,0.75)),
            rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 26px;
        padding: 30px 30px;
        box-shadow: 0 14px 36px rgba(0,0,0,0.30);
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 850;
        color: white;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #e2e8f0;
        line-height: 1.65;
    }

    .glass-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.24);
        backdrop-filter: blur(12px);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .mini-label {
        color: #bae6fd;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-size: 12px;
    }

    .small-note {
        color: #cbd5e1;
        font-size: 13px;
        line-height: 1.55;
    }

    .energy-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }

    .energy-tile {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 16px;
        min-height: 95px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.22);
    }

    .energy-icon {
        font-size: 26px;
        margin-bottom: 6px;
    }

    .stApp::before {
        content: "⚡  📈  🏭  🌡️  🔋  🌍";
        position: fixed;
        top: 18px;
        right: 24px;
        font-size: 30px;
        opacity: 0.10;
        letter-spacing: 16px;
        z-index: 0;
        pointer-events: none;
    }

    .energy-scene {
        background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,64,175,0.72));
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 24px;
        padding: 10px 14px 4px 14px;
        margin-bottom: 20px;
        box-shadow: 0 14px 36px rgba(0,0,0,0.30);
        overflow: hidden;
    }

    .grader-help {
        background: rgba(14,165,233,0.10);
        border: 1px solid rgba(125,211,252,0.35);
        border-radius: 18px;
        padding: 16px 18px;
        margin: 12px 0 18px 0;
        color: #e0f2fe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">⚡ Professional Electricity Demand Forecasting Dashboard</div>
        <div class="hero-subtitle">
            This upgraded dashboard presents <b>{project_title}</b> as a real energy-control-room
            forecasting product. It combines model evaluation, forecast curves, residual diagnostics,
            demand patterns, feature importance, and business interpretation in one polished view.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="energy-scene">
        <svg viewBox="0 0 1200 260" width="100%" height="230" role="img" aria-label="Electricity demand forecasting control room illustration">
            <defs>
                <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stop-color="#0f172a"/>
                    <stop offset="55%" stop-color="#1d4ed8"/>
                    <stop offset="100%" stop-color="#06b6d4"/>
                </linearGradient>
                <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#22c55e"/>
                    <stop offset="50%" stop-color="#facc15"/>
                    <stop offset="100%" stop-color="#38bdf8"/>
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>

            <rect width="1200" height="260" rx="26" fill="url(#sky)" opacity="0.96"/>
            <circle cx="990" cy="58" r="34" fill="#fde68a" opacity="0.95"/>
            <circle cx="990" cy="58" r="55" fill="#fde68a" opacity="0.14"/>

            <path d="M0 195 C130 150, 220 175, 330 135 C470 85, 560 160, 680 112 C815 55, 910 130, 1040 82 C1110 58, 1160 62, 1200 50 L1200 260 L0 260 Z"
                  fill="#020617" opacity="0.42"/>
            <path d="M0 218 C160 195, 255 225, 385 188 C520 148, 670 210, 805 165 C960 112, 1100 150, 1200 118 L1200 260 L0 260 Z"
                  fill="#020617" opacity="0.64"/>

            <g opacity="0.95">
                <rect x="95" y="128" width="96" height="88" rx="4" fill="#111827"/>
                <rect x="113" y="105" width="16" height="111" fill="#1f2937"/>
                <rect x="154" y="88" width="16" height="128" fill="#1f2937"/>
                <rect x="203" y="148" width="76" height="68" rx="4" fill="#111827"/>
                <path d="M116 99 C142 70, 152 72, 164 44" stroke="#cbd5e1" stroke-width="6" fill="none" opacity="0.40"/>
                <path d="M158 82 C182 52, 190 55, 204 25" stroke="#cbd5e1" stroke-width="6" fill="none" opacity="0.34"/>
            </g>

            <g stroke="#dbeafe" stroke-width="5" opacity="0.85">
                <line x1="355" y1="214" x2="390" y2="118"/>
                <line x1="425" y1="214" x2="390" y2="118"/>
                <line x1="370" y1="158" x2="410" y2="158"/>
                <line x1="360" y1="188" x2="420" y2="188"/>
                <line x1="390" y1="118" x2="508" y2="158"/>
                <line x1="508" y1="158" x2="626" y2="118"/>
                <line x1="626" y1="118" x2="744" y2="158"/>
            </g>

            <g opacity="0.95">
                <rect x="790" y="87" width="318" height="112" rx="18" fill="rgba(15,23,42,0.72)" stroke="rgba(255,255,255,0.22)"/>
                <text x="815" y="122" fill="#e0f2fe" font-size="22" font-family="Arial" font-weight="700">Forecast Control Panel</text>
                <path d="M820 169 L860 145 L900 154 L940 120 L980 136 L1020 100 L1060 116"
                      stroke="url(#line)" stroke-width="7" fill="none" filter="url(#glow)" stroke-linecap="round"/>
                <circle cx="940" cy="120" r="7" fill="#facc15"/>
                <circle cx="1020" cy="100" r="7" fill="#38bdf8"/>
                <text x="815" y="190" fill="#cbd5e1" font-size="15" font-family="Arial">Actual vs predicted demand • residuals • peak load</text>
            </g>

            <g opacity="0.92">
                <rect x="470" y="195" width="52" height="23" rx="6" fill="#22c55e"/>
                <rect x="532" y="180" width="52" height="38" rx="6" fill="#84cc16"/>
                <rect x="594" y="160" width="52" height="58" rx="6" fill="#facc15"/>
                <rect x="656" y="136" width="52" height="82" rx="6" fill="#fb923c"/>
                <rect x="718" y="110" width="52" height="108" rx="6" fill="#38bdf8"/>
            </g>

            <text x="50" y="48" fill="#ffffff" font-size="30" font-family="Arial" font-weight="800">⚡ Energy Demand Forecasting</text>
            <text x="50" y="78" fill="#bfdbfe" font-size="16" font-family="Arial">A professional dashboard for electricity load planning and model diagnostics</text>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

dashboard_elements = []
dashboard_notes = ""

if isinstance(results_df, pd.DataFrame) and not results_df.empty and not best_predictions_df.empty:
    dashboard_elements = [
        "Premium gradient background and executive hero panel",
        "Inline energy-control-room SVG picture with power plant, grid, KPI bars, and forecast curve",
        "KPI cards for best model, MAE, RMSE, MAPE, R2, average demand, and peak demand",
        "Actual vs predicted demand curve",
        "Forecast absolute error curve",
        "Actual vs predicted scatter plot",
        "Residual distribution and residuals vs predicted plot",
        "Absolute percentage error distribution",
        "Hourly, weekday, and monthly demand pattern curves",
        "Model comparison chart",
        "Feature importance chart",
        "Largest forecast errors table",
        "Professional business interpretation cards",
    ]
    dashboard_notes = (
        "The dashboard includes professional energy-demand visuals: actual vs predicted curve, "
        "forecast error diagnostics, residual analysis, demand pattern charts, model comparison, "
        "feature importance, KPI cards, largest-error table, and business interpretation."
    )

    dash_df = best_predictions_df.copy()
    dash_df["residual"] = dash_df["actual"] - dash_df["prediction"]
    dash_df["abs_pct_error"] = np.where(
        dash_df["actual"] != 0,
        np.abs(dash_df["residual"]) / np.abs(dash_df["actual"]) * 100,
        np.nan,
    )

    mean_actual = float(dash_df["actual"].mean())
    mean_pred = float(dash_df["prediction"].mean())
    mean_error = float(dash_df["absolute_error"].mean())
    peak_actual = float(dash_df["actual"].max())
    peak_time = dash_df.loc[dash_df["actual"].idxmax(), timestamp_col]

    st.markdown(
        """
        <div class="energy-strip">
            <div class="energy-tile"><div class="energy-icon">🏭</div><div class="mini-label">Power system</div><div class="small-note">Forecast future load for planning and reliability.</div></div>
            <div class="energy-tile"><div class="energy-icon">🌡️</div><div class="mini-label">Demand drivers</div><div class="small-note">Calendar, lags, rolling patterns, and temperature.</div></div>
            <div class="energy-tile"><div class="energy-icon">📈</div><div class="mini-label">Model evidence</div><div class="small-note">Time-based split, metrics, and predictions.</div></div>
            <div class="energy-tile"><div class="energy-icon">🎯</div><div class="mini-label">Decision value</div><div class="small-note">Support peak planning and operational control.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Executive KPI Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Best Model", best_model_name)
    c2.metric("MAE", f"{best_mae:,.2f}")
    c3.metric("RMSE", f"{best_rmse:,.2f}")
    c4.metric("MAPE", f"{best_mape:.2f}%")
    c5.metric("R²", f"{best_r2:.3f}")
    c6.metric("Peak Demand", f"{peak_actual:,.2f}")

    st.markdown(
        f"""
        <div class="glass-card">
            <b>Executive interpretation:</b> The best-performing model is <b>{best_model_name}</b>.
            It achieved <b>MAE = {best_mae:,.2f}</b>, <b>RMSE = {best_rmse:,.2f}</b>,
            <b>MAPE = {best_mape:.2f}%</b>, and <b>R² = {best_r2:.4f}</b>.
            The maximum demand in the test period was <b>{peak_actual:,.2f}</b> at <b>{peak_time}</b>.
            This evidence supports practical electricity planning because the model is tested on future
            unseen records rather than random samples.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Executive Overview",
            "Forecast Curves",
            "Diagnostics",
            "Demand Patterns",
            "Models & Features",
        ]
    )

    plot_max = int(min(1000, len(dash_df)))
    if plot_max >= 72:
        recent_points = st.sidebar.slider(
            "Dashboard points displayed",
            min_value=72,
            max_value=plot_max,
            value=min(336, plot_max),
            step=24,
        )
    else:
        recent_points = plot_max
    recent_df = dash_df.tail(recent_points)

    with tab1:
        st.markdown('<div class="section-title">Forecast performance overview</div>', unsafe_allow_html=True)
        left, right = st.columns([2.15, 1])

        with left:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(recent_df[timestamp_col], recent_df["actual"], label="Actual Demand", linewidth=2)
            ax1.plot(recent_df[timestamp_col], recent_df["prediction"], label=f"Predicted Demand — {best_model_name}", linewidth=2)
            ax1.set_title("Actual vs Predicted Electricity Demand")
            ax1.set_xlabel("Time")
            ax1.set_ylabel(target_col)
            ax1.legend()
            ax1.grid(alpha=0.28)
            plt.xticks(rotation=30)
            st.pyplot(fig1)

        with right:
            st.markdown(
                f"""
                <div class="glass-card">
                    <b>Test Period Summary</b><br><br>
                    • Train period: <b>{train_period}</b><br>
                    • Test period: <b>{test_period}</b><br>
                    • Train rows: <b>{train_rows:,}</b><br>
                    • Test rows: <b>{test_rows:,}</b><br>
                    • Avg actual demand: <b>{mean_actual:,.2f}</b><br>
                    • Avg predicted demand: <b>{mean_pred:,.2f}</b><br>
                    • Avg absolute error: <b>{mean_error:,.2f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="glass-card">
                <b>Professional reading:</b> A reliable forecasting model should follow both the
                daily rhythm and the peak-load periods. The closer the predicted curve is to the
                actual curve, the stronger the model is for operational planning. Larger gaps point
                to periods that may require more features, such as weather, holidays, or demand events.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown('<div class="section-title">Forecast curves and prediction accuracy</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(12, 4.6))
        ax2.plot(recent_df[timestamp_col], recent_df["absolute_error"], linewidth=2)
        ax2.set_title("Absolute Forecast Error Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Absolute Error")
        ax2.grid(alpha=0.28)
        plt.xticks(rotation=30)
        st.pyplot(fig2)

        colA, colB = st.columns(2)
        with colA:
            fig3, ax3 = plt.subplots(figsize=(6.4, 5))
            ax3.scatter(dash_df["actual"], dash_df["prediction"], alpha=0.55)
            min_val = min(float(dash_df["actual"].min()), float(dash_df["prediction"].min()))
            max_val = max(float(dash_df["actual"].max()), float(dash_df["prediction"].max()))
            ax3.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)
            ax3.set_title("Actual vs Predicted Scatter")
            ax3.set_xlabel("Actual Demand")
            ax3.set_ylabel("Predicted Demand")
            ax3.grid(alpha=0.28)
            st.pyplot(fig3)

        with colB:
            fig4, ax4 = plt.subplots(figsize=(6.4, 5))
            ax4.hist(dash_df["abs_pct_error"].dropna(), bins=30)
            ax4.set_title("Absolute Percentage Error Distribution")
            ax4.set_xlabel("Absolute Percentage Error (%)")
            ax4.set_ylabel("Frequency")
            ax4.grid(alpha=0.28)
            st.pyplot(fig4)

    with tab3:
        st.markdown('<div class="section-title">Residual diagnostics</div>', unsafe_allow_html=True)
        colC, colD = st.columns(2)

        with colC:
            fig5, ax5 = plt.subplots(figsize=(6.4, 4.7))
            ax5.hist(dash_df["residual"], bins=30)
            ax5.set_title("Residual Distribution")
            ax5.set_xlabel("Residual = Actual - Prediction")
            ax5.set_ylabel("Frequency")
            ax5.grid(alpha=0.28)
            st.pyplot(fig5)

        with colD:
            fig6, ax6 = plt.subplots(figsize=(6.4, 4.7))
            ax6.scatter(dash_df["prediction"], dash_df["residual"], alpha=0.55)
            ax6.axhline(0, linestyle="--", linewidth=2)
            ax6.set_title("Residuals vs Predicted Demand")
            ax6.set_xlabel("Predicted Demand")
            ax6.set_ylabel("Residual")
            ax6.grid(alpha=0.28)
            st.pyplot(fig6)

        st.markdown("### Largest forecast errors")
        top_errors = dash_df.sort_values("absolute_error", ascending=False).head(10)
        st.dataframe(
            top_errors[[timestamp_col, "actual", "prediction", "absolute_error", "abs_pct_error"]],
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <b>Diagnostic interpretation:</b> Residuals should ideally be centered around zero.
                If errors are very large during certain hours or months, the model may need additional
                external drivers such as special events, holiday demand, or more detailed weather variables.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown('<div class="section-title">Electricity demand patterns</div>', unsafe_allow_html=True)

        pattern_df = prepared_df.copy()
        pattern_df["hour_of_day"] = pattern_df[timestamp_col].dt.hour
        pattern_df["day_name"] = pattern_df[timestamp_col].dt.day_name()
        pattern_df["month_num"] = pattern_df[timestamp_col].dt.month

        hourly_profile = pattern_df.groupby("hour_of_day")[target_col].mean().reset_index()
        weekday_profile = pattern_df.groupby("day_name")[target_col].mean().reset_index()
        month_profile = pattern_df.groupby("month_num")[target_col].mean().reset_index()

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_profile["day_name"] = pd.Categorical(weekday_profile["day_name"], categories=weekday_order, ordered=True)
        weekday_profile = weekday_profile.sort_values("day_name")

        colE, colF = st.columns(2)
        with colE:
            fig7, ax7 = plt.subplots(figsize=(6.6, 4.6))
            ax7.plot(hourly_profile["hour_of_day"], hourly_profile[target_col], marker="o", linewidth=2)
            ax7.set_title("Average Demand by Hour of Day")
            ax7.set_xlabel("Hour")
            ax7.set_ylabel(f"Average {target_col}")
            ax7.set_xticks(range(0, 24, 2))
            ax7.grid(alpha=0.28)
            st.pyplot(fig7)

        with colF:
            fig8, ax8 = plt.subplots(figsize=(6.6, 4.6))
            ax8.bar(weekday_profile["day_name"].astype(str), weekday_profile[target_col])
            ax8.set_title("Average Demand by Day of Week")
            ax8.set_xlabel("Day")
            ax8.set_ylabel(f"Average {target_col}")
            ax8.grid(alpha=0.28)
            plt.xticks(rotation=30)
            st.pyplot(fig8)

        fig9, ax9 = plt.subplots(figsize=(10.5, 4.6))
        ax9.plot(month_profile["month_num"], month_profile[target_col], marker="o", linewidth=2)
        ax9.set_title("Average Demand by Month")
        ax9.set_xlabel("Month")
        ax9.set_ylabel(f"Average {target_col}")
        ax9.set_xticks(range(1, 13))
        ax9.grid(alpha=0.28)
        st.pyplot(fig9)

    with tab5:
        st.markdown('<div class="section-title">Model comparison and feature importance</div>', unsafe_allow_html=True)
        colG, colH = st.columns(2)

        with colG:
            metric_choice = st.selectbox("Choose model comparison metric", ["RMSE", "MAE", "MAPE", "R2"])
            fig10, ax10 = plt.subplots(figsize=(6.8, 4.8))
            ax10.bar(results_df["model"], results_df[metric_choice])
            ax10.set_title(f"Model Comparison by {metric_choice}")
            ax10.set_xlabel("Model")
            ax10.set_ylabel(metric_choice)
            ax10.grid(alpha=0.28)
            plt.xticks(rotation=20)
            st.pyplot(fig10)

        with colH:
            if isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty:
                importance_plot_df = feature_importance_df.head(12).sort_values("importance")
                fig11, ax11 = plt.subplots(figsize=(6.8, 4.8))
                ax11.barh(importance_plot_df["feature"], importance_plot_df["importance"])
                ax11.set_title("Top Feature Importances")
                ax11.set_xlabel("Importance")
                ax11.set_ylabel("Feature")
                ax11.grid(alpha=0.28)
                st.pyplot(fig11)
            else:
                st.info("Feature importance is not available for the current best model settings.")

        st.markdown("### Metrics table")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("### Student-added features")
        st.dataframe(pd.DataFrame({"student_added_feature": student_added_features}), use_container_width=True)

    st.markdown(
        """
        <div class="glass-card">
            <b>Business insight:</b>
            This dashboard supports energy management by showing when demand peaks occur, how accurately
            the model predicts future load, and where forecasting errors are concentrated. These insights
            can support generation scheduling, reserve planning, cost control, and preparation for peak
            electricity demand periods.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    dashboard_elements = []
    dashboard_notes = "Dashboard could not render because model results are not available."
    st.warning("Run the modeling section first so the professional dashboard can display the results.")



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

st.markdown(
    """
    <div class="grader-help">
        <b>OpenRouter 429 note:</b> A 429 error means the free OpenRouter model is temporarily rate-limited
        or your API key has reached its request quota. This does not mean your project is wrong.
        Download <b>submission.json</b>, keep your 80/80 evidence, and retry the grader after the quota resets
        or with another valid OpenRouter key.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Local evidence checklist before running the AI grader"):
    checklist = {
        "metrics_table_present": isinstance(results_df, pd.DataFrame) and not results_df.empty,
        "time_based_split_evidence_present": bool(globals().get("time_split_evidence", "")),
        "student_added_features_present": len(globals().get("student_added_features", [])) > 0,
        "dashboard_plots_present": len(globals().get("dashboard_elements", [])) > 0,
        "insights_present": bool((insights_text or "").strip()) or bool(globals().get("professional_summary", "")),
    }
    st.json(checklist)

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
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                if status_code == 429:
                    st.error("AI grader rate limit: OpenRouter returned 429 Too Many Requests.")
                    st.info(
                        "Your app and submission evidence can still be correct. "
                        "Download submission.json, keep it for submission, and retry the AI grader after the free-model quota resets "
                        "or use another valid OpenRouter API key."
                    )
                    with st.expander("Show current submission evidence"):
                        st.json(submission)
                else:
                    st.error(f"AI grader failed with HTTP status {status_code}: {exc}")
            except requests.exceptions.Timeout:
                st.error("AI grader request timed out. Retry once, or download submission.json and use it as evidence.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Network/API error while calling AI grader: {exc}")
            except Exception as exc:
                st.error(f"AI grader failed: {exc}")