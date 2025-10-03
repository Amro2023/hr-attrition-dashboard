import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------- CONFIG ----------
DEFAULT_CSV_PATH = "/Users/amroosman/Downloads/hr_data.csv"  # fallback if no upload
st.set_page_config(page_title="HR Analytics + Attrition Risk", layout="wide")

# ---------- HELPERS ----------
def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_hr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        pick_col(df, ["satisfaction_level", "satisfaction"]): "satisfaction_level",
        pick_col(df, ["last_evaluation", "evaluation"]): "last_evaluation",
        pick_col(df, ["number_project", "projects"]): "number_project",
        pick_col(df, ["average_montly_hours", "average_monthly_hours", "avg_monthly_hours"]): "average_montly_hours",
        pick_col(df, ["time_spend_company", "tenure_years"]): "time_spend_company",
        pick_col(df, ["Work_accident", "work_accident", "accident"]): "Work_accident",
        pick_col(df, ["promotion_last_5years", "promoted_last_5_years", "promotion_5y"]): "promotion_last_5years",
        pick_col(df, ["left", "attrition", "churn"]): "left",
        pick_col(df, ["department", "dept", "sales"]): "department",
        pick_col(df, ["salary", "pay_band", "comp_band"]): "salary",
    }
    mapping = {k: v for k, v in mapping.items() if k is not None}
    return df.rename(columns=mapping)

def confusion_matrix_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    z = cm.astype(int)
    x = ["Pred Stay", "Pred Leave"]
    y = ["True Stay", "True Leave"]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Blues", showscale=True)
    fig.update_layout(title="Confusion Matrix")
    return fig

def roc_curve_fig(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def pr_curve_fig(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
    fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig

# ---------- DATA SOURCE (upload OR default path) ----------
st.sidebar.header("Data")

DEFAULT_CSV_PATH = "sample_hr_data.csv"  # file you just added to the repo

uploaded = st.sidebar.file_uploader("Upload HR CSV", type=["csv"])
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success("Using uploaded file.")
else:
    try:
        df_raw = pd.read_csv(DEFAULT_CSV_PATH)
        st.sidebar.info(f"Using default CSV: {DEFAULT_CSV_PATH}")
    except FileNotFoundError:
        st.error("No CSV available. Upload an HR CSV or add a sample CSV to the repo.")
        st.stop()


# ---------- SIDEBAR FILTERS ----------
dept_col  = "department" if "department" in df.columns else None
sal_col   = "salary" if "salary" in df.columns else None
promo_col = "promotion_last_5years" if "promotion_last_5years" in df.columns else None
acc_col   = "Work_accident" if "Work_accident" in df.columns else None

if dept_col:
    sel_dept = st.sidebar.selectbox("Department", ["All"] + sorted(df[dept_col].dropna().unique()))
else:
    sel_dept = "All"

if sal_col:
    sel_salary = st.sidebar.selectbox("Salary Band", ["All"] + sorted(df[sal_col].dropna().unique()))
else:
    sel_salary = "All"

sat_min, sat_max = float(df["satisfaction_level"].min()), float(df["satisfaction_level"].max())
sat_range = st.sidebar.slider("Satisfaction", sat_min, sat_max, (sat_min, sat_max), step=0.01)

hrs_min, hrs_max = int(df["average_montly_hours"].min()), int(df["average_montly_hours"].max())
hrs_range = st.sidebar.slider("Monthly hours", hrs_min, hrs_max, (hrs_min, hrs_max), step=1)

ten_min, ten_max = int(df["time_spend_company"].min()), int(df["time_spend_company"].max())
ten_range = st.sidebar.slider("Tenure (years)", ten_min, ten_max, (ten_min, ten_max), step=1)

promo_filter = st.sidebar.multiselect("Promotion (last 5y)", [0,1], default=[0,1]) if promo_col else [0,1]
acc_filter   = st.sidebar.multiselect("Work accident", [0,1], default=[0,1]) if acc_col else [0,1]

# apply filters
filtered = df.copy()
if dept_col and sel_dept != "All":
    filtered = filtered[filtered[dept_col] == sel_dept]
if sal_col and sel_salary != "All":
    filtered = filtered[filtered[sal_col] == sel_salary]
filtered = filtered[
    (filtered["satisfaction_level"].between(*sat_range)) &
    (filtered["average_montly_hours"].between(*hrs_range)) &
    (filtered["time_spend_company"].between(*ten_range))
]
if promo_col:
    filtered = filtered[filtered[promo_col].isin(promo_filter)]
if acc_col:
    filtered = filtered[filtered[acc_col].isin(acc_filter)]

# ---------- HEADER ----------
st.title("Alpha Manufacturing • HR Analytics & Attrition Risk")

# ---------- KPIs ----------
st.subheader("Key Metrics (Filtered)")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Attrition Rate", f"{filtered['left'].mean()*100:.1f}%")
k2.metric("Avg Satisfaction", f"{filtered['satisfaction_level'].mean():.2f}")
k3.metric("Avg Monthly Hours", f"{filtered['average_montly_hours'].mean():.1f}")
k4.metric("Avg Tenure (yrs)", f"{filtered['time_spend_company'].mean():.1f}")

# ---------- EXPLORATORY VISUALS ----------
c1,c2 = st.columns(2)
with c1:
    st.subheader("Attrition by Tenure")
    fig1 = px.histogram(filtered, x="time_spend_company",
                        color=filtered["left"].map({0:"Stayed",1:"Left"}),
                        barmode="group", title="By Years at Company")
    st.plotly_chart(fig1, use_container_width=True)
with c2:
    st.subheader("Hours vs Satisfaction by Attrition")
    fig2 = px.scatter(filtered, x="average_montly_hours", y="satisfaction_level",
                      color=filtered["left"].map({0:"Stayed",1:"Left"}),
                      title="Workload vs Satisfaction")
    st.plotly_chart(fig2, use_container_width=True)

c3,c4 = st.columns(2)
with c3:
    if promo_col:
        st.subheader("Attrition by Promotion")
        fig3 = px.histogram(filtered, x=promo_col,
                            color=filtered["left"].map({0:"Stayed",1:"Left"}),
                            barmode="group", title="Promotion Flag (0/1)")
        st.plotly_chart(fig3, use_container_width=True)
with c4:
    if dept_col:
        st.subheader("Attrition Rate by Department")
        dept_rates = filtered.groupby(dept_col)["left"].mean().sort_values()*100
        st.plotly_chart(px.bar(dept_rates, title="Percent Leaving by Department"), use_container_width=True)

# ---------- MODEL & RISK SCORING ----------
st.markdown("---")
st.header("Predictive Model & Risk Scoring")

clf_choice = st.sidebar.radio("Classifier", ["Random Forest", "Logistic Regression"], index=0)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.30, 0.05)
rf_n      = st.sidebar.slider("RF: n_estimators", 100, 800, 300, 50)
rf_depth  = st.sidebar.slider("RF: max_depth (0=None)", 0, 30, 0, 1)
rf_depth  = None if rf_depth == 0 else rf_depth
lr_c      = st.sidebar.slider("LR: C", 0.01, 5.0, 1.0)

target = "left"
y = filtered[target]
if y.nunique() < 2:
    st.warning("Filtered data has only one class (all stay or all leave). Loosen filters to train a model.")
    st.stop()

num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != target]
cat_cols = [c for c in filtered.columns if c not in num_cols + [target]]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

clf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, random_state=42, n_jobs=-1, class_weight="balanced") \
      if clf_choice == "Random Forest" else \
      LogisticRegression(C=lr_c, max_iter=2000, solver="liblinear", class_weight="balanced")

pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])

X = filtered.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Accuracy",  f"{accuracy_score(y_test,y_pred):.3f}")
m2.metric("Precision", f"{precision_score(y_test,y_pred, zero_division=0):.3f}")
m3.metric("Recall",    f"{recall_score(y_test,y_pred, zero_division=0):.3f}")
m4.metric("F1",        f"{f1_score(y_test,y_pred, zero_division=0):.3f}")
m5.metric("ROC-AUC",   f"{roc_auc_score(y_test,y_prob):.3f}")

c5,c6,c7 = st.columns(3)
with c5: st.plotly_chart(confusion_matrix_fig(y_test, y_pred), use_container_width=True)
with c6: st.plotly_chart(roc_curve_fig(y_test, y_prob), use_container_width=True)
with c7: st.plotly_chart(pr_curve_fig(y_test, y_prob), use_container_width=True)

# Explainability
st.subheader("Model Explainability")
try:
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_cols) if cat_cols else []
    feat_names = list(num_cols) + list(cat_names)
    if clf_choice == "Random Forest":
        importances = pipe.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)
        st.plotly_chart(px.bar(fi.head(20), x="Importance", y="Feature", orientation="h", title="Top Features"), use_container_width=True)
    else:
        coefs = pipe.named_steps["clf"].coef_.ravel()
        fi = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs}).sort_values("Coefficient", ascending=False)
        st.plotly_chart(px.bar(fi.head(20), x="Coefficient", y="Feature", orientation="h", title="Top Positive Coefficients"), use_container_width=True)
        st.plotly_chart(px.bar(fi.tail(20), x="Coefficient", y="Feature", orientation="h", title="Top Negative Coefficients"), use_container_width=True)
except Exception as e:
    st.info(f"Explainability note: {e}")

# Employee-level risk
st.markdown("---")
st.header("Employee-Level Risk Scoring")

threshold = st.slider("Risk threshold (probability of leaving)", 0.05, 0.95, 0.50, 0.05)
all_probs = pipe.predict_proba(X)[:,1]
scored = filtered.copy()
scored["attrition_prob"] = all_probs
scored["risk_flag"] = (scored["attrition_prob"] >= threshold).astype(int)

st.write(f"Employees at/above threshold: **{int(scored['risk_flag'].sum())}** / {len(scored)}")

topn = st.number_input("Show top N highest-risk employees", 5, 500, 25, 5)
cols_to_show = [c for c in ["department","salary","satisfaction_level","last_evaluation","number_project",
                            "average_montly_hours","time_spend_company","promotion_last_5years",
                            "Work_accident","attrition_prob","risk_flag"] if c in scored.columns]
st.dataframe(scored.sort_values("attrition_prob", ascending=False)[cols_to_show].head(int(topn)), use_container_width=True)

csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download scored employees (CSV)", data=csv_bytes, file_name="employee_attrition_scores.csv", mime="text/csv")
