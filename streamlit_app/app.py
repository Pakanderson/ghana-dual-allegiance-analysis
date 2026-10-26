"""
Dual Allegiance in Practice — Streamlit app
Trust, performance, and corruption perceptions of traditional authorities vs.
elected local government in Ghana, built on Afrobarometer Round 9 (2022) data.

Run locally:
    streamlit run app.py

Data: data/ghana_r9_clean.csv (recoded) and data/ghana_r9_raw.csv (original labels).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants / theme
# ---------------------------------------------------------------------------
BLUE = "#2E6F8E"    # elected local government
GOLD = "#C9762C"    # traditional leaders
NEUTRAL = "#8C8C8C"

EDU_LEVELS = [
    "No formal schooling",
    "Informal schooling only (incl. Koranic)",
    "Some primary schooling",
    "Primary school completed",
    "Intermediate / some secondary school",
    "Secondary school completed",
    "Post-secondary (non-university)",
    "University completed",
    "Post-graduate",
]

st.set_page_config(
    page_title="Dual Allegiance in Practice — Ghana",
    page_icon="🇬🇭",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_clean():
    return pd.read_csv(DATA_DIR / "ghana_r9_clean.csv")


@st.cache_data
def load_raw():
    return pd.read_csv(DATA_DIR / "ghana_r9_raw.csv")


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train the same logistic regression + Random Forest pair used in the
    original analysis notebook, on the full clean dataset. Returns fitted
    models, the feature-column order, the scaler, and held-out metrics."""
    model_df = df.dropna(
        subset=["leans_traditional", "urban", "age", "female",
                "education_level", "lived_poverty", "region"]
    ).copy()
    model_df["leans_traditional"] = model_df["leans_traditional"].astype(int)

    region_dummies = pd.get_dummies(model_df["region"], prefix="region", drop_first=True)
    X = pd.concat([
        model_df[["urban", "age", "female", "education_level", "lived_poverty"]].astype(float),
        region_dummies.astype(float),
    ], axis=1)
    y = model_df["leans_traditional"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(X_train_sc, y_train)
    logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test_sc)[:, 1])

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=15,
        random_state=42, class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    return {
        "logreg": logreg, "rf": rf, "scaler": scaler,
        "columns": list(X.columns), "regions": sorted(model_df["region"].unique()),
        "logreg_auc": logreg_auc, "rf_auc": rf_auc,
        "base_rate": y.mean(), "n": len(model_df),
    }


clean = load_clean()
raw = load_raw()
models = train_models(clean)

# ---------------------------------------------------------------------------
# Sidebar filters (apply to Dashboard + Data Explorer)
# ---------------------------------------------------------------------------
st.sidebar.title("🇬🇭 Filters")
st.sidebar.caption("Applied to the Dashboard and Data Explorer tabs.")

all_regions = sorted(clean["region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)

area_choice = st.sidebar.radio("Area", ["All", "Urban", "Rural"], horizontal=True)
gender_choice = st.sidebar.radio("Gender", ["All", "Woman", "Man"], horizontal=True)

filtered = clean[clean["region"].isin(sel_regions)].copy()
if area_choice != "All":
    filtered = filtered[filtered["urban"] == (1 if area_choice == "Urban" else 0)]
if gender_choice != "All":
    filtered = filtered[filtered["female"] == (1 if gender_choice == "Woman" else 0)]

st.sidebar.metric("Respondents in filter", f"{len(filtered):,}", f"of {len(clean):,} total")

if len(filtered) < 20:
    st.sidebar.warning("Fewer than 20 respondents match this filter — figures below will be noisy.")

st.sidebar.divider()
st.sidebar.caption(
    "Data: Afrobarometer Round 9 (2022), Ghana (n=2,369). "
    "Built for practice — see the About tab for methodology and caveats."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Dual Allegiance in Practice")
st.subheader("Trust, performance, and corruption: traditional authorities vs. elected local government in Ghana")

tab_dash, tab_predict, tab_explore, tab_about = st.tabs(
    ["📊 Dashboard", "🔮 Predict", "🗂️ Data Explorer", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
with tab_dash:
    if filtered.empty:
        st.info("No respondents match the current filters. Adjust the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trust: local gov't", f"{filtered['trust_local_gov'].mean():.2f}", help="0 (not at all) – 3 (a lot)")
        c2.metric("Trust: traditional", f"{filtered['trust_traditional'].mean():.2f}", help="0 (not at all) – 3 (a lot)")
        disapprove = filtered["perf_local_gov"].isin([0, 1]).mean() * 100
        c3.metric("Disapprove councillor performance", f"{disapprove:.0f}%")
        gap = filtered["trust_gap"].mean()
        c4.metric("Trust gap (traditional − local)", f"{gap:+.2f}",
                   help="Positive = leans traditional, negative = leans local government")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Mean trust: local government vs. traditional leaders**")
            means = filtered[["trust_local_gov", "trust_traditional"]].mean()
            sems = filtered[["trust_local_gov", "trust_traditional"]].sem()
            fig = go.Figure(go.Bar(
                x=["Local Government Council", "Traditional Leaders"],
                y=means.values,
                error_y=dict(type="data", array=sems.values),
                marker_color=[BLUE, GOLD],
            ))
            fig.update_layout(yaxis_title="Mean trust (0–3)", yaxis_range=[0, 2],
                               margin=dict(t=10, b=10), height=350)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("**Trust by area**")
            by_area = filtered.groupby(filtered["urban"].map({1: "Urban", 0: "Rural"}))[
                ["trust_local_gov", "trust_traditional"]
            ].mean().reindex(["Rural", "Urban"])
            fig = go.Figure()
            fig.add_bar(name="Local Government", x=by_area.index, y=by_area["trust_local_gov"], marker_color=BLUE)
            fig.add_bar(name="Traditional Leaders", x=by_area.index, y=by_area["trust_traditional"], marker_color=GOLD)
            fig.update_layout(barmode="group", yaxis_title="Mean trust (0–3)",
                               margin=dict(t=10, b=10), height=350,
                               legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Regional trust gap** (traditional − local government; regions with n ≥ 10 in current filter)")
        region_gap = (filtered.groupby("region")["trust_gap"]
                      .agg(["mean", "count"]).query("count >= 10").sort_values("mean"))
        if region_gap.empty:
            st.caption("No region in the current filter has at least 10 respondents.")
        else:
            colors = [BLUE if v < 0 else GOLD for v in region_gap["mean"]]
            fig = go.Figure(go.Bar(
                x=region_gap["mean"], y=region_gap.index, orientation="h",
                marker_color=colors,
                customdata=region_gap["count"],
                hovertemplate="%{y}: %{x:.2f} (n=%{customdata})<extra></extra>",
            ))
            fig.add_vline(x=0, line_color="black", line_width=1)
            fig.update_layout(xaxis_title="Trust gap", margin=dict(t=10, b=10),
                               height=max(300, 28 * len(region_gap)))
            st.plotly_chart(fig, width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Perceived corruption** (0=none/some → 3=all)")
            corr = filtered[["corrupt_local_gov", "corrupt_traditional"]].mean()
            fig = go.Figure(go.Bar(
                x=["Local Gov't Councillors", "Traditional Leaders"], y=corr.values,
                marker_color=[BLUE, GOLD],
            ))
            fig.update_layout(yaxis_range=[0, 2], margin=dict(t=10, b=10), height=320)
            st.plotly_chart(fig, width="stretch")

        with col4:
            st.markdown("**Contact behaviour** (% contacted in the past year)")
            contact = filtered[["contacted_local_gov", "contacted_traditional"]].mean() * 100
            fig = go.Figure(go.Bar(
                x=["Contacted Councillor", "Contacted Traditional Leader"], y=contact.values,
                marker_color=[BLUE, GOLD], text=[f"{v:.0f}%" for v in contact.values],
                textposition="outside",
            ))
            fig.update_layout(yaxis_title="%", margin=dict(t=10, b=10), height=320)
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    st.markdown(
        "Enter a hypothetical respondent's profile. The app runs the **same two models** "
        "from the analysis notebook — a logistic regression and a Random Forest — trained live "
        "on the full Ghana Round 9 sample, and shows each model's predicted probability that this "
        "person trusts **traditional leaders more than their elected local government council**."
    )
    st.warning(
        f"Read this as descriptive, not predictive: on held-out data these models score "
        f"ROC-AUC ≈ {models['logreg_auc']:.2f} (logistic) / {models['rf_auc']:.2f} (Random Forest) — "
        f"barely above chance (0.50). Demographics alone explain little of who leans traditional; "
        f"region and lived context matter more. See the About tab for the full caveat.",
        icon="⚠️",
    )

    with st.form("predict_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_urban = st.selectbox("Area", ["Urban", "Rural"])
            f_gender = st.selectbox("Gender", ["Man", "Woman"])
        with fc2:
            f_age = st.slider("Age", 18, 100, 35)
            f_region = st.selectbox("Region", models["regions"],
                                     index=models["regions"].index("Greater Accra")
                                     if "Greater Accra" in models["regions"] else 0)
        with fc3:
            f_edu = st.selectbox("Education", EDU_LEVELS, index=5)
            f_poverty = st.slider("Lived Poverty Index (0=none, 4=high)", 0.0, 4.0, 1.5, 0.1)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {c: 0.0 for c in models["columns"]}
        row["urban"] = 1.0 if f_urban == "Urban" else 0.0
        row["age"] = float(f_age)
        row["female"] = 1.0 if f_gender == "Woman" else 0.0
        row["education_level"] = float(EDU_LEVELS.index(f_edu))
        row["lived_poverty"] = float(f_poverty)
        region_col = f"region_{f_region}"
        if region_col in row:
            row[region_col] = 1.0
        # if the region was the dropped reference category, all region_* stay 0 — correct.

        X_new = pd.DataFrame([row], columns=models["columns"])
        X_new_sc = models["scaler"].transform(X_new)

        p_logreg = models["logreg"].predict_proba(X_new_sc)[0, 1]
        p_rf = models["rf"].predict_proba(X_new)[0, 1]

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Logistic Regression", f"{p_logreg*100:.0f}%", help="Predicted probability of leaning traditional")
            st.progress(min(max(p_logreg, 0.0), 1.0))
        with r2:
            st.metric("Random Forest", f"{p_rf*100:.0f}%", help="Predicted probability of leaning traditional")
            st.progress(min(max(p_rf, 0.0), 1.0))

        st.caption(
            f"For reference, {models['base_rate']*100:.0f}% of respondents in the training sample "
            f"lean traditional overall."
        )

# ---------------------------------------------------------------------------
# Data Explorer tab
# ---------------------------------------------------------------------------
with tab_explore:
    st.markdown("Browse and download the underlying data.")

    which = st.radio("Dataset", ["Clean (recoded, numeric)", "Raw (original Afrobarometer labels)"], horizontal=True)
    source = clean if which.startswith("Clean") else raw

    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        text_query = st.text_input("Filter rows (matches any cell, case-insensitive)", "")
    with search_col2:
        show_cols = st.multiselect("Columns to show", list(source.columns), default=list(source.columns))

    view = source[show_cols] if show_cols else source
    if text_query:
        mask = view.astype(str).apply(lambda col: col.str.contains(text_query, case=False, na=False)).any(axis=1)
        view = view[mask]

    st.dataframe(view, width="stretch", height=480)
    st.caption(f"Showing {len(view):,} of {len(source):,} rows.")

    st.download_button(
        "⬇️ Download filtered CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"ghana_r9_{'clean' if which.startswith('Clean') else 'raw'}_filtered.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### About this app

Ghana runs two parallel local governance institutions: elected **District Assemblies**
(Local Government Act, Act 936; 1992 Constitution Art. 270–277) and the **chieftaincy**
institution (Chieftaincy Act, 2008). This app explores how citizens navigate that
"dual allegiance" empirically, using **Afrobarometer Round 9 (2022)** survey data for
Ghana (n = 2,369).

**Tabs**
- **Dashboard** — the report's core comparisons (trust, corruption, performance, contact
  behaviour), filterable by region, area, and gender.
- **Predict** — a live demographic-only classifier (logistic regression + Random Forest),
  trained in your browser session on the full sample, for exploring the (weak) relationship
  between demographics and institutional leaning.
- **Data Explorer** — search, filter, and download the raw or recoded CSV directly.

**Key finding from the original analysis:** overall trust in the two institutions is
statistically indistinguishable, but a real urban–rural split emerges underneath that
average — and stated trust does not reliably predict who citizens actually turn to for
help. Full write-up and notebook are in the companion project deliverables.

**Data source:** Afrobarometer Round 9 (2022), Ghana country file, retrieved via the public
[Afrobarometer-Data-Explorer](https://github.com/MafiAtUN/Afrobarometer-Data-Explorer) mirror
of the official Afrobarometer microdata (afrobarometer.org), redistributed for research/
educational use.

**Limitations:** cross-sectional survey data support association, not causal claims. The
Predict tab's models are weak (ROC-AUC ≈ 0.57–0.58) and should be read as descriptive, not
as reliable individual-level predictions.
"""
    )
