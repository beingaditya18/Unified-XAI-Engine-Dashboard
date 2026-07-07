import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import yaml

from explainability.lime_engine import LimeExplainerEngine
from explainability.shap_engine import ShapExplainerEngine
from src.data.data_pipeline import AdultIncomePipeline
from src.fairness.fairness_audit import FairnessAuditEngine
from src.models.model_service import ModelService

# Force Matplotlib non-interactive mode
plt.switch_backend("Agg")

# --- CONFIG & STYLING ---
st.set_page_config(
    page_title="Unified XAI Governance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Mode Glassmorphism Theme Styling
st.markdown(
    """
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .highlight-card {
        background: linear-gradient(135deg, rgba(29, 78, 216, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
    }
    .warning-badge {
        background: rgba(220, 38, 38, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        border-radius: 4px;
        padding: 4px 8px;
        font-weight: bold;
    }
    .success-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        border-radius: 4px;
        padding: 4px 8px;
        font-weight: bold;
    }
    .divergent-card {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def get_system_services(config_path="config/config.yaml"):
    pipeline = AdultIncomePipeline(config_path)
    model_service = ModelService(config_path)
    shap_engine = ShapExplainerEngine(config_path)
    lime_engine = LimeExplainerEngine(config_path)
    fairness_engine = FairnessAuditEngine(config_path)

    # Run pipeline to ensure test files exist and encoders are ready
    try:
        pipeline.load_encoders()
    except Exception:
        # Encoders not trained, execute fit
        pipeline.run_pipeline()

    try:
        model_service.load_models()
    except Exception:
        # Train models if not present
        from models.train_models import train_and_save

        train_and_save(config_path)
        model_service.load_models()

    return pipeline, model_service, shap_engine, lime_engine, fairness_engine


pipeline, model_service, shap_engine, lime_engine, fairness_engine = get_system_services()

# Load Config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load split test set
test_path = config["data"]["test_path"]
if os.path.exists(test_path):
    df_test_full = pd.read_csv(test_path)
else:
    _, df_test_encoded, _, y_test = pipeline.run_pipeline()
    df_test_full = df_test_encoded.copy()
    df_test_full[pipeline.target_col] = y_test

X_test = df_test_full.drop(pipeline.target_col, axis=1)
y_test = df_test_full[pipeline.target_col]

# Reconstruct raw categorical labels for user selection display
df_test_raw = pipeline.inverse_transform_instance(X_test)
df_test_raw[pipeline.target_col] = y_test

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("XAI Control Center")
    st.markdown("Configure runtime parameters & models below:")
    st.markdown("---")

    model_arch = st.selectbox(
        "Active Model Architecture",
        ["XGBoost (Tree Classifier)", "Neural Network (MLP Classifier)"],
    )
    model_key = "xgboost" if "XGBoost" in model_arch else "neural_network"

    st.markdown("### Instance Selector")
    select_mode = st.radio("Selection Mode", ["Test Row Index", "Search by Features"])

    if select_mode == "Test Row Index":
        user_idx = st.slider("Select Instance Row Index", 0, len(X_test) - 1, 42)
    else:
        # Filter matching instances
        gender_filter = st.selectbox("Filter Sex", ["All", "Male", "Female"])
        age_min, age_max = st.slider("Filter Age Range", 17, 90, (25, 55))

        filtered_df = df_test_raw[(df_test_raw["age"] >= age_min) & (df_test_raw["age"] <= age_max)]
        if gender_filter != "All":
            filtered_df = filtered_df[filtered_df["sex"] == gender_filter]

        if filtered_df.empty:
            st.warning("No matching records found. Using default index 42.")
            user_idx = 42
        else:
            options = {
                f"Row {idx} (Age: {row['age']}, {row['sex']}, Education: {row['education']})": idx
                for idx, row in filtered_df.head(20).iterrows()
            }
            selected_label = st.selectbox("Select Matching Individual", list(options.keys()))
            user_idx = options[selected_label]

    st.markdown("---")
    st.markdown("### Pipeline Actions")
    if st.button("🔄 Retrain Models & Pipeline"):
        with st.spinner("Retraining model architectures from scratch..."):
            from models.train_models import train_and_save

            train_and_save()
            st.cache_resource.clear()
            st.success("Re-training completed. Page refreshing!")
            st.rerun()

# --- MAIN DASHBOARD HEADER ---
st.markdown(
    """
<div class="highlight-card">
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">🛡️ Model-Agnostic Explainability & Governance</h1>
    <p style="margin: 5px 0 0 0; font-size: 1.05rem; opacity: 0.85;">
        Unified validation platform auditing algorithm decisions, predictive disparities, and local feature attributions.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Fetch Instance & Perform Prediction
instance_encoded = X_test.iloc[[user_idx]]
instance_raw = df_test_raw.iloc[[user_idx]].drop(pipeline.target_col, axis=1)
true_income = df_test_raw.iloc[user_idx][pipeline.target_col]

proba = model_service.predict_proba(model_key, instance_encoded)[0][1]
pred_label = "High Income (>50K)" if proba > 0.5 else "Low Income (<=50K)"
true_label = "High Income (>50K)" if true_income == 1 else "Low Income (<=50K)"

# Performance metrics
try:
    with open("models/model_metrics.json") as f:
        stored_metrics = json.load(f)
    model_accuracy = stored_metrics[model_key]["accuracy"]
    model_auc = stored_metrics[model_key]["roc_auc"]
except Exception:
    metrics = model_service.evaluate_model(model_key, X_test, y_test)
    model_accuracy = metrics["accuracy"]
    model_auc = metrics["roc_auc"]

# KPI Metrics Grid
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
    <div class="metric-card">
        <span style="font-size: 0.9rem; opacity: 0.7;">Prediction</span>
        <h3 style="margin: 5px 0; color: {'#10b981' if proba > 0.5 else '#ef4444'};">{pred_label}</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
    <div class="metric-card">
        <span style="font-size: 0.9rem; opacity: 0.7;">Confidence Probability</span>
        <h3 style="margin: 5px 0;">{proba*100:.1f}%</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
    <div class="metric-card">
        <span style="font-size: 0.9rem; opacity: 0.7;">Ground Truth Status</span>
        <h3 style="margin: 5px 0; color: {'#10b981' if true_income == 1 else '#ef4444'};">{true_label}</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
    <div class="metric-card">
        <span style="font-size: 0.9rem; opacity: 0.7;">Model ROC-AUC / Accuracy</span>
        <h3 style="margin: 5px 0;">{model_auc:.2f} / {model_accuracy:.2%}</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# --- TABS WORKSPACE ---
tab_local, tab_global, tab_fairness, tab_divergence = st.tabs(
    [
        "🎯 Instance attributions",
        "🌎 Global feature impacts",
        "⚖️ Governance & Fairness",
        "🔀 Model Divergence",
    ]
)

# ==============================================================================
# TAB 1: INSTANCE ATTRIBUTIONS (SHAP vs LIME)
# ==============================================================================
with tab_local:
    st.subheader("Local Model Interpretability analysis")
    st.markdown(
        "Observe how input features influenced this decision. Attributions represent the quantitative pull from base reference values."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.write("#### 🎯 LIME Feature Contribution Rules")
        with st.spinner("Executing LIME explanation..."):
            explainer_lime = lime_engine.get_explainer(X_test)
            predict_fn = model_service._get_model(model_key).predict_proba
            lime_exp = lime_engine.explain_instance(
                explainer_lime, instance_encoded.iloc[0], predict_fn
            )

            # Reconstruct interactive Plotly chart for LIME weights
            rules = [x["rule"] for x in lime_exp["predictions"]]
            weights = [x["weight"] for x in lime_exp["predictions"]]

            fig_lime = go.Figure()
            fig_lime.add_trace(
                go.Bar(
                    x=weights,
                    y=rules,
                    orientation="h",
                    marker_color=["#10b981" if w >= 0 else "#ef4444" for w in weights],
                )
            )
            fig_lime.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Surrogate Linear Weight"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_lime, use_container_width=True)

    with col_r:
        st.write("#### 🎯 SHAP Local Attributions (Waterfalls)")
        with st.spinner("Calculating SHAP force values..."):
            model = model_service._get_model(model_key)
            shap_exp = shap_engine.explain_instance(model, instance_encoded, X_test, model_key)

            # Interactive Plotly waterfall chart representing SHAP features
            base_val = shap_exp["base_value"]
            features = [x["feature"] for x in shap_exp["predictions"]]
            sv_vals = [x["shap_value"] for x in shap_exp["predictions"]]
            disp_vals = [x["value"] for x in shap_exp["predictions"]]
            labels = [f"{f} ({v})" for f, v in zip(features, disp_vals)]

            fig_shap = go.Figure()
            # Waterfall logic
            cumulative = base_val
            for label, sv in zip(labels[:8], sv_vals[:8]):
                fig_shap.add_trace(
                    go.Bar(
                        name=label,
                        x=[sv],
                        y=[label],
                        orientation="h",
                        base=cumulative,
                        marker_color="#10b981" if sv >= 0 else "#ef4444",
                    )
                )
                cumulative += sv

            fig_shap.update_layout(
                showlegend=False,  # Hide legend clutter
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    title=f"Prediction score trajectory (Base: {base_val:.3f})",
                ),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    # Detailed instance facts panel
    st.write("---")
    st.write("#### 🔍 Recorded Profile Attributes (Original values)")
    cols_feat = st.columns(6)
    for idx, (col_name, col_val) in enumerate(instance_raw.iloc[0].items()):
        cols_feat[idx % 6].metric(label=col_name, value=str(col_val))

# ==============================================================================
# TAB 2: GLOBAL FEATURE IMPACTS
# ==============================================================================
with tab_global:
    st.subheader("Global Cohort Feature Impact Analysis")
    st.markdown(
        "Visualizes feature priorities compiled across a random sample from the validation cohort."
    )

    with st.spinner("Generating SHAP cohort analysis..."):
        model = model_service._get_model(model_key)

        # Determine sampling bounds
        shap_samples = config["explainability"]["shap"]["test_samples"]
        X_sample = X_test.sample(n=min(shap_samples, len(X_test)), random_state=42)

        shap_values = shap_engine.calculate_shap_values(model, X_sample, X_test, model_key)

        c_plots_l, c_plots_r = st.columns([2, 1])

        with c_plots_l:
            st.write("##### SHAP Summary Plot (Distribution of feature values)")
            fig_g, ax_g = plt.subplots(figsize=(8, 5))
            fig_g.patch.set_facecolor("#0e1117")
            ax_g.set_facecolor("#0e1117")

            # Format text colors for dark UI integration
            plt.rcParams["text.color"] = "#fafafa"
            plt.rcParams["axes.labelcolor"] = "#fafafa"
            plt.rcParams["xtick.color"] = "#fafafa"
            plt.rcParams["ytick.color"] = "#fafafa"

            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig_g)
            plt.close()

        with c_plots_r:
            st.write("##### Feature Priority Ranking")
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feat_imp_df = pd.DataFrame(
                {"Feature": X_test.columns, "Mean |SHAP|": mean_abs_shap}
            ).sort_values(by="Mean |SHAP|", ascending=False)

            fig_imp = px.bar(
                feat_imp_df,
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                color="Mean |SHAP|",
                color_continuous_scale="Purples",
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# TAB 3: GOVERNANCE & FAIRNESS AUDIT
# ==============================================================================
with tab_fairness:
    st.subheader("Demographic & Algorithmic Fairness Audit")
    st.markdown(
        "Examines potential disparate impacts against protected demographics (Gender & Race) using predictions over the test cohort."
    )

    preds_all = model_service.predict(model_key, X_test)
    fairness_metrics = fairness_engine.calculate_fairness_metrics(X_test, y_test, preds_all)

    for attr, metrics in fairness_metrics.items():
        st.markdown(f"### ⚖️ Protected Attribute: **{attr.upper()}**")

        col_m_l, col_m_r = st.columns(2)

        with col_m_l:
            st.markdown("#### Selection Rate Differences")
            # Create interactive bar chart comparing groups
            fig_select = go.Figure()
            fig_select.add_trace(
                go.Bar(
                    x=["Privileged Group", "Unprivileged Group"],
                    y=[
                        metrics["privileged_selection_rate"],
                        metrics["unprivileged_selection_rate"],
                    ],
                    marker_color=["#4f46e5", "#818cf8"],
                    text=[
                        f"{metrics['privileged_selection_rate']:.2%}",
                        f"{metrics['unprivileged_selection_rate']:.2%}",
                    ],
                    textposition="auto",
                )
            )
            fig_select.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1], title="Approval Probability Rate"),
                margin=dict(l=20, r=20, t=10, b=10),
            )
            st.plotly_chart(fig_select, use_container_width=True)

        with col_m_r:
            st.markdown("#### Fairness Metric Summary")

            # Print metrics as cards
            dpr = metrics["disparate_impact_ratio"]
            dp_diff = metrics["demographic_parity_difference"]
            eq_opp = metrics["equal_opportunity_difference"]

            status_text = (
                "BIAS ALERT: Disparate Impact Detected"
                if metrics["is_biased"]
                else "FAIRNESS PASS: Disparity thresholds satisfied"
            )
            status_class = "warning-badge" if metrics["is_biased"] else "success-badge"

            st.markdown(
                f"<span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True
            )
            st.write("")

            # Row stats table
            stats_df = pd.DataFrame(
                {
                    "Metric Name": [
                        "Disparate Impact Ratio (DIR / DPR)",
                        "Demographic Parity Difference",
                        "Equal Opportunity Difference (TPR Diff)",
                        "False Positive Rate Difference",
                    ],
                    "Value": [
                        f"{dpr:.3f}",
                        f"{dp_diff:.3f}",
                        f"{eq_opp:.3f}",
                        f"{metrics['false_positive_rate_difference']:.3f}",
                    ],
                    "Reference Status": [
                        "0.80 - 1.25 (Ideal: 1.0)",
                        f"<= {config['explainability']['fairness']['disparity_threshold']} (Ideal: 0.0)",
                        "Ideal: 0.0",
                        "Ideal: 0.0",
                    ],
                }
            )
            st.table(stats_df.set_index("Metric Name"))

# ==============================================================================
# TAB 4: MODEL DIVERGENCE ANALYSIS (NEW & ADVANCED)
# ==============================================================================
with tab_divergence:
    st.subheader("🔍 Model Divergence Investigation")
    st.markdown(
        "Finds cohort cases where **XGBoost (Tree Logic)** and the **Neural Network (MLP)** disagree. Such conflict points highlight model margin areas and help explain architectural assumptions."
    )

    divergent_cases = model_service.get_divergent_instances(X_test)

    if divergent_cases.empty:
        st.success(
            "Perfect alignment! XGBoost and MLP predictions agreed on 100% of tested instances."
        )
    else:
        st.warning(
            f"Detected {len(divergent_cases)} prediction disagreement instances in this cohort split."
        )

        # Display as DataFrame with nice columns
        display_cols = [
            "age",
            "workclass",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "hours-per-week",
        ]
        disp_df = divergent_cases.copy()

        # Inverse transform labels to make them readable
        disp_df_decoded = pipeline.inverse_transform_instance(disp_df[display_cols])
        disp_df_decoded["XGBoost Pred (Prob)"] = (
            disp_df["pred_xgb"].map({1: ">50K", 0: "<=50K"})
            + " ("
            + disp_df["prob_xgb"].round(3).astype(str)
            + ")"
        )
        disp_df_decoded["MLP Pred (Prob)"] = (
            disp_df["pred_nn"].map({1: ">50K", 0: "<=50K"})
            + " ("
            + disp_df["prob_nn"].round(3).astype(str)
            + ")"
        )
        disp_df_decoded["Original Index"] = disp_df.index

        st.dataframe(disp_df_decoded.set_index("Original Index"))

        st.markdown("### Investigate a specific Divergent Case")
        selected_divergent_idx = st.selectbox(
            "Select an original index to load details into main sliders",
            options=divergent_cases.index.tolist(),
        )

        if st.button("Load this profile for instance attribution analysis"):
            st.info(
                f"Loaded index {selected_divergent_idx}. Switch back to the 'Instance attributions' tab to see explanation comparisons."
            )
            # Set index slider state via query params or force load (simplest is query param/session state if required,
            # but we can instruct the user to use the sidebar slider to target it).
            st.markdown(
                f"👈 **Please set the sidebar instance index slider directly to: `{selected_divergent_idx}`**"
            )

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.85rem; opacity: 0.5;'>🛡️ Unified XAI Governance Framework | Powered by FastAPI, Streamlit, and SHAP/LIME</p>",
    unsafe_allow_html=True,
)
