
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="NexGen Predictive Delivery Optimizer", layout="wide")

st.markdown("""
    <style>
    html, body, div, span, label, section, h1, h2, h3, h4, h5, h6, p, td, th, button {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Apply to Streamlit widgets & charts */
    .stMarkdown, .stDataFrame, .stPlotlyChart, .stTextInput, .stSelectbox, .stMultiSelect, .stSlider {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Optional: header tweaks */
    h1, h2, h3 {
        color: #002147 !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("NexGen Predictive Delivery Delay Optimizer")
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_data(show_spinner=False)
def safe_read(csv_name):
    path = os.path.join(DATA_DIR, csv_name)
    if not os.path.exists(path):
        return None
    try:
    
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1", engine="python")
        except Exception as e:
            st.error(f"Error reading {csv_name}: {e}")
            return None

def clean_columns(df):
    """Lowercase and replace non-alphanumeric sequences with underscore."""
    df = df.copy()

    cols = []
    for c in df.columns.astype(str):
        c2 = c.strip()

        c2 = pd.Series([c2]).str.replace(r"[ \/\-\.,\(\)&]+", "_", regex=True).iloc[0]
        c2 = c2.strip("_").lower()
        cols.append(c2)
    df.columns = cols
    return df

def standardize_common_names(df, kind=None):
    """Rename common alternative column names to a canonical set."""
    if df is None:
        return None
    df = df.copy()

    mapping = {
        # orders
        "order_id": "order_id",
        "orderid": "order_id",
        "order__id": "order_id",
        "order_number": "order_id",
        "order": "order_id",
        "order_date": "order_date",
        "order_date_": "order_date",
        "order_value": "order_value_inr",
        "order_value_inr": "order_value_inr",
        "order_value_inrs": "order_value_inr",
        "order_value_inr_": "order_value_inr",
        # priority / category
        "priority": "priority",
        "product_category": "product_category",
        "product": "product_category",
        # delivery performance
        "promised_delivery_days": "promised_days",
        "promised_days": "promised_days",
        "actual_delivery_days": "actual_days",
        "actual_days": "actual_days",
        "delivery_status": "delivery_status",
        "delivery_cost_inr": "delivery_cost",
        "delivery_cost": "delivery_cost",
        "customer_rating": "customer_rating",
        "rating": "customer_rating",
        # routes
        "distance_km": "distance_km",
        "distance": "distance_km",
        "fuel_consumption_l": "fuel_consumption",
        "fuel_consumption": "fuel_consumption",
        "toll_charges_inr": "toll_charges",
        "toll_charges": "toll_charges",
        "traffic_delay_minutes": "traffic_delay",
        "traffic_delay": "traffic_delay",
        "weather_impact": "weather_impact",
        # feedback
        "feedback_text": "feedback_text",
        "feedback": "feedback_text",
        "would_recommend": "would_recommend",
        "would_recommend_": "would_recommend",
        "issue_category": "issue_category",
        "issue": "issue_category",
        # vehicle
        "vehicle_age": "vehicle_age",
        "age_years": "vehicle_age",
        # origin/destination
        "origin": "origin",
        "destination": "destination",
        "route": "route",
    }

    rename_map = {}
    for col in df.columns:
        if col in mapping:
            rename_map[col] = mapping[col]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

with st.spinner("Loading CSV files..."):
    df_orders = safe_read("orders.csv")
    df_delivery = safe_read("delivery_performance.csv")
    df_routes = safe_read("routes_distance.csv")
    df_fleet = safe_read("vehicle_fleet.csv")
    df_warehouse = safe_read("warehouse_inventory.csv")
    df_feedback = safe_read("customer_feedback.csv")
    df_costs = safe_read("cost_breakdown.csv")

if df_orders is None:
    st.error("orders.csv not found in ./data — please put your CSVs in the data folder.")
    st.stop()


for name, df in [
    ("orders", df_orders),
    ("delivery", df_delivery),
    ("routes", df_routes),
    ("fleet", df_fleet),
    ("warehouse", df_warehouse),
    ("feedback", df_feedback),
    ("costs", df_costs),
]:
    if df is not None:
        df_clean = clean_columns(df)
        df_clean = standardize_common_names(df_clean)
    
        if name == "orders":
            df_orders = df_clean
        elif name == "delivery":
            df_delivery = df_clean
        elif name == "routes":
            df_routes = df_clean
        elif name == "fleet":
            df_fleet = df_clean
        elif name == "warehouse":
            df_warehouse = df_clean
        elif name == "feedback":
            df_feedback = df_clean
        elif name == "costs":
            df_costs = df_clean

master = df_orders.copy()
key = "order_id"
if key not in master.columns:
    st.error("orders.csv must contain 'order_id' column (after cleaning). Found: " + ", ".join(master.columns))
    st.stop()

def safe_merge(left, right, on=key):
    if right is None:
        return left
    if on not in right.columns:
        return left
    return left.merge(right, on=on, how="left")

master = safe_merge(master, df_delivery)
master = safe_merge(master, df_routes)
master = safe_merge(master, df_costs)


if df_feedback is not None and "order_id" in df_feedback.columns:

    desired = {
        "order_id": "order_id",
        "customer_rating": "customer_rating",
        "issue_category": "issue_category",
        "feedback_text": "feedback_text",
        "would_recommend": "would_recommend",
        "feedback_date": "feedback_date",
    }
    
    available = [c for c in df_feedback.columns if c in desired.keys()]
    if available:
        fb_subset = df_feedback[available].rename(columns={c: desired[c] for c in available})
        master = master.merge(fb_subset, on="order_id", how="left")
    else:

        fallback_map = {}
        for col in df_feedback.columns:
            if "rating" in col:
                fallback_map[col] = "customer_rating"
            elif "issue" in col:
                fallback_map[col] = "issue_category"
            elif "feedback" in col and "text" in col:
                fallback_map[col] = "feedback_text"
            elif "recommend" in col:
                fallback_map[col] = "would_recommend"
            elif "order" in col:
                fallback_map[col] = "order_id"
        if fallback_map:
            fb2 = df_feedback.rename(columns=fallback_map)
            cols_to_use = [c for c in ["order_id", "customer_rating", "issue_category", "feedback_text", "would_recommend"] if c in fb2.columns]
            if "order_id" in cols_to_use:
                master = master.merge(fb2[cols_to_use], on="order_id", how="left")


if {"actual_days", "promised_days"}.issubset(master.columns):
    master["delay_days"] = master["actual_days"].astype(float) - master["promised_days"].astype(float)
else:
    
    if "actual_delivery_days" in master.columns and "promised_delivery_days" in master.columns:
        master["delay_days"] = master["actual_delivery_days"].astype(float) - master["promised_delivery_days"].astype(float)
    elif "actual_days" in master.columns and "promised_delivery_days" in master.columns:
        master["delay_days"] = master["actual_days"].astype(float) - master["promised_delivery_days"].astype(float)
    elif "actual_delivery_days" in master.columns and "promised_days" in master.columns:
        master["delay_days"] = master["actual_delivery_days"].astype(float) - master["promised_days"].astype(float)
    else:
        master["delay_days"] = np.nan


if master["delay_days"].notna().any():
    default_thresh = 2
    master["is_delayed_label"] = (master["delay_days"] > default_thresh).astype(int)
else:
    master["is_delayed_label"] = np.nan


if "order_value_inr" not in master.columns and "order_value" in master.columns:
    master["order_value_inr"] = pd.to_numeric(master["order_value"], errors="coerce")


if "distance_km" not in master.columns:
    if "distance" in master.columns:
        master["distance_km"] = pd.to_numeric(master["distance"], errors="coerce")
    elif "distance_km_" in master.columns:
        master["distance_km"] = pd.to_numeric(master["distance_km_"], errors="coerce")


if "fuel_consumption" not in master.columns:
    if "fuel_consumption_l" in master.columns:
        master["fuel_consumption"] = pd.to_numeric(master["fuel_consumption_l"], errors="coerce")


if "vehicle_age" not in master.columns and df_fleet is not None and "vehicle_id" in master.columns:
    
    if "vehicle_id" in df_fleet.columns and "age_years" in df_fleet.columns:
        tmp = df_fleet[["vehicle_id", "age_years"]].rename(columns={"age_years": "vehicle_age"})
        if "vehicle_id" in master.columns:
            master = master.merge(tmp, on="vehicle_id", how="left")


with st.expander("Master dataset sample & stats"):
    st.write("Shape:", master.shape)
    st.dataframe(master.head(10))
    st.write(master.describe(include="all"))


st.header("1. Exploratory Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Delivery Status by Priority")
    if "delivery_status" in master.columns and "priority" in master.columns:
        fig = px.histogram(master, x="delivery_status", color="priority", title="Delivery Status by Priority")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'delivery_status' and 'priority' columns for this chart.")

with col2:
    st.subheader("Delay Distribution by Priority")
    if "delay_days" in master.columns and "priority" in master.columns:
        fig = px.box(master, x="priority", y="delay_days", points="all", title="Delay Days by Priority")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No delay_days data available to plot.")

st.subheader("Customer Rating vs Delay")
if "customer_rating" in master.columns and "delay_days" in master.columns:
    fig = px.scatter(master, x="delay_days", y="customer_rating", color="priority", title="Rating vs Delay")
    st.plotly_chart(fig, use_container_width=True)


if "issue_category" in master.columns:
    st.subheader("Top Issue Categories")
    issue_counts = master["issue_category"].value_counts().reset_index()
    issue_counts.columns = ["issue", "count"]
    st.plotly_chart(px.bar(issue_counts, x="issue", y="count", title="Top Reported Issues"), use_container_width=True)


st.header("2. Predictive Delay Model (Random Forest baseline)")


possible_features = []
heuristic_candidates = [
    "priority",
    "product_category",
    "distance_km",
    "distance",
    "fuel_consumption",
    "toll_charges",
    "traffic_delay",
    "weather_impact",
    "order_value_inr",
    "vehicle_age",
    "delivery_cost",
]
for c in heuristic_candidates:
    if c in master.columns:
        if c == "distance":
            master["distance_km"] = master[c]
        possible_features.append(c if c != "distance" else "distance_km")

possible_features = list(dict.fromkeys(possible_features))

available_features = [c for c in possible_features if c in master.columns]
st.write("Available features detected:", available_features if available_features else "No heuristic features found.")

selected_features = st.multiselect(
    "Choose features to use for model",
    options=available_features,
    default=available_features[:5] if len(available_features) >= 3 else available_features,
)


if master["is_delayed_label"].isna().all():
    st.warning("No delay label available in dataset (cannot train supervised model). Use rule-based approach or create labels.")
else:
    min_rows_needed = 30
    if len(selected_features) == 0:
        st.info("Select features above to train a model.")
    else:
        df_model = master.dropna(subset=selected_features + ["is_delayed_label"])
        st.write("Rows available for training after dropping missing values:", df_model.shape[0])

        if df_model.shape[0] < min_rows_needed:
            st.warning(
                f"Not enough rows ({df_model.shape[0]}) to train robust model—recommend using rule-based or gather more labeled data."
            )

        
        if len(selected_features) > 0 and df_model.shape[0] >= 10:
            X = df_model[selected_features].copy()
            y = df_model["is_delayed_label"].astype(int)

            
            cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"]
            num_cols = [c for c in X.columns if c not in cat_cols]

            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ],
                remainder="drop",
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            pipe = Pipeline([("pre", preprocessor), ("clf", model)])

            test_size = st.slider("Test set size (%)", 10, 40, 20)
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, random_state=42, stratify=y)
            except Exception:
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, random_state=42)

            with st.spinner("Training model..."):
                pipe.fit(X_train, y_train)

            
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) > 1 else None

            st.subheader("Model performance on test set")
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("Precision", f"{prec:.3f}")
            st.metric("Recall", f"{rec:.3f}")
            st.metric("F1", f"{f1:.3f}")
            if auc:
                st.metric("ROC AUC", f"{auc:.3f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, val, ha="center", va="center", color="black")
            st.pyplot(fig_cm)

            # ROC curve
            if y_proba is not None and auc is not None:
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                fig_roc = px.area(
                    x=fpr, y=tpr,
                    title=f"ROC Curve (AUC={auc:.3f})",
                    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                )
                fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"))
                st.plotly_chart(fig_roc, use_container_width=True)

            
            st.subheader("Feature importance (approximate)")
            try:
                pre = pipe.named_steps["pre"]
                feat_names = []
                if num_cols:
                    feat_names.extend(num_cols)
                if cat_cols:
                    ohe = pre.named_transformers_["cat"]
                    if hasattr(ohe, "get_feature_names_out"):
                        cat_names = list(ohe.get_feature_names_out(cat_cols))
                    else:
                        cat_names = []
                        for c in cat_cols:
                            cats = X[c].astype(str).unique().tolist()
                            cat_names.extend([f"{c}__{v}" for v in cats])
                    feat_names.extend(cat_names)
                importances = pipe.named_steps["clf"].feature_importances_
                fi = pd.DataFrame({"feature": feat_names, "importance": importances})
                fi = fi.sort_values("importance", ascending=False).head(20)
                st.dataframe(fi)
                st.plotly_chart(px.bar(fi, x="feature", y="importance", title="Top feature importances"), use_container_width=True)
            except Exception as e:
                st.info("Could not compute feature importances: " + str(e))

            
            st.subheader("Threshold tuning (use to trade off precision vs recall)")
            threshold = st.slider("Probability threshold to consider delayed", 0.0, 1.0, 0.5, 0.01)
            if y_proba is not None:
                y_pred_thresh = (y_proba >= threshold).astype(int)
                st.write("Precision:", precision_score(y_test, y_pred_thresh, zero_division=0))
                st.write("Recall:", recall_score(y_test, y_pred_thresh, zero_division=0))
                st.write("F1:", f1_score(y_test, y_pred_thresh, zero_division=0))

            
            st.subheader("Predict for an order from dataset")
            if "order_id" in df_model.columns:
                order_choices = df_model["order_id"].dropna().unique().tolist()
                order_choice = st.selectbox("Pick order_id to predict", options=order_choices[:200])
                if order_choice:
                    row = df_model[df_model["order_id"] == order_choice].iloc[0]
                    X_single = pd.DataFrame([row[selected_features]])
                    proba = pipe.predict_proba(X_single)[:, 1][0] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
                    pred_label = int(proba >= threshold) if proba is not None else pipe.predict(X_single)[0]
                    st.write("Predicted probability of delay:", f"{proba:.3f}" if proba is not None else "N/A")
                    st.write("Predicted label (delayed=1):", pred_label)
                    recs = []
                    if pred_label == 1:
                        recs.append("Prioritize this order for allocation of faster vehicle / express routing.")
                        recs.append("Notify customer proactively with ETA + reason.")
                    else:
                        recs.append("No immediate action required. Monitor in-transit updates.")
                    st.markdown("**Recommendations:**")
                    for r in recs:
                        st.markdown(f"- {r}")

            
            if st.button("Download test predictions (CSV)"):
                out = X_test.copy()
                out["true_is_delayed"] = y_test.values
                out["pred_prob"] = y_proba if y_proba is not None else pipe.predict(X_test)
                out["pred_label"] = (out["pred_prob"] >= threshold).astype(int)
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "predictions_test.csv", "text/csv")

        else:
            st.info("Select at least 1 feature and ensure there are enough rows to train a model.")

st.header("3. Quick Rule-based Insights (fallback)")
if "delay_days" in master.columns:
    delay_thresh = st.slider("Delay threshold (days) for rule-based classification", 0, 10, 2)
    master["is_delayed_rule"] = (master["delay_days"] > delay_thresh).astype(int)
    pct_delayed = master["is_delayed_rule"].mean() * 100 if master["is_delayed_rule"].size > 0 else 0
    st.write(f"Overall delayed percent (rule): {pct_delayed:.1f}%")
    
    if "origin" in master.columns and "destination" in master.columns:
        master["corridor"] = master["origin"].astype(str) + " → " + master["destination"].astype(str)
        top_corr = (
            master.groupby("corridor")["is_delayed_rule"].mean().reset_index().sort_values("is_delayed_rule", ascending=False).head(10)
        )
        st.plotly_chart(px.bar(top_corr, x="corridor", y="is_delayed_rule", title="Top corridors by delay rate", text_auto=".2f"), use_container_width=True)
else:
    st.info("No delay_days column available for rule-based insights.")

st.header("4. Recommendations & Next steps")
st.markdown(
    """
- Use the model above as a baseline — evaluate its precision/recall for business use.
- If recall is more important (catch most delays), lower the threshold; if precision is critical, raise the threshold.
- For better performance: more labeled data, temporal validation, and feature engineering (carrier reliability, warehouse pick time, time-of-day).
- Consider implementing a small pilot (predict-only) for 1–2 weeks and measure real-world performance.
"""
)


st.header("5. Export")
csv = master.to_csv(index=False).encode("utf-8")
st.download_button("Download Master Dataset (CSV)", csv, "nexgen_master.csv", "text/csv")
