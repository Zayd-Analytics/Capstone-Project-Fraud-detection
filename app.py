import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import shap
import time
import os

# ---------------------------#
#         APP CONFIG
# ---------------------------#
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Fraud Detection System")
st.markdown("### End-to-End Machine Learning App for Detecting Transaction Fraud")

# ---------------------------#
#         LOAD MODEL
# ---------------------------#
@st.cache_resource
def load_model():
    return joblib.load("random_forest_realtime.pkl")

model = load_model()

# ---------------------------#
#         FUNCTIONS
# ---------------------------#
def preprocess_data(df):
    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['log_amount'] = np.log1p(df['amount'])
    df['relative_amount'] = df['amount'] / (df['oldbalanceOrg'] + 1e-5)
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24
    df['high_amount'] = (df['amount'] > 100000).astype(int)
    df['high_transfer'] = df['high_amount'] * (df['type'] == 'TRANSFER').astype(int)

    # Always create all expected type columns
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    for col in ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    df = pd.concat([df, type_dummies], axis=1)
    df = df.drop(columns=['type', 'nameOrig', 'nameDest'], errors='ignore')

    expected_cols = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'balanceDiffOrig', 'balanceDiffDest', 'log_amount', 'relative_amount', 'hour', 'day',
        'high_amount', 'high_transfer',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match model
    if hasattr(model, 'feature_names_in_'):
        df = df[model.feature_names_in_]
    else:
        df = df[expected_cols]

    # Scale numeric columns (if needed)
    scaler = StandardScaler()
    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest', 'log_amount']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def download_csv(df):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

def plot_feature_importance(model, features):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='coolwarm', ax=ax)
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ---------------------------#
#         NAVIGATION
# ---------------------------#
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "📈 Data Overview",
    "⚙️ Real-time Prediction",
    "📤 Bulk Prediction",
    "📊 Model Insights",
    "💡 DATASET OVERVIEW SECTION",
])

# ---------------------------#
#         HOME PAGE
# ---------------------------#
if page == "🏠 Home":
    st.subheader("🚀 Welcome to the Fraud Detection Dashboard")
    st.write("""
    This interactive web app demonstrates a **Random Forest–based Fraud Detection System**.
    You can:
    - Explore the dataset  
    - Perform **real-time predictions** on single transactions  
    - Upload CSVs for **bulk fraud detection**
    - Analyze **model insights and feature importances**
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721260.png", width=250)
    st.success("Model loaded successfully and ready to detect fraud in real time.")

# ---------------------------#
#     DATA OVERVIEW PAGE
# ---------------------------#
elif page == "📈 Data Overview":
    st.subheader("📊 Data Exploration")
    uploaded_file = st.file_uploader("Upload a sample dataset for EDA", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        if 'isFraud' in df.columns:
            st.write("#### Class Distribution")
            st.bar_chart(df['isFraud'].value_counts())

        try:
            numeric_df = df.select_dtypes(include=['number'])
            st.write("#### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

# ---------------------------#
#    REAL-TIME PREDICTION
# ---------------------------#
elif page == "⚙️ Real-time Prediction":
    st.subheader("🔮 Real-time Fraud Prediction")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0)
        newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=4000.0)
    with col2:
        oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=2000.0)
        newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=3000.0)
        step = st.number_input("Transaction Step (hour index)", min_value=1, value=5)
        type_choice = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT"])

    if st.button("🚨 Predict Transaction"):
        input_df = pd.DataFrame({
            'step': [step],
            'type': [type_choice],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
            'nameOrig': ['C12345'],
            'nameDest': ['M12345']
        })

        processed = preprocess_data(input_df)
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1] * 100

        if pred == 1:
            st.error(f"🚨 Fraud Detected! Probability: {prob:.2f}%")
        else:
            st.success(f"✅ Legit Transaction. Fraud Probability: {prob:.2f}%")

        st.progress(int(prob))

# ---------------------------#
#    BULK PREDICTION
# ---------------------------#
elif page == "📤 Bulk Prediction":
    st.subheader("📦 Bulk Fraud Prediction")
    uploaded_csv = st.file_uploader("Upload a CSV file (similar to training data)", type=["csv"])

    if uploaded_csv:
        df_input = pd.read_csv(uploaded_csv)
        st.write("### Uploaded Data Preview")
        st.dataframe(df_input.head())

        df_processed = preprocess_data(df_input.copy())
        preds = model.predict(df_processed)
        df_input["Predicted_Fraud"] = preds

        st.write("### Predictions")
        st.dataframe(df_input.head())

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=download_csv(df_input),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        fraud_rate = df_input["Predicted_Fraud"].mean() * 100
        st.metric("Fraudulent Transactions (%)", f"{fraud_rate:.2f}%")

# ---------------------------#
#    MODEL INSIGHTS PAGE
# ---------------------------#
elif page == "📊 Model Insights":
    st.subheader("🧠 Model Insights & Explainability")

    uploaded_eval = st.file_uploader("📤 Upload labeled dataset (for evaluation)", type=["csv"])

    if uploaded_eval:
        df_eval = pd.read_csv(uploaded_eval)
        if 'isFraud' not in df_eval.columns:
            st.error("❌ Dataset must include 'isFraud' column for evaluation.")
        else:
            # Preprocess & Predict
            df_processed = preprocess_data(df_eval.copy())
            preds = model.predict(df_processed)
            probs = model.predict_proba(df_processed)[:, 1]

            # Compute Metrics
            from sklearn.metrics import (
                confusion_matrix,
                classification_report,
                roc_auc_score,
                precision_score,
                recall_score,
                f1_score,
                accuracy_score,
                roc_curve
            )

            acc = accuracy_score(df_eval["isFraud"], preds)
            rec = recall_score(df_eval["isFraud"], preds)
            prec = precision_score(df_eval["isFraud"], preds)
            f1 = f1_score(df_eval["isFraud"], preds)
            auc = roc_auc_score(df_eval["isFraud"], probs)

            # 📊 Metric Summary
            st.write("### 🧾 Evaluation Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Recall", f"{rec:.3f}")
            col3.metric("Precision", f"{prec:.3f}")
            col4.metric("F1 Score", f"{f1:.3f}")
            col5.metric("AUC", f"{auc:.3f}")

            # Confusion Matrix
            st.write("### 📉 Confusion Matrix")
            cm = confusion_matrix(df_eval["isFraud"], preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ROC Curve
            st.write("### 🩸 ROC Curve")
            fpr, tpr, _ = roc_curve(df_eval["isFraud"], probs)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

            # Classification Report
            st.write("### 📋 Detailed Classification Report")
            report = classification_report(df_eval["isFraud"], preds, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='coolwarm', axis=0))

            # Fraud Detection Summary
            fraud_rate = df_eval["isFraud"].mean() * 100
            pred_fraud_rate = preds.mean() * 100
            st.write("### 🧮 Fraud Detection Summary")
            colA, colB = st.columns(2)
            colA.metric("Actual Fraud Rate", f"{fraud_rate:.2f}%")
            colB.metric("Predicted Fraud Rate", f"{pred_fraud_rate:.2f}%")

            # -------------------- Dashboard Metrics --------------------
            st.subheader("📊 Dashboard Metrics")

            # Basic metrics
            total_txns = len(df_eval)
            fraud_txns = df_eval["isFraud"].sum()
            fraud_rate = (fraud_txns / total_txns) * 100
            avg_amount = df_eval["amount"].mean()
            common_type = df_eval["type"].mode()[0] if "type" in df_eval.columns else "N/A"

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total_txns)
            col2.metric("Fraudulent Transactions", fraud_txns)
            col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

            col4, col5 = st.columns(2)
            col4.metric("Avg. Transaction Amount", f"{avg_amount:,.2f}")
            col5.metric("Most Common Type", common_type)

            # Time-based fraud trends
            if "step" in df_eval.columns:
                df_eval["hour"] = df_eval["step"] % 24
                df_eval["day"] = df_eval["step"] // 24

                st.write("### ⏱️ Fraud by Hour of Day")
                hourly_fraud = df_eval[df_eval["isFraud"] == 1]["hour"].value_counts().sort_index()
                st.bar_chart(hourly_fraud)
                st.write("### 📅 Fraud by Day")
                daily_fraud = df_eval[df_eval["isFraud"] == 1]["day"].value_counts().sort_index()
                st.line_chart(daily_fraud)

# ---------------------------#
#    DATASET OVERVIEW SECTION
# ---------------------------#
elif page == "💡 DATASET OVERVIEW SECTION":
    st.subheader("📊 Dataset Overview")

    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded successfully! Shape: {df.shape}")

        # --- Basic Info ---
        st.write("### 🧾 Basic Information")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")

        # --- Column Types ---
        st.write("### 📂 Column Type Summary")
        col_types = df.dtypes.value_counts()
        st.bar_chart(col_types)

        # --- Missing Values ---
        st.write("### ⚠️ Missing Values Summary")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.dataframe(missing.rename("Missing Values"))
        else:
            st.info("No missing values detected ✅")

        # --- Numeric Summary ---
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.write(f"### 🔢 Numeric Columns ({len(numeric_cols)})")
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().T)
        else:
            st.warning("No numeric columns found.")

        # --- Categorical Summary ---
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        st.write(f"### 🏷️ Categorical Columns ({len(categorical_cols)})")
        if len(categorical_cols) > 0:
            cat_summary = pd.DataFrame({
                "Unique Values": [df[col].nunique() for col in categorical_cols],
                "Most Frequent": [df[col].mode()[0] if not df[col].mode().empty else None for col in categorical_cols]
            }, index=categorical_cols)
            st.dataframe(cat_summary)
        else:
            st.warning("No categorical columns found.")

        # --- Summary Insights ---
        st.markdown("### 💡 Dataset Summary Report")
        with st.expander("Click to view summary insights"):
            st.write(f"**Total Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

            if 'amount' in df.columns:
                st.write(f"- 💰 **Average Transaction Amount:** {df['amount'].mean():,.2f}")
                st.write(f"- 💵 **Median Transaction Amount:** {df['amount'].median():,.2f}")
                st.write(f"- 💸 **Transaction Range:** {df['amount'].min():,.2f} → {df['amount'].max():,.2f}")

            if 'type' in df.columns:
                common_type = df['type'].mode()[0]
                st.write(f"- 🔁 **Most Common Transaction Type:** {common_type}")
                st.markdown("#### 🔁 Transaction Type Breakdown")
                st.bar_chart(df['type'].value_counts())

            if 'isFraud' in df.columns:
                total_frauds = df['isFraud'].sum()
                fraud_rate = (total_frauds / len(df)) * 100
                st.write(f"- ⚠️ **Total Frauds:** {total_frauds}")
                st.write(f"- 🚨 **Fraud Rate:** {fraud_rate:.2f}%")

                st.markdown("#### 🚨 Fraud Distribution")
                c1, c2 = st.columns(2)
                c1.metric("⚠️ Total Frauds", f"{total_frauds:,}")
                c2.metric("🔥 Fraud Rate (%)", f"{fraud_rate:.2f}%")

                fig, ax = plt.subplots(figsize=(4, 4))
                labels = ['Legit', 'Fraud']
                sizes = [len(df) - total_frauds, total_frauds]
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#16a34a', '#dc2626'])
                st.pyplot(fig)

            st.success("✅ Summary generated successfully!")

        # --- Retrain Model Section ---
        st.markdown("---")
        st.subheader("🔁 Retrain Model")

        model_type = st.selectbox("Choose model type", ["RandomForest", "XGBoost"])
        uploaded_train = st.file_uploader("Upload CSV with 'isFraud' column", type=["csv"], key="retrain_csv")

        if uploaded_train:
            try:
                df_train = pd.read_csv(uploaded_train)

                if 'isFraud' not in df_train.columns:
                    st.error("Uploaded data must include an 'isFraud' column.")
                else:
                    st.write("### Preview of Uploaded Data")
                    st.dataframe(df_train.head())

                    df_X = preprocess_data(df_train.drop(columns=['isFraud']))
                    df_y = df_train['isFraud']

                    if st.button("⚠️ Confirm Retraining"):
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                        model_filename = f"{model_type}_model_{timestamp}.pkl"

                        if model_type == "RandomForest":
                            from sklearn.ensemble import RandomForestClassifier
                            new_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif model_type == "XGBoost":
                            from xgboost import XGBClassifier
                            new_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')

                        new_model.fit(df_X, df_y)

                        from sklearn.metrics import roc_auc_score, recall_score
                        y_pred = new_model.predict(df_X)
                        y_prob = new_model.predict_proba(df_X)[:, 1]
                        auc = roc_auc_score(df_y, y_prob)
                        recall = recall_score(df_y, y_pred)

                        joblib.dump(new_model, model_filename)

                        with open("retraining_log.txt", "a") as log_file:
                            log_file.write(f"{timestamp} | {model_type} | {len(df_train)} samples | Fraud rate: {df_y.mean():.4f} | AUC: {auc:.3f} | Recall: {recall:.3f}\n")

                        st.success(f"✅ {model_type} model retrained and saved as `{model_filename}`")
                        st.metric("Training Samples", len(df_train))
                        st.metric("Fraud Rate in Training Data", f"{df_y.mean() * 100:.2f}%")
                        st.metric("AUC Score", f"{auc:.3f}")
                        st.metric("Recall", f"{recall:.3f}")

                        c1, c2 = st.columns(2)
                        total_frauds = df_y.sum()
                        fraud_rate = (total_frauds / len(df_y)) * 100
                        c1.metric("⚠️ Total Frauds", f"{total_frauds:,}")
                        c2.metric("🔥 Fraud Rate (%)", f"{fraud_rate:.2f}%")

                        fig, ax = plt.subplots(figsize=(4, 4))
                        labels = ['Legit', 'Fraud']
                        sizes = [len(df_y) - total_frauds, total_frauds]
                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#16a34a', '#dc2626'])
                        st.pyplot(fig)

                        st.success("✅ Data summary generated successfully!")

            except Exception as e:
                st.error(f"Could not process training data: {e}")

        # --- Show saved models ---
        st.write("### 📁 Saved Models")
        model_files = [f for f in os.listdir() if f.endswith(".pkl")]
        selected_model = st.selectbox("🎯 Select a model to activate", model_files if model_files else ["No models found"])
        if selected_model and selected_model != "No models found":
            st.success(f"✅ Selected model: `{selected_model}`")
            # You can load and use this model later like:
            # active_model = joblib.load(selected_model)

        for file in model_files:
            st.download_button(label=f"Download {file}", data=open(file, "rb").read(), file_name=file)

        # --- Visualize retraining history ---
        st.write("### 📈 Retraining History")
        if os.path.exists("retraining_log.txt"):
            with open("retraining_log.txt", "r") as log_file:
                lines = log_file.readlines()
            history = [line.strip().split(" | ") for line in lines]
            if history:
                hist_df = pd.DataFrame(history, columns=["Timestamp", "Model", "Samples", "Fraud Rate", "AUC", "Recall"])
                hist_df["Samples"] = hist_df["Samples"].str.extract(r"(\d+)").astype(int)
                hist_df["Fraud Rate"] = hist_df["Fraud Rate"].str.extract(r"([0-9.]+)").astype(float)
                hist_df["AUC"] = hist_df["AUC"].str.extract(r"([0-9.]+)").astype(float)
                hist_df["Recall"] = hist_df["Recall"].str.extract(r"([0-9.]+)").astype(float)
                st.dataframe(hist_df)
                st.line_chart(hist_df.set_index("Timestamp")[["Fraud Rate", "AUC", "Recall"]])
            else:
                st.info("No retraining history found.")
        else:
            st.info("Retraining log file not found.")

        # --- Auto-delete old models ---
        st.write("### 🧹 Auto-Delete Old Models")
        enable_cleanup = st.checkbox("Enable auto-delete for models older than 7 days", value=True)

        def delete_old_models(days=7):
            if enable_cleanup:
                now = time.time()
                for file in os.listdir():
                    if file.endswith(".pkl"):
                        file_time = os.path.getmtime(file)
                        if (now - file_time) > (days * 86400):
                            os.remove(file)
                            print(f"Deleted old model: {file}")

        delete_old_models(days=7)

    else:
        st.info("Upload a dataset above to begin analysis.")

        st.markdown("### 💡 Dataset Summary Report")
        with st.expander("Click to view summary insights"):
            st.warning("No dataset loaded. Please upload a CSV file first.")