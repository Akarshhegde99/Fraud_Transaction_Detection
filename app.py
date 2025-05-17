import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# Load model
model = joblib.load('fraud_model.pkl')

# File uploader
uploaded_file = st.file_uploader("📥 Upload transaction CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:")
    st.dataframe(data.head())

    if 'Class' in data.columns:
        X = data.drop('Class', axis=1)
        y_true = data['Class']
    else:
        X = data
        y_true = None

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Results
    st.subheader("🔍 Prediction Results")
    results_df = X.copy()
    results_df['Predicted Label'] = y_pred
    if y_true is not None:
        results_df['Actual Label'] = y_true
        st.dataframe(results_df.head(20))
    else:
        st.write("No 'Class' column found. Showing predicted labels only.")
        st.dataframe(results_df.head(20))

    # Metrics & Plots
    if y_true is not None:
        st.subheader("📉 Metrics & Evaluation")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True)
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label="Precision-Recall curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Recall")
        ax.legend()
        st.pyplot(fig)

    # Display counts of Fraud and Legit predictions
    st.subheader("📊 Transaction Summary")
    fraud_count = np.sum(y_pred == 1)
    legit_count = np.sum(y_pred == 0)
    st.write(f"**Fraud Transactions Detected:** {fraud_count}")
    st.write(f"**Legitimate Transactions Detected:** {legit_count}")
