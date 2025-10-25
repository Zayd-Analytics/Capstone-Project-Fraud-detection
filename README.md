# 🔍 Fraud Detection System

An end-to-end machine learning application built with **Streamlit** to detect fraudulent financial transactions. This interactive dashboard supports real-time and bulk predictions, model evaluation, and retraining — all in one place.

---

## 🚀 Features

- 📊 **Dataset Overview**  
  Upload and explore transaction data with insights on column types, missing values, fraud distribution, and feature correlations.

- 🔮 **Real-Time Prediction**  
  Manually input transaction details and instantly predict fraud probability using a trained model.

- 📦 **Bulk Prediction**  
  Upload a CSV file with multiple transactions and receive fraud predictions for each row. Download results and monitor fraud rate.

- 🧠 **Model Insights**  
  Evaluate model performance using labeled data. View metrics like Accuracy, Precision, Recall, F1 Score, AUC, Confusion Matrix, and ROC Curve.

- 📋 **Classification Report & Fraud Summary**  
  Analyze detailed classification metrics and time-based fraud trends (hour/day). Includes dashboard-level summaries.

- 📊 **Dataset Profiling**  
  Automatically generate numeric and categorical summaries, transaction amount stats, and fraud distribution pie charts.

- 🔁 **Model Retraining**  
  Retrain your model using new labeled data. Choose between RandomForest and XGBoost. Save models with timestamped filenames and track performance metrics.

- 📁 **Saved Models & History**  
  View and download saved models. Track retraining history with AUC, Recall, and Fraud Rate trends. Auto-delete old models to manage storage.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python, Pandas, Scikit-learn, XGBoost  
- **Visualization**: Matplotlib, Seaborn  
- **Model Storage**: Joblib  
- **Logging**: Text-based retraining logs

---
├── app.py               # Main Streamlit app ├── preprocess.py        # Data preprocessing logic ├── retraining_log.txt   # Model retraining history ├── models/              # Saved model files (.pkl) ├── assets/              # Screenshots and visuals

---

## 📸 Screenshots

> Add screenshots from your PowerPoint slides here to showcase each feature.

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py



📬 Contact
Made with ❤️ by Mohd Zayd
Feel free to reach out for collaboration or feedback!

---


## 📂 Folder Structure

