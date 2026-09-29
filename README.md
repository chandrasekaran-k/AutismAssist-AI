# 🧩 AutismAssist AI — Screening & Therapy Support Prototype

An end-to-end machine learning project for **screening support and therapy/support-area prediction** using child developmental and behavioral features.

> **Important:** This is a portfolio/research prototype, not a diagnostic system or medical device. Predictions must not be used as a substitute for assessment by qualified professionals.

## 🎯 Project Objectives

- Build a machine-learning screening support model for `ASD_Status`
- Explore developmental and behavioral indicators
- Build a multi-label model for support areas
- Provide explainable model outputs
- Create a simple Streamlit interface for demonstration
- Package the work as a reproducible GitHub portfolio project

## 🔄 Workflow

```text
Raw Dataset
    ↓
Data Understanding & EDA
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
ASD Screening Model
    ↓
Therapy/Support Multi-Label Model
    ↓
Explainability & Evaluation
    ↓
Streamlit Demo
```

## 📊 Dataset

The supplied dataset contains **3,000 records and 18 columns**.

### Screening features
- Age
- Gender
- Eye Contact
- Response to Name
- Communication Level
- Joint Attention
- Sensory Regulation
- Motor Skills
- ADL Skills
- Hyperactivity Control
- Social Interaction

### Targets
- `ASD_Status`
- `Speech_Therapy_Need`
- `Occupational_Therapy_Need`
- `Behaviour_Therapy_Need`
- `Parent_Training_Need`
- `Social_Skills_Support_Need`

## 🤖 Machine Learning

### ASD Screening
A Random Forest classification pipeline is provided with:
- categorical encoding
- numerical scaling
- class balancing
- train/test evaluation
- probability output

### Support Recommendation
A MultiOutput Random Forest model predicts five support-area labels simultaneously.

## 📈 Current Baseline Results

The included dataset produced the following hold-out baseline for the screening model:

- Accuracy: approximately **60%**
- F1: approximately **45%**
- ROC-AUC: approximately **59%**

These are **dataset-specific baseline results**, not clinical performance claims. The relatively modest performance is useful for demonstrating why validation, calibration, better data quality, and external testing matter.

## 🧠 Explainability

The project includes feature-importance output and a notebook for model comparison and explainability.

## 🖥️ Streamlit Demo

Run:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app provides:
1. ASD screening-support probability
2. Multi-label support-area predictions

## 📁 Project Structure

```text
AutismAssist_AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── asd_screening_dataset.csv
│   └── processed/
│       └── asd_screening_processed.csv
│
├── models/
│   ├── asd_screening_model.joblib
│   └── therapy_recommendation_model.joblib
│
├── notebooks/
│   ├── 01_Data_Understanding_EDA.ipynb
│   ├── 02_ASD_Screening_Model.ipynb
│   ├── 03_Therapy_Recommendation_Model.ipynb
│   └── 04_Model_Comparison_And_Explainability.ipynb
│
├── reports/
│   ├── screening_model_metrics.csv
│   ├── screening_feature_importance.csv
│   └── therapy_model_metrics.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_screening.py
│   └── recommend_therapy.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Joblib
- Streamlit

## ⚠️ Responsible AI

This project demonstrates machine-learning engineering concepts using a supplied dataset. It does **not** establish a clinical diagnosis, treatment plan, or medical recommendation. Real-world deployment would require clinically validated labels, representative data, bias/fairness testing, calibration, external validation, privacy controls, human oversight, and appropriate regulatory review.

## 👨‍💻 Author

**Chandrasekaran K**
