# 🏦 End-to-End Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)

## 📖 Overview

This project is an end-to-End Machine Learning system designed to predict **Bank Customer Churn**. It helps financial institutions identify customers at risk of leaving (churning) so they can take proactive retention measures.

The system is not just a model; it is a full-stack ML application that includes:
1.  **Robust ML Pipeline**: Custom feature engineering and automated preprocessing.
2.  **Model Optimization**: Bayesian hyperparameter tuning using **Optuna**.
3.  **Microservices Architecture**: Decoupled **FastAPI** backend and **Streamlit** frontend.
4.  **Containerization**: Fully Dockerized for easy deployment (e.g., on DigitalOcean).

## 🚀 Key Features

* **Advanced Feature Engineering**:
    * Implements custom transformers for feature creation (e.g., `Balance/Salary Ratio`, `Tenure/Age Ratio`).
    * Handles long-tail distributions (e.g., `Age`, `Balance`) using Log Transformations.
* **AutoML & Tuning**:
    * Utilizes **Optuna** to optimize hyperparameters for 4 different algorithms: **Ridge, HistGradientBoosting, XGBoost, and LightGBM**.
    * Compares model performance with and without **PCA** (Principal Component Analysis).
* **Production-Ready API**:
    * FastAPI endpoint (`/predict`) that validates input schema and handles real-time inference.
    * Ensures training-serving skew is minimized by sharing the exact same pipeline (`churning_pipeline.py`) across training and inference.
* **User-Friendly Interface**:
    * An interactive Streamlit dashboard for business users to input customer data and visualize risk probabilities.

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, Pandas, NumPy.
* **Experiment Tracking & Tuning**: Optuna, MLflow.
* **Web Frameworks**: FastAPI (Backend), Streamlit (Frontend).
* **DevOps**: Docker, Docker Compose.

## 📂 Project Structure

```bash
├── api/
│   ├── Dockerfile              # Backend container config
│   ├── app.py                  # FastAPI application entry point
│   ├── churning_pipeline.py    # Shared ML pipeline (copied for inference)
│   └── requirements.txt
├── streamlit/
│   ├── Dockerfile              # Frontend container config
│   ├── ui.py                   # Streamlit dashboard code
│   └── requirements.txt
├── models/
│   └── global_best_model_optuna.pkl  # Trained serialized model
├── notebooks/
│   ├── 04_train_model_without_optuna.ipynb
│   └── 05_train_models_with_optuna.ipynb # Main training & tuning workflow
├── churning_pipeline.py        # SOURCE OF TRUTH: Custom Transformers & Pipeline
├── docker-compose.yml          # Orchestration for API & Streamlit services
└── README.md