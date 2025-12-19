🏦 End-to-End Bank Customer Churn Prediction📖 OverviewThis project is an end-to-End Machine Learning system designed to predict Bank Customer Churn. It helps financial institutions identify customers at risk of leaving (churning) so they can take proactive retention measures.The system is not just a model; it is a full-stack ML application that includes:Robust ML Pipeline: Custom feature engineering and automated preprocessing.Model Optimization: Bayesian hyperparameter tuning using Optuna.Microservices Architecture: Decoupled FastAPI backend and Streamlit frontend.Containerization: Fully Dockerized for easy deployment (e.g., on DigitalOcean).🚀 Key FeaturesAdvanced Feature Engineering:Implements custom transformers for feature creation (e.g., Balance/Salary Ratio, Tenure/Age Ratio).Handles long-tail distributions (e.g., Age, Balance) using Log Transformations.AutoML & Tuning:Utilizes Optuna to optimize hyperparameters for 4 different algorithms: Ridge, HistGradientBoosting, XGBoost, and LightGBM.Compares model performance with and without PCA (Principal Component Analysis).Production-Ready API:FastAPI endpoint (/predict) that validates input schema and handles real-time inference.Ensures training-serving skew is minimized by sharing the exact same pipeline (churning_pipeline.py) across training and inference.User-Friendly Interface:An interactive Streamlit dashboard for business users to input customer data and visualize risk probabilities.🛠️ Tech StackLanguage: Python 3.10+Machine Learning: Scikit-Learn, XGBoost, LightGBM, Pandas, NumPy.Experiment Tracking & Tuning: Optuna, MLflow.Web Frameworks: FastAPI (Backend), Streamlit (Frontend).DevOps: Docker, Docker Compose.📂 Project StructureBash├── api/
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
📊 Model PerformanceAfter running 80 trials using Optuna to optimize four different model families, LightGBM was selected as the champion model.MetricScoreDetailsModelLightGBMBest performing algorithmTest F1 Score0.6061Harmonized metric for precision/recallStrategyNo PCARaw features + Custom Engineering performed bestSource: 05_train_models_with_optuna.ipynb⚡ Quick Start (Run with Docker)The easiest way to run the application is using Docker Compose.Prerequisites: Ensure Docker Desktop is installed.Clone the repository:Bashgit clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
Build and Start Services:Bashdocker-compose up -d --build
Access the Application:Frontend (Streamlit): Open http://localhost:8501 in your browser.Backend Docs (FastAPI): Open http://localhost:8000/docs to test the API directly.Stop the Application:Bashdocker-compose down
🔧 Local Development (Without Docker)If you prefer to run it locally for development:Install Dependencies:Bashpip install -r api/requirements.txt
pip install -r streamlit/requirements.txt
Run the API:Bashcd api
uvicorn app:app --reload --port 8000
Run the Frontend:Bashcd streamlit
streamlit run ui.py
🔮 Future ImprovementsModel Monitoring: Integrate Prometheus/Grafana to monitor data drift and model decay in production.CI/CD Pipeline: Automate testing and deployment using GitHub Actions.Explainability: Integrate SHAP values into the Streamlit dashboard to explain why a customer is flagged as high risk.👨‍💻 AuthorWilliamGraduate Student in Data Science