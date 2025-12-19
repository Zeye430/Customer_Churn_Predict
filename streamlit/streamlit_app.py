import json
import os
import requests
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from typing import Any, Dict

# -----------------------------------------------------------------------------
# 1. Page Config (Must be first)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Retention AI",
    page_icon="🛡️",
    layout="wide",  # Use wide mode for a dashboard feel
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom CSS for "FinTech" Look
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa; 
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Metric cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0f172a;
    }
    /* Button styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        height: 50px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        border: none;
        color: white;
    }
    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Config & Constants
# -----------------------------------------------------------------------------
POSSIBLE_PATHS = [
    Path("data/data_schema.json"),
    Path("../data/data_schema.json"),
    Path("/app/data/data_schema.json"),
]

SCHEMA_PATH = None
for path in POSSIBLE_PATHS:
    if path.exists():
        SCHEMA_PATH = path
        break

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# 4. Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_schema(path: Path) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def create_gauge_chart(probability):
    """Creates a professional gauge chart for churn probability."""
    color = "red" if probability > 0.5 else "#00CC96"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability", 'font': {'size': 24, 'color': "#64748b"}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 204, 150, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(255, 75, 75, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -----------------------------------------------------------------------------
# 5. Load Schema
# -----------------------------------------------------------------------------
if SCHEMA_PATH:
    schema = load_schema(SCHEMA_PATH)
    numerical_features = schema.get("numerical", {})
    categorical_features = schema.get("categorical", {})
else:
    st.error("⚠️ Data schema not found. Please ensure 'data_schema.json' exists.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. Sidebar - Input Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=50) # Placeholder Icon
    st.title("Customer Profile")
    st.markdown("Configure the customer parameters below to assess churn risk.")
    st.markdown("---")

    user_input: Dict[str, Any] = {}
    
    # Grouping inputs for better UX
    with st.expander("👤 Demographics", expanded=True):
        # Specific ordering or grouping could go here
        for feature_name in ["age", "gender", "geography"]:
            if feature_name in numerical_features:
                stats = numerical_features[feature_name]
                user_input[feature_name] = st.slider(
                    "Age", 
                    min_value=int(stats["min"]), 
                    max_value=int(stats["max"]), 
                    value=int(stats.get("median", 30))
                )
            elif feature_name in categorical_features:
                user_input[feature_name] = st.selectbox(
                    feature_name.title(), 
                    options=categorical_features[feature_name]["unique_values"]
                )

    with st.expander("💳 Financial Status", expanded=True):
        for feature_name in ["credit_score", "balance", "estimated_salary"]:
            if feature_name in numerical_features:
                stats = numerical_features[feature_name]
                label = feature_name.replace("_", " ").title()
                # Use number_input for large money values, slider for scores
                if "balance" in feature_name or "salary" in feature_name:
                    user_input[feature_name] = st.number_input(
                        label, 
                        value=float(stats.get("median", 50000))
                    )
                else:
                    user_input[feature_name] = st.slider(
                        label, 
                        min_value=int(stats["min"]), 
                        max_value=int(stats["max"]), 
                        value=int(stats.get("median", 650))
                    )

    with st.expander("🏦 Bank Relationship", expanded=True):
        remaining = [f for f in list(numerical_features) + list(categorical_features) 
                     if f not in user_input]
        
        for feature_name in remaining:
            label = feature_name.replace("_", " ").title()
            if feature_name in numerical_features:
                stats = numerical_features[feature_name]
                user_input[feature_name] = st.slider(
                    label, 
                    min_value=float(stats["min"]), 
                    max_value=float(stats["max"]), 
                    value=float(stats.get("median", 1)),
                    step=1.0
                )
            else:
                # Handle boolean-like categories
                options = categorical_features[feature_name]["unique_values"]
                # Pretty print for binary options
                display_opts = ["No", "Yes"] if set(options) == {0, 1} else options
                val = st.radio(label, options=options, format_func=lambda x: "Yes" if x == 1 else ("No" if x==0 else x), horizontal=True)
                user_input[feature_name] = val

    st.markdown("---")
    predict_btn = st.button("Run Prediction ⚡")

# -----------------------------------------------------------------------------
# 7. Main Dashboard Area
# -----------------------------------------------------------------------------

# Header Section
st.markdown("## 🛡️ Customer Retention Intelligence")
st.markdown("Real-time scoring engine based on **LightGBM** & **FastAPI**.")

if not predict_btn:
    # Default State - Welcome Screen
    st.info("👈 Please adjust the customer profile in the sidebar and click **Run Prediction**.")
    
    # Optional: Display some static stats or info
    st.markdown("""
    ### About this Model
    This system analyzes **10+ behavioral and demographic signals** to predict the likelihood of a customer leaving the bank.
    
    * **Architecture:** FastAPI Backend + Streamlit Frontend
    * **Model:** Optuna-Tuned Classifier
    * **Threshold:** > 50% probability indicates High Risk.
    """)

else:
    # Prediction State
    payload = {"instances": [user_input]}
    
    with st.spinner("🔄 Analyzing patterns... connecting to Neural Core..."):
        try:
            response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                pred = data["predictions"][0]
                prob = data["probabilities"][0]
                
                # Layout: 2 Columns
                col_viz, col_details = st.columns([1, 1.5])
                
                with col_viz:
                    # 1. Gauge Chart
                    st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
                
                with col_details:
                    st.write("### Analysis Report")
                    
                    # 2. Status Banner
                    if pred == 1:
                        st.error(f"**Status: HIGH RISK**")
                        msg = "⚠️ This customer is likely to churn. Immediate retention intervention is recommended."
                    else:
                        st.success(f"**Status: LOYAL**")
                        msg = "✅ This customer shows stable behavior. Maintain current relationship strategy."
                    
                    st.markdown(msg)
                    
                    # 3. Key Metrics Row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Credit Score", user_input['credit_score'])
                    m2.metric("Products", user_input['num_of_products'])
                    m3.metric("Tenure (Yrs)", user_input['tenure'])
                    
                    # 4. JSON Expander (Hidden by default)
                    with st.expander("View Raw API Response"):
                        st.json(data)
                        
            else:
                st.error(f"Server Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.warning(f"Ensure FastAPI is running at `{API_BASE_URL}`")