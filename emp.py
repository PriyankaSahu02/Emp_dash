import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# Dummy user credentials
USER_CREDENTIALS = {
    "user": "password123",
    "admin": "admin123"
}

# Login function
def login():
    st.sidebar.header("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"): 
        if USER_CREDENTIALS.get(username) == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# ------------------- Check Login Status -------------------
def logout():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# Check login status
if not st.session_state.get("logged_in"):
    login()
else:
    logout()


    # ------------------- Load Data & Models -------------------

    # Download function
    def download_file(file_id, filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = f"./{filename}"
        if not os.path.exists(output_path):  # Avoid re-downloading if exists
            gdown.download(url, output_path, quiet=False)

    # Define Google Drive file IDs
    file_ids = {
        "xgb_model": "1Cg7uHDK6NHESc9CZnlOj9H-B9C_4-eqC",
        "rf_model": "1R_wyGXaIvlz41Nr3NYGNsS-BdUCcSfZE",
        "logreg_model": "1qnhYMvLxkO45f9ZZ82VSDill98VCXP0j",
        "training_features": "1ngeoV5hpluHeCPdmGTEdjOfviH0x3PfU",
        "final_table": "1stX1gW9kkbkR3hSIrBvRJLv_W7D7RYI4"
    }

    # Download all necessary files
    for name, file_id in file_ids.items():
        download_file(file_id, f"{name}.pkl" if name != "final_table" else "final_table.csv")

    # Load ML models
    @st.cache_resource
    def load_models():
        return {
            "xgb": joblib.load("xgb_model.pkl"),
            "rf": joblib.load("rf_model.pkl"),
            "logreg": joblib.load("logreg_model.pkl")
        }

    models = load_models()
    xgb_model, rf_model, logreg_model = models["xgb"], models["rf"], models["logreg"]

    # Load training features
    try:
        training_features = joblib.load("training_features.pkl")
    except FileNotFoundError:
        training_features = None  # Handle missing training_features gracefully


    # Load dataset
    df = pd.read_csv("final_table.csv")
    # ------------------- Streamlit UI -------------------

    st.title("🔮 Employee Attrition Prediction Dashboard")

    # Sidebar - User Input
    st.sidebar.header("⚙️ Prediction Inputs")
    tenure = st.sidebar.slider("Years at Company", 0, 29, 10)
    age = st.sidebar.slider("Age", 20, 61, 35)
    salary = st.sidebar.number_input("Salary", min_value=40000, max_value=122000, value=80000)
    no_of_projects = st.sidebar.slider("Number of Projects", 1, 10, 7)
    performance_ratings = ['PIP', 'S', 'C', 'B', 'A']
    genders = ["M", "F"]
    emp_title_id = st.sidebar.selectbox("Job Title ID", df["emp_title_id"].unique().tolist())
    titles = df["title"].unique().tolist() if "title" in df.columns else ["Engineer", "Manager", "Staff"]
    # Extract unique department values
    departments = sorted(df['primary_dept_name'].dropna().unique())
    other_departments = sorted(df['other_dept_name'].dropna().unique())


    # Dropdown Inputs
    performance_rating = st.sidebar.selectbox("Last Performance Rating", performance_ratings)
    sex = st.sidebar.selectbox("Gender", genders)
    primary_dept = st.sidebar.selectbox("Primary Department", departments)
    other_dept = st.sidebar.selectbox("Other Department", other_departments)
    title = st.sidebar.selectbox("Job Title", titles)


    # ------------------- Prediction Function -------------------
    def predict_attrition():
        input_data = pd.DataFrame({
            "tenure": [tenure],
            "age": [age],
            "salary": [salary],
            "no_of_projects": [no_of_projects],
            "last_performance_rating": [performance_rating],
            "sex": [sex],
            "primary_dept": [primary_dept],
            "other_dept": [other_dept],
            "emp_title_id": [emp_title_id],
            "title": [title]
        })

        # Encode categorical variables
        performance_order = {'PIP': 0, 'S': 1, 'C': 2, 'B': 3, 'A': 4}
        input_data['last_performance_rating'] = input_data['last_performance_rating'].map(performance_order).fillna(0).astype(int)
        
        # One-hot encode categorical variables dynamically
        categorical_features = ['emp_title_id', 'sex', 'title']
        input_data = pd.get_dummies(input_data, columns=categorical_features, dtype=bool)

        # Ensure input data matches model features
        if training_features is not None:
            for feature in training_features:
                if feature not in input_data.columns:
                    input_data[feature] = False if feature.startswith(('emp_title_id_', 'sex_', 'title_')) else 0  # Default values

            # Reorder columns to match training features
            input_data = input_data[training_features]
            
        # Predict probabilities
        xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
        rf_prob = rf_model.predict_proba(input_data)[:, 1]
        logreg_prob = logreg_model.predict_proba(input_data)[:, 1]

        # Ensemble Prediction (Threshold tuned at 0.4)
        threshold = 0.4
        ensemble_prob = (xgb_prob + rf_prob + logreg_prob) / 3
        final_pred = (ensemble_prob > threshold).astype(int)[0]

        return xgb_prob, rf_prob, logreg_prob, ensemble_prob, final_pred


    # ------------------- Gauge Chart Function -------------------
    def display_gauge_chart(prob):
        import numpy as np
        
        # Ensure prob is a float
        if isinstance(prob, np.ndarray):
            prob = prob.item()  # Safely extract a single value

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Attrition Probability"},
            gauge={"axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob > 0.5 else "green"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 100], "color": "lightcoral"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100
                }
                }
        ))
        st.plotly_chart(fig, use_container_width=True)

    def plot_model_comparison(xgb_prob, rf_prob, logreg_prob):
        import numpy as np

        # Ensure all values are floats
        xgb_prob = xgb_prob.item() if isinstance(xgb_prob, np.ndarray) else float(xgb_prob)
        rf_prob = rf_prob.item() if isinstance(rf_prob, np.ndarray) else float(rf_prob)
        logreg_prob = logreg_prob.item() if isinstance(logreg_prob, np.ndarray) else float(logreg_prob)

        # Round to 2 decimal places
        probabilities = [round(xgb_prob, 2), round(rf_prob, 2), round(logreg_prob, 2)]
        models = ["XGBoost", "Random Forest", "Logistic Regression"]

        fig = px.bar(
            x=models,
            y=probabilities,
            labels={"x": "Model", "y": "Predicted Probability"},
            title="Model Probability Comparison",
            text=probabilities,
            color=["XGBoost", "Random Forest", "Logistic Regression"]
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig)



    # ------------------- Main Dashboard -------------------

    if st.button("🔮 Predict Attrition"):
        # Get model predictions
        results = predict_attrition()

        # Unpack results properly
        if results and len(results) == 5:
            xgb_prob, rf_prob, logreg_prob, attrition_prob, prediction = results

            # Ensure attrition_prob is a single scalar value
            if isinstance(attrition_prob, np.ndarray):
                attrition_prob = attrition_prob.item()

            # Use it safely without warnings
            st.metric("📊 Attrition Probability", f"{attrition_prob:.2%}")
            display_gauge_chart(attrition_prob)

            xgb_prob = xgb_prob.item()
            rf_prob = rf_prob.item()
            logerg_prob = logreg_prob.item()

            # Show Risk Message
            if prediction == 1:
                st.error("⚠️ High Attrition Risk!")
            else:
                st.success("✅ Low Attrition Risk")

            # Compare Model Predictions
            plot_model_comparison(xgb_prob, rf_prob, logreg_prob)
        else:
            st.error("❌ Prediction function did not return expected values.")
