import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import os
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Dashboard", layout="wide")

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
        "xgb_model": "1u_lABge8raBhST7PRJ3SiK580X2YykDy",
        "rf_model": "1LVxpoUhuzM2FyZbmhogiK74c8InJ_-JL",
        "logreg_model": "1LVxpoUhuzM2FyZbmhogiK74c8InJ_-JL",
        "scaler": "1sh9D1sMUT07oJ-F4DcZKAzKN4IsKTjfv",
        "final_table": "1GZxktRcIr7OjxNXG_CCEzJ2dWvgeH9Bz",
        "X_test": "1Zjh-o0dqnfm_R3qWo3CdixhLi4j7YAJi",
        "y_test": "1izmgtZej0dAvcnp9sWXyHLAtDXN7rbEj"
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

    # Load scaler
    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        scaler = None  # Handle missing scaler gracefully

    # Load dataset
    df = pd.read_csv("final_table.csv")

    # -------------------------------------------------------

    # ------------------- Sidebar Filters -------------------
    st.sidebar.header("🔍 Employee Filters")
    
    # Define department list (including "All" as an option)
    departments = sorted(df["primary_dept_name"].unique())

    primary_departments = ["All"] + sorted(df["primary_dept_name"].unique().tolist())
    selected_primary_dept = st.sidebar.selectbox("Select Primary Department", primary_departments)

    other_departments = ["All"] + sorted(df["other_dept_name"].unique().tolist())
    selected_other_dept = st.sidebar.selectbox("Select Other Department", other_departments)

    # Title Filter
    titles = df["title"].unique()
    selected_title = st.sidebar.selectbox("Select Job Title", ["All"] + list(titles))

    genders = ["All", "M", "F"]
    selected_gender = st.sidebar.selectbox("Select Gender", genders)

    age_range = st.sidebar.slider("Select Age Range", int(df["age"].min()), int(df["age"].max()), (25, 50))
    tenure_range = st.sidebar.slider("Select Tenure Range (Years)", 0, 16, (2, 10))

    performance_ratings = ["All"] + sorted(df["last_performance_rating"].dropna().unique().tolist())
    selected_rating = st.sidebar.selectbox("Select Performance Rating", ["All"] + performance_ratings)

    salary_range = st.sidebar.slider("Salary Range", int(df["salary"].min()), int(df["salary"].max()), (40000, 120000))

        # Apply Filters
    filtered_df = df.copy()
    if selected_primary_dept != "All":
        filtered_df = filtered_df[filtered_df["primary_dept_name"] == selected_primary_dept]
    if selected_other_dept == "None":
        filtered_df = filtered_df[filtered_df["other_dept_name"].isna()]    
    if selected_other_dept != "All":
        filtered_df = filtered_df[filtered_df["other_dept_name"] == selected_other_dept]
    if selected_title != "All":
        filtered_df = filtered_df[filtered_df["title"] == selected_title]    
    if selected_rating != "All":
        filtered_df = filtered_df[filtered_df["last_performance_rating"] == selected_rating]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["sex"] == selected_gender]

    filtered_df = filtered_df[
        (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1]) &
        (filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1]) &
        (filtered_df["salary"] >= salary_range[0]) & (filtered_df["salary"] <= salary_range[1])
    ]

    # ------------------- Display Filtered Data -------------------
    st.subheader(f"📄 Employee Data ({len(filtered_df)} Records)")
    st.dataframe(filtered_df)


    # ------------------- Attrition & Salary Analysis -------------------
    st.subheader("📊 Employee Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Dynamic Attrition Rate by Department (Primary + Other)")

        # Combine Primary & Other Department Data
        dept_data = filtered_df.melt(
            id_vars=["left"], 
            value_vars=["primary_dept_name", "other_dept_name"], 
            var_name="Dept_Type", 
            value_name="Department"
        ).dropna()

        # Calculate Attrition Rate
        attrition_rates = (
            dept_data.groupby("Department")["left"]
            .mean()
            .sort_values()
        )

        # Plot the Attrition Rate
        fig, ax = plt.subplots()
        attrition_rates.plot(kind="bar", ax=ax, color="coral")
        ax.set_ylabel("Attrition Rate")
        ax.set_title("Attrition Rate by Department (Filtered)")
        st.pyplot(fig)

    with col2:
        st.subheader("💰 Salary Distribution by Primary Department")
        sorted_depts = filtered_df.groupby("primary_dept_name")["salary"].median().sort_values().index
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="primary_dept_name", y="salary", data=filtered_df, order=sorted_depts, ax=ax, hue="primary_dept_name", palette="coolwarm", legend=False)
        ax.set_xticks(range(len(ax.get_xticklabels())))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Department")
        ax.set_ylabel("Salary")
        st.pyplot(fig)

    # Additional Insights
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🎂 Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["age"], bins=15, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Employee Count")
        st.pyplot(fig)

    with col4:
        st.subheader("👥 Gender Distribution")
        gender_counts = filtered_df["sex"].value_counts()
        fig, ax = plt.subplots()
        gender_counts.plot(kind="pie", autopct="%1.1f%%", colors=["lightblue", "pink"], ax=ax, startangle=90)
        ax.set_ylabel("")
        st.pyplot(fig)
        
    # Performance Rating Distribution
    st.subheader("📈 Performance Rating Distribution")
    rating_counts = filtered_df["last_performance_rating"].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, hue=rating_counts.index, palette="viridis", ax=ax, legend=False)
    ax.set_xlabel("Performance Rating")
    ax.set_ylabel("Number of Employees")
    st.pyplot(fig)

    # Tenure Analysis
    st.subheader("⌛ Tenure Analysis")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📊 Tenure Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["tenure"], bins=20, kde=True, color="green", ax=ax)
        ax.set_xlabel("Years at Company")
        ax.set_ylabel("Employee Count")
        st.pyplot(fig)

    with col6:
        st.subheader("💰 Salary vs Performance Rating")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="last_performance_rating", y="salary", data=filtered_df, hue="last_performance_rating", palette="coolwarm", ax=ax, legend=False)
        ax.set_xlabel("Performance Rating")
        ax.set_ylabel("Salary")
        ax.set_title("Salary Distribution Across Performance Ratings")
        st.pyplot(fig)

    # ------------------- Model Performance Evaluation -------------------
    st.subheader("📈 Model Performance")

    def evaluate_model(model, model_name):
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")

        X_test = X_test.to_numpy()

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Display Metrics
        st.subheader(f"🔹 {model_name}")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**Precision:** {prec:.2f}")
        st.write(f"**AUC Score:** {auc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model_name} - Confusion Matrix")
        st.pyplot(fig)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} - ROC Curve")
        ax.legend()
        st.pyplot(fig)

    evaluate_model(xgb_model, "XGBoost")
    evaluate_model(rf_model, "Random Forest")
    evaluate_model(logreg_model, "Logistic Regression")
