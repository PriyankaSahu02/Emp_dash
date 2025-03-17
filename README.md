# Employee Attrition Prediction Dashboard

## 📌 Overview
This repository contains a **Streamlit** web application designed to analyze employee data, predict attrition, and visualize key HR insights. The app provides interactive filtering options and machine learning model evaluation for better employee retention strategies.

## 🚀 Features
- **Employee Data Filtering**: Filter employees by department, job title, gender, tenure, salary range, and more.
- **Attrition Analysis**: View attrition rates by department and demographics.
- **Model Performance Evaluation**: Compare the performance of different ML models (XGBoost, Random Forest, Logistic Regression) with accuracy, precision, and ROC curves.
- **Data Visualization**: Interactive charts and confusion matrices to analyze trends.

## 🛠 Tech Stack
- **Python**
- **Streamlit**
- **Pandas & NumPy**
- **Scikit-learn** (ML models & evaluation metrics)
- **Matplotlib & Seaborn** (Visualization)
- **Joblib** (Model persistence)

## 📂 Project Structure
```
📁 Employee-Attrition-Dashboard
│-- app.py                  # Main Streamlit app script
│-- requirements.txt        # Python dependencies
│-- X_test.pkl              # Preprocessed test dataset
│-- y_test.pkl              # Test labels
│-- models/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   ├── rf_model.pkl        # Trained Random Forest model
│   ├── logreg_model.pkl    # Trained Logistic Regression model
│-- data/
│   ├── final_table.csv   # Merged employee data (if applicable)
│-- README.md               # Project documentation
```

## ⚡ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PriyankaSahu02/Emp_dash.git
cd Emp_dash
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit App
```bash
streamlit run emp.py
```

## 🎯 Usage
- Open the **web app** in your browser.
- Apply various filters to analyze employee data.
- Evaluate machine learning models used for attrition prediction.

## 🚀 Deployment
To deploy the app using **Streamlit Cloud**, follow these steps:
1. Push your code to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository and deploy your app.

## 🔗 Live Demo
Employee Attrition Prediction: https://empdash-p.streamlit.app/
Employee Dashboard: https://empdash-p2.streamlit.app/
