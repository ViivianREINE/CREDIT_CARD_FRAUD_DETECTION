# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fdf6e3;
        color: #3e2723;
        font-family: 'Trebuchet MS', sans-serif;
    }
    h1, h2, h3 {
        color: #4e342e;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #a1887f;
        color: #fffbea;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #6d4c41;
        color: #fffbea;
    }
    .metric-card {
        background-color: #fffaf0;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    hr {
        border: 1px solid #8d6e63;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

# ---------------------------
# Train Model
# ---------------------------
X = data.drop(columns=["Class"])
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------
# Web UI
# ---------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown("### 🍂ML dashboard for fraud detection")

# ---------------------------
# Layout - Dataset Preview
# ---------------------------
with st.expander("📂 Show Dataset Preview"):
    st.dataframe(data.head(20))

# ---------------------------
# Layout - Model Performance
# ---------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns([1,1])

with col1:
    acc = accuracy_score(y_test, y_pred) * 100
    st.markdown(f"<div class='metric-card'><h3>Accuracy</h3><p style='font-size:22px;'>{acc:.2f}%</p></div>", unsafe_allow_html=True)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

with col2:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ---------------------------
# Layout - Prediction Form
# ---------------------------
st.subheader("📝 Test a Transaction")

with st.form("prediction_form"):
    st.write("Enter transaction details:")

    input_data = []
    for col in X.columns[:5]:  # Only first 5 inputs for demo (too many otherwise)
        val = st.number_input(f"{col}", value=float(X[col].mean()))
        input_data.append(val)

    submitted = st.form_submit_button("Predict Fraud")

    if submitted:
        input_array = np.array(input_data + [0]*(X.shape[1]-5)).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        if prediction == 0:
            st.success("✅ Transaction looks **Legit**")
        else:
            st.error("🚨 Fraudulent Transaction Detected!")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:#4e342e; font-weight:bold;">
    🍁 Made with ❤️ by Priyam Parashar 🍁
    </div>
    """,
    unsafe_allow_html=True
)
