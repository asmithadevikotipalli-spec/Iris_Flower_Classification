import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🌸 Iris Flower Classification")
st.write(
    "An interactive Machine Learning project using "
    "K-Nearest Neighbors (KNN) to classify Iris flowers."
)


# -----------------------------
# Load Dataset
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(
    data=X,
    columns=iris.feature_names
)

df["species"] = [
    iris.target_names[i] for i in y
]


# -----------------------------
# Dataset Information
# -----------------------------
st.header("📊 Dataset")

st.write(
    "The Iris dataset contains 150 samples of three different Iris flower species."
)

st.dataframe(df)


# -----------------------------
# Visualization
# -----------------------------
st.header("📈 Data Visualization")

st.write("Pairplot showing relationships between the Iris flower features.")

fig = sns.pairplot(
    df,
    hue="species",
    palette="viridis"
)

st.pyplot(fig)


# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Train Model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)


# -----------------------------
# Model Evaluation
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

st.header("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Test Samples",
        len(y_test)
    )


# Classification Report
st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    predictions,
    target_names=iris.target_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)


# -----------------------------
# Interactive Prediction
# -----------------------------
st.header("🌺 Predict Iris Flower")

st.write(
    "Enter the flower measurements below to predict its species."
)

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Flower Species"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)

    predicted_species = iris.target_names[
        prediction[0]
    ]

    st.success(
        f"🌸 Predicted Species: **{predicted_species.title()}**"
    )
