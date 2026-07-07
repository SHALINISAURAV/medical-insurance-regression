import streamlit as st
import pandas as pd
import joblib
import json


# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(

    page_title="Medical Insurance Predictor",

    page_icon="🏥",

    layout="wide"

)


# ---------------------------
# Load Model
# ---------------------------

MODEL_PATH = "artifacts/model.pkl"


model = joblib.load(
    MODEL_PATH
)



# ---------------------------
# Load Metrics
# ---------------------------

with open(
    "artifacts/metrics.json",
    "r"
) as file:

    metrics = json.load(file)



# ---------------------------
# Header Section
# ---------------------------

st.title(
    "🏥 Medical Insurance Cost Predictor"
)


st.markdown(
"""
### Machine Learning powered insurance cost estimation

This application predicts medical insurance expenses using:

- Data preprocessing pipeline
- Machine learning regression
- Gradient Boosting model

"""
)



# ---------------------------
# Sidebar Inputs
# ---------------------------

st.sidebar.header(
    "Enter Patient Details"
)



age = st.sidebar.slider(

    "Age",

    min_value=18,

    max_value=100,

    value=30

)



sex = st.sidebar.selectbox(

    "Sex",

    [
        "male",
        "female"
    ]

)



bmi = st.sidebar.number_input(

    "BMI",

    min_value=10.0,

    max_value=60.0,

    value=25.0

)



children = st.sidebar.number_input(

    "Children",

    min_value=0,

    max_value=10,

    value=0

)



smoker = st.sidebar.selectbox(

    "Smoker",

    [
        "yes",
        "no"
    ]

)



region = st.sidebar.selectbox(

    "Region",

    [

        "southwest",

        "southeast",

        "northwest",

        "northeast"

    ]

)



# ---------------------------
# Create Input DataFrame
# ---------------------------


input_data = pd.DataFrame({

    "age":[age],

    "sex":[sex],

    "bmi":[bmi],

    "children":[children],

    "smoker":[smoker],

    "region":[region]

})



# ---------------------------
# Prediction
# ---------------------------


if st.button(
    "Predict Insurance Cost"
):

    prediction = model.predict(
        input_data
    )[0]


    st.success(

        f"Estimated Insurance Cost: ${prediction:,.2f}"

    )



# ---------------------------
# Model Information
# ---------------------------

st.divider()


st.subheader(
    "📊 Model Performance"
)


for model_name, result in metrics.items():

    st.write(
        f"**{model_name}**"
    )

    st.write(
        f"R² Score: {result['R2 Score']:.3f}"
    )

    st.write(
        f"MAE: ${result['MAE']:,.2f}"
    )

    st.write(
        f"RMSE: ${result['RMSE']:,.2f}"
    )

    st.divider()