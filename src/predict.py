import joblib
import pandas as pd


# Model path (relative to project root)
MODEL_PATH = "artifacts/model.pkl"



# Load trained model

model = joblib.load(
    MODEL_PATH
)



# Sample input data

sample = pd.DataFrame({

    "age": [35],

    "sex": ["male"],

    "bmi": [28.5],

    "children": [2],

    "smoker": ["no"],

    "region": ["southwest"]

})



# Prediction

prediction = model.predict(
    sample
)



print(
    f"Estimated Insurance Cost: ${prediction[0]:,.2f}"
)