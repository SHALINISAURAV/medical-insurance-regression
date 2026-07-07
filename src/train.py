import json
import joblib


from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


from sklearn.ensemble import GradientBoostingRegressor



from data_preprocessing import (
    load_data,
    split_features_target,
    create_preprocessor
)


from utils import evaluate_model



# Paths

DATA_PATH = "data/insurance.csv"

MODEL_PATH = "artifacts/model.pkl"



# Load dataset

df = load_data(DATA_PATH)


print(df.head())



# Split features and target

X, y = split_features_target(df)



# Train test split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)



# Preprocessor

preprocessor = create_preprocessor()



# Models

models = {


    "Linear Regression":
        LinearRegression(),


    "Ridge Regression":
        Ridge(),


    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )

}



results = {}

trained_models = {}



# Train all models

for name, model in models.items():


    pipeline = Pipeline(

        [

            (
                "preprocessor",
                preprocessor
            ),


            (
                "model",
                model
            )

        ]

    )


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    results[name] = evaluate_model(

        y_test,
        predictions

    )


    trained_models[name] = pipeline



    print(
        name,
        "trained"
    )



print("\nModel Results")

print(results)

with open(
    "artifacts/metrics.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

# Select best model

best_model_name = max(

    results,

    key=lambda x:
    results[x]["R2 Score"]

)



best_model = trained_models[best_model_name]



print(
    "\nBest Model:",
    best_model_name
)



# Save model

joblib.dump(

    best_model,

    MODEL_PATH

)



print(
    "Model Saved Successfully"
)