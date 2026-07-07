import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)



def load_data(path):

    df = pd.read_csv(path)

    return df



def split_features_target(df):

    X = df.drop(
        "expenses",
        axis=1
    )

    y = df["expenses"]

    return X, y



def create_preprocessor():

    numerical_features = [

        "age",
        "bmi",
        "children"

    ]


    categorical_features = [

        "sex",
        "smoker",
        "region"

    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                StandardScaler(),
                numerical_features
            ),


            (
                "cat",
                OneHotEncoder(),
                categorical_features
            )

        ]

    )


    return preprocessor