import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

EDU_ORDER = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
]

def load_student_data(csv_path="data/students_performance.csv"):
    df = pd.read_csv(csv_path)
    df["parental level of education"] = pd.Categorical(
        df["parental level of education"],
        categories=EDU_ORDER,
        ordered=True
    )
    return df

def train_score_predictors(df):
    """Trains regression models to predict math, reading, and writing scores."""
    categorical_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )
    
    X = df[categorical_features]
    
    models = {}
    for target in ["math score", "reading score", "writing score"]:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42))
        ])
        model.fit(X, df[target])
        models[target] = model
        
    return models

