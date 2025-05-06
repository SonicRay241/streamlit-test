from fastapi import FastAPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

model = joblib.load("model.pkl")

app = FastAPI()

@app.post("/predict")
def predict(sepal_length: int, sepal_width: int, petal_length: int, petal_width: int):
    arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return {
        "species": model.decode(model.predict(arr))[0][0]
    }