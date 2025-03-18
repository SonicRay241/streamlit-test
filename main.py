import streamlit as st
import joblib
import numpy as np
from oop import RFModel, DataLoader


data = DataLoader("./dataset/Iris.csv")
data.remove_column("Id")
data.separate_y("Species")
model = RFModel(data)
model.load("model.pkl")
model.split_data()
model.encode_y()

st.title("Spinni boi 🐈")
st.text("O I I A I A O I I I A I")
st.image("./assets/images/oiia.gif")

sepal_length = st.slider("sepal_length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.slider("sepal_width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.slider("petal_length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.slider("petal_width", min_value=0.0, max_value=10.0, step=0.1)

pred = st.button("Predict")

if (pred):
    arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    st.text(model.decode(model.predict(arr))[0][0])