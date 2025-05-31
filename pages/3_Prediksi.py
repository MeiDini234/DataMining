import streamlit as st
import pandas as pd
import joblib

st.title('Halaman 3: Formulir Prediksi')

model = joblib.load('data/model.pkl')

pclass = st.selectbox('Pclass', [1, 2, 3])
age = st.slider('Age', 1, 80, 25)
fare = st.number_input('Fare', min_value=0.0, value=10.0)

if st.button('Prediksi'):
    X_new = pd.DataFrame([[pclass, age, fare]], columns=['Pclass', 'Age', 'Fare'])
    pred = model.predict(X_new)[0]
    st.write(f'Prediksi: **{pred}**')
