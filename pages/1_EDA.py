import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Halaman 1: EDA Dataset Titanic')

df = pd.read_csv('data/titanic.csv')
st.write('Dataset:')
st.dataframe(df.head())

st.write('Statistik Ringkas:')
st.write(df.describe())

st.write('Korelasi:')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
