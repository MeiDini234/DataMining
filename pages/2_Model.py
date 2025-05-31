import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Halaman 2: Hasil Model')

model = joblib.load('data/model.pkl')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

y_pred = model.predict(X_test)

st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))

st.write('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
