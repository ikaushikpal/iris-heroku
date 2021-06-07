import streamlit as st
import pandas as pd
import joblib
import numpy as np


@st.cache
def load_data():
    clf_svm = joblib.load('model/iris_svm_model.sav')
    std_scalar_obj = joblib.load('model/std_scalar_obj.sav')
    types_iris = pd.DataFrame({'Class Labels':['Setosa','Versicolor','Virginica']})
    return (clf_svm, std_scalar_obj, types_iris)


clf_svm, std_scalar_obj, types_iris = load_data()

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

    data = {'Sepal Length': sepal_length,
            'Sepal Width': sepal_width,
            'Petal Length': petal_length,
            'Petal Width': petal_width}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write("""
If sidebar is not present then click/touch on upper left **>** symbol and 4 slider will be present. You can change any parameter values and ML model will predict class according to it.
""")
st.write(df)

X_scaled = std_scalar_obj.transform(df)
y_pred = clf_svm.predict(X_scaled)


st.subheader('Class labels and their corresponding index number')
st.write(types_iris)

st.subheader('Model Prediction')
st.write(types_iris.iloc[y_pred])


st.write('''I made this webapp using streamlit library. The Model I used was SVM. Github link for more details 
https://github.com/ikaushikpal/iris-heroku

Thanks For Checking 😀
''')
