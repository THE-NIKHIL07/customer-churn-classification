import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

x_train_columns = [
    'CreditScore',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
    'Geography_France',
    'Geography_Germany',
    'Geography_Spain'
]

#load the trained model[scaler,ohe,pickle]
model=load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file :
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file :
    onehot_encoder=pickle.load(file)
    
with open('scaler.pkl','rb') as file :
    scaler=pickle.load(file)


#streamlit app

# Title
st.title("Customer Churn Prediction")

# Sidebar inputs
st.sidebar.header("Input Customer Data")

def user_input_features():
    CreditScore = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=600)
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    Tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
    Balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0)
    NumOfProducts = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=1)
    HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)
    Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    
    data = {
        "CreditScore": CreditScore,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Geography": Geography
    }
    
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# --- Preprocess inputs ---
# Encode Gender
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# One-hot encode Geography
geo_encoded = onehot_encoder.transform(input_df[['Geography']])
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Drop original Geography column and concat encoded columns
input_df = input_df.drop('Geography', axis=1)
input_df = pd.concat([input_df, geo_df], axis=1)

# Scale numerical features
input_df[input_df.columns] = scaler.transform(input_df)

input_df = input_df[x_train_columns]

# Prediction
prediction_prob = model.predict(input_df)
prediction_class = (prediction_prob > 0.5).astype(int)

st.subheader("Prediction Probability")
st.write(prediction_prob[0][0])

st.subheader("Prediction Class")
st.write("Will Exit" if prediction_class[0][0]==1 else "Will Stay")
    
        