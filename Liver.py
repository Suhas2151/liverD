import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler

model = pickle.load(open('GaussianNB_prediction_model1.py', 'rb'))

def preprocess_input(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
                     aspartate_aminotransferase, total_protiens, albumin, albumin_and_globulin_ratio):
    values = [[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
               aspartate_aminotransferase, total_protiens, albumin, albumin_and_globulin_ratio]]
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(values)
    return scaled_values

def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
                          aspartate_aminotransferase, total_protiens, albumin, albumin_and_globulin_ratio):
    scaled_values = preprocess_input(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                     alamine_aminotransferase, aspartate_aminotransferase, total_protiens, albumin,
                                     albumin_and_globulin_ratio)
    prediction = model.predict(scaled_values)
    return prediction

def main():
    st.title('Liver Disease Prediction')

    age = st.number_input('Age', min_value=0, max_value=150, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0, value=0.5)
    direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0, value=0.1)
    alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0, value=150)
    alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0, value=80)
    aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0, value=100)
    total_protiens = st.number_input('Total Protiens', min_value=0.0, value=6.0)
    albumin = st.number_input('Albumin', min_value=0.0, value=3.5)
    albumin_and_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, value=0.8)

    if st.button('Predict'):
        gender_code = 1 if gender == 'Male' else 0
        prediction = predict_liver_disease(age, gender_code, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                           alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                                           albumin, albumin_and_globulin_ratio)

        if prediction[0] == 1:
            st.error('Prediction: Liver Disease Detected')
        else:
            st.success('Prediction: No Liver Disease Detected')

if __name__ == '__main__':
    main()
