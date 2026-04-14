import streamlit as st
import pandas as pd
import joblib
model=joblib.load('logistic_heart.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('columns.pkl')
st.title('Heart stroke prediction❤️')
st.markdown('please fill the form below to predict the risk of heart stroke')
age=st.slider('Age',10,100,25)
sex=st.selectbox("Sex",['Male', 'Female'])
chest_pain=st.selectbox("Chest Pain Type",['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
resting_blood_pressure=st.number_input('Resting Blood Pressure',80,200,120)
serum_cholesterol=st.number_input('Serum Cholesterol',100,400,200)
fasting_blood_sugar=st.selectbox("Fasting Blood Sugar > 120 mg/dl",['Yes', 'No'])
resting_ecg=st.selectbox("Resting ECG",['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
max_heart_rate=st.slider('Max Heart Rate',60,220,150)
exercise_induced_angina=st.selectbox("Exercise Induced Angina",['Yes', 'No'])
st_depression=st.number_input('ST Depression',0.0,10.0,1.0)
slope=st.selectbox("Slope of the Peak Exercise ST Segment",['Upsloping', 'Flat', 'Downsloping'])
num_major_vessels=st.slider('Number of Major Vessels',0,4,0)
thalassemia=st.selectbox("Thalassemia",['Normal', 'Fixed Defect', 'Reversible Defect'])
if st.button('Predict'):
    input_data={
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'chest_pain': chest_pain,
        'resting_blood_pressure': resting_blood_pressure,
        'serum_cholesterol': serum_cholesterol,
        'fasting_blood_sugar': 1 if fasting_blood_sugar == 'Yes' else 0,
        'resting_ecg': resting_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': 1 if exercise_induced_angina == 'Yes' else 0,
        'st_depression': st_depression,
        'slope': slope,
        'num_major_vessels': num_major_vessels,
        'thalassemia': thalassemia
    }
    input_df=pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_columns]
    input_df_scaled=scaler.transform(input_df)
    prediction=model.predict(input_df_scaled)[0]
    if prediction==1:
        st.error('High risk of heart stroke. Please consult a doctor immediately.')
    else:
        st.success('Low risk of heart stroke. Keep maintaining a healthy lifestyle!')