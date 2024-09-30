import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

model = pickle.load(open('Diabeties_Deployment', 'rb'))

def diabeties_pred (input_data):
    numpy_array = np.asarray(input_data)
    reshaped_array = numpy_array.reshape(1, -1)
    
    prediction = model.predict(reshaped_array)
    if prediction == 0:
        return 'Non-Diabetic'
    else:
        return 'Diabetic'
    

def main():
    st.title('Diabeties Prediction Model :')
    
    Pregnacies = st.text_input('No. Of Pregnacies :')
    Glucose = st.text_input('Glucose Level :')
    BloodPressure = st.text_input('Blood Pressure :')
    SkinThickness = st.text_input('Skin Thickness : ')
    Insulin = st.text_input('Insulin Level :')
    BMI = st.text_input('BMI :')
    DPF = st.text_input('Diabeties Pedigree Function :')
    Age = st.text_input("Age :")

    Diagnosis = ''

    if st.button('Predict'):
        Diagnosis = diabeties_pred(
            [int(Pregnacies), int(Glucose), float(BloodPressure), int(SkinThickness), int(Insulin), float(BMI), float(DPF), int(Age)]
            )
    st.success(Diagnosis)
    
if __name__ == '__main__':
    main()

