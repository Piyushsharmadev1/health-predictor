import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
#from dotenv import load_dotenv
import numpy as np


API_KEY = st.secrets["GOOGLE_API_KEY"]


genai.configure(api_key=API_KEY)
model_gen = genai.GenerativeModel("gemini-flash-latest")


def get_ai_response(prompt):
    try:
       response = model_gen.generate_content(prompt)
       return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"


st.set_page_config(
    page_title="Health Predictor",
    layout="wide",
    page_icon=""
)


working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'rf_classifier.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(working_dir, 'saved_models', 'scaler.pkl'), 'rb'))


disease_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'disease_model.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(working_dir, 'saved_models', 'label_encoder.pkl'), 'rb'))
symptoms_list = pickle.load(open(os.path.join(working_dir, 'saved_models', 'symptoms.pkl'), 'rb'))


def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds,
            prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
            diaBP, BMI, heartRate, glucose):

    male = 1 if male == 'male' else 0
    currentSmoker = 1 if currentSmoker == 'yes' else 0
    BPMeds = 1 if BPMeds == 'yes' else 0
    prevalentStroke = 1 if prevalentStroke == 'yes' else 0
    prevalentHyp = 1 if prevalentHyp == 'yes' else 0
    diabetes = 1 if diabetes == 'yes' else 0

    data = [[male, age, currentSmoker, cigsPerDay, BPMeds,
             prevalentStroke, prevalentHyp, diabetes, totChol,
             sysBP, diaBP, BMI, heartRate, glucose]]

    data = scaler.transform(data)
    return model.predict(data)[0]


NORMAL_RANGES = {
    "Glucose": (70, 140),
    "BloodPressure": (80, 120),
    "BMI": (18.5, 24.9),
    "Cholesterol": (125, 200),
    "SysBP": (90, 120),
    "DiaBP": (60, 80),
    "HeartRate": (60, 100),
}

def check_normal_ranges(values):
    abnormal = []
    for key, val in values.items():
        if key in NORMAL_RANGES:
            low, high = NORMAL_RANGES[key]
            if val < low or val > high:
                abnormal.append(f"{key} ({val})")
    return abnormal


with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Common Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'stethoscope'],
        default_index=0
    )




if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.selectbox('Number of Pregnancies', list(range(0, 21)))
        SkinThickness = st.selectbox('Skin Thickness', list(range(0, 101, 5)))
        DiabetesPedigreeFunction = st.selectbox('Diabetes Pedigree Function', [round(x*0.1,1) for x in range(0,31)])
    with col2:
        Glucose = st.selectbox('Glucose Level', list(range(70, 301, 5)))
        Insulin = st.selectbox('Insulin Level', list(range(0,901,25)))
        Age = st.selectbox('Age', list(range(1,121)))
    with col3:
        BloodPressure = st.selectbox('Blood Pressure', list(range(80,201,5)))
        BMI = st.selectbox('BMI', [round(x*0.5,1) for x in range(20,141)])

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

        diab_prediction = diabetes_model.predict([user_input])

        abnormal_params = check_normal_ranges({"Glucose": Glucose, "BloodPressure": BloodPressure, "BMI": BMI})

        if diab_prediction[0] == 1 or abnormal_params:
            st.error('⚠️ The person may have diabetes or abnormal values detected')
            if abnormal_params:
                st.warning(f"⚠️ Abnormal values: {', '.join(abnormal_params)}")
            prompt = f"Patient has diabetes or abnormal values: {', '.join(abnormal_params)}. Glucose: {Glucose}, BMI: {BMI}, Age: {Age}. Give diet, precautions, lifestyle changes."
        else:
            st.success('✅ The person is healthy')
            prompt = "Patient is healthy. Give health tips, diet plan, exercise routine."

        st.subheader("🤖 AI Response")
        st.write(get_ai_response(prompt))
        st.warning("⚠️ AI advice only. Consult a doctor.")




if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    with col1: male = st.selectbox('Gender', ['male','female']); prevalentHyp = st.selectbox('Hypertension', ['yes','no'])
    with col2: age = st.number_input('Age', 1,120); diabetes = st.selectbox('Diabetes', ['yes','no'])
    with col3: currentSmoker = st.selectbox('Current Smoker', ['yes','no']); totChol = st.number_input('Cholesterol',100.0,600.0)
    with col1: cigsPerDay = st.number_input('Cigarettes Per Day',0.0,50.0)
    with col2: BPMeds = st.selectbox('BP Medicines', ['yes','no'])
    with col3: prevalentStroke = st.selectbox('Stroke History',['yes','no'])
    with col1: sysBP = st.number_input('Systolic BP',80.0,250.0)
    with col2: diaBP = st.number_input('Diastolic BP',50.0,150.0)
    with col3: BMI = st.number_input('BMI',10.0,50.0)
    with col1: heartRate = st.number_input('Heart Rate',40.0,200.0)
    with col2: glucose = st.number_input('Glucose',50.0,400.0)

    if st.button('Heart Test Result'):
        result = predict(heart_model, scaler, male, age, currentSmoker, cigsPerDay,
                         BPMeds, prevalentStroke, prevalentHyp, diabetes,
                         totChol, sysBP, diaBP, BMI, heartRate, glucose)

        abnormal_params = check_normal_ranges({"SysBP": sysBP,"DiaBP": diaBP,"Cholesterol": totChol,"BMI": BMI,"HeartRate": heartRate,"Glucose": glucose})

        if result==1 or abnormal_params:
            st.error("⚠️ The patient may have heart disease or abnormal values")
            if abnormal_params: st.warning(f"⚠️ Abnormal values: {', '.join(abnormal_params)}")
            prompt = f"Patient details with abnormal values: {', '.join(abnormal_params)}. Age: {age}, BP: {sysBP}/{diaBP}, Cholesterol: {totChol}. Give diet, precautions, lifestyle changes."
        else:
            st.success("✅ No Heart Disease")
            prompt = "Patient is healthy. Give health tips, diet plan, exercise routine."

        st.subheader("🤖 AI Response")
        st.write(get_ai_response(prompt))
        st.warning("⚠️ AI advice only. Not a substitute for doctor.")






if selected == 'Common Disease Prediction':
    st.title("Common Disease Predictor")
    
    st.write("Select your symptoms from below:")
    
    
    selected_symptoms = st.multiselect(
        "Select Symptoms",
        symptoms_list
    )
    
    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
            input_vector = np.array(input_vector).reshape(1, -1)
            
            
            pred = disease_model.predict(input_vector)
            predicted_disease = label_encoder.inverse_transform(pred)[0]
            
            st.success(f"Predicted Disease: {predicted_disease}")
            
            
            prompt = f"""
            Patient shows symptoms: {', '.join(selected_symptoms)}.
            Predicted disease: {predicted_disease}.
            Give:
            - Diet recommendations
            - Home remedies
            - Precautions
            - Warning signs to consult a doctor
            """
            st.subheader("🤖 AI Response")
            st.write(get_ai_response(prompt))
            st.warning("⚠️ AI advice only. Consult a doctor if symptoms persist.")