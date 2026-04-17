import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

# -------------------- ENV LOAD --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model_gen = genai.GenerativeModel("gemini-1.5-flash")

# -------------------- AI FUNCTION --------------------
def get_ai_response(prompt):
    try:
        response = model_gen.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Health Predictor", layout="wide")

# -------------------- SAFE MODEL LOADING --------------------
def load_model(path):
    try:
        return pickle.load(open(path, "rb"))
    except:
        st.error(f"❌ Model not found: {path}")
        return None

working_dir = os.path.dirname(os.path.abspath(__file__))
heart_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'rf_classifier.pkl'), 'rb'))
diabetes_model = load_model(os.path.join(working_dir, "saved_models/diabetes_model.sav"))
scaler = load_model(os.path.join(working_dir, "saved_models/scaler.pkl"))

disease_model = load_model(os.path.join(working_dir, "saved_models/disease_model.pkl"))
label_encoder = load_model(os.path.join(working_dir, "saved_models/label_encoder.pkl"))
symptoms_list = load_model(os.path.join(working_dir, "saved_models/symptoms.pkl"))

# -------------------- SIDEBAR --------------------
with st.sidebar:
    selected = option_menu(
        "Disease Prediction",
        ["Diabetes", "Heart", "Common Disease"],
        icons=["activity", "heart", "stethoscope"]
    )

# =========================================================
# 🩺 DIABETES
# =========================================================
if selected == "Diabetes":
    st.title("🩺 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose", 0, 300)
        BloodPressure = st.number_input("BP", 0, 200)
        SkinThickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        Insulin = st.number_input("Insulin", 0, 900)
        BMI = st.number_input("BMI", 0.0, 50.0)
        DPF = st.number_input("DPF", 0.0, 2.5)
        Age = st.number_input("Age", 1, 120)

    if st.button("Predict"):
        if diabetes_model:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DPF, Age]

            result = diabetes_model.predict([input_data])

            if result[0] == 1:
                st.error("⚠️ Diabetic")
                prompt = "Patient has diabetes. Give diet & precautions."
            else:
                st.success("✅ Healthy")
                prompt = "Patient is healthy. Give fitness tips."

            st.write(get_ai_response(prompt))

# =========================================================
# ❤️ HEART
# =========================================================
if selected == "Heart":
    st.title("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    chol = st.number_input("Cholesterol", 100, 600)
    sysBP = st.number_input("Systolic BP", 80, 250)
    diaBP = st.number_input("Diastolic BP", 50, 150)
    BMI = st.number_input("BMI", 10.0, 50.0)

    if st.button("Predict Heart"):
        if heart_model and scaler:
            data = [[age, chol, sysBP, diaBP, BMI]]
            data = scaler.transform(data)

            result = heart_model.predict(data)

            if result[0] == 1:
                st.error("⚠️ Heart Disease Risk")
                prompt = "Heart disease patient. Give precautions."
            else:
                st.success("✅ Healthy")
                prompt = "Healthy heart. Give lifestyle tips."

            st.write(get_ai_response(prompt))

# =========================================================
# 🧠 COMMON DISEASE
# =========================================================
if selected == "Common Disease":
    st.title("🧠 Disease Predictor")

    if symptoms_list:
        selected_symptoms = st.multiselect("Select Symptoms", symptoms_list)

        if st.button("Predict Disease"):
            if disease_model and label_encoder:

                input_vector = [1 if s in selected_symptoms else 0 for s in symptoms_list]
                input_vector = np.array(input_vector).reshape(1, -1)

                pred = disease_model.predict(input_vector)
                disease = label_encoder.inverse_transform(pred)[0]

                st.success(f"Predicted: {disease}")

                prompt = f"Patient has {disease}. Give treatment advice."
                st.write(get_ai_response(prompt))