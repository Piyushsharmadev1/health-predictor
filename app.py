import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------- Load ENV --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

# -------------------- Configure Gemini --------------------
genai.configure(api_key=GOOGLE_API_KEY)
model_gen = genai.GenerativeModel("gemini-flash-latest")

# -------------------- Safe AI Function --------------------
def get_ai_response(prompt):
    try:
       response = model_gen.generate_content(prompt)
       return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Health Predictor",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# -------------------- Load Models --------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'rf_classifier.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(working_dir, 'saved_models', 'scaler.pkl'), 'rb'))

# -------------------- Heart Predict Function --------------------
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

# -------------------- Normal Ranges --------------------
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

# -------------------- Sidebar --------------------
with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart'],
        default_index=0
    )

## =========================================================
# 🩺 Diabetes Prediction
# =========================================================
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    # Number of Pregnancies (0-20)
    with col1:
        Pregnancies = st.selectbox(
            'Number of Pregnancies',
            list(range(0, 21))
        )

    # Glucose Level (70-300, step 5)
    with col2:
        Glucose = st.selectbox(
            'Glucose Level',
            list(range(70, 301, 5))
        )

    # Blood Pressure (80-200, step 5)
    with col3:
        BloodPressure = st.selectbox(
            'Blood Pressure',
            list(range(80, 201, 5))
        )

    # Skin Thickness (0-100, step 5)
    with col1:
        SkinThickness = st.selectbox(
            'Skin Thickness',
            list(range(0, 101, 5))
        )

    # Insulin Level (0-900, step 25)
    with col2:
        Insulin = st.selectbox(
            'Insulin Level',
            list(range(0, 901, 25))
        )

    # BMI (10.0-70.0, step 0.5)
    with col3:
        BMI = st.selectbox(
            'BMI',
            [round(x * 0.5, 1) for x in range(20, 141)]  # 10.0 to 70.0
        )

    # Diabetes Pedigree Function (0.0-3.0, step 0.1)
    with col1:
        DiabetesPedigreeFunction = st.selectbox(
            'Diabetes Pedigree Function',
            [round(x * 0.1, 1) for x in range(0, 31)]
        )

    # Age (1-120)
    with col2:
        Age = st.selectbox(
            'Age',
            list(range(1, 121))
        )

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

        diab_prediction = diabetes_model.predict([user_input])

        # Check normal ranges
        abnormal_params = check_normal_ranges({
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "BMI": BMI
        })

        if diab_prediction[0] == 1 or abnormal_params:
            st.error('⚠️ The person may have diabetes or abnormal values detected')
            if abnormal_params:
                st.warning(f"⚠️ Abnormal values: {', '.join(abnormal_params)}")
            prompt = f"""
            Patient has diabetes or abnormal values: {', '.join(abnormal_params)}.
            Glucose: {Glucose}, BMI: {BMI}, Age: {Age}

            Give:
            - Diet plan
            - Precautions
            - Lifestyle changes
            """
        else:
            st.success('✅ The person is healthy')
            prompt = f"""
            Patient is healthy.

            Give:
            - Health tips
            - Diet plan
            - Exercise routine
            """

        st.subheader("🤖 AI Response")
        st.write(get_ai_response(prompt))
        st.warning("⚠️ AI advice only. Consult a doctor.")
# =========================================================
# ❤️ Heart Disease Prediction
# =========================================================
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)
    with col1:
        male = st.selectbox('Gender', ['male', 'female'])
    with col2:
        age = st.number_input('Age', 1, 120)
    with col3:
        currentSmoker = st.selectbox('Current Smoker', ['yes', 'no'])

    with col1:
        cigsPerDay = st.number_input('Cigarettes Per Day', 0.0, 50.0)
    with col2:
        BPMeds = st.selectbox('BP Medicines', ['yes', 'no'])
    with col3:
        prevalentStroke = st.selectbox('Stroke History', ['yes', 'no'])

    with col1:
        prevalentHyp = st.selectbox('Hypertension', ['yes', 'no'])
    with col2:
        diabetes = st.selectbox('Diabetes', ['yes', 'no'])
    with col3:
        totChol = st.number_input('Cholesterol', 100.0, 600.0)

    with col1:
        sysBP = st.number_input('Systolic BP', 80.0, 250.0)
    with col2:
        diaBP = st.number_input('Diastolic BP', 50.0, 150.0)
    with col3:
        BMI = st.number_input('BMI', 10.0, 50.0)

    with col1:
        heartRate = st.number_input('Heart Rate', 40.0, 200.0)
    with col2:
        glucose = st.number_input('Glucose', 50.0, 400.0)

    if st.button('Heart Test Result'):
        result = predict(
            heart_model, scaler,
            male, age, currentSmoker, cigsPerDay,
            BPMeds, prevalentStroke, prevalentHyp,
            diabetes, totChol, sysBP, diaBP,
            BMI, heartRate, glucose
        )

        # Check normal ranges
        abnormal_params = check_normal_ranges({
            "SysBP": sysBP,
            "DiaBP": diaBP,
            "Cholesterol": totChol,
            "BMI": BMI,
            "HeartRate": heartRate,
            "Glucose": glucose
        })

        if result == 1 or abnormal_params:
            st.error("⚠️ The patient may have heart disease or abnormal values")
            if abnormal_params:
                st.warning(f"⚠️ Abnormal values: {', '.join(abnormal_params)}")
            prompt = f"""
            Patient details with abnormal values: {', '.join(abnormal_params)}.
            Age: {age}, BP: {sysBP}/{diaBP}, Cholesterol: {totChol}

            Give:
            - Diet plan
            - Precautions
            - Lifestyle changes
            """
        else:
            st.success("✅ No Heart Disease")
            prompt = f"""
            Patient is healthy.

            Give:
            - Health tips
            - Diet plan
            - Exercise routine
            """

        st.subheader("🤖 AI Response")
        st.write(get_ai_response(prompt))
        st.warning("⚠️ AI advice only. Not a substitute for doctor.")