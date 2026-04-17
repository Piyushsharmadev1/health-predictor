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

def get_ai_response(prompt):
    try:
        response = model_gen.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# -------------------- Page Config --------------------
st.set_page_config(page_title="Health Predictor", layout="wide", page_icon="🧑‍⚕️")

# -------------------- Session State --------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================================================
# 🏠 LANDING PAGE
# =========================================================
if st.session_state.page == "home":
    st.markdown("""
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                              url("https://images.unsplash.com/photo-1576091160550-2173dba999ef");
            background-size: cover;
            background-position: center;
        }
        .title {
            text-align: center;
            color: white;
            font-size: 70px;
            font-weight: bold;
            margin-top: 180px;
        }
        .subtitle {
            text-align: center;
            color: white;
            font-size: 22px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🧑‍⚕️ Health Predictor AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict diseases using AI in seconds</div>', unsafe_allow_html=True)

    if st.button("🚀 Get Started"):
        st.session_state.page = "app"
        st.rerun()

    st.stop()

# =========================================================
# 🧠 MAIN APP (WITH BACKGROUND)
# =========================================================

# 🔥 Background inside app
st.markdown("""
<style>

/* 🌌 Background Image + Dark Overlay */
.stApp {
    background-image: 
        linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
        url("https://images.unsplash.com/photo-1584982751601-97dcc096659c");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* 🧑‍⚕️ Titles */
h1, h2, h3 {
    color: white !important;
    text-align: center;
}

/* 📦 Input Boxes (Glass Effect) */
.stSelectbox, .stNumberInput {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 8px;
    color: white !important;
}

/* 📝 Labels */
label {
    color: #ffffff !important;
    font-weight: 500;
}

/* 🔘 Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border-radius: 12px;
    height: 3.5em;
    font-size: 16px;
    border: none;
}

/* 📊 Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.85);
}

/* 🧾 Result Text */
.stSuccess, .stError {
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load Models --------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'rf_classifier.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(working_dir, 'saved_models', 'scaler.pkl'), 'rb'))

# -------------------- Predict Function --------------------
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

# -------------------- Sidebar --------------------
with st.sidebar:
    if st.button("🏠 Home"):
        st.session_state.page = "home"
        st.rerun()

    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        icons=['activity', 'heart'],
        default_index=0
    )

# =========================================================
# 🩺 Diabetes Prediction
# =========================================================
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.selectbox('Pregnancies', list(range(0, 21)))
    with col2:
        Glucose = st.selectbox('Glucose', list(range(70, 301, 5)))
    with col3:
        BloodPressure = st.selectbox('Blood Pressure', list(range(80, 201, 5)))

    with col1:
        SkinThickness = st.selectbox('Skin Thickness', list(range(0, 101, 5)))
    with col2:
        Insulin = st.selectbox('Insulin', list(range(0, 901, 25)))
    with col3:
        BMI = st.selectbox('BMI', [round(x * 0.5, 1) for x in range(20, 141)])

    with col1:
        DiabetesPedigreeFunction = st.selectbox('DPF', [round(x * 0.1, 1) for x in range(0, 31)])
    with col2:
        Age = st.selectbox('Age', list(range(1, 121)))

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

        prediction = diabetes_model.predict([user_input])

        if prediction[0] == 1:
            st.error('⚠️ Diabetes Risk')
            prompt = "Patient has diabetes. Give diet and precautions."
        else:
            st.success('✅ Healthy')
            prompt = "Give health tips and diet plan."

        st.write(get_ai_response(prompt))

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
        currentSmoker = st.selectbox('Smoker', ['yes', 'no'])

    with col1:
        cigsPerDay = st.number_input('Cigs/Day', 0.0, 50.0)
    with col2:
        BPMeds = st.selectbox('BP Meds', ['yes', 'no'])
    with col3:
        prevalentStroke = st.selectbox('Stroke', ['yes', 'no'])

    with col1:
        prevalentHyp = st.selectbox('Hypertension', ['yes', 'no'])
    with col2:
        diabetes = st.selectbox('Diabetes', ['yes', 'no'])
    with col3:
        totChol = st.number_input('Cholesterol', 100.0, 600.0)

    with col1:
        sysBP = st.number_input('Sys BP', 80.0, 250.0)
    with col2:
        diaBP = st.number_input('Dia BP', 50.0, 150.0)
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

        if result == 1:
            st.error("⚠️ Heart Disease Risk")
            prompt = "Patient has heart risk. Give precautions."
        else:
            st.success("✅ Healthy")
            prompt = "Give health tips and exercise."

        st.write(get_ai_response(prompt))