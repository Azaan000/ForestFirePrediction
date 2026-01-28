import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide"
)

# =========================
# CSS Styling
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');
/* App background */
.stApp {
    background: linear-gradient(135deg, #0b1f14, #06140c);
    font-family: 'Segoe UI', sans-serif;
    color: #e0f2e9;
}

/* Glass cards */
.glass {
    background: rgba(20, 40, 30, 0.75);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(46, 204, 113, 0.25);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(10px);
}

/* Headings */
h1 {
    text-align: center;
    color: #2ecc71;
    font-size: 3rem;
}
h2, h3 {
    color: #a8ffcb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(5, 20, 12, 0.95);
    border-right: 2px solid rgba(46, 204, 113, 0.3);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2ecc71, #27ae60);
    color: black;
    border-radius: 30px;
    font-size: 20px;
    height: 3.2em;
    font-weight: bold;
    border: none;
    transition: 0.3s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #27ae60, #1e8449);
    box-shadow: 0px 10px 25px rgba(46, 204, 113, 0.6);
}

/* Risk cards */
.risk-low {
    background: #2ecc71;
    padding: 18px;
    border-radius: 15px;
    font-size: 20px;
    color: black;
    text-align: center;
    font-weight: bold;
}
.risk-medium {
    background: #f1c40f;
    padding: 18px;
    border-radius: 15px;
    font-size: 20px;
    color: black;
    text-align: center;
    font-weight: bold;
}
.risk-high {
    background: #e74c3c;
    padding: 18px;
    border-radius: 15px;
    font-size: 20px;
    color: white;
    text-align: center;
    font-weight: bold;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #2ecc71;
}

/* Footer */
.footer {
    text-align: center;
    opacity: 0.75;
    margin-top: 40px;
    color: #9cffc2;
    font-size: 14px;
}
/* Slider track */
.stSlider > div > div > div > div {
    background-color: #2ecc71 !important;
}

/* Slider thumb */
.stSlider > div > div > div > div > div {
    background-color: #2ecc71 !important;
}


/* Apply font everywhere */
html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
}

/* Glowy title */
.glow-title {
    color: #2ecc71;
    text-shadow:
        0 0 5px #2ecc71,
        0 0 10px #2ecc71,
        0 0 20px #27ae60,
        0 0 40px #1e8449;
}

/* Subtle glow text */
.glow-text {
    color: #a8ffcb;
    text-shadow:
        0 0 5px rgba(46, 204, 113, 0.6),
        0 0 10px rgba(46, 204, 113, 0.4);
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
model = joblib.load("data/forest_fire_model.pkl")

# =========================
# App Title
# =========================
st.markdown(
    "<h1 class='glow-title'>🔥 Forest Fire Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='glow-text' style='text-align:center;'>AI-Powered Environmental Risk Analysis</p>",
    unsafe_allow_html=True
)


# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🌍 Environmental Inputs")

X = st.sidebar.slider("X Coordinate", 0, 10, 5)
Y = st.sidebar.slider("Y Coordinate", 0, 10, 5)
month = st.sidebar.selectbox("Month", list(range(12)))
day = st.sidebar.selectbox("Day", list(range(7)))

FFMC = st.sidebar.slider("🔥 Fine Fuel Moisture Code", 0.0, 100.0, 50.0)
DMC = st.sidebar.slider("🌾 Duff Moisture Code", 0.0, 300.0, 100.0)
DC = st.sidebar.slider("🌲 Drought Code", 0.0, 800.0, 400.0)
ISI = st.sidebar.slider("⚡ Initial Spread Index", 0.0, 60.0, 10.0)

temp = st.sidebar.slider("🌡 Temperature (°C)", -5.0, 40.0, 20.0)
RH = st.sidebar.slider("💧 Humidity (%)", 0, 100, 40)
wind = st.sidebar.slider("🌬 Wind Speed", 0.0, 10.0, 3.0)
rain = st.sidebar.slider("🌧 Rain (mm)", 0.0, 10.0, 0.0)

# =========================
# Prediction Section
# =========================
left, right = st.columns([2, 1])

if "history" not in st.session_state:
    st.session_state.history = []

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Fire Risk Prediction")

    if st.button("🔥 Analyze Fire Risk"):
        input_data = np.array([[X, Y, month, day, FFMC, DMC, DC, ISI,
                                 temp, RH, wind, rain]])

        prob = model.predict_proba(input_data)[0][1]

        st.progress(int(prob * 100))

        if prob < 0.4:
            st.markdown(f"<div class='risk-low'>🟢 LOW RISK<br>{prob:.2%}</div>", unsafe_allow_html=True)
            risk = "Low"
        elif prob < 0.7:
            st.markdown(f"<div class='risk-medium'>🟠 MEDIUM RISK<br>{prob:.2%}</div>", unsafe_allow_html=True)
            risk = "Medium"
        else:
            st.markdown(f"<div class='risk-high'>🔴 HIGH RISK<br>{prob:.2%}</div>", unsafe_allow_html=True)
            risk = "High"

        st.session_state.history.append({
            "X": X, "Y": Y,
            "Probability": prob,
            "Risk": risk
        })

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# System Info
# =========================
with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("ℹ️ System Info")
    st.write("""
    • **Algorithm:** Random Forest  
    • **Dataset:** UCI Forest Fires  
    • **Frontend:** Streamlit  
    • **Purpose:** Early Fire Detection  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Feature Importance
# =========================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("📈 Feature Importance")

features = ["X","Y","Month","Day","FFMC","DMC","DC","ISI","Temp","RH","Wind","Rain"]
importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance,color="#2ecc71")
ax.set_title("Feature Importance")
st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Prediction History
# =========================
if st.session_state.history:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🕒 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Fire Location Visualization
# =========================
if st.session_state.history:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🗺️ Fire Risk Locations")

    map_df = pd.DataFrame(st.session_state.history)
    st.scatter_chart(map_df, x="X", y="Y")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
    Forest Fire Prediction System 🔥  
    Machine Learning Academic Project
</div>
""", unsafe_allow_html=True)
