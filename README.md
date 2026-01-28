🔥 Forest Fire Prediction System
📌 Overview

The Forest Fire Prediction System is a machine learning–based web application that predicts the risk of forest fires using environmental and meteorological data.
It leverages a Random Forest classifier trained on the UCI Forest Fires dataset and provides real-time predictions through an interactive Streamlit dashboard with a modern green–black themed UI.

This system aims to assist in early fire risk assessment, helping improve preparedness and prevention strategies.

🎯 Objectives

Predict the probability of forest fire occurrence

Analyze the impact of environmental and weather factors

Visualize feature importance for model interpretability

Provide a clean, interactive, and user-friendly web interface

🧠 Machine Learning Model

Algorithm: Random Forest Classifier

Why Random Forest?

Handles non-linear relationships well

Robust against overfitting

Works effectively with mixed feature types

Provides feature importance for interpretability

📊 Dataset

Source: UCI Machine Learning Repository – Forest Fires Dataset

Total Samples: 517

Target Variable: fire (0 = No Fire, 1 = Fire)

🔍 Features Used
Feature	Description
X, Y	Spatial coordinates in the forest
Month	Encoded month (0–11)
Day	Encoded day of the week (0–6)
FFMC	Fine Fuel Moisture Code
DMC	Duff Moisture Code
DC	Drought Code
ISI	Initial Spread Index
Temp	Temperature (°C)
RH	Relative Humidity (%)
Wind	Wind speed
Rain	Rainfall (mm)
📘 Fire Weather Index Abbreviations

FFMC – Fine Fuel Moisture Code

DMC – Duff Moisture Code

DC – Drought Code

ISI – Initial Spread Index

These indices are part of the Canadian Forest Fire Weather Index (FWI) system.

🖥️ System Architecture
Dataset → Data Preprocessing → ML Model (Random Forest)
        → Model Evaluation → Model Serialization
        → Streamlit Frontend → User Prediction

🎨 Features

✔ Real-time fire risk prediction
✔ Probability-based risk classification (Low / Medium / High)
✔ Feature importance visualization
✔ Prediction history tracking
✔ Fire location visualization
✔ Green-black themed modern UI
✔ Interactive sliders and charts

🛠️ Technologies Used
Backend / ML

Python

Scikit-learn

Pandas

NumPy

Frontend / Visualization

Streamlit

Matplotlib

Custom CSS (Green–Black Theme)
