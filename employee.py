import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample dataset (Replace with real employee data)
data = {
    "experience": [1, 3, 5, 7, 10, 12, 2, 6, 8, 15],
    "education": ["Bachelor", "Master", "PhD", "Bachelor", "Master", "PhD", "Bachelor", "Master", "PhD", "Bachelor"],
    "working_hours": [40, 45, 50, 35, 60, 55, 30, 50, 40, 45],
    "performance": [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Encoding categorical data
le = LabelEncoder()
df["education"] = le.fit_transform(df["education"])

# Splitting data
X = df.drop(columns=["performance"])
y = df["performance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_performance(experience, education, working_hours):
    education_encoded = le.transform([education])[0]
    prediction = model.predict(np.array([[experience, education_encoded, working_hours]]))
    return "High Performance" if prediction[0] == 1 else "Low Performance"

iface = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Number(label="Years of Experience"),
        gr.Dropdown(choices=["Bachelor", "Master", "PhD"], label="Education Level"),
        gr.Number(label="Working Hours per Week")
    ],
    outputs="text",
    title="Employee Performance Predictor",
    description="Predicts if an employee is likely to have high or low performance based on experience, education, and working hours."
)

iface.launch()
