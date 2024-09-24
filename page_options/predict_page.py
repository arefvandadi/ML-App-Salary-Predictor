import streamlit as st
import pickle
import numpy as np

# Open the pickled ML model
def load_model():
    with open('./data/trained-ML-model/Full_ML_Model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data["model"], data["le_Country"], data["le_EdLevel"]

model, le_Country, le_EdLevel = load_model()


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",    
        "United Kingdom",    
        "Germany",   
        "Canada",          
        "Brazil",           
        "France",           
        "Spain",           
        "Australia",         
        "Netherlands",       
        "Poland",      
        "Italy", 
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education", education)

    experience = st.slider("Years of Experience", 0, 50, 3, 1)

    salary_button_clicked = st.button("Calculate Salary")

    if salary_button_clicked:
        input = np.array([[country, education_level, experience]])
        input[:, 0] = le_Country.fit_transform(input[:, 0])
        input[:, 1] = le_EdLevel.fit_transform(input[:, 1])
        input = input.astype(float)
        predicted_salary = model.predict(input)
        st.subheader(f"The estimated salary is ${predicted_salary[0]:.2f}")