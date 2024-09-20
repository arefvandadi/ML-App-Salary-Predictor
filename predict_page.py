import streamlit as st
import pickle
import numpy as np

# Open the pickled ML model
def load_model():
    with open('./Full_ML_Model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data["model"], data["le_Country"], data["le_EdLevel"]

model, le_Country, le_EdLevel = load_model()

