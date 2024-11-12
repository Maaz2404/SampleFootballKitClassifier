import streamlit as st
from fastai.vision.all import *
from PIL import Image
from pathlib import Path
import pathlib

import platform


learn_inf = load_learner('export.pkl')
# Load the pre-trained model


# Streamlit app layout
st.title("Football Kit Classifier")
st.write("Upload an image of a football kit, and the app will classify the team!")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Make prediction
    pred, pred_idx, probs = learn_inf.predict(image)

    # Display the results
    st.write(f"Prediction: **{pred}**")
    st.write(f"Probability: **{probs[pred_idx]*100:.2f}%**")
