### Script for CS329s ML Deployment Lec 
import os
import json
import requests
import time
from io import BytesIO
import streamlit as st
from PIL import Image
from predict import predict_endpoint


# Setup environment credentials (you'll need to change these)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cmpt756-final-project-pose-742126ca93fa.json" # change for your GCP key
# PROJECT="cmpt756-final-project-pose" # change for your GCP project
# PROJECT_NUMBER="669397337286" # project number can be found on the main console page
# REGION = "us-central1" # change for your GCP region (where your model is hosted)
# VERTEX_ENDPOINT_ID = "8806795668392771584" # Vertex AI endpoint ID, can be found in Deploy and Test in Vertex AI.

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(img_bytes):
    """
    Takes an image and uses a model from Compute Engine to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    preds = predict_endpoint(
                                # project=PROJECT,
    #                         endpoint_id=VERTEX_ENDPOINT_ID,
    #                         project_number=PROJECT_NUMBER,
                            img_bytes=img_bytes
                            )
    image = Image.open(BytesIO(img_bytes))
    return image, preds


### Streamlit code (works as a straigtht-forward script) ###
def streamlit_app():
    st.title("Welcome to Image Classification Model")
    st.header("Image Classification Model Deployed through GCP Serverful offerings: GCP Compute Engine and Docker.")
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader(label="Upload an image with a person",
                                    type=["png", "jpeg", "jpg"])
    st.session_state.pred_button = False
    # Create logic for app flow
    if not uploaded_file:
        st.warning("Please upload an image.")
        st.stop()
    else:
        st.session_state.uploaded_image = uploaded_file.read()
        st.image(st.session_state.uploaded_image, use_column_width=True)
        pred_button = st.button("Predict")
    # Did the user press the predict button?
    if pred_button:
        st.session_state.pred_button = True 

    # And if they did...
    if st.session_state.pred_button:
        start = time.time()
        st.session_state.image, st.session_state.preds = make_prediction(st.session_state.uploaded_image)
        end = time.time()
        st.write(f"Prediction: {st.session_state.preds}, Time taken (Docker Call): {end-start}")

# TODO: code could be cleaned up to work with a main() function...
if __name__ == "__main__":
    streamlit_app()