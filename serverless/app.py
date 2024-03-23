### Script for CS329s ML Deployment Lec 
import os
import json
import requests
from io import BytesIO
import streamlit as st
from PIL import Image
from predict import predict_endpoint


# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="" # change for your GCP key
PROJECT="" # change for your GCP project
PROJECT_NUMBER="" # project number can be found on the main console page
REGION="" # change for your GCP region (where your model is hosted)
VERTEX_ENDPOINT_ID="" # Vertex AI endpoint ID, can be found in Deploy and Test in Vertex AI.

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(img_bytes):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    preds = predict_endpoint(project=PROJECT,
                            endpoint_id=VERTEX_ENDPOINT_ID,
                            project_number=PROJECT_NUMBER,
                            img_bytes=img_bytes,
                            )
    image = Image.open(BytesIO(img_bytes))
    return image, preds


### Streamlit code (works as a straigtht-forward script) ###
def streamlit_app():
    st.title("Welcome to Pose Estimation")
    st.header("Pose Estimation Model Deployed through GCP Serverless offerings: GCP AI Platform and App Engine.")
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
        st.session_state.image, st.session_state.preds = make_prediction(st.session_state.uploaded_image)
        st.write(f"Prediction: {st.session_state.preds}")

# TODO: code could be cleaned up to work with a main() function...
if __name__ == "__main__":
    streamlit_app()