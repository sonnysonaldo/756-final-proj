### Script for CS329s ML Deployment Lec 
import os
import json
import requests
import streamlit as st
import torch
import torchvision
from utils import load_and_prep_image, update_logger, predict_json

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pose-estimation-417318-f4da158a7162.json" # change for your GCP key
PROJECT = "pose-estimation-417318" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Pose Estimation")
st.header("Pose Estimation Model Deployed through GCP Serverless offerings: GCP AI Platform and App Engine.")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = load_and_prep_image(image, img_shape=100)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = image.flatten()
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(model=model,
                         project=PROJECT,
                         region=REGION,
                         instances=image)
    return image, preds

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

MODEL = "keypoint_rcnn"

# And if they did...
if st.session_state.pred_button:
    st.session_state.image, st.session_state.pred = make_prediction(st.session_state.uploaded_image, model=MODEL, class_names=None)
    st.write(f"Prediction: {st.session_state.preds}, \
               Confidence: {st.session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    st.session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if st.session_state.feedback == "Select an option":
        pass
    elif st.session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=st.session_state.image,
                            model_used=MODEL,
                            pred_class=st.session_state.pred_class,
                            pred_conf=st.session_state.pred_conf,
                            correct=True))
    elif st.session_state.feedback == "No":
        st.session_state.correct_class = st.text_input("What should the correct label be?")
        if st.session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=st.session_state.image,
                                model_used=MODEL,
                                pred_class=st.session_state.pred_class,
                                pred_conf=st.session_state.pred_conf,
                                correct=False,
                                user_label=st.session_state.correct_class))

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()