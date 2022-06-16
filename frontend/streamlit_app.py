import json
import requests
from PIL import Image
import streamlit as st


FASTAPI_ENDPOINT = r"http://127.0.0.1:12345/predict"


st.title("Image Classification Basic")
st.header("Imagenet Image Classifiation using Resnet-18 model")
st.text("The ResNet-18 model will return class probablities for all of the 1000 classes present in Imagenet.")


with st.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose an Image ...", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Predict!")

if submitted and uploaded_file is not None:
    st.write("UPLOADED!")
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded image.")
    st.text(uploaded_file.name)

    st.write("\nClassifying...")

    # ==============================================================
    send_data = uploaded_file.getvalue()  # byte data
    file = {"image_data": send_data}

    resp_post = requests.post(url=FASTAPI_ENDPOINT, files=file)
    decoded_predictions = resp_post.json()
    # ==============================================================

    for key, value_dict in decoded_predictions.items():
        pred_class = value_dict["class"]
        pred_conf = value_dict["confidence"]

        message = f"Class: {pred_class:<20} Score: {pred_conf}"
        st.text(message)

    # ==============================================================

    submitted = False
    uploaded_file = None
