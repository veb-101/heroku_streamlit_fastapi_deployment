import json
import requests
from PIL import Image
import streamlit as st


FASTAPI_ENDPOINT = "https://veb-fastapi-back.herokuapp.com/predict"


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

    # # ==============================================================
    # resp_get = requests.get(url=FASTAPI_ENDPOINT)
    # get_response = json.loads(resp_get.text)
    # st.write(get_response)

    # ==============================================================
    send_data = uploaded_file.getvalue()  # byte data
    file = {"image_data": send_data}

    st.write("\nClassifying...")

    resp_post = requests.post(url=FASTAPI_ENDPOINT, files=file)
    predictions = json.loads(resp_post.text)

    # ==============================================================
    st.text("CLASS | SCORE")

    for pred_dict in predictions:
        st.write(f"{pred_dict['class']} | {pred_dict['confidence']}")

    submitted = False
    uploaded_file = None
