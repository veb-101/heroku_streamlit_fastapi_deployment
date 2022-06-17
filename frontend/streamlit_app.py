import requests
from PIL import Image
import streamlit as st

# Change this with your deployed app URL
FASTAPI_ENDPOINT = r"https://veb-fastapi-back.herokuapp.com/predict"


st.title("Streamlit Application")
st.header("Image Classifiation using Resnet-18 model.")
st.text("The model will return confidence scores for the 1000 classes present in ImageNet.\nWe will print the TOP-5 predictions.")


with st.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose an Image ...", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Predict!")
    st.write("UPLOADED!")

if submitted and uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption=uploaded_file.name)

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
