import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions


def get_model():
    model = load_model(r"./resnet18", compile=False)
    return model


def read_imagefile(file, resize_to=(224, 224)):
    image = Image.open(BytesIO(file)).convert("RGB")
    image = image.resize(resize_to)
    image = np.asarray(image).astype(np.float32)
    return image


async def get_predictions(model, image):
    image = preprocess_input(image, mode="torch")
    image = tf.expand_dims(image, axis=0)
    preds = model.predict(image)

    post_preds = decode_predictions(preds, top=5)[0]

    response = []

    for res in post_preds:
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = float(res[2])

        response.append(resp)

    return response
