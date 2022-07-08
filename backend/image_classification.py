import uvicorn
from fastapi import FastAPI, UploadFile, File

# ===========================================================================
# Imports related to Model loading and image processing.
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

# ===========================================================================

# Create our application
MODEL = None

app = FastAPI(title="Image Classification using ResNet-18")


@app.on_event("startup")
def get_model():
    global MODEL
    MODEL = load_model("resnet18", compile=False)
    # As the first pass through takes longer to run,
    # we'll pass a dummy input through the model as first pass

    # Passing dummy input through the model
    MODEL.predict(tf.random.normal((1, 224, 224, 3)))
    return MODEL


async def read_imagefile(file, resize_to=(224, 224)):
    image = Image.open(BytesIO(file)).convert("RGB")
    image = image.resize(resize_to)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image


async def predict(image):
    global MODEL
    if not MODEL:
        MODEL = get_model()

    image = preprocess_input(image, mode="torch")
    image = tf.expand_dims(image, axis=0)
    preds = MODEL.predict(image)

    # Return the top-5 prediction classes and confidence scores.
    post_preds = decode_predictions(preds, top=5)[0]

    prediction_dictionary = {}

    for idx, res in enumerate(post_preds, 1):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = float(res[2])

        prediction_dictionary[idx] = resp

    return prediction_dictionary


@app.get("/")
async def root_page():  # user-defined Asynchronous function, the function name can be anything.
    return {"message": "Connection established, Welcome!!!"}


@app.post("/predict")
async def custom_prediction(image_data: UploadFile = File()):

    # # Validate file type
    # extension = image_data.filename.split(".")[-1] in ("jpg", "jpeg", "png")

    # if not extension:
    #     return "Image must be jpg or png or jpeg format!"

    # Data upload
    data = await image_data.read()

    # Reading uploaded data and making predictions
    image = await read_imagefile(data)
    prediction = await predict(image)

    return prediction


if __name__ == "__main__":
    uvicorn.run("image_classification:app", reload=True, port=12345)
