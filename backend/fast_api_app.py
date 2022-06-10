# Uvicorn is an ASGI web server implementation for Python.
# Uvicorn crates a web server that handles requests made to the server
# and then forwards it the application.

import uvicorn  #
from fastapi import FastAPI
from fastapi import UploadFile, File

# ===========================================================================
from utilities import get_model
from utilities import read_imagefile
from utilities import get_predictions

MODEL = None

app = FastAPI(title="Image Classification using ResNet-18")


@app.on_event("startup")
def load_saved_model():
    global MODEL
    MODEL = get_model()
    return MODEL


# the function is triggered automatically
@app.get("/")
async def IDK():  # any name for the function
    return {"message": "Hello World"}
    # return "adsadasdsa"


@app.get("/welcome")
async def get_name(name: str):
    return f"welcome to the page: {name}"


@app.get("/{name}")
async def dynamic_page(name: str):
    return f"welcome to your page: {name}"


# https://fastapi.tiangolo.com/tutorial/request-files/
@app.post("/predict")
async def custom_prediction(image_data: UploadFile = File()):
    # Data load
    data = await image_data.read()

    # Convert to tensorflow tensor and predict
    image = read_imagefile(data)
    prediction = await get_predictions(MODEL, image)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=12345)
