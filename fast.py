from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import uuid
import os

from ultralytics import YOLO

app = FastAPI()

# Mount static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- LOAD MODELS ----------------
YOLO_MODEL = YOLO("acl.pt")

def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224,224,3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(base.input, out)

    try:
        model.load_weights("model.weights.h5")
    except:
        model.load_weights("model.weights.h5", by_name=True, skip_mismatch=True)

    return model

CNN_MODEL = build_model()

# ---------------- GRADCAM ----------------
def preprocess(img):
    img = cv2.resize(img, (224,224)) / 255.0
    return np.expand_dims(img, 0)

def make_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        CNN_MODEL.input,
        [CNN_MODEL.get_layer("top_conv").output, CNN_MODEL.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.GaussianBlur(heatmap, (7,7), 0)
    heatmap = np.power(heatmap, 1.5)
    heatmap[heatmap < 0.2] = 0

    return heatmap

def overlay(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    h, w = img.shape[:2]
    colormap = cv2.resize(colormap, (w, h))

    return cv2.addWeighted(img, 0.6, colormap, 0.4, 0)

# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # YOLO
    results = YOLO_MODEL(img)
    yolo_img = results[0].plot(conf=False)

    # Crop
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return {"error": "No region detected"}

    x1,y1,x2,y2 = map(int, boxes[0])
    crop = img[y1:y2, x1:x2]

    # GradCAM
    img_array = preprocess(crop)
    heatmap = make_gradcam(img_array)
    gradcam = overlay(crop, heatmap)

    # Save
    uid = str(uuid.uuid4())
    path = f"outputs/{uid}.png"
    cv2.imwrite(path, gradcam)

    return {"image": f"/{path}"}