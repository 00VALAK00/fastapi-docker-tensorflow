from fastapi import FastAPI
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
from tensorflow.keras.models import load_model
from pydantic import BaseModel

app = FastAPI(title="Water-potability")


class water_characteristics(BaseModel):
    aluminium: float
    ammonia: float
    arsenic: float
    barium: float
    cadmium: float
    chloramine: float
    chromium: float
    copper: float
    flouride: float
    bacteria: float
    viruses: float
    lead: float
    nitrates: float
    nitrites: float
    mercury: float
    perchlorate: float
    radium: float
    selenium: float
    silver: float
    uranium: float


def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)

    def f1_score_func(y_true, y_pred):
        return f1_score(y_true, y_pred, average='micro')

    f1 = tf.py_function(f1_score_func, (y_true, y_pred), tf.float32)
    return f1


def load_tf_model():
    global model
    custom_objects = {"f1_score": f1_score}
    with tf.keras.saving.custom_object_scope(custom_objects):
        reconstructed = tf.keras.models.load_model("model.keras")
    return reconstructed


@app.on_event("startup")
def load_tf_model():
    global model
    custom_objects = {"f1_score": f1_score}
    with tf.keras.saving.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model("model.keras")


@app.get("/")
def home():
    return "Water Potability prediction application"


@app.post("/predict")
def predict(water: water_characteristics):
    data = np.array([water.aluminium,
                     water.ammonia,
                     water.arsenic,
                     water.barium,
                     water.cadmium,
                     water.chloramine,
                     water.chromium,
                     water.copper,
                     water.flouride,
                     water.bacteria,
                     water.viruses,
                     water.lead,
                     water.nitrates,
                     water.nitrites,
                     water.mercury,
                     water.perchlorate,
                     water.radium,
                     water.selenium,
                     water.silver,
                     water.uranium])
    data = np.expand_dims(data, axis=0)
    pred = model.predict(data)
    return {"prediction": pred}






