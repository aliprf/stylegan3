import imgui
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from numpy import save, load, asarray
import csv
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
import os

from image_utility import ImageUtilities

class FER:
    def __init__(self, h5_address: str, GPU=True):
        self._exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

        if GPU:
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self._model = self._load_model(h5_address=h5_address)

    def _load_model(self, h5_address: str):
        """load weight file and create the model once"""
        model = tf.keras.models.load_model(h5_address, custom_objects={'tf': tf})
        return model

    def recognize_fer(self, npy_img):
        """create and image from the path and recognize expression"""
        img_util = ImageUtilities()
        # resize img to 1*224*224*3
        img = img_util.resize_image(npy_img=npy_img, w=224, h=224, ch=3)
        img = np.expand_dims(img, axis=0)
        #
        prediction = self._model.predict_on_batch([img])
        exp_vector = np.array(prediction[0])
        # embeddings = prediction[1:]
        return self._exps[np.argmax(exp_vector)], exp_vector

    def load_and_recognize_fer(self, img_path: str):
        """create and image from the path and recognize expression"""
        img_util = ImageUtilities()

        img = img_util.read_image(img_path=img_path)
        # resize img to 1*224*224*3
        img = img_util.resize_image(npy_img=img, w=224, h=224, ch=3)
        img = np.expand_dims(img, axis=0)
        #
        prediction = self._model.predict_on_batch([img])
        exp_vector = np.array(prediction[0])
        # embeddings = prediction[1:]
        return self._exps[np.argmax(exp_vector)], exp_vector




