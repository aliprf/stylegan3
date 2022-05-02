import imgui
import tensorflow as tf
# import tensorflow.keras as keras
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
from tqdm import tqdm


class FER:
    def __init__(self, h5_address: str = None, GPU=True):
        self._exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

        if GPU:
            physical_devices = tf.config.list_physical_devices('GPU')
            print(len(physical_devices))
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if h5_address is not None:
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

    def create_noise(self, h5_addresses, exp_id, num):
        encoder = tf.keras.models.load_model(h5_addresses[0], custom_objects={'tf': tf, 'Sampling': Sampling})
        decoder = tf.keras.models.load_model(h5_addresses[1], custom_objects={'tf': tf, 'Sampling': Sampling})

        noises = []
        i = 1
        npy_noise = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)
        mu, log_var = encoder.predict_on_batch(npy_noise)
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        while i < num:
            p_exp = [0.1, 0.0, 0.0, 0.00, 0.0, 0.0, 0.9]
            f_exp = np.expand_dims(np.array(p_exp), 0)

            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            latent = mu + tf.exp(0.5 * log_var) * epsilon
            noise = decoder.predict_on_batch([latent, f_exp])
            # noises.append(npy_noise)
            noises.append(noise)
            i += 1
        model = None
        return noises

    def _get_sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=z_mean,
                                                 stddev=tf.exp(0.5 * z_log_var))
        epsilon1 = tf.keras.backend.random_normal(shape=(batch, dim))

        XXX = z_mean + tf.exp(0.5 * z_log_var) * epsilon1
        YYY = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        epsilon = tf.keras.backend.random_uniform(shape=(batch, dim), minval=-10000, maxval=10000)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # return epsilon

    def create_total_cvs(self, img_path, fer_path, age_path, race_path, gender_path, cvs_file):
        f = open(cvs_file, "a")
        k_fem = 0
        k_male = 0
        FEMALE = 0
        MALE = 1
        for file in tqdm(os.listdir(img_path)):
            if file.endswith('.jpg') and (k_fem < 5000 or k_male < 5000):
                img_f = os.path.join(img_path, file)
                bare_f = str(file).split('.')[0]
                fer_f = os.path.join(fer_path, 'exp_' + bare_f + '.npy')
                fer = np.load(fer_f)[0]
                age_f = os.path.join(age_path, bare_f + '_age.npy')
                age = np.load(age_f)
                race_f = os.path.join(race_path, bare_f + '_race.npy')
                race = np.load(race_f)
                gender_f = os.path.join(gender_path, bare_f + '_gender.npy')
                gender = np.load(gender_f)

                if np.argmax(fer) == 6 or np.argmax(fer) == 0:
                    if np.argmax(gender) == FEMALE and k_fem <= 5000:
                        k_fem += 1
                        f.write(file + ',' +
                                " ".join(str(x) for x in np.round(fer.tolist(), 3).tolist()) + ',' +
                                " ".join(str(x) for x in np.round(age.tolist(), 3).tolist()) + ',' +
                                " ".join(str(x) for x in np.round(race.tolist(), 3).tolist()) + ',' +
                                " ".join(str(x) for x in np.round(gender.tolist(), 3).tolist()) + '\n')

                    elif np.argmax(gender) == MALE and k_male <= 5000:
                        k_male += 1
                        f.write(file + ',' +
                                " ".join(str(x) for x in np.round(fer.tolist(), 3).tolist()) + ',' +
                                " ".join(str(x) for x in np.round(age.tolist(), 3).tolist()) + ',' +
                                " ".join(str(x) for x in np.round(race.tolist(), 3).tolist()) + ',' +
                                " ".join(str(x) for x in np.round(gender.tolist(), 3).tolist()) + '\n')
        print("DONE")


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
