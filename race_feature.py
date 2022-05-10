import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate, linalg
from image_utility import ImageUtilities


class RaceExtraction:
    def __init__(self, model_path, GPU=True):
        if GPU:
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self._model = tf.keras.models.load_model(model_path)

    def predict_single_path(self, img_path):
        img_util = ImageUtilities()
        image = cv2.imread(img_path)
        image = img_util.resize_image(npy_img=image, w=224, h=224, ch=3)
        image = image / 255
        image = np.expand_dims(image, axis=0)
        race = self._model.predict(image)[0, :]
        return race

    def create_histogram(self, race_path):
        histogram = np.zeros(shape=6, dtype=np.int)
        for file in tqdm(os.listdir(race_path)):
            if file.endswith('.npy'):
                _f = os.path.join(race_path, file)
                _vector = np.load(file=_f)
                race = np.argmax(_vector)
                histogram[race] += 1
        '''plot'''
        fig, ax = plt.subplots()
        ax.bar(['asian', 'indian', 'black', 'white', 'mid-east', 'latino'], histogram, color='#219ebc')
        for i, v in enumerate(histogram):
            ax.text(i - .1, v + 3, str(v), color='#ff006e')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.xlabel('Race', fontweight='bold')
        plt.savefig('race_hist.png')

    def predict_and_save(self, img_dir, out_dir, csv_file_path):
        img_util = ImageUtilities()
        images_name_list = sorted(os.listdir(img_dir))
        f = open(csv_file_path, "w")
        for image_name in tqdm(images_name_list):
            image_dir = img_dir + '/' + image_name
            image = cv2.imread(image_dir)
            image = image / 255
            image = img_util.resize_image(npy_img=image, w=224, h=224, ch=3)
            image = np.expand_dims(image, axis=0)
            race = self._model.predict(image)[0]

            race_dir_name = out_dir + '/' + os.path.splitext(image_name)[0] + '_race.npy'
            np.save(race_dir_name, race)
            f.write(str(os.path.splitext(image_name)[0]) + ',' +
                    " ".join(str(x) for x in  np.round(race.tolist(),3)) + ' \n')
        f.close()
