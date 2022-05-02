import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate, linalg
from image_utility import ImageUtilities


class GenderExtraction:
    def __init__(self, model_path, proto_path):
        self._model = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def predict_single_path(self, img_path):
        image = cv2.imread(img_path)
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(224, 224))
        self._model.setInput(blob)
        gender = self._model.forward()
        gender = gender[0]
        return gender

    def create_histogram(self, gender_path):
        histogram = np.zeros(shape=2, dtype=np.int)
        for file in tqdm(os.listdir(gender_path)):
            if file.endswith('.npy'):
                _f = os.path.join(gender_path, file)
                _vector = np.load(file=_f)
                gender = np.argmax(_vector)
                histogram[gender] += 1
        '''plot'''
        fig, ax = plt.subplots()
        ax.bar(['Female', 'Male'], histogram, color='b')
        for i, v in enumerate(histogram):
            ax.text(i - .1, v + 3, str(v), color='red')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.xlabel('Gender', fontweight='bold')
        plt.savefig('Gender_hist.png')

    def predict_and_save(self, img_dir, out_dir, csv_file_path):
        images_name_list = sorted(os.listdir(img_dir))
        f = open(csv_file_path, "w")
        for image_name in tqdm(images_name_list):
            image_dir = img_dir + '/' + image_name
            image = cv2.imread(image_dir)
            blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(224, 224))
            self._model.setInput(blob)
            gender = self._model.forward()
            gender = gender[0]

            age_dir_name = out_dir + '/' + os.path.splitext(image_name)[0] + '_gender.npy'
            np.save(age_dir_name, gender)
            f.write(str(os.path.splitext(image_name)[0]) + ',' +
                    " ".join(str(x) for x in np.round(gender.tolist(), 3).tolist()) + ' \n')
        f.close()
