import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate, linalg
from deepface import DeepFace


class AgeExtraction:
    def __init__(self, model_path=None, proto_path=None):
        self._model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        if model_path is not None and proto_path is not None:
            self._model = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def predict_single_path(self, img_path):
        image = cv2.imread(img_path)
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(227, 227), mean=self._model_mean_values,
                                     swapRB=False)
        self._model.setInput(blob)
        age_8_classes = self._model.forward()
        age_4_classes = np.zeros(4)
        age_4_classes[0] = np.sum(age_8_classes[0, 0:3])
        age_4_classes[1] = np.sum(age_8_classes[0, 3:5])
        age_4_classes[2] = np.sum(age_8_classes[0, 5:7])
        age_4_classes[3] = np.sum(age_8_classes[0, 7])
        return age_4_classes

    def create_histogram(self, age_path, name=''):
        histogram = np.zeros(shape=4, dtype=np.int)
        for file in tqdm(os.listdir(age_path)):
            if file.endswith('.npy'):
                age_f = os.path.join(age_path, file)
                age_vector = np.load(file=age_f)
                age = np.argmax(age_vector)
                histogram[age] += 1
        '''plot'''
        fig, ax = plt.subplots()
        ax.bar(['0-15', '16-32', '33-53', '54-100'], histogram, color=['#0096c7', '#57cc99', '#f77f00', '#7209b7'])
        for i, v in enumerate(histogram):
            ax.text(i - .1, v + 3, str(v))
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.xlabel('Age', fontweight='bold')
        plt.savefig(name + 'age_hist.png')

    def predict_and_save_deep_face(self, img_dir, out_dir, csv_file_path):
        images_name_list = sorted(os.listdir(img_dir))
        f = open(csv_file_path, "w")
        for image_name in tqdm(images_name_list):
            img_path = img_dir + image_name

            age_4_classes = np.zeros(4)
            try:
                obj = DeepFace.analyze(img_path=img_path, actions=['age'],  enforce_detection=False)
                age_pr = obj['age']
                if 0 <= age_pr <= 15:
                    age_4_classes[0] = 1.0
                elif 16 <= age_pr <= 32:
                    age_4_classes[1] = 1.0
                elif 33 <= age_pr <= 53:
                    age_4_classes[2] = 1.0
                elif 54 <= age_pr <= 100:
                    age_4_classes[3] = 1.0
            except Exception as e:
                print(str(e))

            age_dir_name = out_dir + os.path.splitext(image_name)[0] + '_age.npy'
            np.save(age_dir_name, age_4_classes)
            f.write(str(os.path.splitext(image_name)[0]) + ',' + " ".join(
                str(x) for x in np.round(age_4_classes.tolist(), 3).tolist()) + ' \n')
        ''''''
        f.close()

    def predict_and_save(self, img_dir, out_dir, csv_file_path):
        images_name_list = sorted(os.listdir(img_dir))
        f = open(csv_file_path, "w")
        for image_name in tqdm(images_name_list):
            image_dir = img_dir + '/' + image_name
            image = cv2.imread(image_dir)
            blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(227, 227), mean=self._model_mean_values,
                                         swapRB=False)

            self._model.setInput(blob)
            age_8_classes = self._model.forward()
            age_4_classes = np.zeros(4)
            age_4_classes[0] = np.round(np.sum(age_8_classes[0, 0:3]), 3)
            age_4_classes[1] = np.round(np.sum(age_8_classes[0, 3:5]), 3)
            age_4_classes[2] = np.round(np.sum(age_8_classes[0, 5:7]), 3)
            age_4_classes[3] = np.round(np.sum(age_8_classes[0, 7]), 3)

            age_dir_name = out_dir + '/' + os.path.splitext(image_name)[0] + '_age.npy'
            np.save(age_dir_name, age_4_classes)
            f.write(str(os.path.splitext(image_name)[0]) + ',' + " ".join(
                str(x) for x in np.round(age_4_classes.tolist(), 3).tolist()) + ' \n')
        f.close()
