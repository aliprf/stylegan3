#!pip install deepface
from deepface import DeepFace
import numpy as np
from numpy import save, load
import numpy as np
import pickle
import os
from tqdm import tqdm
from numpy import save, load
import math
from PIL import Image
from numpy import save, load
from scipy.linalg import svd


class FaceRecognition:
    def compare_by_path(self, s_path, d_path):
        matched_list = []
        diff_list = []
        for file in tqdm(os.listdir(s_path)):
            img_add_1 = os.path.join(s_path, file)
            img_add_2 = os.path.join(d_path, file)
            try:
                obj = DeepFace.verify(img_add_1, img_add_2
                                      , model_name='ArcFace',
                                      detector_backend='retinaface')
                if obj['verified']:
                    matched_list.append(file)
                else:
                    diff_list.append(file)
            except Exception as e:
                pass
        return len(matched_list)/(len(matched_list) + len(diff_list))

    def analyze_all_by_path(self, images_path, save_path):
        for file in tqdm(os.listdir(images_path)):
            if str(file).endswith('jpg'):
                img_file = os.path.join(images_path, file)
                obj = DeepFace.analyze(img_path=img_file, actions=['age'])#actions=['age', 'gender', 'race', 'emotion'])

    def verify_two_images(self, img_add_1, img_add_2):
        obj = DeepFace.verify(img_add_1, img_add_2
                              , model_name='ArcFace',
                              detector_backend='retinaface',
                              enforce_detection=False)
        print(obj["verified"])
