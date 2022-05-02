import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from numpy import save, load, asarray
import csv
from skimage.io import imread, imsave
from skimage.transform import resize
import pickle
from PIL import Image

import os


class ImageUtilities:
    def read_image(self, img_path: str):
        return imread(img_path)

    def resize_image(self, npy_img, w: int, h: int, ch: int):
        img = resize(npy_img, (w, h, ch))
        return img

    def save_image(self, npy_img, save_path: str, save_name: str):
        imsave(fname=f'{save_path}/{save_name}.jpg', arr=npy_img)
        # Image.fromarray(npy_img, 'RGB').save(f'{save_path}/{save_name}.jpg')

