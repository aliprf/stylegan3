import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from scipy import interpolate, linalg
import tensorflow as tf

class Analyze:

    def __init__(self, exp_path: str, noise_path: str):
        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self._exp_path = exp_path
        self._noise_path = noise_path
        self._histogram = np.zeros(shape=7, dtype=np.int)
        self._histogram_2d = np.zeros(shape=(7, 7), dtype=np.int)  # 7 dimensions & 30-40-50-60-70-80-90
        self._expressions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

    def calculate_exp_histogram_2d(self):
        for file in tqdm(os.listdir(self._exp_path)):
            if file.endswith('.npy'):
                exp_f = os.path.join(self._exp_path, file)
                exp_vector = np.load(file=exp_f)
                exp_indic = np.argmax(exp_vector)
                exp_value = exp_vector[0, exp_indic]
                self._histogram_2d[exp_indic, int(exp_value * 10) - 3] += 1

    def calculate_exp_histogram(self):
        for file in tqdm(os.listdir(self._exp_path)):
            if file.endswith('.npy'):
                exp_f = os.path.join(self._exp_path, file)
                exp_vector = np.load(file=exp_f)
                exp = np.argmax(exp_vector)
                # if exp ==6:
                #     print(file)
                self._histogram[exp] += 1

    def plot_histogram_2d(self, file_name):
        fig, ax = plt.subplots()
        ind = np.arange(len(self._expressions))
        chart_w = 1 / (len(self._expressions) + 5)
        for i in tqdm(range(len(self._expressions))):
            ax.bar(ind + i * chart_w, np.log(self._histogram_2d[:, i] + 1), chart_w)
        plt.xticks(ind + chart_w, ('Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger'))
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.xlabel('Expressions', fontweight='bold')
        plt.savefig(file_name)

    def plot_histogram(self, file_name):
        fig, ax = plt.subplots()
        ax.bar(self._expressions, self._histogram, color='g')
        for i, v in enumerate(self._histogram):
            ax.text(i - .1, v + 3, str(v), color='red')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.xlabel('Expressions', fontweight='bold')
        plt.savefig(file_name)

    def create_categories(self, exp_path: str, out_dir: str):
        names_arr = [[], [], [], [], [], [], []]
        for file in tqdm(os.listdir(self._exp_path)):
            if file.endswith('.npy'):
                exp_f = os.path.join(self._exp_path, file)
                exp_vector = np.load(file=exp_f)
                exp = np.argmax(exp_vector)
                names_arr[exp].append(file)
        for i in range(len(names_arr)):
            f = open(f'{out_dir}{self._expressions[i]}.txt', "w")
            for name in names_arr[i]:
                f.write(name + '\n')
            f.close()

    def interpolate_hypersphere(self, exp_cat_path):
        noise_vectors = []
        f = open(f'{exp_cat_path}', "r")
        lines = f.readlines()
        for file in tqdm(lines):
            noise_f = os.path.join(self._noise_path, str(file.strip()).split('_')[1])
            noise_vectors.append(np.load(file=noise_f))

        v1 = noise_vectors[0]
        v2 = noise_vectors[70]

        num_steps = 250
        v1_norm = linalg.norm(v1)
        v2_norm = linalg.norm(v2)
        v2_normalized = v2 * (v1_norm / v2_norm)

        vectors = []
        for step in range(num_steps):
            interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
            interpolated_norm = linalg.norm(interpolated)
            interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
            vectors.append(interpolated_normalized)
        return np.stack(vectors)

    def interpolate(self, exp_cat_path: str):
        noise_vectors = []
        f = open(f'{exp_cat_path}', "r")
        lines = f.readlines()
        for file in tqdm(lines):
            noise_f = os.path.join(self._noise_path, str(file.strip()).split('_')[1])
            noise_vectors.append(np.load(file=noise_f))
        # interpolation:
        x_values = np.arange(start=0, stop=len(noise_vectors), step=1)
        # x_values = np.linspace(0.0, 1.0, num=len(noise_vectors))
        y_values = np.zeros(shape=[512, len(x_values)])

        inter_functions = []
        for i in range(len(y_values)):
            y_values[i] = np.array(noise_vectors)[:, 0, i]
            inter_functions.append(interpolate.interp1d(x_values, y_values[i]))
            # inter_functions.append(interpolate.CubicSpline(x_values, y_values[i]))
            # inter_functions.append(interpolate.Akima1DInterpolator(x_values, y_values[i]))
            # inter_functions.append(interpolate.PchipInterpolator(x_values, y_values[i]))

        return inter_functions

        # ynew = f(0.5)
        # print(1)


