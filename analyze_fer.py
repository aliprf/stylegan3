import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from scipy import interpolate, linalg
import tensorflow as tf
import cv2


class AnalyzeFer:

    def __init__(self, exp_path: str, noise_path: str):
        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if exp_path is not None:
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
        ax.bar(self._expressions, self._histogram, color='#219ebc')
        for i, v in enumerate(self._histogram):
            ax.text(i - .1, v + 3, str(v), color='#ff006e')
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

    def interpolate_by_semantic(self, noise_path: str,
                                anno_path_fer: str, task_id_fer: int,
                                anno_path_gender: str, task_id_gender: int,
                                anno_path_race: str, task_id_race: int,
                                anno_path_age: str, task_id_age: int,
                                ):
        noise_vectors_0 = []
        noise_vectors_1 = []
        noise_vectors_2 = []
        noise_vectors_3 = []
        for file in tqdm(os.listdir(noise_path)):
            if file.endswith(".npy"):
                bare_f = str(file).split('.')[0]
                noise_f = os.path.join(noise_path, file)
                noise = np.load(noise_f)[0]

                fer_f = os.path.join(anno_path_fer, 'exp_' + bare_f + '.npy')
                fer = np.load(fer_f)[0]
                gender_f = os.path.join(anno_path_gender, bare_f + '_gender.npy')
                gender = np.load(gender_f)
                race_f = os.path.join(anno_path_race, bare_f + '_race.npy')
                race = np.load(race_f)
                age_f = os.path.join(anno_path_age, bare_f + '_age.npy')
                age = np.load(age_f)
                # fer + gender
                if np.argmax(fer) == task_id_fer and fer[task_id_fer] >= 0.6 and \
                        np.argmax(gender) == task_id_gender and gender[task_id_gender] >= 0.70:
                    noise_vectors_0.append(noise)
                # fer + race
                if np.argmax(fer) == task_id_fer and fer[task_id_fer] >= 0.6 and \
                        np.argmax(race) == task_id_race and race[task_id_race] >= 0.70:
                    noise_vectors_1.append(noise)
                # fer + age
                if np.argmax(fer) == task_id_fer and fer[task_id_fer] >= 0.6 and \
                        np.argmax(age) == task_id_age and age[task_id_age] >= 0.70:
                    noise_vectors_2.append(noise)
                # fer + race + gender
                if np.argmax(fer) == task_id_fer and fer[task_id_fer] >= 0.6 and \
                        np.argmax(gender) == task_id_gender and gender[task_id_gender] >= 0.70 and \
                        np.argmax(race) == task_id_race and race[task_id_race] >= 0.70:
                    noise_vectors_3.append(noise)
        '''interpolation'''
        inter_functions_arr = []
        nvs = [noise_vectors_0, noise_vectors_1, noise_vectors_2, noise_vectors_3]
        for nv in nvs:
            x_values = np.arange(start=0, stop=len(nv), step=1)
            y_values = np.zeros(shape=[512, len(x_values)])
            inter_functions = []
            for i in range(len(y_values)):
                y_values[i] = np.array(nv)[:, i]
                inter_functions.append(interpolate.CubicSpline(x_values, y_values[i]))
            inter_functions_arr.append(inter_functions)
        # ['a_fe', 'a_bl', 'a_ch', 'a_bl_fe']
        return {'a_fe': inter_functions_arr[0],
                'a_bl': inter_functions_arr[1],
                'a_ch': inter_functions_arr[2],
                'a_bl_fe': inter_functions_arr[3]
                }
        # return inter_functions_arr

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

    def create_fer_noises(self, indices_path, exp_stat_path):
        indices = np.load(indices_path)
        exp_stat_path = np.load(exp_stat_path)
        gen_noise = []
        # npy_noise_orig = np.round(np.random.RandomState(27).randn(1, 512), decimals=3)
        npy_noise_orig = np.round(np.random.RandomState(27).randn(1, 512), decimals=3)
        gen_noise.append(npy_noise_orig)

        # indices = np.arange(145, 150)
        indices = [125, 464]

        for i in indices:
            npy_noise = np.copy(npy_noise_orig)
            mu, std = -5.0, 0.1
            epsilon = np.round(np.random.normal(), decimals=3)
            ind_sample = mu + 1.0 * std * epsilon
            npy_noise[0, i] = ind_sample
            gen_noise.append(npy_noise)

        # for i in range(10):
        #     npy_noise = np.copy(npy_noise_orig)
        #     for ind in range(len(indices)):
        #         mu, std = exp_stat_path[ind]
        #         epsilon = np.round(np.random.normal(), decimals=3)
        #         ind_sample = mu + 2.0 * std * epsilon
        #         npy_noise[0, indices[ind]] = ind_sample
        #     gen_noise.append(npy_noise)
        return gen_noise

    def create_csv(self, img_path, fer_path, csv_file_path):
        f = open(csv_file_path, "w")
        for file in tqdm(os.listdir(img_path)):
            if file.endswith('.jpg'):
                img_f = os.path.join(img_path, file)
                bare_f = str(file).split('.')[0]
                fer_f = os.path.join(fer_path, 'exp_' + bare_f + '.npy')
                if os.path.exists(fer_f):
                    fer = np.load(fer_f)
                    f.write(str(os.path.splitext(file)[0]) + ',' +
                            " ".join(str(x) for x in np.round(fer.tolist(), 3).tolist()) + ' \n')
        f.close()
