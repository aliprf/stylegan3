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
import csv
import shutil
from config import FolderStructures
from image_utility import ImageUtilities
from tqdm import tqdm
from sklearn.utils import shuffle
import glob


class FER:
    def __init__(self, h5_address: str = None, GPU=True):
        self._exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        if GPU:
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if h5_address is not None:
            self._model = self._load_model(h5_address=h5_address)

    def _load_model(self, h5_address: str):
        """load weight file and create the model once"""
        model = tf.keras.models.load_model(h5_address, custom_objects={'tf': tf})
        return model

    def recognize_fer(self, npy_img):
        """create and image from the path and recognize expression"""
        # resize img to 1*224*224*3
        img = ImageUtilities().resize_image(npy_img=npy_img, w=224, h=224, ch=3)
        img = np.expand_dims(img, axis=0)
        #
        prediction = self._model.predict_on_batch([img])
        exp_vector = np.array(prediction[0])

        prediction = None
        img = None
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

    def copy_all(self, cvs_file, s_img_folder, s_exp_path, s_noise_path, s_age_path, s_gender_path, s_race_path,
                 d_img_folder, d_exp_path, d_noise_path, d_age_path, d_gender_path, d_race_path):
        """"""
        '''deleting files'''
        files = glob.glob(d_img_folder + '*.jpg', recursive=True)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                pass
        _paths = [d_exp_path, d_noise_path, d_age_path, d_gender_path, d_race_path]
        for _path in tqdm(_paths):
            files = glob.glob(_path + '*.npy', recursive=True)
            print('deleting ->')
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    pass

        #  open csv file, and read img. find and copy it.
        with open(cvs_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in tqdm(csv_reader):
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    source_prefix = row[5]
                    img_name = row[0]
                    shutil.copy2(source_prefix+s_img_folder + img_name, d_img_folder + img_name)
                    f_name = img_name.split('.')[0]

                    shutil.copy2(source_prefix+s_noise_path + f_name + '.npy', d_noise_path + f_name + '.npy')
                    shutil.copy2(source_prefix+s_exp_path + 'exp_' + f_name + '.npy', d_exp_path + 'exp_' + f_name + '.npy')
                    shutil.copy2(source_prefix+s_age_path + f_name + '_age.npy', d_age_path + f_name + '_age.npy')
                    shutil.copy2(source_prefix+s_gender_path + f_name + '_gender.npy', d_gender_path + f_name + '_gender.npy')
                    shutil.copy2(source_prefix+s_race_path + f_name + '_race.npy', d_race_path + f_name + '_race.npy')

                    line_count += 1

    def copy_final_images(self, cvs_file, s_img_folder, d_img_folder):
        #  open csv file, and read img. find and copy it.
        with open(cvs_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in tqdm(csv_reader):
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    img_name = row[0]
                    shutil.copy2(s_img_folder + img_name, d_img_folder + img_name)
                    line_count += 1

    def query_images(self, cvs_query_file, query, final_csv, number_of_samples, prefix):
        import random
        f_saver = open(final_csv, "a")
        with open(cvs_query_file) as csv_file:
            csv_reader = list(csv.reader(csv_file, delimiter=','))
            random.shuffle(csv_reader)
            for row in tqdm(csv_reader):
                save_or_not = []
                img_name = row[0]
                fer = row[1].split(' ')
                age = row[2].split(' ')
                race = row[3].split(' ')
                gender = row[4].split(' ')
                '''filtering '''
                if query['fer'] is not None:
                    task_ids = query['fer']
                    if float(np.argmax(fer)) in task_ids:  # and float(fer[np.argmax(fer)]) >= 0.4:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                if query['gender'] is not None:
                    task_ids = query['gender']
                    if float(np.argmax(gender)) in task_ids:  # and float(gender[np.argmax(gender)]) >= 0.9:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                if query['race'] is not None:
                    task_ids = query['race']
                    if float(np.argmax(race)) in task_ids:  # and float(race[np.argmax(race)]) >= 0.5:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                if query['age'] is not None:
                    task_ids = query['age']
                    if float(np.argmax(age)) in task_ids:  # and float(age[np.argmax(age)]) >= 0.5:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                '''save'''
                avg = np.mean(save_or_not)
                if avg >= 1:
                    f_saver.write(img_name + ',' +
                                  " ".join(x for x in fer) + ',' +
                                  " ".join(x for x in age) + ',' +
                                  " ".join(x for x in race) + ',' +
                                  " ".join(x for x in gender) + ',' +
                                  prefix+'\n')
                    number_of_samples -= 1
                    if number_of_samples == 0:
                        break
                ''''''
        f_saver.close()

    def create_histogram_csv_reduced(self, cvs_file, f_index, task, file_name):
        if task == 'age':
            histogram = np.zeros(shape=3, dtype=np.int)
        if task == 'race':
            histogram = np.zeros(shape=3, dtype=np.int)
        if task == 'gender':
            histogram = np.zeros(shape=2, dtype=np.int)
        if task == 'exp':
            histogram = np.zeros(shape=3, dtype=np.int)

        with open(cvs_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in tqdm(csv_reader):
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    fea_vec = row[f_index].split(' ')
                    feature = np.argmax(fea_vec)
                    if task == 'exp':
                        if feature == 6: feature = 2
                    if task == 'race':
                        if feature == 3: feature = 2
                        elif feature == 2: feature = 0
                        else: feature = 1

                    histogram[feature] += 1
                    line_count += 1
        '''plot'''
        fig, ax = plt.subplots()
        if task == 'exp':
            ax.bar(['Neutral', 'Happy', 'Angry'], histogram, color=['#0096c7', '#57cc99', '#f77f00'])
            plt.xlabel('Expression', fontweight='bold')
        if task == 'age':
            ax.bar(['0-15', '16-32', '33-53', '54-100'], histogram, color=['#0096c7', '#57cc99', '#f77f00', '#7209b7'])
            plt.xlabel('Age', fontweight='bold')
        if task == 'race':
            ax.bar(['Black', 'Brown', 'White'], histogram, color=['#0096c7', '#57cc99', '#f77f00'])
            plt.xlabel('Skin Color', fontweight='bold')
        if task == 'gender':
            ax.bar(['Female', 'Male'], histogram, color=['#0096c7', '#f77f00'])
            plt.xlabel('Gender', fontweight='bold')

        for i, v in enumerate(histogram):
            ax.text(i - .1, v + 50, str(v), color='#0d1b2a')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.savefig(file_name + '.png')

    def create_histogram_csv(self, cvs_file, f_index, task, file_name):
        if task == 'age':
            histogram = np.zeros(shape=4, dtype=np.int)
        if task == 'race':
            histogram = np.zeros(shape=6, dtype=np.int)
        if task == 'gender':
            histogram = np.zeros(shape=2, dtype=np.int)

        with open(cvs_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in tqdm(csv_reader):
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    fea_vec = row[f_index].split(' ')
                    feature = np.argmax(fea_vec)
                    histogram[feature] += 1
                    line_count += 1
        '''plot'''
        fig, ax = plt.subplots()
        if task == 'age':
            ax.bar(['0-15', '16-32', '33-53', '54-100'], histogram, color='#219ebc')
            plt.xlabel('Age', fontweight='bold')
        if task == 'race':
            ax.bar(['asian', 'indian', 'black', 'white', 'mid-east', 'latino'], histogram, color='#219ebc')
            plt.xlabel('Race', fontweight='bold')
        if task == 'gender':
            ax.bar(['Female', 'Male'], histogram, color='#219ebc')
            plt.xlabel('Gender', fontweight='bold')

        for i, v in enumerate(histogram):
            ax.text(i - .1, v + 3, str(v), color='#ff006e')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.savefig(file_name + '.png')

    def create_total_cvs_raw(self, img_path, fer_path, age_path, race_path, gender_path, cvs_file):
        f = open(cvs_file, "a")
        k_fem = 0
        k_male = 0
        FEMALE = 0
        MALE = 1
        for file in tqdm(os.listdir(img_path)):
            if file.endswith('.jpg'):
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

                # if np.argmax(fer) == 1:  # Happy
                if np.argmax(fer) == 6 or np.argmax(fer) == 0:  # Angry
                    f.write(file + ',' +
                            " ".join(str(x) for x in np.round(fer.tolist(), 3).tolist()) + ',' +
                            " ".join(str(x) for x in np.round(age.tolist(), 3).tolist()) + ',' +
                            " ".join(str(x) for x in np.round(race.tolist(), 3).tolist()) + ',' +
                            " ".join(str(x) for x in np.round(gender.tolist(), 3).tolist()) + '\n')
        print("DONE")

    def create_total_cvs(self, img_path, fer_path, age_path, race_path, gender_path, cvs_file):
        f = open(cvs_file, "a")
        k_fem = 0
        k_male = 0
        FEMALE = 0
        MALE = 1
        for file in tqdm(os.listdir(img_path)):
            if file.endswith('.jpg') and (k_fem < 5000 or k_male < 5000):
                # if file.endswith('.jpg'):
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

                if np.argmax(fer) == 6 or np.argmax(fer) == 0:  # Angry
                    # if np.argmax(fer) == 1:  # Happy
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

    def query_images_by_path(self, folder_names, query, num_samples):
        noise_arr = []
        noise_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/noise_vectors' for fn in folder_names]
        fer_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/feature_vectors' for fn in folder_names]
        race_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/race_extraction' for fn in folder_names]
        gender_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/gender_extraction' for fn in folder_names]
        age_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/age_extraction' for fn in folder_names]
        for p_index in range(len(noise_paths)):
            noise_path = noise_paths[p_index]
            fer_path = fer_paths[p_index]
            race_path = race_paths[p_index]
            gender_path = gender_paths[p_index]
            age_path = age_paths[p_index]
            i = 0
            item_list = shuffle(os.listdir(noise_path))
            for file in tqdm(item_list):
                if file.endswith(".npy"):
                    try:
                        bare_f = str(file).split('.')[0]
                        noise_f = os.path.join(noise_path, file)
                        noise = np.load(noise_f)[0]
                        fer_f = os.path.join(fer_path, 'exp_' + bare_f + '.npy')
                        fer = np.load(fer_f)

                        gender_f = os.path.join(gender_path, bare_f + '_gender.npy')
                        gender = np.load(gender_f)

                        race_f = os.path.join(race_path, bare_f + '_race.npy')
                        race = np.load(race_f)

                        age_f = os.path.join(age_path, bare_f + '_age.npy')
                        age = np.load(age_f)

                        save_or_not = []

                        if len(fer) != 7:
                            fer = fer[0]

                        if query['fer'] is not None:
                            task_ids = query['fer']
                            if np.argmax(fer) in task_ids and fer[np.argmax(fer)] >= 0.4:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)
                        if query['gender'] is not None:
                            task_ids = query['gender']
                            if np.argmax(gender) in task_ids and gender[np.argmax(gender)] >= 0.7:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)
                        if query['race'] is not None:
                            task_ids = query['race']
                            if np.argmax(race) in task_ids and race[np.argmax(race)] >= 0.4:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)
                        if query['age'] is not None:
                            task_ids = query['age']
                            if np.argmax(age) in task_ids and age[np.argmax(age)] >= 0.4:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)

                        avg = np.mean(save_or_not)
                        if avg >= 1:
                            noise_arr.append(noise)
                            i += 1
                    except Exception as e:
                        print(e)

                    if i > num_samples:
                        break
        return noise_arr


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
