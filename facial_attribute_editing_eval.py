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
from config import FolderStructures, PreTrainedModels, Age_codes, Race_codes, Expression_codes, Gender_codes


class FacialAttributeEditingEval:
    _eigenvalues_prefix = "_eigenvalues_"
    _eigenvectors_prefix = "_eigenvectors_"
    _meanvector_prefix = "_meanvector_"

    def __init__(self, source_image_path=None, source_noise_path=None, source_expression_path=None,
                 source_age_path=None,
                 source_gender_path=None, source_race_path=None,
                 target_image_path=None, target_noise_path=None, target_expression_path=None,
                 target_age_path=None, target_gender_path=None, target_race_path=None):

        self._source_image_path = source_image_path
        self._source_noise_path = source_noise_path
        self._source_expression_path = source_expression_path
        self._source_age_path = source_age_path
        self._source_gender_path = source_gender_path
        self._source_race_path = source_race_path

        self._target_image_path = target_image_path
        self._target_noise_path = target_noise_path
        self._target_expression_path = target_expression_path
        self._target_age_path = target_age_path
        self._target_gender_path = target_gender_path
        self._target_race_path = target_race_path
        if self._target_image_path is not None:
            if not os.path.exists(self._target_image_path): os.makedirs(self._target_image_path)
            if not os.path.exists(self._target_noise_path): os.makedirs(self._target_noise_path)
            if not os.path.exists(self._target_expression_path): os.makedirs(self._target_expression_path)
            if not os.path.exists(self._target_age_path): os.makedirs(self._target_age_path)
            if not os.path.exists(self._target_gender_path): os.makedirs(self._target_gender_path)
            if not os.path.exists(self._target_race_path): os.makedirs(self._target_race_path)
    def create_all_features_histogram(self):
        pass

    def calculate_semantic_score(self, task, s_path, d_path):
        if task == 'FEMALE' or task == 'MALE':
            s_path += 'gender_extraction'
            d_path += 'gender_extraction'
        if task == 'ANGRY' or task == 'NEUTRAL':
            s_path += 'feature_vectors'
            d_path += 'feature_vectors'
        if task == 'BLACK':
            s_path += 'race_extraction'
            d_path += 'race_extraction'
        if task == 'OLD':
            s_path += 'age_extraction'
            d_path += 'age_extraction'
        correct = 0
        all_samples = 0
        to_female = {
            'all_to_female': 0,
            'male_to_female': 0,
            'all': 0,
            'male': 0,
        }
        to_black = {
            'all_to_black': 0,
            'all_to_brown': 0,
            'white_to_black': 0,
            'brown_to_black': 0,
            'white_to_brown': 0,
            'all': 0,
            'white': 0,
            'brown': 0
        }
        to_angry_neutral = {
            'all_to_angry_neutral': 0,
            'all_to_angry': 0,
            'all_to_neutral': 0,
            'happy_to_angry_neutral': 0,
            'happy_to_angry': 0,
            'happy_to_neutral': 0,
            'neutral_to_anger': 0,
            'all': 0,
            'happy': 0,
            'neutral': 0,
            'to_others': 0
        }
        to_old = {
            'all_to_old': 0,
            'ch_to_older': 0,
            'yo_to_older': 0,
            'mid_to_older': 0,
            'all': 0,
            'ch': 0,
            'yo': 0,
            'mid': 0,
            'old': 0
        }

        for file in tqdm(os.listdir(s_path)):
            s_ftr = np.load(os.path.join(s_path, file), allow_pickle=True)
            d_ftr = np.load(os.path.join(d_path, file), allow_pickle=True)
            if task == 'OLD':
                if np.argmax(s_ftr) == Age_codes.CHILD:
                    to_old['ch'] += 1
                if np.argmax(s_ftr) == Age_codes.YOUTH:
                    to_old['yo'] += 1
                if np.argmax(s_ftr) == Age_codes.MIDDLE:
                    to_old['mid'] += 1
                if np.argmax(s_ftr) == Age_codes.OLD:
                    to_old['old'] += 1
                if np.argmax(s_ftr) == Age_codes.CHILD and \
                        (np.argmax(d_ftr) == Age_codes.YOUTH or
                         np.argmax(d_ftr) == Age_codes.MIDDLE or
                         np.argmax(d_ftr) == Age_codes.OLD):
                    to_old['ch_to_older'] += 1
                if np.argmax(s_ftr) == Age_codes.YOUTH and \
                        (np.argmax(d_ftr) == Age_codes.MIDDLE or
                         np.argmax(d_ftr) == Age_codes.OLD):
                    to_old['yo_to_older'] += 1
                if np.argmax(s_ftr) == Age_codes.MIDDLE and np.argmax(d_ftr) == Age_codes.OLD:
                    to_old['mid_to_older'] += 1
                to_old['all'] += 1

            if task == 'FEMALE':
                if np.argmax(d_ftr) == Gender_codes.FEMALE:
                    to_female['all_to_female'] += 1
                if np.argmax(s_ftr) == Gender_codes.MALE:
                    to_female['male'] += 1
                    if np.argmax(d_ftr) == Gender_codes.FEMALE:
                        to_female['male_to_female'] += 1
                to_female['all'] += 1

            elif task == 'BLACK':
                if np.argmax(d_ftr) == Race_codes.BLACK:
                    to_black['all_to_black'] += 1
                if np.argmax(d_ftr) == Race_codes.INDIAN or np.argmax(d_ftr) == Race_codes.MID_EST:
                    to_black['all_to_brown'] += 1
                if np.argmax(s_ftr) == Race_codes.WHITE:
                    to_black['white'] += 1
                    if np.argmax(d_ftr) == Race_codes.BLACK:
                        to_black['white_to_black'] += 1
                    if np.argmax(d_ftr) == Race_codes.INDIAN or np.argmax(d_ftr) == Race_codes.MID_EST:
                        to_black['white_to_brown'] += 1
                if np.argmax(s_ftr) == Race_codes.INDIAN or np.argmax(s_ftr) == Race_codes.MID_EST:
                    to_black['brown'] += 1
                    if np.argmax(d_ftr) == Race_codes.BLACK:
                        to_black['brown_to_black'] += 1
                to_black['all'] += 1

            elif task == 'ANGRY':
                if not (np.argmax(d_ftr) == Expression_codes.ANGER or np.argmax(d_ftr) == Expression_codes.NEUTRAL):
                    to_angry_neutral['to_others'] += 1
                if np.argmax(d_ftr) == Expression_codes.ANGER or np.argmax(d_ftr) == Expression_codes.NEUTRAL:
                    to_angry_neutral['all_to_angry_neutral'] += 1
                if np.argmax(d_ftr) == Expression_codes.ANGER:
                    to_angry_neutral['all_to_angry'] += 1
                if np.argmax(d_ftr) == Expression_codes.NEUTRAL:
                    to_angry_neutral['all_to_neutral'] += 1
                if np.argmax(s_ftr) == Expression_codes.HAPPY:
                    to_angry_neutral['happy'] += 1
                    if np.argmax(d_ftr) == Expression_codes.ANGER:
                        to_angry_neutral['happy_to_angry'] += 1
                    if np.argmax(d_ftr) == Expression_codes.NEUTRAL:
                        to_angry_neutral['happy_to_neutral'] += 1
                    if np.argmax(d_ftr) == Expression_codes.ANGER or np.argmax(d_ftr) == Expression_codes.NEUTRAL:
                        to_angry_neutral['happy_to_angry_neutral'] += 1
                if np.argmax(s_ftr) == Expression_codes.NEUTRAL:
                    to_angry_neutral['neutral'] += 1
                    if np.argmax(d_ftr) == Expression_codes.ANGER:
                        to_angry_neutral['neutral_to_anger'] += 1
                to_angry_neutral['all'] += 1

        return to_female, to_angry_neutral, to_black, to_old

    def get_modified_noises(self, noise_path):
        modified_noises = []
        names = []
        for file in tqdm(os.listdir(noise_path)):
            latent_vector = load(noise_path + file)
            modified_noises.append(latent_vector)
            names.append(file.split('.')[0])
        return modified_noises, names

    def modify_noise(self, task_name, s_p, i_p, alpha, pca_accuracy):
        eigenvalues = load(
            'pca_obj/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        for file in tqdm(os.listdir(self._source_image_path)):
            base_name = file.split('.')[0]
            noise_f = base_name + '.npy'
            latent_vector = load(self._source_noise_path + noise_f)
            _, latent_vector_id = self._make_single_semantic_noise(latent_vector=latent_vector[-1, :],
                                                                   s_p=s_p,
                                                                   i_p=i_p,
                                                                   alpha=alpha,
                                                                   eigenvalues=eigenvalues,
                                                                   eigenvectors=eigenvectors,
                                                                   meanvector=meanvector)
            np.save(self._target_noise_path + noise_f, latent_vector_id)

    def _make_single_semantic_noise(self, latent_vector, s_p, i_p, alpha, eigenvalues, eigenvectors, meanvector):
        k_seg = int(s_p * len(eigenvalues))
        k_id = int(i_p * len(eigenvalues))

        eigenvectors_sem = eigenvectors[:, :k_seg]
        eigenvalues_sem = eigenvalues[:k_seg]

        eigenvectors_id = eigenvectors[:, k_id:]
        eigenvalues_id = eigenvalues[k_id:]

        b_vector_p_sem = self._calculate_b_vector(latent_vector, True, eigenvalues_sem, eigenvectors_sem, meanvector)
        b_vector_p_id = self._calculate_b_vector(latent_vector, True, eigenvalues_id, eigenvectors_id, meanvector)

        vec_sem = np.expand_dims((alpha * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem)), 0)
        vec_id = np.expand_dims((alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id)), 0)
        return vec_sem, vec_id

    def _calculate_b_vector(self, sample, correction, eigenvalues, eigenvectors, meanvector):
        tmp1 = sample - meanvector
        b_vector = np.dot(eigenvectors.T, tmp1)
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    b_item = max(b_item, -1 * lambda_i_sqr)
                b_vector[i] = b_item
                i += 1

        return b_vector
