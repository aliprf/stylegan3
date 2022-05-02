from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle
import os
from tqdm import tqdm
from numpy import save, load
import math
from PIL import Image
from numpy import save, load
from scipy.linalg import svd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

class ASM:
    _eigenvalues_prefix = "_eigenvalues_"
    _eigenvectors_prefix = "_eigenvectors_"
    _meanvector_prefix = "_meanvector_"

    _s_prefix = 's'
    _u_prefix = 'U'
    _vt_prefix = 'V_t'

    def create_lda(self, noise_path, task_0, task_id_0, anno_path_0, task_1, task_id_1, anno_path_1):
        noise_arr_0 = []
        noise_arr_1 = []

        i = 0
        j = 0
        for file in tqdm(os.listdir(noise_path)):
            if file.endswith(".npy"):
                bare_f = str(file).split('.')[0]
                noise_f = os.path.join(noise_path, file)
                noise = np.load(noise_f)[0]
                if task_0 == 'fer':
                    fer_f = os.path.join(anno_path_0, 'exp_' + bare_f + '.npy')
                    fer = np.load(fer_f)[0]
                    if np.argmax(fer) == task_id_0 and fer[task_id_0] >= 0.7:
                        noise_arr_0.append(noise)
                        i += 1
                if task_1 == 'race':
                    race_f = os.path.join(anno_path_1, bare_f + '_race.npy')
                    race = np.load(race_f)
                    if np.argmax(race) == task_id_1 and race[task_id_1] >= 0.7:
                        noise_arr_1.append(noise)
                        j += 1
                if i > 10 and j > 10:
                    break
        noise_arr_0 = np.array(noise_arr_0)
        noise_arr_1 = np.array(noise_arr_1)
        X = np.concatenate([noise_arr_0, noise_arr_1])
        Y = np.concatenate([np.zeros(len(noise_arr_0)), np.ones(len(noise_arr_1))])

        lda = LDA(store_covariance=True)
        lda.fit(X, Y)

        print('')
        return np.expand_dims(lda.means_[0], 0), \
               np.expand_dims(lda.means_[1], 0), \
               np.expand_dims(np.mean(noise_arr_0,0), 0) , \
               np.expand_dims(np.mean(noise_arr_1,0), 0)


    def create_pca_from_npy(self, task, noise_path, anno_path, task_id, pca_accuracy=99):
        print('PCA calculation started: loading labels')
        noise_arr = []
        i = 0
        for file in tqdm(os.listdir(noise_path)):
            if file.endswith(".npy"):
                bare_f = str(file).split('.')[0]
                noise_f = os.path.join(noise_path, file)
                noise = np.load(noise_f)[0]
                if task == 'fer':
                    fer_f = os.path.join(anno_path, 'exp_' + bare_f + '.npy')
                    fer = np.load(fer_f)[0]
                    if np.argmax(fer) == task_id and fer[task_id] >= 0.6:
                        xxx = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/gender_extraction'
                        xxx_f = os.path.join(xxx, bare_f + '_gender.npy')
                        sem = np.load(xxx_f)
                        if np.argmax(sem) == 0:
                            noise_arr.append(noise)
                            i += 1
                if task == 'gender':
                    gender_f = os.path.join(anno_path, bare_f + '_gender.npy')
                    gender = np.load(gender_f)
                    if np.argmax(gender) == task_id and gender[task_id] >= 0.95:
                        noise_arr.append(noise)
                        i += 1
                if task == 'race':
                    race_f = os.path.join(anno_path, bare_f + '_race.npy')
                    race = np.load(race_f)
                    if np.argmax(race) == task_id and race[task_id] >= 0.8:
                        noise_arr.append(noise)
                        i += 1
                if task == 'age':
                    age_f = os.path.join(anno_path, bare_f + '_age.npy')
                    age = np.load(age_f)
                    if np.argmax(age) == task_id and age[task_id] >= 0.9:
                        noise_arr.append(noise)
                        i += 1
                if i > 10000:
                    break

        ''' no normalization is needed, since we want to generate hm'''
        mean_lbl_arr = np.mean(noise_arr, axis=0)
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(noise_arr, pca_accuracy)
        eigenvectors = eigenvectors.T

        u, s, vh = np.linalg.svd(np.array(noise_arr).T, full_matrices=True)

        save('pca_obj/_' + task + str(task_id) + self._eigenvalues_prefix + str(pca_accuracy), eigenvalues)
        save('pca_obj/_' + task + str(task_id) + self._eigenvectors_prefix + str(pca_accuracy), eigenvectors)
        save('pca_obj/_' + task + str(task_id) + self._meanvector_prefix + str(pca_accuracy), mean_lbl_arr)

        save('pca_obj/_' + task + str(task_id) + self._u_prefix + str(pca_accuracy), u)
        save('pca_obj/_' + task + str(task_id) + self._s_prefix + str(pca_accuracy), s)
        save('pca_obj/_' + task + str(task_id) + self._vt_prefix + str(pca_accuracy), vh)

    def get_asm(self, task_id, pca_accuracy, num, task, alpha=1.0):
        noises = []
        for i in range(num):
            eigenvalues = load(
                'pca_obj/_' + task + str(task_id) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + task + str(task_id) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
            meanvector = load('pca_obj/_' + task + str(task_id) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]

            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            out = alpha * meanvector + np.dot(eigenvectors, b_vector_p)
            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out, 0))
        return noises

    def get_asm_multiple(self, task_ids, tasks, pca_accuracy, num, alpha=1.0):
        noises = []
        eigenvalues_0 = load(
            'pca_obj/_' + tasks[0] + str(task_ids[0]) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors_0 = load(
            'pca_obj/_' + tasks[0] + str(task_ids[0]) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector_0 = load(
            'pca_obj/_' + tasks[0] + str(task_ids[0]) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        eigenvalues_1 = load(
            'pca_obj/_' + tasks[1] + str(task_ids[1]) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors_1 = load(
            'pca_obj/_' + tasks[1] + str(task_ids[1]) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector_1 = load(
            'pca_obj/_' + tasks[1] + str(task_ids[1]) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        for i in range(num):
            beta_0 = 1.0
            beta_1 = 1.0
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p_0 = self._calculate_b_vector(sample, True, eigenvalues_0, eigenvectors_0, meanvector_0)
            b_vector_p_1 = self._calculate_b_vector(sample, True, eigenvalues_1, eigenvectors_1, meanvector_1)

            noises.append(np.expand_dims(sample, 0))

            out0 = beta_0 * meanvector_0 + beta_0 * np.dot(eigenvectors_0, b_vector_p_0)
            out1 = beta_1 * meanvector_1 + beta_1 * np.dot(eigenvectors_1, b_vector_p_1)

            mean_t = meanvector_0 + meanvector_1

            d = out0 - meanvector_0
            out00 = meanvector_0 - d

            noises.append(np.expand_dims(out0, 0))
            noises.append(np.expand_dims(out1, 0))
            # noises.append(np.expand_dims(out00, 0))

            # noises.append(np.expand_dims(meanvector_1 + out0, 0))
            # noises.append(np.expand_dims(meanvector_0 + out1, 0))

            # noises.append(np.expand_dims(mean_t - out0, 0))
            # noises.append(np.expand_dims(mean_t - out1, 0))

        return noises

    def get_asm_multiple_0(self, task_ids, tasks, pca_accuracy, num, alpha=1.0):
        noises = []
        eigenvalues_0 = load(
            'pca_obj/_' + tasks[0] + str(task_ids[0]) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors_0 = load(
            'pca_obj/_' + tasks[0] + str(task_ids[0]) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector_0 = load(
            'pca_obj/_' + tasks[0] + str(task_ids[0]) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        eigenvalues_1 = load(
            'pca_obj/_' + tasks[1] + str(task_ids[1]) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors_1 = load(
            'pca_obj/_' + tasks[1] + str(task_ids[1]) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector_1 = load(
            'pca_obj/_' + tasks[1] + str(task_ids[1]) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        for i in range(num):
            beta_0 = 1.0
            beta_1 = 1.0
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p_0 = self._calculate_b_vector(sample, True, eigenvalues_0, eigenvectors_0, meanvector_0)
            b_vector_p_1 = self._calculate_b_vector(sample, True, eigenvalues_1, eigenvectors_1, meanvector_1)

            w_0 = []
            w_1 = []
            for j in range(len(b_vector_p_0)):
                if j >= 0.5 * len(b_vector_p_0):  # identity
                    w_0.append(1.0)
                else:
                    w_0.append(0.0)
            for j in range(len(b_vector_p_1)):
                if j <= 0.25 * len(b_vector_p_1):  # semantic
                    w_1.append(1.0)
                else:
                    w_1.append(0.0)

            b_vector_p_new_0 = b_vector_p_0 * np.array(w_0)
            b_vector_p_new_1 = b_vector_p_1 * np.array(w_1)

            noises.append(np.expand_dims(sample, 0))

            out0 = beta_0 * meanvector_0 + beta_0 * np.dot(eigenvectors_0, b_vector_p_new_0)
            out1 = beta_1 * meanvector_1 + beta_1 * np.dot(eigenvectors_1, b_vector_p_new_1)

            out2 = beta_0 * meanvector_0 + beta_1 * meanvector_1 + \
                   beta_0 * np.dot(eigenvectors_0, b_vector_p_new_0) + beta_1 * np.dot(eigenvectors_1, b_vector_p_new_1)

            # out0 = beta_0 * meanvector_0 + beta_0 * np.dot(eigenvectors_0, b_vector_p_0)

            # b_vector_p_1 = self._calculate_b_vector(out0, True, eigenvalues_1, eigenvectors_1, meanvector_1)
            # out1 = beta_1 * meanvector_1 + beta_1 * np.dot(eigenvectors_1, b_vector_p_1)

            # for j in range(20):
            #     beta_1 += 0.1
            #     out = beta_0 * meanvector_0 + beta_1 * meanvector_1 + \
            #           beta_0 * np.dot(eigenvectors_0, b_vector_p_0) + beta_1 * np.dot(eigenvectors_1, b_vector_p_1)
            #
            #     # o_0 = np.dot(eigenvectors_1, (-0.5*b_vector_p_1))
            #     # noises.append(np.expand_dims(o_0, 0))

            noises.append(np.expand_dims(out0, 0))
            noises.append(np.expand_dims(out1, 0))
            noises.append(np.expand_dims(out2, 0))

        return noises

    def get_asm_b(self, task_id, pca_accuracy, num, task, alpha=1.0):
        noises = []
        eigenvalues = load('pca_obj/_' + task + str(task_id) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load('pca_obj/_' + task + str(task_id) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task + str(task_id) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            # b_vector_p_rand = np.random.normal(loc=np.mean(b_vector_p), scale=np.max(b_vector_p), size=b_vector_p.shape)

            out = alpha * meanvector + np.dot(eigenvectors, b_vector_p)
            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out, 0))

            for k in range(10):
                w = []
                w_p = []
                for j in range(len(b_vector_p)):
                    if j < k * 0.1 * len(b_vector_p):
                        w.append(1.0)
                        w_p.append(0.0)
                    else:
                        w.append(0.0)
                        w_p.append(1.0)
                out_w = alpha * meanvector + np.dot(eigenvectors, b_vector_p * np.array(w))
                # out_w_p = alpha * meanvector + np.dot(eigenvectors,  b_vector_p * np.array(w_p))
                out_w_n = 0.3 * sample + 0.7 * out_w
                # noises.append(np.expand_dims(out_w, 0))
                noises.append(np.expand_dims(out_w_n, 0))
                # noises.append(np.expand_dims(out_w_p, 0))
            # for k in range(10):
            #     b_vector_p_rand = np.random.normal(loc=np.mean(b_vector_p), scale=0.2, size=b_vector_p.shape)
            #     b_vector_p_new = b_vector_p * np.array(w) + b_vector_p_rand * np.array(w)
            #     out_new = alpha * meanvector + np.dot(eigenvectors, b_vector_p_new)
            #     noises.append(np.expand_dims(out_new, 0))
        return noises

    def get_asm_svd(self, task_id, pca_accuracy, num, task, alpha=1.0):
        noises = []
        eigenvalues = load('pca_obj/_' + task + str(task_id) + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load('pca_obj/_' + task + str(task_id) + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task + str(task_id) + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        u = load('pca_obj/_' + task + str(task_id) + self._u_prefix + str(pca_accuracy) + ".npy")
        s = load('pca_obj/_' + task + str(task_id) + self._s_prefix + str(pca_accuracy) + ".npy")
        vh = load('pca_obj/_' + task + str(task_id) + self._vt_prefix + str(pca_accuracy) + ".npy")

        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)

            # b_vector_p_0 = self._calculate_b_vector(sample, True, s[:10], u[:, :10], meanvector)
            # b_vector_p_rand = np.random.normal(loc=np.mean(b_vector_p), scale=np.max(b_vector_p), size=b_vector_p.shape)

            out = alpha * meanvector + np.dot(eigenvectors, b_vector_p)

            # out_0 = alpha * meanvector + np.dot(u[:, :10], b_vector_p_0)
            num_eigen = 500

            # out_0 = np.mean((u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :])[:, :3], -1)

            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out, 0))
            # noises.append(np.expand_dims(meanvector, 0))
            noises.append(np.expand_dims(meanvector-out, 0))
            # noises.append(np.expand_dims(out-meanvector, 0))

            # out_0 = (u[:, :num_eigen] @ np.diag(s)[:num_eigen, :num_eigen] @ vh[:num_eigen, :])
            # for k in range(200):
            #     o_0 = out_0[:,k]
            #     out_1 = 0.6 * sample + 0.4 * o_0
            #     noises.append(np.expand_dims(o_0, 0))
            #     noises.append(np.expand_dims(out_1, 0))



            # # SVD
            # n_fact = 1
            # w = []
            # for k in range(1):
            #     for j in range(len(b_vector_p)):
            #         if j > 0.90 * len(b_vector_p):
            #             w.append(1.0)
            #         else:
            #             w.append(0.0)
            #     out_n = np.dot(np.dot(u[:, :n_fact], np.diag(s)[:n_fact, :n_fact]), vh[:n_fact, :])
            #     # out_n *= w
            #     # out_w = alpha * meanvector + np.dot(eigenvectors_red, b_vector_p)
            #     noises.append(np.expand_dims(out_n, 0))
            #     # n_fact = int(n_fact - 0.01 * n_fact)

        return noises

    def _func_PCA(self, input_data, pca_postfix):
        input_data = np.array(input_data)
        n_components = pca_postfix / 100
        if n_components >= 1:
            n_components = int(n_components)
        pca = PCA(n_components=n_components)
        pca.fit(input_data)
        pca_input_data = pca.transform(input_data)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        return pca_input_data, eigenvalues, eigenvectors

    def _calculate_b_vector(self, sample, correction, eigenvalues, eigenvectors, meanvector):
        tmp1 = sample - meanvector
        # tmp1 = meanvector - sample
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
