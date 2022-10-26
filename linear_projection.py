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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from scipy.fft import fft, fftfreq, dct, dctn
from numpy.fft import *
from sklearn.utils import shuffle
import numpy as np
import dask.array as da
from dask_ml.decomposition import PCA as DPCA


class LinearProjection:
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
               np.expand_dims(np.mean(noise_arr_0, 0), 0), \
               np.expand_dims(np.mean(noise_arr_1, 0), 0)

    def create_pca_from_npy_old(self, tasks, noise_path, fer_path, race_path, gender_path, age_path, name,
                                pca_accuracy=99):
        print('PCA calculation started: loading labels')
        noise_arr = []
        i = 0
        for file in tqdm(os.listdir(noise_path)):
            if file.endswith(".npy"):
                bare_f = str(file).split('.')[0]
                noise_f = os.path.join(noise_path, file)
                noise = np.load(noise_f)[0]
                noise = np.clip(noise, -5.0, +5.0)

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

                if tasks['fer'] is not None:
                    task_ids = tasks['fer']
                    if np.argmax(fer) in task_ids and fer[np.argmax(fer)] >= 0.4:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                if tasks['gender'] is not None:
                    task_ids = tasks['gender']
                    if np.argmax(gender) in task_ids and gender[np.argmax(gender)] >= 0.9:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                if tasks['race'] is not None:
                    task_ids = tasks['race']
                    if np.argmax(race) in task_ids and race[np.argmax(race)] >= 0.5:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)
                if tasks['age'] is not None:
                    task_ids = tasks['age']
                    if np.argmax(age) in task_ids and age[np.argmax(age)] >= 0.5:
                        save_or_not.append(1)
                    else:
                        save_or_not.append(0)

                avg = np.mean(save_or_not)
                if avg >= 1:
                    noise_arr.append(noise)
                    i += 1
                #
                # if i > 100:
                #     break

        '''svd'''
        noise_arr = np.array(noise_arr)
        u, s, vh = np.linalg.svd(noise_arr.T, full_matrices=True)

        '''PCA via SVD:'''
        # B = noise_arr - mean_lbl_arr
        # U, S, V = np.linalg.svd(B.T/ np.sqrt(len(noise_arr)), full_matrices=True)
        save('pca_obj/_' + name + self._u_prefix + str(pca_accuracy), u)
        save('pca_obj/_' + name + self._s_prefix + str(pca_accuracy), s)
        save('pca_obj/_' + name + self._vt_prefix + str(pca_accuracy), vh)

        ''' no normalization is needed, since we want to generate hm'''
        print(len(noise_arr))
        mean_lbl_arr = np.mean(noise_arr, axis=0)
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(noise_arr, pca_accuracy)
        eigenvectors = eigenvectors.T
        save('pca_obj/_' + name + self._eigenvalues_prefix + str(pca_accuracy), eigenvalues)
        save('pca_obj/_' + name + self._eigenvectors_prefix + str(pca_accuracy), eigenvectors)
        save('pca_obj/_' + name + self._meanvector_prefix + str(pca_accuracy), mean_lbl_arr)

    def create_pca_from_npy(self, tasks, noise_paths, fer_paths, race_paths, gender_paths, age_paths,
                            name, pca_accuracy=99, sample_limit=20000, save_dir='./pca_obj/_'):
        print('PCA loading labels => ' + name)
        noise_arr = []

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

                        if tasks['fer'] is not None:
                            task_ids = tasks['fer']
                            if np.argmax(fer) in task_ids and fer[np.argmax(fer)] >= 0.4:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)
                        if tasks['gender'] is not None:
                            task_ids = tasks['gender']
                            if np.argmax(gender) in task_ids and gender[np.argmax(gender)] >= 0.9:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)
                        if tasks['race'] is not None:
                            task_ids = tasks['race']
                            if np.argmax(race) in task_ids and race[np.argmax(race)] >= 0.5:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)
                        if tasks['age'] is not None:
                            task_ids = tasks['age']
                            if np.argmax(age) in task_ids and age[np.argmax(age)] >= 0.5:
                                save_or_not.append(1)
                            else:
                                save_or_not.append(0)

                        avg = np.mean(save_or_not)
                        if avg >= 1:
                            noise_arr.append(noise)
                            i += 1
                    except Exception as e:
                        print(e)

                    if i > sample_limit:
                        break
        '''svd'''
        # noise_arr = np.array(noise_arr)
        # u, s, vh = np.linalg.svd(noise_arr.T, full_matrices=True)
        # save('pca_obj_fft/_' + name + self._u_prefix + str(pca_accuracy), u)
        # save('pca_obj_fft/_' + name + self._s_prefix + str(pca_accuracy), s)
        # save('pca_obj_fft/_' + name + self._vt_prefix + str(pca_accuracy), vh)

        ''' no normalization is needed, since we want to generate hm'''
        print('PCA calculation => ' + name)
        print(len(noise_arr))
        mean_lbl_arr = np.mean(noise_arr, axis=0)
        # reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(noise_arr, pca_accuracy)
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA_dask(noise_arr, pca_accuracy)
        eigenvectors = eigenvectors.T
        save(save_dir + name + self._eigenvalues_prefix + str(pca_accuracy), eigenvalues)
        save(save_dir + name + self._eigenvectors_prefix + str(pca_accuracy), eigenvectors)
        save(save_dir + name + self._meanvector_prefix + str(pca_accuracy), mean_lbl_arr)
        mean_lbl_arr = None
        reduced_lbl_arr = None
        eigenvalues = None
        eigenvectors = None

    def create_pca_from_list(self, noise_arr, name, save_dir='./pca_obj/_', pca_accuracy=99):
        print('create_pca_from_list => ' + name)
        print(len(noise_arr))
        mean_lbl_arr = np.mean(noise_arr, axis=0)
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA_dask(noise_arr)
        eigenvectors = eigenvectors.T
        save(save_dir + name + self._eigenvalues_prefix + str(pca_accuracy), eigenvalues)
        save(save_dir + name + self._eigenvectors_prefix + str(pca_accuracy), eigenvectors)
        save(save_dir + name + self._meanvector_prefix + str(pca_accuracy), mean_lbl_arr)
        mean_lbl_arr = None
        reduced_lbl_arr = None
        eigenvalues = None
        eigenvectors = None

    def _modify_weight(self, x, index=1):
        """"""
        f_x = x
        '''flip over the mean'''
        # avg = np.mean(x, 0)
        # f_x = avg - x
        '''-random--'''
        # w = np.round(np.random.randn(1, 512), decimals=3)[0]
        '''-sin--'''
        # w = 1.0 * np.array([np.sin(i * 1 / (np.shape(x)[0] / 1.0*np.pi)) for i in range(0, np.shape(x)[0])])
        # f_x = np.array([w.T * x[:, i] for i in range(0, np.shape(x)[1])]).T
        '''-cos--'''
        # w = 2.0*np.array([np.cos(i * (1 / (np.shape(x)[0] / (1.0*2.0 * np.pi)))) for i in range(0, np.shape(x)[0])])
        # if len(np.shape(x)) == 1:
        #     f_x = w.T * x
        # else:
        #     f_x = np.array([w.T * x[:, i] for i in range(0, np.shape(x)[1])]).T
        '''--discrete fourier transform--'''
        # SAMPLE_RATE = 1
        # DURATION = 512
        # N = SAMPLE_RATE * DURATION
        #
        # f_x = fft(f_x)
        # yf = fft(f_x)
        # f_x = 10.1 * dctn(f_x)
        # xf = range(0,512)
        #
        # # plt.plot(xf, np.abs(yf[:, :4]))
        #
        # # plt.plot(xf, yf[:, :2])
        # # plt.plot(xf, f_x[:, :2])
        # # plt.show()

        #
        # f_x = self._f_denoise(f_x, index)
        # f_x = self.filter_signal(f_x, 10000, index)
        f_x = self._binarize_sig(f_x, index)
        # f_x = self.filter_signal(f_x, 10000, index)

        # f_x = yf
        ''''''
        return f_x

    def _binarize_sig(self, sigal, index):
        f_sig = []
        for element in sigal:
            if element > 0:
                f_sig.append(.5)
            elif element < 0:
                f_sig.append(-.5)
            else:
                f_sig.append(0)
        return 10 * np.array(f_sig)

    def filter_signal(self, signal, threshold, index, do_save):
        fourier = rfft(signal)
        frequencies = rfftfreq(signal.size, d=20e-3 / signal.size)
        fourier[frequencies > threshold] = 0
        filtered = irfft(fourier)
        if do_save:
            plt.figure(figsize=(15, 10))
            plt.plot(signal, label='Raw')
            plt.plot(filtered, label='Filtered')
            plt.legend()
            plt.title("FFT Denoising with threshold = " + str(threshold), size=15)
            plt.savefig('sig_img/' + str(index) + 'sm.png', bbox_inches='tight', dpi=80)
        return filtered

    def _f_denoise(self, signal, index):
        dt = 0.001
        t = np.arange(0, 1, dt)
        if len(np.shape(signal)) == 2:
            n = np.shape(signal)[1]  # len(t)
        else:
            n = np.shape(signal)[0]  # len(t)
        fhat = np.fft.fft(signal, n)  # computes the fft
        psd = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)  # frequency array
        idxs_half = np.arange(1, np.floor(n / 2), dtype=np.int32)  # first half index

        minsignal, maxsignal = signal.min(), signal.max()

        fhat = np.fft.fft(signal, n)  # computes the fft
        psd = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)  # frequency array
        idxs_half = np.arange(1, np.floor(n / 2), dtype=np.int32)  # first half index

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(freq[idxs_half], np.abs(psd[idxs_half]), color='b', lw=1.0, label='PSD noisy')
        ax[0].set_xlabel('Frequencies in Hz')
        ax[0].set_ylabel('Amplitude')
        ''''''
        threshold = .3 * psd.max()
        # psd_idxs = [True if (x < threshold and x > .3 * psd.max()) else False for x in psd ]
        psd_idxs = psd > threshold
        psd_clean = psd * psd_idxs  # zero out all the unnecessary powers
        fhat_clean = psd_idxs * fhat  # used to retrieve the signal
        signal_filtered = -1 * np.fft.ifft(fhat_clean)  # inverse fourier transform

        ax[1].plot(freq[idxs_half], np.abs(psd_clean[idxs_half]), color='r', lw=1, label='PSD clean')
        ax[1].set_xlabel('Frequencies in Hz')
        ax[1].set_ylabel('Amplitude')
        plt.savefig('sig_img/' + str(index) + 'signal-main.png', bbox_inches='tight', dpi=80)

        return signal_filtered

    def _component_svd_weight(self, x):
        f_x = x
        u, s, vh = np.linalg.svd(x, full_matrices=True)
        f_x = np.sum(
            np.array([s[i] * np.dot(np.expand_dims(u[i], 1), np.expand_dims(vh[i], 0)) for i in range(50, 400)]),
            0)
        # u, s, vh = np.linalg.svd(x.T, full_matrices=True)
        # f_x = np.sum(np.array([s[i] * np.dot(np.expand_dims(u[i], 1), np.expand_dims(vh[i], 0)) for i in range(10,49)]), 0).T
        return f_x

    def make_single_semantic_noise_fou(self, task_name, pca_accuracy, num, vec_percent_sem, vec_percent_id, alpha=1.0):
        noises = []
        eigenvalues = load(
            'pca_obj/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")
        #
        # u = load('pca_obj/_' + task_name + self._u_prefix + str(pca_accuracy) + ".npy")
        # s = load('pca_obj/_' + task_name + self._s_prefix + str(pca_accuracy) + ".npy")
        # vh = load('pca_obj/_' + task_name + self._vt_prefix + str(pca_accuracy) + ".npy")
        ''''''
        k_sem = int(vec_percent_sem * len(eigenvalues))
        k_id = int(len(eigenvalues) - vec_percent_id * len(eigenvalues))

        eigenvalues_sem = eigenvalues[:k_sem]
        eigenvectors_sem = eigenvectors[:, :k_sem]

        eigenvalues_id = eigenvalues[k_id:]
        eigenvectors_id = eigenvectors[:, k_id:]

        # eigenvectors_sem_soft = softmax(eigenvectors_sem, axis=0)

        # eigenvectors_sem_fx = self._modify_weight(eigenvectors)
        # eigenvalues_sem_fx = eigenvalues

        eigenvectors_sem_fx = eigenvectors
        eigenvalues_sem_fx = eigenvalues

        # eigenvalues_sem_fx = eigenvalues_sem  # abs(self._modify_weight(eigenvalues_sem))

        # eigenvectors_sem_fx = self._component_svd_weight(self._modify_weight(eigenvectors_sem))

        for i in range(num):
            sample = np.round(np.random.RandomState(i).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            b_vector_p_sem = self._calculate_b_vector(sample, True, eigenvalues_sem, eigenvectors_sem, meanvector)
            b_vector_p_id = self._calculate_b_vector(sample, True, eigenvalues_id, eigenvectors_id, meanvector)

            b_vector_p_sem_fx = self._calculate_b_vector(sample, False, eigenvalues_sem_fx, eigenvectors_sem_fx,
                                                         meanvector)

            out = alpha * meanvector + np.dot(eigenvectors, b_vector_p)
            out_sem = (alpha * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem))
            out_id = alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id)

            out_sem_fx = (alpha * meanvector + np.dot(eigenvectors_sem_fx, b_vector_p_sem_fx))

            # sample = self._modify_weight(sample, index=i)
            out_sem_fx = self._modify_weight(out_sem, index=1000 + i)

            noises.append(np.expand_dims(sample, 0))
            # noises.append(np.expand_dims(out, 0))
            noises.append(np.expand_dims(out_sem, 0))
            noises.append(np.expand_dims(out_sem_fx, 0))
            # noises.append(np.expand_dims(out_id, 0))

            # noises.append(np.expand_dims(1.0 * out_sem_fx + 1.0 * out_id, 0))

            # '''SVD'''
            # u, s, vh = np.linalg.svd(np.array(np.expand_dims(sample, 0)).T, full_matrices=True)
            # noises.append(np.expand_dims(u.diagonal(), 0))

            # for j in range(5):
            #     out_svd = u[j:j+1, :]
            #     noises.append(out_svd)

            # # out_svd = np.mean((u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :])[:, :3], -1)
            # # out_svd = u[:, 6:7] @ np.diag(s)[:1, :1]
            # out_svd = u[:, :1] @ eigenvalues[:1]
            # noises.append(np.expand_dims(out_svd, 0))
            # # for j in range(k):
            # #     out_svd = u[:, j:j+1] @ eigenvalues[j:j+1]
            # #     noises.append(np.expand_dims(out_svd, 0))
        return noises

    def make_single_semantic_noise_n(self, task_name, pca_accuracy, num, vec_percent_sem, vec_percent_id, alpha=1.0):
        noises = []
        eigenvalues = load(
            'pca_obj/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")
        #
        # u = load('pca_obj/_' + task_name + self._u_prefix + str(pca_accuracy) + ".npy")
        # s = load('pca_obj/_' + task_name + self._s_prefix + str(pca_accuracy) + ".npy")
        # vh = load('pca_obj/_' + task_name + self._vt_prefix + str(pca_accuracy) + ".npy")
        ''''''
        k_sem = int(vec_percent_sem * len(eigenvalues))
        k_id = int(len(eigenvalues) - vec_percent_id * len(eigenvalues))

        eigenvalues_sem = eigenvalues[:k_sem]
        eigenvectors_sem = eigenvectors[:, :k_sem]

        eigenvalues_id = eigenvalues[k_id:]
        eigenvectors_id = eigenvectors[:, k_id:]

        # eigenvectors_sem_soft = softmax(eigenvectors_sem, axis=0)

        eigenvectors_sem_fx = self._modify_weight(eigenvectors_sem)
        eigenvalues_sem_fx = eigenvalues_sem  # abs(self._modify_weight(eigenvalues_sem))

        # eigenvectors_sem_fx = self._component_svd_weight(self._modify_weight(eigenvectors_sem))

        for i in range(num):
            sample = np.round(np.random.RandomState(i).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            b_vector_p_sem = self._calculate_b_vector(sample, True, eigenvalues_sem, eigenvectors_sem, meanvector)
            b_vector_p_id = self._calculate_b_vector(sample, True, eigenvalues_id, eigenvectors_id, meanvector)

            b_vector_p_sem_fx = self._calculate_b_vector(sample, False, eigenvalues_sem_fx, eigenvectors_sem_fx,
                                                         meanvector)

            out = alpha * meanvector + np.dot(eigenvectors, b_vector_p)
            out_sem = (alpha * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem))
            out_id = alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id)

            out_sem_fx = (alpha * meanvector + np.dot(eigenvectors_sem_fx, b_vector_p_sem_fx))

            # noises.append(np.expand_dims(sample, 0))
            # noises.append(np.expand_dims(out, 0))
            # noises.append(np.expand_dims(out_sem, 0))
            noises.append(np.expand_dims(out_sem_fx, 0))
            # noises.append(np.expand_dims(out_id, 0))

            # noises.append(np.expand_dims(1.0 * out_sem_fx + 1.0 * out_id, 0))

            # '''SVD'''
            # u, s, vh = np.linalg.svd(np.array(np.expand_dims(sample, 0)).T, full_matrices=True)
            # noises.append(np.expand_dims(u.diagonal(), 0))

            # for j in range(5):
            #     out_svd = u[j:j+1, :]
            #     noises.append(out_svd)

            # # out_svd = np.mean((u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :])[:, :3], -1)
            # # out_svd = u[:, 6:7] @ np.diag(s)[:1, :1]
            # out_svd = u[:, :1] @ eigenvalues[:1]
            # noises.append(np.expand_dims(out_svd, 0))
            # # for j in range(k):
            # #     out_svd = u[:, j:j+1] @ eigenvalues[j:j+1]
            # #     noises.append(np.expand_dims(out_svd, 0))
        return noises

    def make_single_semantic_noise_disentangle(self, task_name, pca_accuracy, num, s_p, i_p, alpha=1.0):
        noises = []
        eigenvalues = load(
            'pca_obj/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        #
        # u = load('pca_obj/_' + task_name + self._u_prefix + str(pca_accuracy) + ".npy")
        # s = load('pca_obj/_' + task_name + self._s_prefix + str(pca_accuracy) + ".npy")
        # vh = load('pca_obj/_' + task_name + self._vt_prefix + str(pca_accuracy) + ".npy")
        ''''''
        k_seg = int(s_p * len(eigenvalues))
        k_id = int(i_p * len(eigenvalues))

        eigenvectors_sem = eigenvectors[:, :k_seg]
        eigenvalues_sem = eigenvalues[:k_seg]

        eigenvectors_id = eigenvectors[:, k_id:]
        eigenvalues_id = eigenvalues[k_id:]

        meanvector_m = load('pca_obj/_' + 'MALE' + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        for i in range(num):
            sample = np.round(np.random.RandomState(i).randn(1, 512), decimals=3)[0]
            b_vector_p_sem = self._calculate_b_vector(sample, True, eigenvalues_sem, eigenvectors_sem, meanvector)
            b_vector_p_id = self._calculate_b_vector(sample, True, eigenvalues_id, eigenvectors_id, meanvector)

            out_sem = (alpha * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem))
            out_id = alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id)
            out_id_dis = alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id) - 0.8 * meanvector_m

            noises.append(np.expand_dims(sample, 0))
            # noises.append(np.expand_dims(out_sem, 0))
            noises.append(np.expand_dims(out_id, 0))
            noises.append(np.expand_dims(out_id_dis, 0))

            # '''SVD'''
            # u, s, vh = np.linalg.svd(np.array(np.expand_dims(sample, 0)).T, full_matrices=True)
            # noises.append(np.expand_dims(u.diagonal(), 0))

            # for j in range(5):
            #     out_svd = u[j:j+1, :]
            #     noises.append(out_svd)

            # # out_svd = np.mean((u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :])[:, :3], -1)
            # # out_svd = u[:, 6:7] @ np.diag(s)[:1, :1]
            # out_svd = u[:, :1] @ eigenvalues[:1]
            # noises.append(np.expand_dims(out_svd, 0))
            # # for j in range(k):
            # #     out_svd = u[:, j:j+1] @ eigenvalues[j:j+1]
            # #     noises.append(np.expand_dims(out_svd, 0))
        return noises

    def make_single_semantic_noise(self, task_name, pca_accuracy, num, s_p, i_p, alpha=1.0):
        noises = []
        eigenvalues = load(
            'pca_obj/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        #
        # u = load('pca_obj/_' + task_name + self._u_prefix + str(pca_accuracy) + ".npy")
        # s = load('pca_obj/_' + task_name + self._s_prefix + str(pca_accuracy) + ".npy")
        # vh = load('pca_obj/_' + task_name + self._vt_prefix + str(pca_accuracy) + ".npy")
        ''''''
        k_seg = int(s_p * len(eigenvalues))
        k_id = int(i_p * len(eigenvalues))

        eigenvectors_sem = eigenvectors[:, :k_seg]
        eigenvalues_sem = eigenvalues[:k_seg]

        eigenvectors_id = eigenvectors[:, k_id:]
        eigenvalues_id = eigenvalues[k_id:]
        for i in range(num):
            sample = np.round(np.random.RandomState(i).randn(1, 512), decimals=3)[0]
            b_vector_p_sem = self._calculate_b_vector(sample, True, eigenvalues_sem, eigenvectors_sem, meanvector)
            b_vector_p_id = self._calculate_b_vector(sample, True, eigenvalues_id, eigenvectors_id, meanvector)

            out_sem = (alpha * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem))
            # out_sem = -(out_sem / np.linalg.norm(out_sem))
            out_id = alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id)

            noises.append(np.expand_dims(sample, 0))
            # noises.append(np.expand_dims(meanvector, 0))
            # noises.append(np.expand_dims(out_sem, 0))
            noises.append(np.expand_dims(out_id, 0))

            # '''SVD'''
            # u, s, vh = np.linalg.svd(np.array(np.expand_dims(sample, 0)).T, full_matrices=True)
            # noises.append(np.expand_dims(u.diagonal(), 0))

            # for j in range(5):
            #     out_svd = u[j:j+1, :]
            #     noises.append(out_svd)

            # # out_svd = np.mean((u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :])[:, :3], -1)
            # # out_svd = u[:, 6:7] @ np.diag(s)[:1, :1]
            # out_svd = u[:, :1] @ eigenvalues[:1]
            # noises.append(np.expand_dims(out_svd, 0))
            # # for j in range(k):
            # #     out_svd = u[:, j:j+1] @ eigenvalues[j:j+1]
            # #     noises.append(np.expand_dims(out_svd, 0))
        return noises

    def make_comp_semantic_noise(self, task_names, pca_accuracy, num, s_ps, i_ps, alpha=1.0):
        noises = []
        eigenvalues_0 = load(
            'pca_obj/_' + task_names[0] + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors_0 = load(
            'pca_obj_fft/_' + task_names[0] + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector_0 = load('pca_obj/_' + task_names[0] + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        eigenvalues_1 = load(
            'pca_obj/_' + task_names[1] + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors_1 = load(
            'pca_obj/_' + task_names[1] + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector_1 = load('pca_obj/_' + task_names[1] + self._meanvector_prefix + str(pca_accuracy) + ".npy")

        ''''''
        k_sem_0 = int(s_ps[0] * len(eigenvalues_0))
        k_id_0 = int(i_ps[0] * len(eigenvalues_0))
        eigenvectors_sem_0 = eigenvectors_0[:, :k_sem_0]
        eigenvalues_sem_0 = eigenvalues_0[:k_sem_0]
        eigenvectors_id_0 = eigenvectors_0[:, k_id_0:]
        eigenvalues_id_0 = eigenvalues_0[k_id_0:]

        k_sem_1 = int(s_ps[1] * len(eigenvalues_1))
        k_id_1 = int(i_ps[1] * len(eigenvalues_1))
        eigenvectors_sem_1 = eigenvectors_1[:, :k_sem_1]
        eigenvalues_sem_1 = eigenvalues_1[:k_sem_1]
        eigenvectors_id_1 = eigenvectors_1[:, k_id_1:]
        eigenvalues_id_1 = eigenvalues_1[k_id_1:]

        for i in range(num):
            sample = np.round(np.random.RandomState(i).randn(1, 512), decimals=3)[0]

            b_vector_p_sem_0 = self._calculate_b_vector(sample, True, eigenvalues_sem_0, eigenvectors_sem_0,
                                                        meanvector_0)
            b_vector_p_id_0 = self._calculate_b_vector(sample, True, eigenvalues_id_0, eigenvectors_id_0, meanvector_0)
            out_sem_0 = alpha * meanvector_0 + np.dot(eigenvectors_sem_0, b_vector_p_sem_0)
            out_id_0 = alpha * meanvector_0 + np.dot(eigenvectors_id_0, b_vector_p_id_0)

            b_vector_p_sem_1 = self._calculate_b_vector(sample, True, eigenvalues_sem_1, eigenvectors_sem_1,
                                                        meanvector_1)
            b_vector_p_id_1 = self._calculate_b_vector(sample, True, eigenvalues_id_1, eigenvectors_id_1, meanvector_1)
            out_sem_1 = alpha * meanvector_1 + np.dot(eigenvectors_sem_1, b_vector_p_sem_1)
            out_id_1 = alpha * meanvector_1 + np.dot(eigenvectors_id_1, b_vector_p_id_1)

            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out_sem_0, 0))
            noises.append(np.expand_dims(out_sem_1, 0))
            out_sem_m = (0.3 * out_sem_0) + (0.7 * out_sem_1)
            noises.append(np.expand_dims(out_sem_m, 0))
        return noises

    def make_single_semantic_noise_fft(self, task_name, pca_accuracy, num, s_p, i_p, alpha=1.0):
        noises = []
        eigenvalues = load(
            'pca_obj_fft/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj_fft/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj_fft/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")
        #
        # u = load('pca_obj_fft/_' + task_name + self._u_prefix + str(pca_accuracy) + ".npy")
        # s = load('pca_obj_fft/_' + task_name + self._s_prefix + str(pca_accuracy) + ".npy")
        # vh = load('pca_obj_fft/_' + task_name + self._vt_prefix + str(pca_accuracy) + ".npy")
        ''''''
        k_sem = int(s_p * len(eigenvalues))
        k_id = int(i_p * len(eigenvalues))

        eigenvectors_sem = eigenvectors[:, :k_sem]
        eigenvalues_sem = eigenvalues[:k_sem]

        eigenvectors_id = eigenvectors[:, k_id:]
        eigenvalues_id = eigenvalues[k_id:]

        '''semantic decompose by SVD'''
        # u_sem = u[:, :k]
        # v_sem = vh[:, :k]
        # u_cov = np.dot(u_sem, u_sem.T)
        for i in range(num):
            sample = np.round(np.random.RandomState(i).randn(1, 512), decimals=3)[0]
            sample = np.clip(sample, -10.0, +10.0)
            sample = self.filter_signal(sample, 3000, i, False)
            #
            b_vector_p_sem = self._calculate_b_vector(sample, True, eigenvalues_sem, eigenvectors_sem, meanvector)
            b_vector_p_id = self._calculate_b_vector(sample, True, eigenvalues_id, eigenvectors_id, meanvector)
            out_sem = alpha * meanvector + np.dot(eigenvectors_sem, b_vector_p_sem)
            out_id = alpha * meanvector + np.dot(eigenvectors_id, b_vector_p_id)
            # out_svd = u_cov @ sample

            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out_sem, 0))
            noises.append(np.expand_dims(out_id, 0))
            # noises.append(np.expand_dims(out_svd, 0))

            # # out_svd = np.mean((u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :])[:, :3], -1)
            # # out_svd = u[:, 6:7] @ np.diag(s)[:1, :1]
            # out_svd = u[:, :1] @ eigenvalues[:1]
            # noises.append(np.expand_dims(out_svd, 0))
            # # for j in range(k):
            # #     out_svd = u[:, j:j+1] @ eigenvalues[j:j+1]
            # #     noises.append(np.expand_dims(out_svd, 0))
        return noises

    def make_compound_semantic_noise(self, data, num, pca_accuracy=99):
        eigenvalues_arr = []
        eigenvectors_arr = []
        meanvector_arr = []

        for i in range(len(data)):
            eigenvalues = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvalues_prefix + str(data[i]['p_ac']) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvectors_prefix + str(data[i]['p_ac']) + ".npy")
            meanvector_arr.append(load(
                'pca_obj/_' + data[i]['t_n'] + self._meanvector_prefix + str(data[i]['p_ac']) + ".npy"))
            eigenvectors_arr.append(eigenvectors)
            eigenvalues_arr.append(eigenvalues)
        '''calculate Mean'''
        sem_p_0 = int(data[0]['s_p'] * len(eigenvalues_arr[0]))
        id_p_0 = int(data[0]['i_p'] * len(eigenvalues_arr[0]))
        alpha_0 = int(data[0]['alpha'])

        sem_p_1 = int(data[1]['s_p'] * len(eigenvalues_arr[1]))
        id_p_1 = int(data[1]['i_p'] * len(eigenvalues_arr[1]))
        alpha_1 = int(data[1]['alpha'])

        ''' feature 0 '''
        eigenvalues_sem_0 = eigenvalues_arr[0][:sem_p_0]
        eigenvectors_sem_0 = eigenvectors_arr[0][:, :sem_p_0]
        eigenvalues_id_0 = eigenvalues_arr[0][id_p_0:]
        eigenvectors_id_0 = eigenvectors_arr[0][:, id_p_0:]
        ''' feature 1 '''
        eigenvalues_sem_1 = eigenvalues_arr[1][:sem_p_1]
        eigenvectors_sem_1 = eigenvectors_arr[1][:, :sem_p_1]
        eigenvalues_id_1 = eigenvalues_arr[1][id_p_1:]
        eigenvectors_id_1 = eigenvectors_arr[1][:, id_p_1:]
        '''feature fusion'''

        '''make noise'''
        noises = []
        sem_noises_0 = []
        sem_noises_1 = []
        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]

            b_vector_p_sem_0 = self._calculate_b_vector(sample, True, eigenvalues_sem_0, eigenvectors_sem_0,
                                                        meanvector_arr[0])
            b_vector_p_id_0 = self._calculate_b_vector(sample, True, eigenvalues_id_0, eigenvectors_id_0,
                                                       meanvector_arr[0])
            out_sem_0 = alpha_0 * meanvector_arr[0] + np.dot(eigenvectors_sem_0, b_vector_p_sem_0)
            out_id_0 = alpha_0 * meanvector_arr[0] + np.dot(eigenvectors_id_0, b_vector_p_id_0)

            b_vector_p_sem_1 = self._calculate_b_vector(out_sem_0, True, eigenvalues_sem_1, eigenvectors_sem_1,
                                                        meanvector_arr[1])
            b_vector_p_id_1 = self._calculate_b_vector(out_id_0, True, eigenvalues_id_1, eigenvectors_id_1,
                                                       meanvector_arr[1])
            out_sem_1 = alpha_1 * meanvector_arr[1] + np.dot(eigenvectors_sem_1, b_vector_p_sem_1)
            out_id_1 = alpha_1 * meanvector_arr[1] + np.dot(eigenvectors_id_1, b_vector_p_id_1)

            out_id2 = alpha_1 * meanvector_arr[1] + np.dot(eigenvectors_id_1, b_vector_p_id_1) + \
                       1.0 * meanvector_arr[1]

            noises.append(np.expand_dims(sample, 0))
            '''semantic-based synthesis'''
            # noises.append(np.expand_dims(out_sem_0, 0))
            # noises.append(np.expand_dims(out_sem_1, 0))
            '''facial attribute manipulation'''
            noises.append(np.expand_dims(out_id_0, 0))
            noises.append(np.expand_dims(out_id_1, 0))
            # noises.append(np.expand_dims(out_id2, 0))

            # noises.append(np.expand_dims(0.9*out_id_1+0.1*out_sem_1, 0))

            # both_a = 0.5*(out_sem_0+out_id_0) + 0.5*(out_sem_1+out_id_1)
            # both_a = 0.5*(out_id_0) + 0.5*(out_id_1)
            # noises.append(np.expand_dims(both_a, 0))

            # noises.append(np.expand_dims(out_sem_0, 0))

            # sem_noises_0.append(np.expand_dims(out_sem_0, 0))
            # sem_noises_1.append(np.expand_dims(out_sem_1, 0))

            # noises.append(np.expand_dims(out_id_0, 0))
            # noises.append(np.expand_dims(out_id_1, 0))

            # x = sample - 0.5 * (out_sem_0+out_sem_1) # Cumulative identity

            # noises.append(np.expand_dims(sample-out_id_1, 0)) # semantcie
            # noises.append(np.expand_dims(sample-out_sem_0, 0)) # identity

            # noises.append(np.expand_dims(sample-out_sem_1, 0))
            # noises.append(np.expand_dims(x, 0))

            # noises.append(np.expand_dims(1.0 * (out_sem_0)
            #                              + 0.0 * ( out_sem_1)
            #                              , 0))

            # noises.append(np.expand_dims(0.5 * (out_sem_0+out_sem_1)
            #                              + 0.5 * (out_id_0+out_id_1)
            #                              , 0))
        return noises
        # sem_avg_0 = [np.mean(np.array(sem_noises_0)[:, :, :], axis=0)]
        # sem_avg_1 = [np.mean(np.array(sem_noises_1)[:, :, :], axis=0)]
        # return [sem_avg_1[0] - sem_avg_0[0]]

    def make_compound_semantic_noise_3(self, data, num, pca_accuracy=99, alpha=1.0):
        eigenvalues_arr = []
        eigenvectors_arr = []
        meanvector_arr = []

        for i in range(len(data)):
            eigenvalues = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvalues_prefix + str(data[i]['p_ac']) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvectors_prefix + str(data[i]['p_ac']) + ".npy")
            meanvector_arr.append(load(
                'pca_obj/_' + data[i]['t_n'] + self._meanvector_prefix + str(data[i]['p_ac']) + ".npy"))
            eigenvectors_arr.append(eigenvectors)
            eigenvalues_arr.append(eigenvalues)
        '''calculate Mean'''
        k_0 = int(data[0]['vec_p'] * len(eigenvalues_arr[0]))
        k_1 = int(data[1]['vec_p'] * len(eigenvalues_arr[1]))
        ''' feature 0 '''
        eigenvalues_sem_0 = eigenvalues_arr[0][:k_0]
        eigenvectors_sem_0 = eigenvectors_arr[0][:, :k_0]
        eigenvalues_id_0 = eigenvalues_arr[0][k_0:]
        eigenvalues_id_0 = eigenvalues_arr[0][k_0:]

        eigenvalues_id_0 = eigenvalues_arr[0]
        eigenvectors_id_0 = eigenvectors_arr[0]
        ''' feature 1 '''
        eigenvalues_sem_1 = eigenvalues_arr[1][:k_1]
        eigenvectors_sem_1 = eigenvectors_arr[1][:, :k_1]
        eigenvalues_id_1 = eigenvalues_arr[1][k_1:]
        eigenvectors_id_1 = eigenvectors_arr[1][:, k_1:]
        '''feature fusion'''

        '''make noise'''
        noises = []
        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]

            b_vector_p_sem_0 = self._calculate_b_vector(sample, True, eigenvalues_sem_0, eigenvectors_sem_0,
                                                        meanvector_arr[0])
            b_vector_p_id_0 = self._calculate_b_vector(sample, True, eigenvalues_id_0, eigenvectors_id_0,
                                                       meanvector_arr[0])

            b_vector_p_sem_1 = self._calculate_b_vector(sample, True, eigenvalues_sem_1, eigenvectors_sem_1,
                                                        meanvector_arr[1])
            b_vector_p_id_1 = self._calculate_b_vector(sample, True, eigenvalues_id_1, eigenvectors_id_1,
                                                       meanvector_arr[1])

            out_sem_0 = (alpha * meanvector_arr[0] + np.dot(eigenvectors_sem_0, b_vector_p_sem_0))
            out_id_0 = alpha * meanvector_arr[0] + np.dot(eigenvectors_id_0, b_vector_p_id_0)

            out_sem_1 = (alpha * meanvector_arr[1] + np.dot(eigenvectors_sem_1, b_vector_p_sem_1))
            out_id_1 = alpha * meanvector_arr[1] + np.dot(eigenvectors_id_1, b_vector_p_id_1)

            noises.append(np.expand_dims(sample, 0))
            # noises.append(np.expand_dims(out_sem_0, 0))
            # noises.append(np.expand_dims(out_sem_1, 0))
            # noises.append(np.expand_dims(out_id_0, 0))
            # noises.append(np.expand_dims(out_id_1, 0))

            noises.append(np.expand_dims(0.5 * (out_sem_0)
                                         + 0.5 * (out_sem_1)
                                         , 0))

            # noises.append(np.expand_dims(0.5 * (out_sem_0+out_sem_1)
            #                              + 0.5 * (out_id_0+out_id_1)
            #                              , 0))
        return noises

    def make_compound_semantic_noise_2(self, data, num, vec_percent, pca_accuracy=99):
        eigenvalues_arr = []
        eigenvectors_arr = []
        meanvector_arr = []

        for i in range(len(data)):
            eigenvalues = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvalues_prefix + str(data[i]['p_ac']) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvectors_prefix + str(data[i]['p_ac']) + ".npy")
            meanvector_arr.append(load(
                'pca_obj/_' + data[i]['t_n'] + self._meanvector_prefix + str(data[i]['p_ac']) + ".npy"))
            eigenvectors_arr.append(eigenvectors[:, :int(data[i]['vec_p'] * len(eigenvalues))])
            eigenvalues_arr.append(eigenvalues[:int(data[i]['vec_p'] * len(eigenvalues))])
        '''calculate Mean'''
        meanvector = meanvector_arr[1]
        ''''''
        n_egv_0 = eigenvalues_arr[0] / max(eigenvalues_arr[0])
        n_egv_1 = eigenvalues_arr[1] / max(eigenvalues_arr[1])
        ''''''
        eigenvectors = np.concatenate([eigenvectors_arr[1], eigenvectors_arr[0]], axis=1)
        eigenvalues = np.concatenate([eigenvalues_arr[1], eigenvalues_arr[0]], axis=0)

        n_egvs = np.concatenate([n_egv_0, n_egv_0], axis=0)
        #
        eigenvectors = eigenvectors[:, :int(vec_percent * len(eigenvalues))]
        eigenvalues = eigenvalues[:int(vec_percent * len(eigenvalues))]

        n_egvs = n_egvs[:int(vec_percent * len(eigenvalues))]

        '''sort based on the importance'''
        eigenvalues = eigenvalues[np.argsort(n_egvs)[::-1]]
        eigenvectors = eigenvectors[:, np.argsort(n_egvs)[::-1]]

        '''make noise'''
        noises = []
        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            out = meanvector + np.dot(eigenvectors, b_vector_p)
            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out, 0))
        return noises

    def make_compound_semantic_noise_1(self, data, num, vec_percent, pca_accuracy=99):
        eigenvalues_arr = []
        eigenvectors_arr = []
        meanvector_arr = []

        for i in range(len(data)):
            eigenvalues = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvalues_prefix + str(data[i]['p_ac']) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvectors_prefix + str(data[i]['p_ac']) + ".npy")
            meanvector_arr.append(load(
                'pca_obj/_' + data[i]['t_n'] + self._meanvector_prefix + str(data[i]['p_ac']) + ".npy"))
            eigenvectors_arr.append(eigenvectors[:, :int(data[i]['vec_p'] * len(eigenvalues))])
            eigenvalues_arr.append(eigenvalues[:int(data[i]['vec_p'] * len(eigenvalues))])

        '''create the covariance matrix between eigenvectors'''
        # A = eigenvalues_arr[0] * eigenvectors_arr[0]
        # B = eigenvalues_arr[1] * eigenvectors_arr[1]

        ''''''
        # AB = np.dot(A.T, B)
        # AB_m = AB - np.mean(AB)
        # U, S, V = np.linalg.svd(AB_m / np.sqrt(len(AB)))
        # u, s, vh = np.linalg.svd(AB)
        ''''''
        # A_m = A - np.mean(A)
        # B_m = B - np.mean(B)
        #
        # u_A, s_A, vh_A = np.linalg.svd(A)
        # u_B, s_B, vh_B = np.linalg.svd(B)

        # eigenvalues = np.concatenate([s_B[1:3], s_A[10:100], ])
        # eigenvectors = np.concatenate([u_B[:, 1:3], u_A[:, 10:100], ], axis=1)

        # cov_m = np.cov(np.concatenate([eigenvectors_arr[0], eigenvectors_arr[1]], axis=1))
        # cov_m_T = np.cov(np.concatenate([eigenvectors_arr[0], eigenvectors_arr[1]], axis=1).T)
        # dot_p = np.dot(eigenvectors_arr[0].T, eigenvectors_arr[1])
        # cs = cosine_similarity(eigenvectors_arr[0], eigenvectors_arr[1])

        meanvector = meanvector_arr[1]
        # meanvector = meanvector_arr[0] + meanvector_arr[1]

        eigenvectors = np.concatenate([eigenvectors_arr[1], eigenvectors_arr[0]], axis=1)
        eigenvalues = np.concatenate([eigenvalues_arr[1], eigenvalues_arr[0]], axis=0)
        #
        eigenvectors = eigenvectors[:, :int(vec_percent * len(eigenvalues))]
        eigenvalues = eigenvalues[:int(vec_percent * len(eigenvalues))]

        ''' the similarity between the 2 transformation matrix'''
        # dot_p = np.dot(eigenvectors_arr[0].T, eigenvectors_arr[1])
        dot_p = np.dot(eigenvectors_arr[0].T,
                       eigenvectors_arr[1])
        dot_p_f = dot_p.flatten()
        x = np.arange(0, len(dot_p_f))

        plt.style.use('ggplot')
        plt.plot(x, np.exp(abs(dot_p_f)), linestyle='--', c='b')
        # plt.plot(x, abs(dot_p_f), linestyle='--', c='b')

        # sns.heatmap(dot_p, linewidth=0.0000001,
        #             vmin=np.min(dot_p),
        #             vmax=np.max(dot_p))

        plt.savefig('similarity')

        idx = np.argsort(abs(dot_p_f))[:-1]  # the most less similar
        A_ind = []
        B_ind = []
        ''' adding the most important eigenVectors'''
        # A_ind = [i for i in np.arange(0,5)]
        B_ind = [i for i in np.arange(0, 10)]
        ''' adding less similar '''
        for ids in range(25):
            i = idx[ids] // (len(dot_p[0, :]))
            j = idx[ids] - (i * len(dot_p[0, :]))
            A_ind.append(i)
            B_ind.append(j)
        ''' '''
        A_ind = list(set(A_ind))
        B_ind = list(set(B_ind))

        A_evecs = eigenvectors_arr[0][:, A_ind]
        A_evals = eigenvalues_arr[0][A_ind]

        B_evecs = eigenvectors_arr[1][:, B_ind]
        B_evals = eigenvalues_arr[1][B_ind]

        '''normal'''
        A_evals_N = eigenvalues_arr[0] / max(A_evals)
        B_evals_N = eigenvalues_arr[1] / max(B_evals)
        n_egvs = np.concatenate([A_evals_N, B_evals_N], axis=0)

        ''''''
        eigenvectors = np.concatenate([A_evecs, B_evecs], axis=1)
        eigenvalues = np.concatenate([A_evals, B_evals], axis=0)

        eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

        '''make noise'''
        noises = []
        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            out = meanvector + np.dot(eigenvectors, b_vector_p)
            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out, 0))
        return noises

    def make_compound_semantic_noise_0(self, data, num, vec_percent, pca_accuracy=99):
        eigenvalues_arr = []
        eigenvectors_arr = []
        meanvector_arr = []

        for i in range(len(data)):
            eigenvalues = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvalues_prefix + str(data[i]['p_ac']) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvectors_prefix + str(data[i]['p_ac']) + ".npy")
            meanvector_arr.append(load(
                'pca_obj/_' + data[i]['t_n'] + self._meanvector_prefix + str(data[i]['p_ac']) + ".npy"))
            eigenvectors_arr.append(eigenvectors[:, :int(data[i]['vec_p'] * len(eigenvalues))])
            eigenvalues_arr.append(eigenvalues[:int(data[i]['vec_p'] * len(eigenvalues))])

        meanvector = meanvector_arr[1]
        ''''''
        n_egv_0 = eigenvalues_arr[0] / max(eigenvalues_arr[0])
        n_egv_1 = eigenvalues_arr[1] / max(eigenvalues_arr[1])
        ''''''
        # meanvector = meanvector_arr[0] + meanvector_arr[1]

        eigenvectors = np.concatenate([eigenvectors_arr[1], eigenvectors_arr[0]], axis=1)
        eigenvalues = np.concatenate([eigenvalues_arr[1], eigenvalues_arr[0]], axis=0)

        n_egvs = np.concatenate([n_egv_0, n_egv_0], axis=0)
        #
        eigenvectors = eigenvectors[:, :int(vec_percent * len(eigenvalues))]
        eigenvalues = eigenvalues[:int(vec_percent * len(eigenvalues))]

        n_egvs = n_egvs[:int(vec_percent * len(eigenvalues))]

        '''sort based on the importance'''
        eigenvalues = eigenvalues[np.argsort(n_egvs)[::-1]]
        eigenvectors = eigenvectors[:, np.argsort(n_egvs)[::-1]]

        '''make noise'''
        noises = []
        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            out = meanvector + np.dot(eigenvectors, b_vector_p)
            noises.append(np.expand_dims(sample, 0))
            noises.append(np.expand_dims(out, 0))
        return noises

    def old_make_compound_semantic_noise(self, data, num, vec_percent, pca_accuracy=99):
        eigenvalues_arr = []
        eigenvectors_arr = []
        meanvector_arr = []

        for i in range(len(data)):
            eigenvalues = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvalues_prefix + str(data[i]['p_ac']) + ".npy")
            eigenvectors = load(
                'pca_obj/_' + data[i]['t_n'] + self._eigenvectors_prefix + str(data[i]['p_ac']) + ".npy")
            meanvector_arr.append(load(
                'pca_obj/_' + data[i]['t_n'] + self._meanvector_prefix + str(data[i]['p_ac']) + ".npy"))
            eigenvectors_arr.append(eigenvectors[:, :int(data[i]['vec_p'] * len(eigenvalues))])
            eigenvalues_arr.append(eigenvalues[:int(data[i]['vec_p'] * len(eigenvalues))])

        '''create the covariance matrix between eigenvectors'''
        cov_m = np.cov(np.concatenate([eigenvectors_arr[0], eigenvectors_arr[1]], axis=1))
        cov_m_T = np.cov(np.concatenate([eigenvectors_arr[0], eigenvectors_arr[1]], axis=1).T)
        dot_p = np.dot(eigenvectors_arr[0].T, eigenvectors_arr[1])
        # cs = cosine_similarity(eigenvectors_arr[0], eigenvectors_arr[1])

        meanvector = 0.5 * meanvector_arr[0] + 0.5 * meanvector_arr[1]
        eigenvectors = np.concatenate([eigenvectors_arr[0], eigenvectors_arr[1]], axis=1)
        eigenvalues = np.concatenate([eigenvalues_arr[0], eigenvalues_arr[1]], axis=0)

        eigenvectors = eigenvectors[:, :int(vec_percent * len(eigenvalues))]
        eigenvalues = eigenvalues[:int(vec_percent * len(eigenvalues))]

        '''make noise'''
        noises = []
        for i in range(num):
            sample = np.round(np.random.RandomState(i * 100).randn(1, 512), decimals=3)[0]
            b_vector_p = self._calculate_b_vector(sample, True, eigenvalues, eigenvectors, meanvector)
            out = meanvector + np.dot(eigenvectors, b_vector_p)
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

    def get_asm_b(self, task_name, pca_accuracy, num, alpha=1.0):
        noises = []
        eigenvalues = load(
            'pca_obj/_' + task_name + self._eigenvalues_prefix + str(pca_accuracy) + ".npy")
        eigenvectors = load(
            'pca_obj/_' + task_name + self._eigenvectors_prefix + str(pca_accuracy) + ".npy")
        meanvector = load('pca_obj/_' + task_name + self._meanvector_prefix + str(pca_accuracy) + ".npy")

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
                # out_w = alpha * meanvector + np.dot(eigenvectors, b_vector_p * np.array(w))
                out_w_p = alpha * meanvector + np.dot(eigenvectors, b_vector_p * np.array(w_p))
                # out_w_n = 0.3 * sample + 0.7 * out_w
                # noises.append(np.expand_dims(out_w, 0))
                # noises.append(np.expand_dims(out_w_n, 0))
                noises.append(np.expand_dims(out_w_p, 0))
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
            noises.append(np.expand_dims(meanvector - out, 0))
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

    def _func_PCA_dask(self, input_data):
        input_data = np.array(input_data)
        dX = da.from_array(input_data, chunks='auto')
        pca_d = DPCA()
        pca_d.fit(dX)
        pca_input_data = pca_d.transform(dX)
        eigenvalues = pca_d.explained_variance_
        eigenvectors = pca_d.components_
        return pca_input_data, eigenvalues, eigenvectors

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
