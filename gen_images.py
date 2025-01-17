# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import tensorflow as tf
import os
import re
from typing import List, Optional, Tuple, Union

from age_feature import AgeExtraction
from gender_feature import GenderExtraction
from race_feature import RaceExtraction
from linear_projection import LinearProjection

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from fer import FER

from config import FolderStructures, PreTrainedModels, Age_codes, Race_codes, Expression_codes, Gender_codes
from image_utility import ImageUtilities
from analyze_fer import AnalyzeFer
import random
from tqdm import tqdm
from os.path import exists
from scipy.special import softmax
from scipy.fft import fft, fftfreq, dct, dctn
from numpy.fft import *
from facial_attribute_editing_eval import FacialAttributeEditingEval
from face_recognition import FaceRecognition
import json


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def generate_fixed(network_pkl: str,
                   seeds: List[int],
                   truncation_psi: float,
                   noise_mode: str,
                   outdir: str,
                   translate: Tuple[float, float],
                   rotate: float,
                   class_idx: Optional[int]):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        # label[:, class_idx] = 1
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    '''loading FER model'''
    img_util = ImageUtilities()
    fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/AffectNet_6336.h5')

    # '''loading Age'''
    # age_class = AgeExtraction(model_path='./features/age/age_net.caffemodel',
    #                           proto_path='./features/age/age_net.prototxt')
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        npy_noise = np.round(np.random.RandomState(seed).randn(1, G.z_dim), decimals=3)
        z = torch.from_numpy(npy_noise).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # convert image to npy
        npy_img = img[0].cpu().numpy()
        # extract expression
        exp, exp_vec = fer_class.recognize_fer(npy_img=npy_img)
        # recognize age

        # resize image to 512 * 512 * 3
        resized_npy_img = img_util.resize_image(npy_img=npy_img, w=512, h=512, ch=3)
        # save image
        img_util.save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.images, save_name=str(seed))
        # save expressions
        np.save(file=f'{outdir}{FolderStructures.feature_vectors}/exp_{str(seed)}', arr=np.round(exp_vec, decimals=3))
        # save noise vector
        np.save(file=f'{outdir}{FolderStructures.noise_vectors}/{str(seed)}', arr=npy_noise)


def generate_with_interpolation_function(network_pkl: str,
                                         inter_functions,
                                         num_of_samples: int,
                                         name_pre: str,
                                         _fer_class,
                                         truncation_psi: float,
                                         noise_mode: str,
                                         outdir: str,
                                         translate: Tuple[float, float],
                                         rotate: float,
                                         class_idx: Optional[int]):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        # label[:, class_idx] = 1
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    f = open(f'/media/ali/extradata/styleGAN3_samples/v1/annotation.txt', "w")
    exps_str = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
    jj = 0
    while jj < num_of_samples:
        new_noise = []
        for i in range(512):
            new_noise.append(inter_functions[i](random.uniform(0.0, 511.0)))
        noise = np.expand_dims(np.array(new_noise), axis=0)

        z = torch.from_numpy(noise).to(device)
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # convert image to npy
        npy_img = img[0].cpu().numpy()
        # extract expression
        exp, exp_vec = _fer_class.recognize_fer(npy_img=npy_img)
        exp_vec = exp_vec[0]
        if exp == 'Happy': continue
        # if (exp != 'Anger' and exp_vec[6] < 0.6) or \
        #     (exp != 'Neutral' and exp_vec[0] < 0.6):
        #     #     (exp == 'Disgust' and exp_vec[5] > 0.4) or \
        #     #     (exp == 'Sad' and exp_vec[2] > 0.4):
        #     # npy_img = None
        #     continue

        # resize image to 512 * 512 * 3
        resized_npy_img = ImageUtilities().resize_image(npy_img=npy_img, w=512, h=512, ch=3)
        # save image
        pre = random.randint(0, 100000)
        ImageUtilities().save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.interpolate_images,
                                    save_name=name_pre + str(jj) + str(pre))
        np.save(file=f'{outdir}{FolderStructures.interpolate_feature_vectors}/exp_{name_pre}{str(jj)}{str(pre)}',
                arr=np.round(exp_vec, decimals=3))
        np.save(file=f'{outdir}{FolderStructures.interpolate_noise_vectors}/{name_pre}{str(jj)}{str(pre)}', arr=noise)
        # f.write(
        #     str(jj) + ' : ' + exps_str[np.argmax(exp_vec)] + '===> ' + ''.join(str(e) for e in list(exp_vec)) + '\n\r')
        jj += 1
        print(jj)


def generate_with_noise(network_pkl: str,
                        noises,
                        fer_detection: bool,
                        truncation_psi: float,
                        noise_mode: str,
                        outdir: str,
                        translate: Tuple[float, float],
                        rotate: float,
                        class_idx: Optional[int], names=None, ):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        # label[:, class_idx] = 1
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    '''loading FER model'''
    img_util = ImageUtilities()
    if fer_detection:
        fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/AffectNet_6336.h5')
    # Generate images.
    jj = 10000
    for index, noise in enumerate(noises):
        z = torch.from_numpy(noise).to(device)
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # convert image to npy
        npy_img = img[0].cpu().numpy()
        # resize image to 512 * 512 * 3
        resized_npy_img = img_util.resize_image(npy_img=npy_img, w=512, h=512, ch=3)

        if fer_detection:
            exp, exp_vec = fer_class.recognize_fer(npy_img=npy_img)
            # if exp == 'Happy': continue
            np.save(file=f'{outdir}{FolderStructures.feature_vectors}/exp_{str(jj)}',
                    arr=np.round(exp_vec, decimals=3))
        if names is None:
            np.save(file=f'{outdir}{FolderStructures.noise_vectors}/{str(jj)}', arr=noise)
        jj += 1

        # save image
        if names is None:
            img_util.save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.images,
                                save_name=str(jj))
        else:
            img_util.save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.images,
                                save_name=str(names[index]))
        # extract expression


"============================= histogram ========================================"


def create_histogram_fer(folder_name):
    analyser = AnalyzeFer(noise_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/noise_vectors',
                          exp_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/feature_vectors')
    '''creating 1-d histogram'''
    analyser.calculate_exp_histogram()
    analyser.plot_histogram(file_name='exp_hist.jpg')
    return 0


def create_histogram_age(folder_name, name=''):
    age_class = AgeExtraction(model_path='./features/age/age_net.caffemodel',
                              proto_path='./features/age/age_net.prototxt')
    age_class.create_histogram(age_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/age_extraction/',
                               name=name)


def create_histogram_gender(folder_name):
    gender_class = GenderExtraction(model_path='./features/gender/gender.caffemodel',
                                    proto_path='./features/gender/gender.prototxt')
    gender_class.create_histogram(
        gender_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/gender_extraction/')


def create_histogram_race(folder_name):
    race_class = RaceExtraction(model_path='./features/race/race.h5', GPU=False)
    race_class.create_histogram(race_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/race_extraction/')


"=============================  predicting features ========================================"


def predict_race_images(folder_name):
    """ ['asian','indian','black','white','middle eastern','latino hispanic'] """
    race_class = RaceExtraction(model_path='./features/race/race.h5')
    race_class.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images/',
                                out_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/race_extraction/',
                                csv_file_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/race.csv')


def predict_age_images(folder_name):
    """4 age ranges (0-15) as child, (16-32) as young, (33-53) as adult, and (54-100) as old"""
    age_class = AgeExtraction(model_path='./features/age/age_net.caffemodel',
                              proto_path='./features/age/age_net.prototxt')

    age_class.predict_and_save_deep_face(img_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images/',
                                         out_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/age_extraction/',
                                         csv_file_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/age.csv')

    # age_class.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images/',
    #                            out_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/age_extraction/',
    #                            csv_file_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/age.csv')


def predict_gender_images(folder_name):
    """man or woman"""
    gender_class = GenderExtraction(model_path='./features/gender/gender.caffemodel',
                                    proto_path='./features/gender/gender.prototxt')
    gender_class.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images/',
                                  out_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/gender_extraction/',
                                  csv_file_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/gender.csv')


def predict_fer_images(folder_name):
    f_analyser = AnalyzeFer(noise_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/noise_vectors',
                            exp_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/feature_vectors')
    f_analyser.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images/',
                                out_dir=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/feature_vectors/',
                                csv_file_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/fer.csv')

    '''creating csv'''
    f_analyser.create_csv(img_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images',
                          fer_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/feature_vectors',
                          csv_file_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/fer.csv')


"================================ Fer interpolation ================================"


def interpolate_fer_images(_fer_class):
    analyser = AnalyzeFer(noise_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/noise_vectors',
                          exp_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors')
    NEUTRAL = 1
    HAPPY = 1
    ANGRY = 6
    FEMALE = 0
    MALE = 1
    CHILD = 0
    YOUTH = 1
    MIDDLE = 2
    OLD = 3
    BLACK = 2
    pre_names = ['a_fe', 'a_bl', 'a_ch', 'a_bl_fe']

    inter_functions_arr = analyser.interpolate_by_semantic(
        noise_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/noise_vectors',
        anno_path_fer=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors',
        task_id_fer=ANGRY,
        anno_path_gender=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/gender_extraction',
        task_id_gender=FEMALE,
        anno_path_race=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/race_extraction',
        task_id_race=BLACK,
        anno_path_age=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/age_extraction',
        task_id_age=CHILD)

    for i in range(len(pre_names)):
        generate_with_interpolation_function(
            network_pkl=PreTrainedModels.styleGan_pkl,
            inter_functions=inter_functions_arr[pre_names[i]],
            num_of_samples=10000,
            name_pre=pre_names[i],
            _fer_class=_fer_class,
            truncation_psi=0.7,
            noise_mode='const',  # 'const', 'random', 'none'],
            outdir=FolderStructures.prefix,
            translate=parse_vec2('0,0'),
            rotate=0,
            class_idx=0)
    #
    # inter_functions = analyser.interpolate(
    #     exp_cat_path=f'/media/ali/extradata/styleGAN3_samples/v1/expression_by_category/Anger.txt')
    #
    # generate_with_interpolation_function(
    #         network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #         inter_functions=inter_functions,
    #         num_of_samples=300000,
    #         name_pre='t',
    #         truncation_psi=0.7,
    #         noise_mode='const',  # 'const', 'random', 'none'],
    #         outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #         translate=parse_vec2('0,0'),
    #         rotate=0,
    #         class_idx=0)

    # new_noises = []
    # for j in range(100):
    #     new_noise = []
    #     for i in range(512):
    #         new_noise.append(inter_functions[i](random.uniform(0.0, 511.0)))
    #     new_noises.append(np.expand_dims(np.array(new_noise), axis=0))
    #
    # generate_with_noise(network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #                     noises=new_noises,
    #                     truncation_psi=0.7,
    #                     noise_mode='const',  # 'const', 'random', 'none'],
    #                     outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #                     translate=parse_vec2('0,0'),
    #                     rotate=0,
    #                     class_idx=0)


"=================================================================="


def create_lda_exp(task_0, task_id_0, task_1, task_id_1):
    noise_path = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/noise_vectors'

    if task_0 == 'fer':
        anno_path_0 = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/feature_vectors'
    if task_1 == 'fer':
        anno_path_1 = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/feature_vectors'

    if task_0 == 'race':
        anno_path_0 = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/race_extraction'
    if task_1 == 'race':
        anno_path_1 = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/race_extraction'

    asm = LinearProjection()
    lda_means = asm.create_lda(noise_path=noise_path,
                               anno_path_0=anno_path_0,
                               anno_path_1=anno_path_1,
                               task_0=task_0, task_id_0=task_id_0,
                               task_1=task_1, task_id_1=task_id_1)
    return lda_means


def create_pca(pca_accuracy, tasks, name, folder_names: list, save_dir: str, sample_limit):
    lin_obj = LinearProjection()
    noise_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/noise_vectors' for fn in folder_names]
    fer_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/feature_vectors' for fn in folder_names]
    race_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/race_extraction' for fn in folder_names]
    gender_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/gender_extraction' for fn in folder_names]
    age_paths = [f'{FolderStructures.prefix}zz_productin/' + fn + '/age_extraction' for fn in folder_names]

    lin_obj.create_pca_from_npy(
        noise_paths=noise_paths,
        fer_paths=fer_paths,
        race_paths=race_paths,
        gender_paths=gender_paths,
        age_paths=age_paths,
        tasks=tasks,
        name=name,
        pca_accuracy=pca_accuracy,
        sample_limit=sample_limit,
        save_dir=save_dir
    )


"============================================================================================="
"=============================== move after creation of the images ==========================="


def filter_and_move(s_path_img: str,
                    s_path_noise: str,
                    s_path_feature: str,
                    d_path_img: str,
                    d_path_noise: str,
                    d_path_feature: str,
                    ):
    for file in tqdm(os.listdir(s_path_img)):
        if file.endswith('.jpg'):
            img_f_s = os.path.join(s_path_img, file)
            img_f_d = os.path.join(d_path_img, file)
            bare_f = str(file).split('.')[0]

            fer_f_s = os.path.join(s_path_feature, 'exp_' + bare_f + '.npy')
            fer_f_d = os.path.join(d_path_feature, 'exp_' + bare_f + '.npy')

            noise_f_s = os.path.join(s_path_noise, bare_f + '.npy')
            noise_f_d = os.path.join(d_path_noise, bare_f + '.npy')

            if exists(fer_f_s) and exists(noise_f_s):
                os.rename(img_f_s, img_f_d)
                os.rename(fer_f_s, fer_f_d)
                os.rename(noise_f_s, noise_f_d)


def create_fft(noise_path, save_path):
    for file in tqdm(os.listdir(noise_path)):
        if file.endswith('.npy'):
            noise = np.load(os.path.join(noise_path, file))
            f_noise = fft(noise)
            np.save(save_path + file, f_noise)
    pass


def revise_fft_noises(noise_path, save_path):
    lobj = LinearProjection()
    i = 0
    noises = []
    for file in tqdm(os.listdir(noise_path)):
        if file.endswith('.npy'):
            noise_raw = np.load(os.path.join(noise_path, file))
            # noise_raw = np.clip(noise_raw, -10.0, +10.0)[0, :]
            noise_raw = noise_raw[0, :]
            noise = lobj.filter_signal(noise_raw, 5000, i, True)
            noise = np.expand_dims(noise, 0)
            # np.save(save_path + file, noise)
            noises.append(noise)
            i += 1
    return noises


def revise_filter_noises(noise_path):
    lobj = LinearProjection()
    i = 0
    noises_f = []
    for file in tqdm(os.listdir(noise_path)):
        if file.endswith('.npy'):
            bare_f = str(file).split('.')[0]
            noise_raw = np.load(os.path.join(noise_path, file))
            if np.max(noise_raw) > 10:
                noises_f.append(bare_f)
            i += 1
    return noises_f


def remove_outlier(folder_name, noises_f):
    image_path = f'{FolderStructures.prefix}zz_productin/' + folder_name + '/images/'
    noise_path = f'{FolderStructures.prefix}zz_productin/' + folder_name + '/noise_vectors/'
    fer_path = f'{FolderStructures.prefix}zz_productin/' + folder_name + '/feature_vectors/'
    race_path = f'{FolderStructures.prefix}zz_productin/' + folder_name + '/race_extraction/'
    gender_path = f'{FolderStructures.prefix}zz_productin/' + folder_name + '/gender_extraction/'
    age_path = f'{FolderStructures.prefix}zz_productin/' + folder_name + '/age_extraction/'

    # for file in tqdm(os.listdir(image_path)):
    #     bare_f = str(file).split('.')[0]
    #     noise_f = os.path.join(noise_path, bare_f + '.npy')
    #     if not os.path.exists(noise_f):
    #         os.remove(os.path.join(image_path, file))

    for bare_f in noises_f:
        noise_f = os.path.join(noise_path, bare_f + '.npy')
        img_f = os.path.join(image_path, bare_f + '.jpg')
        fer_f = os.path.join(fer_path, 'exp_' + bare_f + '.npy')
        gender_f = os.path.join(gender_path, bare_f + '_gender.npy')
        race_f = os.path.join(race_path, bare_f + '_race.npy')
        age_f = os.path.join(age_path, bare_f + '_age.npy')
        os.remove(img_f)
        os.remove(noise_f)
        os.remove(fer_f)
        os.remove(gender_f)
        os.remove(race_f)
        os.remove(age_f)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python gen_images.py
    # orig_name = 'orig_new_angry_diverse'  # orig_new_angry_diverse OR orig_100K_normal
    # folder_name = 'orig_new_angry_diverse'  # new_angry_diverse OR new_100K_normal
    # noises = revise_fft_noises(noise_path=f'{FolderStructures.prefix}zz_productin/' + orig_name + '/noise_vectors',
    #                            save_path=f'{FolderStructures.prefix}zz_productin/' + folder_name + '/noise_vectors/')

    # noises_f = revise_filter_noises(noise_path=f'{FolderStructures.prefix}zz_productin/' + orig_name + '/noise_vectors')
    # remove_outlier(folder_name=orig_name, noises_f=noises_f)

    # generate_with_noise(network_pkl=FolderStructures.styleGan_weight_path,
    #                     noises=noises,
    #                     fer_detection=False,
    #                     truncation_psi=0.7,
    #                     noise_mode='const',  # 'const', 'random', 'none'],
    #                     outdir=FolderStructures.prefix + 'zz_productin/'+folder_name+'/',
    #                     translate=parse_vec2('0,0'),
    #                     rotate=0,
    #                     class_idx=0)
    #
    '''create new dataset with interpolation'''
    # _fer_class = FER(h5_address=PreTrainedModels.exp_model, GPU=True)
    # interpolate_fer_images(_fer_class)
    '''======================'''

    '''move images from interpolation to another destination'''
    # filter_and_move(s_path_img=f'{FolderStructures.prefix}{FolderStructures.interpolate_images}',
    #                 d_path_img=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/images',
    #                 s_path_noise=f'{FolderStructures.prefix}{FolderStructures.interpolate_noise_vectors}',
    #                 d_path_noise=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/noise_vectors',
    #                 s_path_feature=f'{FolderStructures.prefix}{FolderStructures.interpolate_feature_vectors}',
    #                 d_path_feature=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors'
    #                 )
    '''========================================='''

    '''batch image generation: '''
    # _fer_class = FER(h5_address=PreTrainedModels.exp_model, GPU=True)
    # lin_obj = LinearProjection()
    # analyze_fer_images(_fer_class)
    '''======================================'''

    '''evaluation_set: analyzing Age, Gender and Race'''
    # predict_fer_images(_fer_class=None, folder_name=folder_name)
    # folder_name = 'evaluation_set'
    # predict_age_images(folder_name=folder_name)
    # create_histogram_age(folder_name=folder_name)

    '''revise age'''
    # folder_names = ['BLACK_2.0', 'FEMALE_2.5', 'FEMALE_3.5', 'FEMALE_4.0']
    # for folder_name in folder_names:
    #     predict_age_images(folder_name='evaluation_set/facial_a_modified/' + folder_name)
    #     create_histogram_age(folder_name='evaluation_set/facial_a_modified/' + folder_name, name=folder_name + '_')

    # folder_names = ['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal']
    # for folder_name in folder_names:
    #     predict_age_images(folder_name=folder_name)
    #     create_histogram_age(folder_name=folder_name, name=folder_name+'_')

    # predict_gender_images(folder_name=folder_name)
    # predict_race_images(folder_name=folder_name)
    '''======================================'''

    '''Histogram Age, Gender and Race'''
    # create_histogram_fer(folder_name=folder_name)
    # create_histogram_age(folder_name=folder_name)
    # create_histogram_gender(folder_name=folder_name)
    # create_histogram_race(folder_name=folder_name)
    '''======================================'''
    '''creating PCA'''
    save_dir = './pca_obj/_'

    '''filter with query'''
    lin_obj = LinearProjection()
    fer_class = FER()
    # pca_list = ['Angry', 'FEMALE', 'MALE', 'BLACK', 'OLD', 'CHILD']
    pca_list = []
    '''WHITE'''
    if 'WHITE' in pca_list:
        print('WHITE')
        noise_male_white = fer_class.query_images_by_path(
            folder_names=['orig_100K_normal', 'orig_new_angry_diverse'],
            query={'fer': None, 'gender': [Gender_codes.FEMALE],
                   'race': [Race_codes.WHITE], 'age': None},
            num_samples=20e3
        )
        noise_female_white = fer_class.query_images_by_path(
            folder_names=['orig_100K_normal', 'orig_new_angry_diverse'],
            query={'fer': None, 'gender': [Gender_codes.MALE],
                   'race': [Race_codes.WHITE], 'age': None},
            num_samples=20e3
        )
        noise_child = noise_male_white + noise_female_white
        lin_obj.create_pca_from_list(noise_arr=noise_child, name='WHITE', save_dir='./pca_obj/_')

    '''CHILD'''
    if 'CHILD' in pca_list:
        print('CHILD')
        noise_male_child = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': None, 'gender': [Gender_codes.MALE],
                   'race': None, 'age': [Age_codes.CHILD, Age_codes.YOUTH]},
            num_samples=2e3
        )
        noise_female_child = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': None, 'gender': [Gender_codes.FEMALE],
                   'race': None, 'age': [Age_codes.CHILD, Age_codes.YOUTH]},
            num_samples=2e3
        )
        noise_child = noise_male_child + noise_female_child
        lin_obj.create_pca_from_list(noise_arr=noise_child, name='CHILD', save_dir='./pca_obj/_')

    '''OLD'''
    if 'OLD' in pca_list:
        print('\n OLD')
        noise_female_old_happy = fer_class.query_images_by_path(
            folder_names=['orig_100K_normal'],
            query={'fer': [Expression_codes.HAPPY], 'gender': [Gender_codes.FEMALE],
                   'race': None, 'age': [Age_codes.OLD, Age_codes.MIDDLE]},
            num_samples=2e3
        )
        base_len = len(noise_female_old_happy)
        print('noise_female_old_happy=> ' + str(len(noise_female_old_happy)))
        '''--'''
        noise_female_old_neutral = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.NEUTRAL], 'gender': [Gender_codes.FEMALE],
                   'race': None, 'age': [Age_codes.OLD, Age_codes.MIDDLE]},
            num_samples=base_len / 3
        )
        if len(noise_female_old_neutral) > base_len:
            noise_female_old_neutral = noise_female_old_neutral[:base_len]
        print('noise_female_old_neutral=> ' + str(len(noise_female_old_neutral)))
        '''--'''
        noise_male_old_happy = fer_class.query_images_by_path(
            folder_names=['orig_100K_normal'],
            query={'fer': [Expression_codes.HAPPY], 'gender': [Gender_codes.MALE],
                   'race': None, 'age': [Age_codes.OLD, Age_codes.MIDDLE]},
            num_samples=base_len
        )
        if len(noise_male_old_happy) > base_len:
            noise_male_old_happy = noise_male_old_happy[:base_len]
        print('noise_male_old_happy=> ' + str(len(noise_male_old_happy)))
        '''--'''
        noise_male_old_neutral = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.NEUTRAL], 'gender': [Gender_codes.MALE],
                   'race': None, 'age': [Age_codes.OLD, Age_codes.MIDDLE]},
            num_samples=base_len / 3
        )
        if len(noise_male_old_neutral) > base_len:
            noise_male_old_neutral = noise_male_old_neutral[:base_len]
        print('noise_male_old_neutral=> ' + str(len(noise_male_old_neutral)))
        '''---'''
        noise_old = noise_female_old_happy + noise_female_old_neutral + noise_male_old_happy + noise_male_old_neutral
        lin_obj.create_pca_from_list(noise_arr=noise_old, name='OLD', save_dir='./pca_obj/_')

    '''BLACK'''
    if 'BLACK' in pca_list:
        print('BLACK')
        noise_male_black = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': None, 'gender': [Gender_codes.MALE],
                   'race': [Race_codes.BLACK], 'age': None},
            num_samples=2e3
        )
        noise_female_black = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': None, 'gender': [Gender_codes.FEMALE],
                   'race': [Race_codes.BLACK], 'age': None},
            num_samples=2e3
        )
        noise_black = noise_male_black + noise_female_black
        lin_obj.create_pca_from_list(noise_arr=noise_black, name='BLACK', save_dir='./pca_obj/_')

    '''Angry'''
    if 'Angry' in pca_list:
        print('Angry')
        noise_angry_female = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.ANGER], 'gender': [Gender_codes.FEMALE], 'race': None,
                   'age': [Age_codes.CHILD, Age_codes.YOUTH]},
            num_samples=2e5
        )
        base_len = len(noise_angry_female)
        print('noise_angry_female=> ' + str(len(noise_angry_female)))

        noise_angry_male = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.ANGER], 'gender': [Gender_codes.MALE], 'race': None,
                   'age': [Age_codes.CHILD, Age_codes.YOUTH]},
            num_samples=base_len / 2
        )
        if len(noise_angry_male) > base_len:
            noise_angry_male = noise_angry_male[:base_len]
        print('noise_angry_male=> ' + str(len(noise_angry_male)))
        noise_angry = noise_angry_female + noise_angry_male
        lin_obj.create_pca_from_list(noise_arr=noise_angry, name='ANGRY', save_dir='./pca_obj/_')

    '''FEMALE'''
    if 'FEMALE' in pca_list:
        print('FEMALE')
        noise_female_angry = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.ANGER], 'gender': [Gender_codes.FEMALE], 'race': None, 'age': None},
            num_samples=5e3
        )
        noise_female_happy = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.HAPPY], 'gender': [Gender_codes.FEMALE], 'race': None, 'age': None},
            num_samples=5e3
        )
        noise_female = noise_female_angry + noise_female_happy
        lin_obj.create_pca_from_list(noise_arr=noise_female, name='FEMALE', save_dir='./pca_obj/_')

    '''MALE'''
    if 'MALE' in pca_list:
        print('MALE')
        noise_male_angry = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.ANGER], 'gender': [Gender_codes.MALE], 'race': None, 'age': None},
            num_samples=5e3
        )
        noise_male_happy = fer_class.query_images_by_path(
            folder_names=['orig_new_angry_diverse', 'orig_50K_moreAngry', 'orig_100K_normal'],
            query={'fer': [Expression_codes.HAPPY], 'gender': [Gender_codes.MALE], 'race': None, 'age': None},
            num_samples=5e3
        )
        noise_male = noise_male_angry + noise_male_happy
        lin_obj.create_pca_from_list(noise_arr=noise_male, name='MALE', save_dir='./pca_obj/_')

    '''------------------'''
    '''     ANGRY'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': None,
    #                   'race': None,
    #                   'age': None},
    #            name='ANGRY',
    #            folder_names=['orig_new_angry_diverse'],
    #            save_dir=save_dir,
    #            sample_limit=25e3)
    # '''     FEMALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': None,
    #                   'gender': [Gender_codes.FEMALE],
    #                   'race': None,
    #                   'age': None},
    #            name='FEMALE',
    #            folder_names=['orig_new_angry_diverse', 'orig_100K_normal'],
    #            save_dir=save_dir,
    #            sample_limit=15e3)
    # '''     MALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': None,
    #                   'gender': [Gender_codes.MALE],
    #                   'race': None,
    #                   'age': None},
    #            name='MALE',
    #            folder_names=['orig_new_angry_diverse', 'orig_100K_normal'],
    #            save_dir=save_dir,
    #            sample_limit=15e3)
    '''     ANGRY_FEMALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': [Gender_codes.FEMALE],
    #                   'race': None,
    #                   'age': None},
    #            name='ANGRY_FEMALE')
    '''     ANGRY_MALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': [Gender_codes.MALE],
    #                   'race': None,
    #                   'age': None},
    #            name='ANGRY_MALE')
    '''     ANGRY_BLACK'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': None,
    #                   'race': [Race_codes.BLACK,
    #                            Race_codes.INDIAN,
    #                            Race_codes.MID_EST],
    #                   'age': None},
    #            name='ANGRY_BLACK')
    '''     ANGRY_WHITE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': None,
    #                   'race': [Race_codes.WHITE],
    #                   'age': None},
    #            name='ANGRY_WHITE')
    '''     ANGRY_CHILD'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': None,
    #                   'race': None,
    #                   'age': [Age_codes.CHILD]},
    #            name='ANGRY_CHILD')

    '''     ANGRY_OLD'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.ANGER],
    #                   'gender': None,
    #                   'race': None,
    #                   'age': [Age_codes.OLD]},
    #            name='ANGRY_OLD')

    '''     HAPPY_MALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.HAPPY],
    #                   'gender': [Gender_codes.MALE],
    #                   'race': None,
    #                   'age': None},
    #            name='HAPPY_MALE')

    '''     HAPPY_FEMALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': [Expression_codes.HAPPY],
    #                   'gender': [Gender_codes.FEMALE],
    #                   'race': None,
    #                   'age': None},
    #            name='HAPPY_FEMALE')

    '''     FEMALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': None,
    #                   'gender': [Gender_codes.FEMALE],
    #                   'race': None,
    #                   'age': None},
    #            name='FEMALE')

    '''     MALE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': None,
    #                   'gender': [Gender_codes.MALE],
    #                   'race': None,
    #                   'age': None},
    #            name='MALE')
    '''     BLACK'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': None,
    #                   'gender': None,
    #                   'race': [Race_codes.BLACK],
    #                   'age': None},
    #            name='BLACK')
    '''     WHITE'''
    # create_pca(pca_accuracy=99,
    #            tasks={'fer': None,
    #                   'gender': None,
    #                   'race': [Race_codes.WHITE],
    #                   'age': None},
    #            name='WHITE')
    '''======================================'''
    # noise_path = f'{FolderStructures.prefix}zz_productin/' + orig_name + '/noise_vectors'
    # limit = 100
    # noise_af = []
    # for file in tqdm(os.listdir(noise_path)):
    #     if file.endswith('.npy'):
    #         noise_raw = np.load(os.path.join(noise_path, file))
    #         if np.max(noise_raw) <= 10: continue
    #         noise_raw_clip = np.clip(noise_raw, -10.0, +10.0)
    #         noise_af.append(noise_raw)
    #         noise_af.append(noise_raw_clip)
    #         limit -= 1
    #         if limit ==0:
    #             break

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # validation_jobs = ['noise_modification', 'synthesis', 'annotate', 'identity_comparison', 'semantic_score']
    validation_jobs = ['identity_comparison', 'semantic_score']

    '''facial attribute editing experiments'''
    # tasks = ['FEMALE', 'ANGRY', 'BLACK']
    # alphas = [3.5, 2.5, 2.0]  # identity = [96.68, 84.67, 79.61]
    # alphas = [2.5, 2.0, 1.5]  # identity= [99.66, 98.87, 99.14]
    # alphas = [4.0, 3.0, 1.0]  # identity= [92.83, 57.22, 99.97]

    tasks = ['OLD', 'OLD']
    alphas = [4.0, 5.0]  # identity= [3.5=> 96.30, ]

    source_prefix = '/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/'

    if 'noise_modification' in validation_jobs:
        for i, task in enumerate(tasks):
            target_prefix = '/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/facial_a_modified/' \
                            + task + '_' + str(alphas[i]) + '/'

            fae_obj = FacialAttributeEditingEval(source_image_path=source_prefix + 'images/',
                                                 source_noise_path=source_prefix + 'noise_vectors/',
                                                 source_expression_path=source_prefix + 'feature_vectors/',
                                                 source_race_path=source_prefix + 'race_extraction/',
                                                 source_age_path=source_prefix + 'age_extraction/',
                                                 source_gender_path=source_prefix + 'gender_extraction/',
                                                 target_image_path=target_prefix + 'images/',
                                                 target_noise_path=target_prefix + 'noise_vectors/',
                                                 target_expression_path=target_prefix + 'feature_vectors/',
                                                 target_race_path=target_prefix + 'race_extraction/',
                                                 target_age_path=target_prefix + 'age_extraction/',
                                                 target_gender_path=target_prefix + 'gender_extraction/',
                                                 )
            fae_obj.modify_noise(task_name=task, pca_accuracy=99, s_p=0.3, i_p=0.0, alpha=alphas[i])
    if 'synthesis' in validation_jobs:
        for i, task in enumerate(tasks):
            fae_obj = FacialAttributeEditingEval()
            target_prefix = '/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/facial_a_modified/'\
                            + task + '_' + str(alphas[i]) + '/'
            noises, names = fae_obj.get_modified_noises(noise_path=target_prefix + 'noise_vectors/')
            '''         generating images:              '''
            generate_with_noise(network_pkl=FolderStructures.styleGan_weight_path,
                                noises=noises,
                                names=names,
                                fer_detection=False,
                                truncation_psi=0.7,
                                noise_mode='const',  # 'const', 'random', 'none'],
                                outdir=target_prefix,
                                translate=parse_vec2('0,0'),
                                rotate=0,
                                class_idx=0)
            ''''''
    if 'annotate' in validation_jobs:
        annotation_jobs = ['fer', 'age', 'race', 'gender']
        for i, task in enumerate(tasks):
            folder_name = 'evaluation_set/facial_a_modified/' + task + '_' + str(alphas[i])
            if 'fer' in annotation_jobs:
                predict_fer_images(folder_name=folder_name)
            if 'age' in annotation_jobs:
                predict_age_images(folder_name=folder_name)
            if 'race' in annotation_jobs:
                predict_race_images(folder_name=folder_name)
            if 'gender' in annotation_jobs:
                predict_gender_images(folder_name=folder_name)

    if 'identity_comparison' in validation_jobs:
        fr_obj = FaceRecognition()
        source_image_path = source_prefix + 'images'
        for i, task in enumerate(tasks):
            target_image_path = '/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/facial_a_modified/' \
                                + task + '_' + str(alphas[i]) + '/images'
            match_score = fr_obj.compare_by_path(source_image_path, target_image_path)
            print(task + ' ' + str(match_score) + '%')

    if 'semantic_score' in validation_jobs:
        fae_obj = FacialAttributeEditingEval()
        for i, task in enumerate(tasks):
            s_path = source_prefix
            d_path = '/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/facial_a_modified/' \
                     + task + '_' + str(alphas[i]) + '/'
            to_female, to_angry_neutral, to_black, to_old = fae_obj.calculate_semantic_score(task, s_path, d_path)
            if task == 'FEMALE':
                to_female['alpha'] = alphas[i]
                with open('to_FEMALE' + str(alphas[i]) + '.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(to_female))
            if task == 'ANGRY':
                to_angry_neutral['alpha'] = alphas[i]
                with open('to_ANGRY' + str(alphas[i]) + '.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(to_angry_neutral))
            if task == 'BLACK':
                to_black['alpha'] = alphas[i]
                with open('to_BLACK' + str(alphas[i]) + '.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(to_black))
            if task == 'OLD':
                to_black['alpha'] = alphas[i]
                with open('to_OLD' + str(alphas[i]) + '.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(to_old))

    '''Generating semantic-based images'''

    '''         creating single-semantic noises     '''
    # noise_af = lin_obj.make_comp_semantic_noise(task_names=['ANGRY', 'FEMALE'],
    #                                             pca_accuracy=99, num=25,
    #                                             s_ps=[0.1, 0.5], i_ps=[0.1, 0.3])
    # noise_af = lin_obj.make_single_semantic_noise_disentangle(task_name='BLACK', pca_accuracy=99, num=150,
    #                                                           s_p=0.999, i_p=0.0, alpha=+2.0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # noise_af = lin_obj.make_single_semantic_noise(task_name='WHITE', pca_accuracy=99, num=50, s_p=0.999, i_p=0.0, alpha=+5.0)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='CHILD', pca_accuracy=99, num=50, s_p=0.999, i_p=0.0, alpha=+5.0)

    # noise_af = lin_obj.make_single_semantic_noise(task_name='OLD', pca_accuracy=99, num=50, s_p=0.999, i_p=0.0, alpha=+5.0)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='BLACK', pca_accuracy=99, num=150, s_p=0.999, i_p=0.0, alpha=+2.0)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='ANGRY', pca_accuracy=99, num=150, s_p=0.999, i_p=0.0, alpha=+3.0)

    # noise_af = lin_obj.make_single_semantic_noise(task_name='ANGRY', pca_accuracy=99, num=50, s_p=0.2, i_p=0.0, alpha=+3.0)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='FEMALE', pca_accuracy=99, num=50, s_p=0.3, i_p=0.0, alpha=1.0)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='OLD', pca_accuracy=99, num=50, s_p=0.8, i_p=0.0, alpha=2.5)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='BLACK', pca_accuracy=99, num=50, s_p=0.99, i_p=0.0, alpha=2.0)
    # noise_af = lin_obj.make_single_semantic_noise(task_name='CHILD', pca_accuracy=99, num=50, s_p=0.8, i_p=0.0, alpha=2.0)
    #
    '''semanti-based config'''
    # noise_af = lin_obj.make_compound_semantic_noise(
    #     data=[
    #             {'t_n': 'ANGRY', 'p_ac': 99, 's_p': 0.4, 'i_p': 0.0, 'alpha': 4.0},
    #             {'t_n': 'FEMALE', 'p_ac': 99, 's_p': 0.2, 'i_p': 0.0, 'alpha': 5.0}
    #           ],
    #     num=25
    # )

    # noise_af = lin_obj.make_compound_semantic_noise(
    #     data=[{'t_n': 'FEMALE', 'p_ac': 99, 's_p': 0.5, 'i_p': 0.0, 'alpha': 4.0},
    #           {'t_n': 'BLACK', 'p_ac': 99, 's_p': 0.5, 'i_p': 0.0, 'alpha': 2.0},
    #           ],
    #     num=25
    #     )

    # noise_A = lin_obj.make_single_semantic_noise(task_name='ANGRY', pca_accuracy=99, num=30, vec_percent=0.1)

    # noise = list(np.array(noise_ao) + np.array(noise_aw))
    # noise = list(np.mean([noise_ao, noise_ach], axis=0))

    '''         compound semantic'''
    # noise = lin_obj.make_compound_semantic_noise(data=[{'t_n': 'ANGRY', 'p_ac': 99, 'vec_p': 0.1},
    #                                                    {'t_n': 'FEMALE', 'p_ac': 99, 'vec_p': 0.5},
    #                                                    ],
    #                                              num=15,
    #                                              vec_percent=0.99  # DON't reduce this
    #                                              )
    #
    # noise = lin_obj.make_compound_semantic_noise(data=[{'t_n': 'ANGRY_BLACK', 'p_ac': 99, 'vec_p': 0.99},
    #                                                    {'t_n': 'FEMALE', 'p_ac': 99, 'vec_p': 0.99},
    #                                                    ],
    #                                              num=15,
    #                                              vec_percent=0.99  # DON't reduce this
    #                                              )

    '''======================================'''
    '''         generating images:              '''
    generate_with_noise(network_pkl=FolderStructures.styleGan_weight_path,
                        noises=noise_af,
                        fer_detection=False,
                        truncation_psi=0.7,
                        noise_mode='const',  # 'const', 'random', 'none'],
                        outdir=FolderStructures.prefix,
                        translate=parse_vec2('0,0'),
                        rotate=0,
                        class_idx=0)
    '''======================================'''

    # noise = asm.get_asm_svd(task_id=6, pca_accuracy=99, num=20, task='fer', alpha=1.0)

    # noise = asm.get_asm_b(task_id=6, pca_accuracy=99, num=20, task='fer', alpha=1.0)
    # noise = asm.get_asm_b(task_id=2, pca_accuracy=99, num=20, task='race', alpha=1.0)
    # noise = asm.get_asm_multiple(task_ids=[6, 2], pca_accuracy=99, num=20, tasks=['fer', 'race'], alpha=1.0)

    # noise = asm.get_asm(task_id=6, pca_accuracy=20, num=20, task='fer')

    # noise = asm.get_asm(task='gender', task_id=0, pca_accuracy=99, num=20)
    # noise = asm.get_asm(task='race', task_id=2, pca_accuracy=25, num=20)
    # noise = asm.get_asm(task='age', task_id=0, pca_accuracy=99, num=20)
    #

    #
    ''' creating 5k dataset'''
    fer_class = FER()
    # fer_class.create_total_cvs_raw(
    #     img_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/images/',
    #     fer_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/feature_vectors',
    #     race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/race_extraction/',
    #     age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/age_extraction/',
    #     gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/gender_extraction/',
    #     cvs_file='./angry_50k.csv')

    # fer_class.create_total_cvs_raw(
    #     img_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/images/',
    #     fer_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/feature_vectors',
    #     race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/race_extraction/',
    #     age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/age_extraction/',
    #     gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/gender_extraction/',
    #     cvs_file='./happy_100k.csv')

    # fer_class.create_total_cvs(img_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/images/',
    #                            fer_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/feature_vectors',
    #                            race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/race_extraction/',
    #                            age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/age_extraction/',
    #                            gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/gender_extraction/',
    #                            cvs_file='./angry.csv')

    # fer_class.create_total_cvs(img_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/images/',
    #                            fer_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/feature_vectors',
    #                            race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/race_extraction/',
    #                            age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/age_extraction/',
    #                            gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/gender_extraction/',
    #                            cvs_file='./happy.csv')
    # prefix = f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/00_production_item/'
    '''         copy happy'''
    # fer_class.copy_final_images(cvs_file=prefix + '0_final_happy.csv',
    #                             s_img_folder=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/images/',
    #                             d_img_folder=prefix + 'happy_images/',
    #                             )
    '''         copy Angry'''
    # fer_class.copy_final_images(cvs_file=prefix + '0_final_angry.csv',
    #                             s_img_folder=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/images/',
    #                             d_img_folder=prefix + 'angry_images/',
    #                             )
    # #
    '''filter with query'''
    # eval_tasks = ['csv_creation', 'copy', 'histogram']
    eval_tasks = []

    if 'csv_creation' in eval_tasks:
        fer_class.query_images(cvs_query_file='./happy_100k.csv',
                               prefix='/media/ali/extradata/styleGAN3_samples/v1/zz_productin/orig_100K_normal/',
                               query={
                                   'fer': [Expression_codes.HAPPY, Expression_codes.NEUTRAL, Expression_codes.ANGER],
                                   'gender': [Gender_codes.FEMALE, Gender_codes.MALE],
                                   'race': [Race_codes.BLACK, Race_codes.WHITE, Race_codes.INDIAN, Race_codes.MID_EST],
                                   'age': None},
                               final_csv='10k_evaluation.csv', number_of_samples=8764)  # 1236 , 8764
        fer_class.query_images(cvs_query_file='./angry_50k.csv',
                               prefix='/media/ali/extradata/styleGAN3_samples/v1/zz_productin/orig_50K_moreAngry/',
                               query={
                                   'fer': [Expression_codes.NEUTRAL, Expression_codes.ANGER],
                                   'gender': [Gender_codes.FEMALE, Gender_codes.MALE],
                                   'race': [Race_codes.BLACK, Race_codes.WHITE, Race_codes.INDIAN, Race_codes.MID_EST],
                                   'age': None},
                               final_csv='10k_evaluation.csv', number_of_samples=1236)  # 1236 , 8764
    '''         copy Evaluation Set'''
    if 'copy' in eval_tasks:
        fer_class.copy_all(cvs_file='10k_evaluation.csv',
                           s_img_folder='images/',
                           s_noise_path='noise_vectors/',
                           s_exp_path='feature_vectors/',
                           s_age_path='age_extraction/',
                           s_gender_path='gender_extraction/',
                           s_race_path='race_extraction/',
                           d_img_folder=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/images/',
                           d_noise_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/noise_vectors/',
                           d_exp_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/feature_vectors/',
                           d_age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/age_extraction/',
                           d_gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/gender_extraction/',
                           d_race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/race_extraction/'
                           )

    # final_csv='HAPPY_MALE_WHITE.csv'
    # final_csv='HAPPY_FEMALE_WHITE.csv'
    # final_csv='HAPPY_FEMALE_BLACK.csv'
    # )

    '''         make the histograms'''
    # image	expression	age	race gender
    if 'histogram' in eval_tasks:
        prefix = '/media/ali/extradata/styleGAN3_samples/v1/zz_productin/evaluation_set/'
        fer_class.create_histogram_csv_reduced(cvs_file=prefix + '10k_evaluation.csv', task='exp', file_name='10K_exp',
                                               f_index=1)
        fer_class.create_histogram_csv_reduced(cvs_file=prefix + '10k_evaluation.csv', task='gender',
                                               file_name='10K_gender', f_index=4)
        fer_class.create_histogram_csv_reduced(cvs_file=prefix + '10k_evaluation.csv', task='race',
                                               file_name='10K_skin_color', f_index=3)
        fer_class.create_histogram_csv_reduced(cvs_file=prefix + '10k_evaluation.csv', task='age', file_name='10K_age',
                                               f_index=2)

    # fer_class.create_histogram_csv(cvs_file=prefix + 'happy.csv', task='v', file_name='Happy_gender', f_index=4)
    # fer_class.create_histogram_csv(cvs_file=prefix + '0_final_angry.csv', task='age', file_name='0_A_age', f_index=2)
    # fer_class.create_histogram_csv(cvs_file=prefix + '0_final_angry.csv', task='race', file_name='0_A_race', f_index=3)
    # fer_class.create_histogram_csv(cvs_file=prefix + '0_final_happy.csv', task='age', file_name='0_H_age', f_index=2)
    # fer_class.create_histogram_csv(cvs_file=prefix + '0_final_happy.csv', task='race', file_name='0_H_race', f_index=3)

    '''====='''
    # # noises = fer_class.create_noise(h5_addresses=['./encoder.h5', './decoder.h5'], exp_id=6, num=50)
    # fer_ana = AnalyzeFer(exp_path=None, noise_path='')
    # noises = fer_ana.create_fer_noises(indices_path='./indices.npy', exp_stat_path='./p_stat.npy')
    #
    # generate_with_noise(network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #                     noises=noises,
    #                     truncation_psi=0.7,
    #                     noise_mode='const',  # 'const', 'random', 'none'],
    #                     outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #                     translate=parse_vec2('0,0'),
    #                     rotate=0,
    #                     class_idx=0)

    # noise = np.load(file=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.noise_vectors}/{str(0)}.npy')
    # exp = np.load(file=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.feature_vectors}/{str(0)}.npy')

    # analyze_images()

    # noise_1 = np.load(file=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.noise_vectors}/{str(79)}.npy')
    # noise_2 = np.load(file=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.noise_vectors}/{str(555)}.npy')
    # noise = np.mean([noise_1, noise_2], axis=0)
    #
    # generate_with_noise(network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #                     noise=noise,
    #                     truncation_psi=0.7,
    #                     noise_mode='const',  # 'const', 'random', 'none'],
    #                     outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #                     translate=parse_vec2('0,0'),
    #                     rotate=0,
    #                     class_idx=0)

    # generate_fixed(network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #                seeds=parse_range('1000-100000'),
    #                truncation_psi=0.7,
    #                noise_mode='const', #'const', 'random', 'none'],
    #                outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #                translate=parse_vec2('0,0'),
    #                rotate=0,
    #                class_idx=0)

# ----------------------------------------------------------------------------
