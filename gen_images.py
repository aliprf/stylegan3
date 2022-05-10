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
                        fer_detection:bool,
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

    '''loading FER model'''
    img_util = ImageUtilities()
    if fer_detection:
        fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/AffectNet_6336.h5')
    # fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/RafDB_8696.h5')

    # Generate images.
    jj = 10000
    for noise in noises:
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
        # save image
        img_util.save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.images,
                            save_name=str(jj))
        # extract expression
        if fer_detection:
            exp, exp_vec = fer_class.recognize_fer(npy_img=npy_img)
            np.save(file=f'{outdir}{FolderStructures.feature_vectors}/exp_{str(jj)}',
                    arr=np.round(exp_vec, decimals=3))
        np.save(file=f'{outdir}{FolderStructures.noise_vectors}/{str(jj)}', arr=noise)
        jj += 1


"============================= histogram ========================================"


def create_histogram_fer():
    analyser = AnalyzeFer(noise_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/noise_vectors',
                          exp_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors')
    '''creating 1-d histogram'''
    analyser.calculate_exp_histogram()
    analyser.plot_histogram(file_name='exp_histogram.jpg')
    return 0


def create_histogram_age():
    age_class = AgeExtraction(model_path='./features/age/age_net.caffemodel',
                              proto_path='./features/age/age_net.prototxt')
    age_class.create_histogram(age_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/age_extraction/')


def create_histogram_gender():
    gender_class = GenderExtraction(model_path='./features/gender/gender.caffemodel',
                                    proto_path='./features/gender/gender.prototxt')
    gender_class.create_histogram(
        gender_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/gender_extraction/')


def create_histogram_race():
    race_class = RaceExtraction(model_path='./features/race/race.h5', GPU=False)
    race_class.create_histogram(race_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/race_extraction/')


"=============================  predicting features ========================================"


def predict_race_images():
    """ ['asian','indian','black','white','middle eastern','latino hispanic'] """
    race_class = RaceExtraction(model_path='./features/race/race.h5')
    race_class.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/images/',
                                out_dir=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/race_extraction/',
                                csv_file_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/race.csv')
    race_class.create_histogram(race_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/race_extraction/')


def predict_age_images():
    """4 age ranges (0-15) as child, (16-32) as young, (33-53) as adult, and (54-100) as old"""
    age_class = AgeExtraction(model_path='./features/age/age_net.caffemodel',
                              proto_path='./features/age/age_net.prototxt')
    age_class.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/images/',
                               out_dir=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/age_extraction/',
                               csv_file_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/age.csv')
    age_class.create_histogram(age_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/age_extraction/')


def predict_gender_images():
    """man or woman"""
    gender_class = GenderExtraction(model_path='./features/gender/gender.caffemodel',
                                    proto_path='./features/gender/gender.prototxt')
    gender_class.predict_and_save(img_dir=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/images/',
                                  out_dir=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/gender_extraction/',
                                  csv_file_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/gender.csv')
    gender_class.create_histogram(
        gender_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/gender_extraction/')


def predict_fer_images(_fer_class):
    analyser = AnalyzeFer(noise_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/noise_vectors',
                          exp_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors')
    '''creating csv'''
    analyser.create_csv(img_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/images',
                        fer_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors',
                        csv_file_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/fer.csv')


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


"=============================== create_pca ==================================="


def create_pca(pca_accuracy, tasks, name):
    lin_obj = LinearProjection()
    lin_obj.create_pca_from_npy(
        noise_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/noise_vectors',
        fer_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/feature_vectors',
        race_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/race_extraction',
        gender_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/gender_extraction',
        age_path=f'{FolderStructures.prefix}zz_productin/new_angry_diverse/age_extraction',
        tasks=tasks,
        name=name,
        pca_accuracy=pca_accuracy)


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


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python gen_images.py

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

    '''analyzing Age, Gender and Race'''
    # predict_fer_images(_fer_class=None)
    # predict_age_images()
    # predict_gender_images()
    # predict_race_images()
    '''======================================'''

    '''Histogram Age, Gender and Race'''
    # create_histogram_fer()
    # create_histogram_age()
    # create_histogram_gender()
    # create_histogram_race()
    '''======================================'''

    '''creating PCA'''
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
    '''======================================'''

    '''Generating semantic-based images'''
    lin_obj = LinearProjection()

    '''         creating single-semantic noises     '''
    # noise_af = lin_obj.make_single_semantic_noise(task_name='ANGRY_FEMALE', pca_accuracy=99, num=10, vec_percent=0.5)
    # noise_am = lin_obj.make_single_semantic_noise(task_name='ANGRY_MALE', pca_accuracy=99, num=10, vec_percent=0.25)
    # noise_aw = lin_obj.make_single_semantic_noise(task_name='ANGRY_WHITE', pca_accuracy=99, num=10, vec_percent=0.25)
    # noise_ab = lin_obj.make_single_semantic_noise(task_name='ANGRY_BLACK', pca_accuracy=99, num=10, vec_percent=0.55)

    # noise = list(np.mean([noise_0, noise_1], axis=0))
    '''         generating images:              '''
    # generate_with_noise(network_pkl=FolderStructures.styleGan_weight_path,
    #                     noises=noise_ab,
    #                     fer_detection=False,
    #                     truncation_psi=0.7,
    #                     noise_mode='const',  # 'const', 'random', 'none'],
    #                     outdir=FolderStructures.prefix,
    #                     translate=parse_vec2('0,0'),
    #                     rotate=0,
    #                     class_idx=0)
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
    # fer_class = FER()
    # fer_class.create_total_cvs(img_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/images/',
    #                            fer_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/feature_vectors',
    #                            race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/race_extraction/',
    #                            age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/age_extraction/',
    #                            gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/100K_normal/gender_extraction/',
    #                            cvs_file='./happy.csv')
    # #
    # fer_class.create_total_cvs(img_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/images/',
    #                            fer_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/feature_vectors',
    #                            race_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/race_extraction/',
    #                            age_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/age_extraction/',
    #                            gender_path=f'/media/ali/extradata/styleGAN3_samples/v1/zz_productin/50K_moreAngry/gender_extraction/',
    #                            cvs_file='./angry.csv')

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
