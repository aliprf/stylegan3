# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from fer import FER

from config import FolderStructures
from image_utility import ImageUtilities
from analyze import Analyze
import random


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
    fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/AffectNet_6336.h5', GPU=True)

    # Generate images.
    f = open(f'/media/ali/extradata/styleGAN3_samples/v1/annotation.txt', "w")
    jj = 0
    exps_str = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

    jj = 0
    while jj < 500:
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
        exp, exp_vec = fer_class.recognize_fer(npy_img=npy_img)
        if exp == 'Happy' or exp == 'Surprise': continue
        if exp != 'Anger' and np.argmax(exp_vec[0]) < 0.7: continue
        if exp == 'Neutral' and np.argmax(exp_vec[0]) > 0.4: continue
        if exp == 'Disgust' and np.argmax(exp_vec[0]) > 0.5: continue
        if exp == 'Sad' and np.argmax(exp_vec[0]) > 0.5: continue

        # resize image to 512 * 512 * 3
        resized_npy_img = img_util.resize_image(npy_img=npy_img, w=512, h=512, ch=3)
        # save image
        img_util.save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.interpolate_images,
                            save_name=str(jj))
        np.save(file=f'{outdir}{FolderStructures.interpolate_feature_vectors}/exp_{str(jj)}',
                arr=np.round(exp_vec, decimals=3))
        np.save(file=f'{outdir}{FolderStructures.interpolate_noise_vectors}/{str(jj)}', arr=noise)

        f.write(
            str(jj) + ' : ' + exps_str[np.argmax(exp_vec)] + '===> ' + ''.join(str(e) for e in list(exp_vec)) + '\n\r')
        jj += 1


def generate_with_noise(network_pkl: str,
                        noises,
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
    # fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/AffectNet_6336.h5')
    fer_class = FER(h5_address='/media/ali/extradata/Ad-Corre-weights/RafDB_8696.h5')

    # Generate images.
    f = open(f'/media/ali/extradata/styleGAN3_samples/v1/annotation.txt', "w")
    jj = 0
    exps_str = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
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
        # extract expression
        exp, exp_vec = fer_class.recognize_fer(npy_img=npy_img)
        # resize image to 512 * 512 * 3
        resized_npy_img = img_util.resize_image(npy_img=npy_img, w=512, h=512, ch=3)
        # save image
        img_util.save_image(npy_img=resized_npy_img, save_path=outdir + FolderStructures.interpolate_images,
                            save_name=str(jj))
        np.save(file=f'{outdir}{FolderStructures.interpolate_feature_vectors}/exp_{str(jj)}',
                arr=np.round(exp_vec, decimals=3))
        np.save(file=f'{outdir}{FolderStructures.interpolate_noise_vectors}/{str(jj)}', arr=noise)

        f.write(
            str(jj) + ' : ' + exps_str[np.argmax(exp_vec)] + '===> ' + ''.join(str(e) for e in list(exp_vec)) + '\n\r')
        jj += 1


def analyze_images():
    # analyser = Analyze(exp_path=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.interpolate_feature_vectors}',
    #                    noise_path=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.interpolate_noise_vectors}')
    #
    analyser = Analyze(exp_path=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.feature_vectors}',
                       noise_path=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.noise_vectors}')
    #
    '''creating categories'''
    # analyser.create_categories(exp_path=f'/media/ali/extradata/styleGAN3_samples/v1/{FolderStructures.interpolate_feature_vectors}',
    #                            out_dir=f'/media/ali/extradata/styleGAN3_samples/v1/expression_by_category/')

    '''creating 1-d histogram'''
    analyser.calculate_exp_histogram_2d()
    analyser.plot_histogram_2d(file_name='histogram2d-50k.jpg')
    '''creating 2-d histogram'''
    analyser.calculate_exp_histogram()
    analyser.plot_histogram(file_name='histogram_int-50k.jpg')

    return 0

    '''interpolate_hypersphere'''
    # new_noises = analyser.interpolate_hypersphere(exp_cat_path=f'/media/ali/extradata/styleGAN3_samples/v1/expression_by_category/Anger.txt')
    #
    # generate_with_noise(network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #                     noises=new_noises,
    #                     truncation_psi=0.7,
    #                     noise_mode='const',  # 'const', 'random', 'none'],
    #                     outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #                     translate=parse_vec2('0,0'),
    #                     rotate=0,
    #                     class_idx=0)
    #
    # return 0

    inter_functions = analyser.interpolate(
        exp_cat_path=f'/media/ali/extradata/styleGAN3_samples/v1/expression_by_category/Anger.txt')

    generate_with_interpolation_function(
        network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
        inter_functions=inter_functions,
        truncation_psi=0.7,
        noise_mode='const',  # 'const', 'random', 'none'],
        outdir='/media/ali/extradata/styleGAN3_samples/v1/',
        translate=parse_vec2('0,0'),
        rotate=0,
        class_idx=0)

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


if __name__ == "__main__":
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

    analyze_images()

    # generate_fixed(network_pkl="/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl",
    #                seeds=parse_range('1000-100000'),
    #                truncation_psi=0.7,
    #                noise_mode='const', #'const', 'random', 'none'],
    #                outdir='/media/ali/extradata/styleGAN3_samples/v1/',
    #                translate=parse_vec2('0,0'),
    #                rotate=0,
    #                class_idx=0)

# ----------------------------------------------------------------------------
