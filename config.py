from dataclasses import dataclass


class FolderStructures:
    prefix = '/media/ali/extradata/styleGAN3_samples/v1/'
    styleGan_weight_path = '/media/ali/extradata/styleGAN3_pkls/stylegan3-r-ffhq-1024x1024.pkl'
    images = 'images'
    noise_vectors = 'noise_vectors'
    feature_vectors = 'feature_vectors'

    interpolate_images = 'interpolate_images'
    interpolate_noise_vectors = 'interpolate_noise_vectors'
    interpolate_feature_vectors = 'interpolate_feature_vectors'


class PreTrainedModels:
    # prefix = '/media/data3/ali/styleGan/source/stylegan3/'
    prefix = '/home/ali/PycharmProjects/stylegan3/'

    styleGan_pkl = prefix + 'features/pkl/stylegan3-r-ffhq-1024x1024.pkl'
    exp_model = prefix + 'features/exp/AffectNet_6336.h5'

    age_model = prefix + 'features/age/age_net.caffemodel'
    age_proto = prefix + 'features/age/age_net.prototxt'

    gender_model = prefix + 'features/gender/gender.caffemodel'
    gender_proto = prefix + 'features/gender/gender.prototxt'

    race_model = prefix + 'features/race/race.h5'


class Race_codes:
    ASIAN: int = 0
    INDIAN: int = 1
    BLACK: int = 2
    WHITE: int = 3
    MID_EST: int = 4
    LATINO: int = 5


class Gender_codes:
    FEMALE: int = 0
    MALE: int = 1


class Age_codes:
    CHILD: int = 0
    YOUTH: int = 1
    MIDDLE: int = 2
    OLD: int = 3


class Expression_codes:
    NEUTRAL: int = 0
    HAPPY: int = 1
    SAD: int = 2
    SURPRISE: int = 3
    FEAR: int = 4
    DISGUST: int = 5
    ANGER: int = 6
