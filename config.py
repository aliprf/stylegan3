class FolderStructures:
    prefix = 'media/data3/ali/styleGan/samples/v1/'

    images = 'images'
    noise_vectors = 'noise_vectors'
    feature_vectors = 'feature_vectors'

    interpolate_images = 'interpolate_images'
    interpolate_noise_vectors = 'interpolate_noise_vectors'
    interpolate_feature_vectors = 'interpolate_feature_vectors'

class PreTrainedModels:
    prefix = '/media/data3/ali/styleGan/source/stylegan3/'

    styleGan_pkl = prefix +'features/pkl/stylegan3-r-ffhq-1024x1024.pkl'
    exp_model = prefix + 'features/exp/AffectNet_6336.h5'

    age_model = prefix + 'features/age/age_net.caffemodel'
    age_proto = prefix + 'features/age/age_net.prototxt'

    gender_model = prefix + 'features/gender/gender.caffemodel'
    gender_proto = prefix + 'features/gender/gender.prototxt'

    race_model = prefix + 'features/race/race.h5'





