import tensorflow as tf
from utils import adjust_dynamic_range
from ops import extract
import numpy as np
import scipy
import pickle
from tqdm import tqdm

class Inception_V3(tf.keras.Model):
    def __init__(self, name='Inception_V3'):
        super(Inception_V3, self).__init__(name=name)

        self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        self.inception_v3.trainable = False

    def call(self, x, training=False, mask=None):
        x = self.inception_v3(x, training=training)

        return x

def torch_normalization(x):
    x /= 255.

    r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=x)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x = tf.concat(axis=-1, values=[
        (r - mean[0]) / std[0],
        (g - mean[1]) / std[1],
        (b - mean[2]) / std[2]
    ])

    return x

def inception_processing(filename):
    x = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
    img = tf.image.resize(img, [256, 256], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
    img = tf.image.resize(img, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

    img = torch_normalization(img)
    return img

def calculate_FID(generator, inception_model,
                  strategy, img_slice, dataset_name, num_samples,
                  shape, timesteps, z_dim,
                  posterior_mean_coef1, posterior_mean_coef2, posterior_log_variance_clipped,
                  real_cache, real_mu=None, real_cov=None):

    batch_size = shape[0]
    x_init = tf.random.normal(shape=shape)

    @tf.function
    def p_sample(x_0, x_t, t):
        x_t_shape = x_t.shape
        mean = extract(posterior_mean_coef1, t, x_t_shape) * x_0 + extract(posterior_mean_coef2, t, x_t_shape) * x_t
        log_var_clipped = extract(posterior_log_variance_clipped, t, x_t_shape) # beta_hat

        noise = tf.random.normal(shape=x_t_shape)
        nonzero_mask = (1 - tf.cast((t == 0), tf.float32))

        x_sample = mean + nonzero_mask[:, None, None, None] * tf.exp(0.5 * log_var_clipped) * noise

        return x_sample

    @tf.function
    def sample_from_model():
        x = x_init

        for i in reversed(range(timesteps)):
            t = tf.ones(x.shape[0], dtype=tf.int32) * i

            latent_z = tf.random.normal(shape=[x.shape[0], z_dim])
            x_0 = generator(x, t, latent_z, training=False)
            x_new = p_sample(x_0, x, t, posterior_mean_coef1, posterior_mean_coef2, posterior_log_variance_clipped)
            x = x_new

        return x

    @tf.function
    def gen_samples_feats():
        # run networks
        fake_img = sample_from_model()
        fake_img = adjust_dynamic_range(fake_img, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.float32)
        fake_img = tf.transpose(fake_img, [0, 2, 3, 1])
        fake_img = tf.image.resize(fake_img, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

        fake_img = torch_normalization(fake_img)

        feats = inception_model(fake_img)

        return feats

    @tf.function
    def get_inception_features(img):
        feats = inception_model(img)
        return feats

    @tf.function
    def get_real_features(img):
        feats = strategy.run(get_inception_features, args=(img))
        feats = tf.concat(strategy.experimental_local_results(feats), axis=0)

        return feats

    @tf.function
    def get_fake_features():

        feats = strategy.run(gen_samples_feats, args=())
        feats = tf.concat(strategy.experimental_local_results(feats), axis=0)

        return feats

    if not real_cache:
        real_feats = tf.zeros([0, 2048])

        for img in img_slice:
            feats = get_real_features(img)
            real_feats = tf.concat([real_feats, feats], axis=0)
            print('real feats:', np.shape(real_feats)[0])

        real_mu = np.mean(real_feats, axis=0)
        real_cov = np.cov(real_feats, rowvar=False)

        with open('{}_mu_cov.pickle'.format(dataset_name), 'wb') as f:
            pickle.dump((real_mu, real_cov), f, protocol=pickle.HIGHEST_PROTOCOL)

        print('{} real pickle save !!!'.format(dataset_name))

        del real_feats

    fake_feats = tf.zeros([0, 2048])
    for _ in tqdm(range(0, num_samples, batch_size)):

        feats = get_fake_features()

        fake_feats = tf.concat([fake_feats, feats], axis=0)

    fake_mu = np.mean(fake_feats, axis=0)
    fake_cov = np.cov(fake_feats, rowvar=False)
    del fake_feats

    # Calculate FID.
    m = np.square(fake_mu - real_mu).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(fake_cov, real_cov), disp=False)  # pylint: disable=no-member
    dist = m + np.trace(fake_cov + real_cov - 2 * s)

    return dist