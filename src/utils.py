import numpy as np
import os
import cv2

import tensorflow as tf
from glob import glob
import string
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


# import math
class Image_data:

    def __init__(self, img_size, z_dim, dataset_path):
        self.img_size = img_size
        self.z_dim = z_dim
        self.dataset_path = dataset_path

    def image_processing(self, filename):
        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_size, self.img_size], antialias=True,
                              method=tf.image.ResizeMethod.BICUBIC)
        img = preprocess_fit_train_image(img)

        latent = tf.random.normal(shape=(self.z_dim,), dtype=tf.float32)

        return img, latent

    def preprocess(self):
        self.train_images = glob(os.path.join(self.dataset_path, '*.png')) + glob(
            os.path.join(self.dataset_path, '*.jpg'))


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images


def random_flip_left_right(images):
    s = tf.shape(images)
    mask = tf.random.uniform([1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [s[0], s[1], s[2]])  # [h, w, c]
    images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[1]))
    return images


def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    images = random_flip_left_right(images)
    images = tf.transpose(images, [2, 0, 1])

    return images


def preprocess_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [2, 0, 1])

    return images


def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [0, 2, 3, 1])
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images


def load_images(image_path, img_width, img_height, img_channel):
    # from PIL import Image
    if img_channel == 1:
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, dsize=(img_width, img_height))
    img = tf.image.resize(img, [img_height, img_width], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
    img = preprocess_image(img)

    if img_channel == 1:
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = np.expand_dims(img, axis=0)

    return img


def save_images(images, size, image_path):
    # size = [height, width]
    return imsave(postprocess_images(images), size, image_path)


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img


def str2bool(x):
    return x.lower() in ('true')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def automatic_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def multi_gpu_loss(x, global_batch_size):
    ndim = len(x.shape)
    no_batch_axis = list(range(1, ndim))
    x = tf.reduce_mean(x, axis=no_batch_axis)
    x = tf.reduce_sum(x) / global_batch_size

    return x


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return tf.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return tf.keras.initializers.VarianceScaling(scale=scale, mode='fan_avg', distribution='uniform')


@tf.function
def update_model_average(old_model, new_model, beta=0.995):
    def update_average(old, new, beta):
        if old is None:
            return new
        return old * beta + (1 - beta) * new

    for old_weight, new_weight in zip(old_model.weights, new_model.weights):
        assert old_weight.shape == new_weight.shape

        old_weight.assign(update_average(old_weight, new_weight, beta))

    return


class CosineAnnealingLR(LearningRateSchedule):
    def __init__(
            self,
            init_lr,
            epoch,
            start_steps,
            batch_size,
            dataset_len,
            eta_min=1e-5,
            name=None):
        super(CosineAnnealingLR, self).__init__()

        self.init_lr = self.cosine_annealing(init_lr, epoch, start_steps, batch_size, dataset_len, eta_min, name=name)
        self.epoch = epoch
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.eta_min = eta_min
        self.name = name

    def cosine_annealing(self, init_lr, epoch, step, batch_size, dataset_len, eta_min, name):
        with tf.name_scope(name or "CosineDecay"):
            init_lr = tf.convert_to_tensor(init_lr, name="initial_learning_rate")
            dtype = init_lr.dtype
            epoch = tf.cast(epoch, dtype)

            iter = step * batch_size
            step_to_epoch = tf.cast(iter, dtype) // dataset_len
            completed_fraction = step_to_epoch / epoch
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf.constant(np.pi, dtype=dtype) * completed_fraction))

            decayed = (1 - eta_min) * cosine_decayed + eta_min

            return tf.multiply(init_lr, decayed)

    def __call__(self, step):
        return self.cosine_annealing(self.init_lr, self.epoch, step, self.batch_size, self.dataset_len, self.eta_min, self.name)

    def get_config(self):
        return {
            "init_lr": self.init_lr,
            "epoch": self.epoch,
            "start_steps": self.start_steps,
            "batch_size": self.batch_size,
            "dataset_len": self.dataset_len,
            "eta_min": self.eta_min,
            "name": self.name
        }
