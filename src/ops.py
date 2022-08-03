import tensorflow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
import numpy as np
import math
from einops import rearrange

# cuda version
from cuda.upfirdn_2d import *

# no cuda version
# from cuda.upfirdn_2d_ref import *

from utils import default_init, contract_inner


class Conv2D(Layer):
    def __init__(self, fmaps, kernel, up=False, down=False, resample_kernel=(1, 3, 3, 1), use_bias=True):
        super(Conv2D, self).__init__()
        self.fmaps = fmaps
        self.kernel = kernel
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.use_bias = use_bias

    def build(self, input_shape):
        w_init = default_init()
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        self.w = tf.Variable(initial_value=w_init(shape=weight_shape), trainable=True, name='conv_w')

        if self.use_bias:
            self.b = tf.Variable(initial_value=tf.zeros([1, self.fmaps, 1, 1]), trainable=True, name='conv_b')

    def call(self, inputs, training=True):
        x = inputs
        w = self.w

        if self.up:
            x = upsample_conv_2d(x, w, self.kernel, self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, w, self.kernel, self.resample_kernel)
        else:
            x = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        if self.use_bias:
            x += self.b

        return x

class NIN(Layer):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super(NIN, self).__init__()
        self.init_value = default_init(scale=init_scale)
        self.in_dim = in_dim
        self.num_units = num_units

    def build(self, input_shape):
        self.w = tf.Variable(initial_value=self.init_value(shape=[self.in_dim, self.num_units]), trainable=True, name='nin_w')
        self.b = tf.Variable(initial_value=tf.zeros([self.num_units]), trainable=True, name='nin_b')

    def call(self, x, training=True):
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # [B, C, H, W] -> [B, H, W, C]
        x = contract_inner(x, self.w) + self.b
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B, H, W, C] -> [B, C, H, W]
        return x

class AdaptiveGroupNorm(Layer):
    def __init__(self, num_groups, in_channel):
        super(AdaptiveGroupNorm, self).__init__()

        self.norm = tfa.layers.GroupNormalization(groups=num_groups, axis=1, center=False, scale=False, epsilon=1e-6)

        self.gamma = nn.Dense(units=in_channel, use_bias=True)
        self.beta = nn.Dense(units=in_channel, use_bias=False)

    def call(self, x, style=None, training=True):
        gamma = rearrange(self.gamma(style), 'b c -> b c 1 1')
        beta = rearrange(self.beta(style), 'b c -> b c 1 1')

        x = self.norm(x, training=training)
        x = gamma * x + beta

        return x

class TimestepEmbedding(Layer):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super(TimestepEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.sinusoidal_pos_embed = SinusoidalPosEmb(dim=self.embedding_dim)

        self.mlp = Sequential([
            nn.Dense(units=hidden_dim),
            act,
            nn.Dense(units=output_dim)
        ])

    def call(self, t, training=True):
        t_emb = self.sinusoidal_pos_embed(t)
        t_emb = self.mlp(t_emb)
        return t_emb


class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        return emb


class Upsample(Layer):
    def __init__(self, fmaps, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super(Upsample, self).__init__()

        kernel_size = 3
        if not fir:
            if with_conv:
                self.conv = nn.Conv2D(filters=fmaps, kernel_size=kernel_size, strides=1, padding='SAME', data_format='channels_first')
            else:
                if with_conv:
                    self.conv = Conv2D(fmaps=fmaps, kernel=kernel_size, up=True, resample_kernel=fir_kernel, use_bias=True)


        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel

    def call(self, x, training=True):
        B, C, H, W = x.shape
        if not self.fir:
            x = tf.transpose(x, perm=[0, 2, 3, 1])
            h = tf.image.resize(x, [H * 2, W * 2], antialias=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            h = tf.transpose(h, perm=[0, 3, 1, 2])
            if self.with_conv:
                h = self.conv(h)
        else:
            if self.with_conv:
                h = self.conv(x)
            else:
                h = upsample_2d(x, self.fir_kernel, factor=2)

        return h

class Downsample(Layer):
    def __init__(self, fmaps, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super(Downsample, self).__init__()
        kernel_size = 3
        if not fir:
            if with_conv:
                self.conv = nn.Conv2D(filters=fmaps, kernel_size=kernel_size, strides=2, padding='VALID', data_format='channels_first')
        else:
            if with_conv:
                self.conv = Conv2D(fmaps=fmaps, kernel=kernel_size, down=True, resample_kernel=fir_kernel, use_bias=True)

        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.fmaps = fmaps

    def call(self, x, training=True):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = tf.pad(x, paddings=[[0, 0], [0, 0], [0, 1], [0, 1]])
                x = self.conv(x)
            else:
                x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID', data_format='NCHW')
        else:
            if self.with_conv:
                x = self.conv(x)
            else:
                x = downsample_2d(x, self.fir_kernel, factor=2)

        return x

class PixelNorm(Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def call(self, x, training=True):
        x = x / tf.sqrt(tf.reduce_mean(x ** 2, axis=1, keepdims=True) + 1e-8)

        return x

class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)


""" Diffusion Coefficient """
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - tf.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = tf.gather(input, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = tf.reshape(out, shape=reshape)

    return out


def get_time_schedule(n_timestep):
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float32)
    t = t / n_timestep
    t = t * (1. - eps_small) + eps_small
    return t


def get_sigma_schedule(n_timestep=4, beta_min=0.1, beta_max=20.0, use_geometric=False):
    t = get_time_schedule(n_timestep)

    if use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = tf.convert_to_tensor(1e-8, dtype=tf.float32)
    betas = tf.concat([first[None], betas], axis=0)
    betas = tf.cast(betas, dtype=tf.float32)
    sigmas = betas ** 0.5
    a_s = tf.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, config):
        n_timestep = config['timesteps'] # 4
        beta_min = config['beta_min'] # 0.1
        beta_max = config['beta_max'] # 20
        use_geometric = config['use_geometric'] # False
        self.sigmas, self.a_s, _ = get_sigma_schedule(n_timestep, beta_min, beta_max, use_geometric)
        self.a_s_cum = tf.math.cumprod(self.a_s)
        self.sigmas_cum = tf.sqrt(1 - self.a_s_cum ** 2)
        # self.a_s_prev = tf.pad(self.a_s[:-1], paddings=[[1, 0]], constant_values=1)
        # tf.concat([tf.convert_to_tensor([1.0], dtype=tf.float32), a_s[:-1]], axis=0)


class Posterior_Coefficients():
    def __init__(self, config):
        n_timestep = config['timesteps'] # 4
        beta_min = config['beta_min'] # 0.1
        beta_max = config['beta_max'] # 20
        use_geometric = config['use_geometric'] # False
        _, _, self.betas = get_sigma_schedule(n_timestep, beta_min, beta_max, use_geometric)

        # we don't need the zeros
        self.betas = tf.cast(self.betas, dtype=tf.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas) # alpha_hat
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], paddings=[[1, 0]], constant_values=1) # alpha_hat (t-1)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) # beta_hat

        # self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = tf.math.rsqrt(self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * tf.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        posterior_clipped = tf.clip_by_value(self.posterior_variance, clip_value_min=1e-20, clip_value_max=tf.reduce_max(self.posterior_variance))
        self.posterior_log_variance_clipped = tf.math.log(posterior_clipped)

class MinibatchStd(Layer):
    def __init__(self, stddev_group, stddev_feat):
        super(MinibatchStd, self).__init__()
        self.stddev_group = stddev_group
        self.stddev_feat = stddev_feat

    def call(self, x, training=True):

        batch, channel, height, width = x.shape
        group = min(batch, self.stddev_group)
        stddev = tf.reshape(x, [group, -1, self.stddev_feat, channel // self.stddev_feat, height, width])
        stddev = tf.sqrt(tf.math.reduce_variance(stddev, axis=0) + 1e-8)
        stddev = tf.squeeze(tf.reduce_mean(stddev, axis=[2, 3, 4], keepdims=True), axis=2)
        stddev = tf.tile(stddev, multiples=[group, 1, height, width])
        x = tf.concat([x, stddev], axis=1)

        return x