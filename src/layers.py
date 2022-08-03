from ops import *

class ResnetBlockBigGAN(Layer):
    def __init__(self, act, in_ch, out_ch=None, t_emb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.0):
        super(ResnetBlockBigGAN, self).__init__()

        out_ch = out_ch if out_ch else in_ch
        self.group_norm_0 = AdaptiveGroupNorm(num_groups=min(in_ch // 4, 32), in_channel=in_ch)

        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.conv_0 = nn.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='SAME', data_format='channels_first')
        if t_emb_dim is not None:
            self.dense = nn.Dense(units=out_ch, kernel_initializer=default_init())
        self.group_norm_1 = AdaptiveGroupNorm(num_groups=min(out_ch // 4, 32), in_channel=out_ch)
        self.dropout = nn.Dropout(rate=dropout)
        self.conv_1 = nn.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=default_init(scale=init_scale), data_format='channels_first')
        if in_ch != out_ch or up or down :
            self.conv_2 = nn.Conv2D(filters=out_ch, kernel_size=1, strides=1, data_format='channels_first')

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def call(self, x, t_emb=None, z_emb=None, training=True):
        h = self.act(self.group_norm_0(x, z_emb))

        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)

        h = self.conv_0(h)
        # Add bias to each feature map conditioned on the time embedding

        if t_emb is not None :
            h += self.dense(self.act(t_emb))[:, :, None, None]

        h = self.act(self.group_norm_1(h, z_emb))
        h = self.dropout(h, training=training)
        h = self.conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class DDPM_AttnBlock(Layer):
    """Channel-wise self-attention block. Modified from DDPM."""
    def __init__(self, fmaps, skip_rescale=False, init_scale=0.0):
        super(DDPM_AttnBlock, self).__init__()
        self.group_norm = tfa.layers.GroupNormalization(groups=min(fmaps // 4, 32), axis=1, epsilon=1e-6)

        self.nins = [
            NIN(fmaps, fmaps),
            NIN(fmaps, fmaps),
            NIN(fmaps, fmaps),
            NIN(fmaps, fmaps, init_scale=init_scale)
        ]
        self.skip_rescale = skip_rescale

    def call(self, x, training=True):
        B, C, H, W = x.shape

        h = self.group_norm(x)
        q = self.nins[0](h)
        k = self.nins[1](h)
        v = self.nins[2](h)

        w = tf.einsum('b c h w, b c i j -> b h w i j', q, k) * (int(C) ** (-0.5))
        w = tf.reshape(w, [B, H, W, H*W])
        w = tf.nn.softmax(w, axis=-1)
        w = tf.reshape(w, [B, H, W, H, W])
        h = tf.einsum('b h w i j, b c i j -> b c h w', w, v)
        h = self.nins[3](h)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class DownConvBlock(Layer):
    def __init__(self, fmaps, kernel_size=3, downsample=False, act=nn.LeakyReLU(0.2), fir_kernel=(1, 3, 3, 1)):
        super(DownConvBlock, self).__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Conv2D(filters=fmaps, kernel_size=kernel_size, strides=1, padding='SAME', data_format='channels_first')
        self.conv2 = nn.Conv2D(filters=fmaps, kernel_size=kernel_size, strides=1, padding='SAME', kernel_initializer=default_init(scale=0.0), data_format='channels_first')

        self.dense = nn.Dense(units=fmaps)

        self.act = act

        self.skip = nn.Conv2D(filters=fmaps, kernel_size=1, strides=1, use_bias=False, data_format='channels_first')

    def call(self, x, t_emb=None, training=True):
        out = self.act(x)
        out = self.conv1(out)
        out += self.dense(t_emb)[..., None, None]

        out = self.act(out)

        if self.downsample:
            out = downsample_2d(out, self.fir_kernel, factor=2)
            x = downsample_2d(x, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(x)
        out = (out + skip) / np.sqrt(2)

        return out