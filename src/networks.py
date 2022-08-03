from layers import *
import functools

class Generator(Model):
    """ NCSN++ model """
    def __init__(self, config, fmaps, z_emb_dim):
        super(Generator, self).__init__()

        self.act = SiLU()
        self.fmaps = fmaps

        self.z_emb_dim = z_emb_dim
        self.ch_mult = config['ch_mult']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolutions = config['attn_resolutions']
        self.dropout = config['dropout']
        self.num_resolutions = len(self.ch_mult)
        self.all_resolutions = [config['img_size'] // (2 ** i) for i in range(self.num_resolutions)]

        self.conditional = config['conditional']  # noise-conditional
        self.fir = config['fir']
        self.fir_kernel = config['fir_kernel']
        self.skip_rescale = config['skip_rescale']
        self.n_mlp = config['n_mlp']
        self.init_scale = 0.0

        # timestep embedding
        self.sinusoidal_pos_embed = SinusoidalPosEmb(dim=self.fmaps)
        self.t_mlp_layers = Sequential([
            nn.Dense(units=self.fmaps * 4, kernel_initializer=default_init()),
            SiLU(),
            nn.Dense(units=self.fmaps * 4, kernel_initializer=default_init())
        ])

        # z mapping block
        mapping_layers = [PixelNorm(),
                          nn.Dense(units=self.z_emb_dim),
                          self.act]
        for _ in range(self.n_mlp):
            mapping_layers.append(nn.Dense(units=self.z_emb_dim))
            mapping_layers.append(self.act)

        self.mapping_layers = Sequential(mapping_layers)

        # initial block
        DownsampleBlock = functools.partial(Downsample, with_conv=True, fir=self.fir, fir_kernel=self.fir_kernel)
        AttnBlock = functools.partial(DDPM_AttnBlock, skip_rescale=self.skip_rescale, init_scale=self.init_scale)
        ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                        act=self.act,
                                        dropout=self.dropout,
                                        fir=self.fir,
                                        fir_kernel=self.fir_kernel,
                                        init_scale=self.init_scale,
                                        skip_rescale=self.skip_rescale,
                                        t_emb_dim=self.fmaps * 4)
        # Downsampling block
        down_layers = []
        down_layers.append(nn.Conv2D(filters=self.fmaps, kernel_size=3, strides=1, padding='SAME', data_format='channels_first'))
        hs_c = [self.fmaps]

        in_ch = self.fmaps
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                out_ch = self.fmaps * self.ch_mult[i_level]
                down_layers.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if self.all_resolutions[i_level] in self.attn_resolutions:
                    down_layers.append(AttnBlock(fmaps=in_ch))
                hs_c.append(in_ch)

            if i_level != self.num_resolutions - 1:
                down_layers.append(ResnetBlock(down=True, in_ch=in_ch))
                down_layers.append(DownsampleBlock(fmaps=in_ch))

                hs_c.append(in_ch)

        self.down_layers = down_layers

        in_ch = hs_c[-1]
        self.middle_layers = [
            ResnetBlock(in_ch=in_ch),
            AttnBlock(fmaps=in_ch),
            ResnetBlock(in_ch=in_ch)
        ]

        up_layers = []
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                out_ch = self.fmaps * self.ch_mult[i_level]
                up_layers.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if self.all_resolutions[i_level] in self.attn_resolutions:
                up_layers.append(AttnBlock(fmaps=in_ch))

            if i_level != 0:
                up_layers.append(ResnetBlock(in_ch=in_ch, up=True))

        up_layers.append(tfa.layers.GroupNormalization(groups=min(in_ch // 4, 32), axis=1, epsilon=1e-6))
        up_layers.append(nn.Conv2D(filters=3, kernel_size=3, strides=1, padding='SAME',
                                   data_format='channels_first', kernel_initializer=default_init(self.init_scale)))
        self.up_layers = up_layers

    def call(self, x, t=None, z=None, training=True, **kwargs):
        # timestep/noise_level embedding; only for continuous training
        z_emb = self.mapping_layers(z)

        # Sinusoidal positional embeddings.
        t_emb = self.sinusoidal_pos_embed(t)
        if self.conditional:
            t_emb = self.t_mlp_layers(t_emb)
        else:
            t_emb = None

        # Downsampling block
        x_init = x
        m_idx = 0

        hs = [self.down_layers[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = self.down_layers[m_idx](hs[-1], t_emb, z_emb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = self.down_layers[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = self.down_layers[m_idx](hs[-1], t_emb, z_emb)
                m_idx += 1

                x_init = self.down_layers[m_idx](x_init)
                m_idx += 1
                if self.skip_rescale:
                    x_init = (x_init + h) / np.sqrt(2.)
                else:
                    x_init = x_init + h
                h = x_init

                hs.append(h)

        # Middle block
        h = hs[-1]
        h = self.middle_layers[0](h, t_emb, z_emb)
        h = self.middle_layers[1](h)
        h = self.middle_layers[2](h, t_emb, z_emb)

        # Upsampling block
        m_idx = 0
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up_layers[m_idx](tf.concat([h, hs.pop()], axis=1), t_emb, z_emb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = self.up_layers[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = self.up_layers[m_idx](h, t_emb, z_emb)
                m_idx += 1

        h = self.act(self.up_layers[m_idx](h))
        m_idx += 1
        h = self.up_layers[m_idx](h)
        m_idx += 1

        return tf.tanh(h)

class Discriminator(Model):
    """A time-dependent discriminator for large images."""
    def __init__(self, fmaps, t_emb_dim=128):
        super(Discriminator, self).__init__()
        # Gaussian random feature embedding layer for time
        self.act = nn.LeakyReLU(0.2)

        self.t_embed = TimestepEmbedding(embedding_dim=t_emb_dim, hidden_dim=t_emb_dim, output_dim=t_emb_dim, act=self.act)

        self.conv0 = nn.Conv2D(filters=fmaps * 2, kernel_size=1, strides=1, data_format='channels_first')
        self.conv1 = DownConvBlock(fmaps=fmaps * 4, downsample=True, act=self.act)

        self.convs = []
        for _ in range(5):
            self.convs.append(
                DownConvBlock(fmaps=fmaps * 8, downsample=True, act=self.act)
            )

        self.conv_last = nn.Conv2D(filters=fmaps * 8, kernel_size=3, strides=1, padding='SAME', data_format='channels_first')
        self.dense_last = nn.Dense(units=1)

        self.minibatch_std = MinibatchStd(stddev_group=4, stddev_feat=1)

    def call(self, x, t=None, x_t=None, training=None, **kwargs):
        t_embed = self.act(self.t_embed(t))
        x = tf.concat([x, x_t], axis=1)

        x = self.conv0(x)
        x = self.conv1(x, t_embed)

        for down_conv_block in self.convs :
            x = down_conv_block(x, t_embed)

        x = self.minibatch_std(x)

        x = self.conv_last(x)
        x = self.act(x)

        x = tf.reduce_sum(tf.reshape(x, shape=[x.shape[0], x.shape[1], -1]), axis=2)
        x = self.dense_last(x)

        return x