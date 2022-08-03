from utils import *
from networks import *
import time
from tensorflow.python.data.experimental import AUTOTUNE
from evaluation.frechet_inception_distance import *
automatic_gpu_usage()

class DDGAN():
    def __init__(self, t_params, strategy):
        super(DDGAN, self).__init__()
        self.model_name = 'Denoising_Diffusion_GANs'
        self.config = t_params
        self.phase = t_params['phase']
        self.checkpoint_dir = t_params['checkpoint_dir']
        self.result_dir = t_params['result_dir']
        self.log_dir = t_params['log_dir']
        self.sample_dir = t_params['sample_dir']
        self.dataset_name = t_params['dataset']
        self.strategy = strategy
        self.NUM_GPUS = t_params['NUM_GPUS']

        """ Network parameters """
        self.timesteps = t_params['timesteps']
        self.g_nf = t_params['g_nf']
        self.d_nf = t_params['d_nf']
        self.z_dim = t_params['z_dim']
        self.z_emb_dim = t_params['z_emb_dim']
        self.t_emb_dim = t_params['t_emb_dim']
        self.diff_coeff = Diffusion_Coefficients(self.config)
        self.pos_coeff = Posterior_Coefficients(self.config)

        """ Training parameters """
        self.g_lr = t_params['g_lr']
        self.d_lr = t_params['d_lr']
        self.beta1 = t_params['beta1']
        self.beta2 = t_params['beta2']
        self.r1_gamma = t_params['r1_gamma']
        self.lazy_reg = t_params['lazy_reg']
        self.ema_decay = t_params['ema_decay']

        self.epoch = t_params['epoch']
        self.batch_size = t_params['batch_size']
        self.each_batch_size = t_params['batch_size'] // t_params['NUM_GPUS']

        """ Print parameters """
        self.img_size = t_params['img_size']
        self.cal_fid = t_params['cal_fid']
        self.print_freq = t_params['print_freq']
        self.save_freq = t_params['save_freq']
        self.log_template = 'step [{}/{}]: epoch [{}/{}]: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, fid: {:.2f}, best_fid: {:.2f}, best_fid_iter: {}'

        """ Directory """
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)

        """ Dataset """
        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)

        if self.cal_fid:
            print("{} pickle loading...".format(self.dataset_name))
            if os.path.exists('{}_mu_cov.pickle'.format(self.dataset_name)):
                with open('{}_mu_cov.pickle'.format(self.dataset_name), 'rb') as f:
                    self.real_mu, self.real_cov = pickle.load(f)
                self.real_cache = True
                print("Pickle load success !!!")
            else:
                print("Pickle load fail !!!")
                self.real_cache = False

        """ Print """
        self.physical_gpus = tf.config.experimental.list_physical_devices('GPU')
        self.logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Dataset Iterator """
            img_class = Image_data(self.img_size, self.z_dim, self.dataset_path)
            img_class.preprocess()

            self.dataset_num = len(img_class.train_images)
            print("Dataset number : ", self.dataset_num)
            print()
            self.iteration = self.epoch * self.dataset_num // self.batch_size

            dataset_slice = tf.data.Dataset.from_tensor_slices(img_class.train_images)
            dataset_iter = dataset_slice.shuffle(buffer_size=self.dataset_num, reshuffle_each_iteration=True).repeat()
            dataset_iter = dataset_iter.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
            dataset_iter = dataset_iter.prefetch(buffer_size=AUTOTUNE)
            dataset_iter = self.strategy.experimental_distribute_dataset(dataset_iter)
            self.dataset_iter = iter(dataset_iter)

            if self.cal_fid:
                """ FID dataset iterator """
                img_slice = dataset_slice.shuffle(buffer_size=self.dataset_num, reshuffle_each_iteration=True, seed=777)
                img_slice = img_slice.map(map_func=inception_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=False)
                img_slice = img_slice.prefetch(buffer_size=AUTOTUNE)
                self.fid_img_slice = self.strategy.experimental_distribute_dataset(img_slice)

                self.inception_model = Inception_V3()
                inception_images = np.ones((1, 299, 299, 3), dtype=np.float32)
                _ = self.inception_model(inception_images)



            """ Network """
            self.generator = Generator(self.config, fmaps=self.g_nf, z_emb_dim=self.z_emb_dim)
            self.generator_ema = Generator(self.config, fmaps=self.g_nf, z_emb_dim=self.z_emb_dim)
            self.discriminator = Discriminator(fmaps=2*self.d_nf, t_emb_dim=self.t_emb_dim)


            """ Finalize model (build) """
            t = np.random.randint(0, self.timesteps, size=[1,])
            latent_z = np.random.normal(size=[1, self.z_dim])
            images = np.ones([1, 3, self.img_size, self.img_size])

            _ = self.generator(images, t, latent_z)
            _, = self.generator_ema(images, t, latent_z)
            _ = self.discriminator(images, t, images)

            self.generator_ema.set_weights(self.generator.get_weights())


            """ Optimizer """
            self.g_lr_schedule = CosineAnnealingLR(self.g_lr, epoch=self.epoch, start_steps=0,
                                                   batch_size=self.batch_size, dataset_len=self.dataset_num)
            self.d_lr_schedule = CosineAnnealingLR(self.d_lr, epoch=self.epoch, start_steps=0,
                                                   batch_size=self.batch_size, dataset_len=self.dataset_num)
            self.g_optimizer = tf.keras.optimizers.Adam(self.g_lr_schedule, self.beta1, self.beta2)
            self.d_optimizer = tf.keras.optimizers.Adam(self.d_lr_schedule, self.beta1, self.beta2)

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(generator=self.generator,
                                            generator_ema=self.generator_ema,
                                            discriminator=self.discriminator,
                                            g_optimizer=self.g_optimizer,
                                            d_optimizer=self.d_optimizer,
                                            g_lr_schedule=self.g_lr_schedule,
                                            d_lr_schedule=self.d_lr_schedule)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])

                # self.g_lr_schedule = CosineAnnealingLR(self.g_lr, epoch=self.epoch, start_steps=self.start_iteration,
                #                                        batch_size=self.batch_size, dataset_len=self.dataset_num)
                # self.d_lr_schedule = CosineAnnealingLR(self.d_lr, epoch=self.epoch, start_steps=self.start_iteration,
                #                                        batch_size=self.batch_size, dataset_len=self.dataset_num)
                #
                # self.g_optimizer = tf.keras.optimizers.Adam(self.g_lr_schedule, self.beta1, self.beta2)
                # self.d_optimizer = tf.keras.optimizers.Adam(self.d_lr_schedule, self.beta1, self.beta2)

                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            """ Test """
            """ Network """
            self.generator_ema = Generator(self.config, fmaps=self.g_nf, z_emb_dim=self.z_emb_dim)

            """ Finalize model (build) """
            t = np.random.randint(0, self.timesteps, size=[1,])
            latent_z = np.random.normal(size=[1, self.z_dim])
            images = np.ones([1, self.img_size, self.img_size, 3])

            _, = self.generator_ema(images, t, latent_z)

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(generator_ema=self.generator_ema)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    # @tf.function
    def q_sample(self, x_start, t):
        x_start_shape = x_start.shape
        noise_t = tf.random.normal(shape=x_start_shape)
        noise_t_1 = tf.random.normal(shape=x_start_shape)
        x_t = extract(self.diff_coeff.a_s_cum, t, x_start_shape) * x_start + extract(self.diff_coeff.sigmas_cum, t, x_start_shape) * noise_t
        x_t_1 = extract(self.diff_coeff.a_s, t+1, x_start_shape) * x_t + extract(self.diff_coeff.sigmas, t+1, x_start_shape) * noise_t_1 # x_t+1

        return x_t, x_t_1

    # @tf.function
    def p_sample(self, x_0, x_t, t):
        x_t_shape = x_t.shape
        mean = extract(self.pos_coeff.posterior_mean_coef1, t, x_t_shape) * x_0 + extract(self.pos_coeff.posterior_mean_coef2, t, x_t_shape) * x_t
        log_var_clipped = extract(self.pos_coeff.posterior_log_variance_clipped, t, x_t_shape) # beta_hat
        # var = extract(pos_coeff.posterior_variance, t, x_t_shape)

        noise = tf.random.normal(shape=x_t_shape)
        nonzero_mask = (1 - tf.cast((t == 0), tf.float32))

        x_sample = mean + nonzero_mask[:, None, None, None] * tf.exp(0.5 * log_var_clipped) * noise

        return x_sample

    @tf.function
    def sample_from_model(self, generator, n_time, z_dim):
        x = tf.random.normal(shape=[self.each_batch_size, 3, self.img_size, self.img_size])

        for i in reversed(range(n_time)):
            t = tf.ones(x.shape[0], dtype=tf.int32) * i

            latent_z = tf.random.normal(shape=[x.shape[0], z_dim])
            x_0 = generator(x, t, latent_z, training=False)
            x_new = self.p_sample(x_0, x, t)
            x = x_new

        return x


    def d_train_step(self, real_images, latent_z):
        with tf.GradientTape() as tape:
            # sample t
            t = tf.random.uniform(shape=[real_images.shape[0]], minval=0, maxval=self.timesteps, dtype=tf.int32)
            x_t, x_t_1 = self.q_sample(x_start=real_images, t=t)
            x_t_1_sg = tf.stop_gradient(x_t_1)

            # train with real
            real_logit = self.discriminator(x_t, t, x_t_1_sg)
            real_loss = tf.math.softplus(-real_logit)

            # train with fake
            x_0_predict = self.generator(x_t_1_sg, t, latent_z)
            x_sample = self.p_sample(x_0=x_0_predict, x_t=x_t_1, t=t)

            fake_logit = self.discriminator(x_sample, t, x_t_1_sg)
            fake_loss = tf.math.softplus(fake_logit)

            d_loss = real_loss + fake_loss
            d_loss = multi_gpu_loss(d_loss, global_batch_size=self.batch_size)

        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return d_loss

    def d_reg_train_step(self, real_images, latent_z):
        with tf.GradientTape() as tape:
            # sample t
            t = tf.random.uniform(shape=[real_images.shape[0]], minval=0, maxval=self.timesteps, dtype=tf.int32)
            x_t, x_t_1 = self.q_sample(x_start=real_images, t=t)
            x_t_1_sg = tf.stop_gradient(x_t_1)

            # train with real
            real_logit = self.discriminator(x_t, t, x_t_1_sg)
            real_loss = tf.math.softplus(-real_logit)

            # train with fake
            x_0_predict = self.generator(x_t_1_sg, t, latent_z)
            x_sample = self.p_sample(x_0=x_0_predict, x_t=x_t_1, t=t)

            fake_logit = self.discriminator(x_sample, t, x_t_1_sg)
            fake_loss = tf.math.softplus(fake_logit)

            # simple GP
            with tf.GradientTape() as p_tape:
                p_tape.watch(x_t)
                real_logit_ptape = tf.reduce_sum(self.discriminator(x_t, t, x_t_1_sg))

            real_grad = p_tape.gradient(real_logit_ptape, x_t)
            real_grad_norm = tf.norm(tf.reshape(real_grad, shape=[real_grad.shape[0], -1]), ord=2, axis=1)
            real_grad_penalty = tf.square(real_grad_norm)

            real_grad_penalty = self.r1_gamma / 2 * real_grad_penalty

            d_loss = real_loss + fake_loss + real_grad_penalty
            d_loss = multi_gpu_loss(d_loss, global_batch_size=self.batch_size)

        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return d_loss

    def g_train_step(self, real_images, latent_z):
        with tf.GradientTape() as tape:
            # sample t
            t = tf.random.uniform(shape=[real_images.shape[0]], minval=0, maxval=self.timesteps, dtype=tf.int32)
            x_t, x_t_1 = self.q_sample(x_start=real_images, t=t)
            x_t_1_sg = tf.stop_gradient(x_t_1)

            # train with fake
            x_0_predict = self.generator(x_t_1_sg, t, latent_z)
            x_sample = self.p_sample(x_0=x_0_predict, x_t=x_t_1, t=t) # x_t sample

            fake_logit = self.discriminator(x_sample, t, x_t_1_sg)
            fake_loss = tf.math.softplus(-fake_logit)

            g_loss = fake_loss
            g_loss = multi_gpu_loss(g_loss, global_batch_size=self.batch_size)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return g_loss

    """ Distribute Train """
    @tf.function
    def distribute_d_train_step(self, real_images, latent_z):
        loss = self.strategy.run(self.d_train_step, args=[real_images, latent_z])
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

        return loss

    @tf.function
    def distribute_d_reg_train_step(self, real_images, latent_z):
        loss = self.strategy.run(self.d_reg_train_step, args=[real_images, latent_z])
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

        return loss

    @tf.function
    def distribute_g_train_step(self, real_images, latent_z):
        loss = self.strategy.run(self.g_train_step, args=[real_images, latent_z])
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

        return loss

    def train(self):
        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # start training
        print()
        print(self.dataset_path)
        print(len(self.physical_gpus), "Physical GPUs,", len(self.logical_gpus), "Logical GPUs")
        print("Each batch size : ", self.each_batch_size)
        print("Global batch size : ", self.batch_size)
        print("Target image size : ", self.img_size)
        print("Print frequency : ", self.print_freq)
        print("Save frequency : ", self.save_freq)
        print("TF Version :", tf.__version__)
        print('max_steps: {}'.format(self.iteration))
        print()
        losses = {'d_loss': 0.0, 'g_loss': 0.0,
                  'fid': 0.0, 'best_fid': 0.0, 'best_fid_iter': 0}
        fid = 0
        best_fid = 1000
        best_fid_iter = 0
        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            x_real, latent_z = next(self.dataset_iter)

            if idx == 0:
                g_params = self.generator.count_params()
                d_params = self.discriminator.count_params()
                params = g_params + d_params
                print("G network parameters : ", format(g_params, ','))
                print("D network parameters : ", format(d_params, ','))
                print("Total network parameters : ", format(params, ','))

            # update discriminator
            # At first time, each function takes 1~2 min to make the graph.
            if (idx + 1) % self.lazy_reg == 0:
                d_loss = self.distribute_d_reg_train_step(x_real, latent_z)
            else:
                d_loss = self.distribute_d_train_step(x_real, latent_z)
            losses['d_loss'] = np.float64(d_loss)

            # update generator
            g_loss = self.distribute_g_train_step(x_real, latent_z)
            losses['g_loss'] = np.float64(g_loss)

            # update g_clone
            update_model_average(self.generator_ema, self.generator, self.ema_decay)

            # calculate FID
            if self.cal_fid:
                if np.mod(idx, self.save_freq) == 0 or idx == self.iteration-1 :
                    fid = calculate_FID(self.generator_ema, self.inception_model,
                                        self.strategy, self.fid_img_slice, self.dataset_name, self.dataset_num,
                                        shape=[self.batch_size, 3, self.img_size, self.img_size], timesteps=self.timesteps, z_dim=self.z_dim,
                                        posterior_mean_coef1=self.pos_coeff.posterior_mean_coef1, posterior_mean_coef2=self.pos_coeff.posterior_mean_coef2,
                                        posterior_log_variance_clipped=self.pos_coeff.posterior_log_variance_clipped,
                                        real_cache=self.real_cache, real_mu=self.real_mu, real_cov=self.real_cov)
                    if fid < best_fid:
                        print("BEST FID UPDATED")
                        best_fid = fid
                        best_fid_iter = idx
                        self.manager.save(checkpoint_number=idx)

                        losses['best_fid'] = np.float64(best_fid)
                        losses['best_fid_iter'] = np.float64(best_fid_iter)
                    losses['fid'] = np.float64(fid)
            else:
                if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                    self.manager.save(checkpoint_number=idx)


            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('d_loss', losses['d_loss'], step=idx)
                tf.summary.scalar('g_loss', losses['g_loss'], step=idx)


            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:
                total_num_samples = self.each_batch_size
                partial_size = int(np.floor(np.sqrt(total_num_samples)))

                sampled_images = self.sample_from_model(self.generator_ema, self.timesteps, self.z_dim)
                save_images(images=sampled_images[:partial_size * partial_size, :, :, :],
                            size=[partial_size, partial_size],
                            image_path='./{}/sampled_{:06d}.png'.format(self.sample_dir, idx + 1))


            elapsed = time.time() - iter_start_time
            curr_epoch = idx * self.batch_size // self.dataset_num
            print(self.log_template.format(idx, self.iteration, curr_epoch, self.epoch, elapsed,
                                           losses['d_loss'], losses['g_loss'],losses['fid'], losses['best_fid'], losses['best_fid_iter']))
        # save model for final step
        fid = calculate_FID(self.generator_ema, self.inception_model,
                            self.strategy, self.fid_img_slice, self.dataset_name, self.dataset_num,
                            shape=[self.batch_size, 3, self.img_size, self.img_size],
                            timesteps=self.timesteps, z_dim=self.z_dim,
                            posterior_mean_coef1=self.pos_coeff.posterior_mean_coef1,
                            posterior_mean_coef2=self.pos_coeff.posterior_mean_coef2,
                            posterior_log_variance_clipped=self.pos_coeff.posterior_log_variance_clipped,
                            real_cache=self.real_cache, real_mu=self.real_mu, real_cov=self.real_cov)
        self.manager.save(checkpoint_number=self.iteration)
        print("Total train time: %4.4f" % (time.time() - start_time))
        print("FID: ".format(fid))

    def test(self):
        total_num_samples = self.each_batch_size
        partial_size = int(np.floor(np.sqrt(total_num_samples)))

        sampled_images = self.sample_from_model(self.generator_ema, self.timesteps, self.z_dim)
        save_images(images=sampled_images[:partial_size * partial_size, :, :, :],
                    size=[partial_size, partial_size],
                    image_path='./{}/test_sampled.png'.format(self.result_dir))

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)