from utils import *
from networks import *
import time
from tensorflow.python.data.experimental import AUTOTUNE

automatic_gpu_usage()

class Diffusion():
    def __init__(self, t_params, strategy):
        super(Diffusion, self).__init__()
        self.model_name = 'Diffusion'
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
        self.objective = t_params['objective']
        self.eta = t_params['eta']
        self.beta_schedule = t_params['beta_schedule']

        """ Training parameters """
        self.ema_decay = t_params['ema_decay']

        self.batch_size = t_params['batch_size']
        self.each_batch_size = t_params['batch_size'] // t_params['NUM_GPUS']
        self.iteration = t_params['iteration']
        self.learning_rate = t_params['learning_rate']

        """ Print parameters """
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.img_size = t_params['img_size']
        self.print_freq = t_params['print_freq']
        self.save_freq = t_params['save_freq']
        self.log_template = 'step [{}/{}]: elapsed: {:.2f}s, loss: {:.3f}'

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

        print(self.dataset_path)
        print()

        """ Print """
        physical_gpus = tf.config.experimental.list_physical_devices('GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        print("Each batch size : ", self.each_batch_size)
        print("Global batch size : ", self.batch_size)
        print("Target image size : ", self.img_size)
        print("Print frequency : ", self.print_freq)
        print("Save frequency : ", self.save_freq)
        print("TF Version :", tf.__version__)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""
            img_class = Image_data(self.img_size,  self.dataset_path)
            img_class.preprocess()

            dataset_num = len(img_class.train_images)
            print("Dataset number : ", dataset_num)
            print()

            dataset_slice = tf.data.Dataset.from_tensor_slices(img_class.train_images)
            dataset_iter = dataset_slice.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
            dataset_iter = dataset_iter.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
            dataset_iter = dataset_iter.prefetch(buffer_size=AUTOTUNE)
            dataset_iter = self.strategy.experimental_distribute_dataset(dataset_iter)
            self.dataset_iter = iter(dataset_iter)


            """ Network """
            self.ema = EMA(self.ema_decay)
            self.unet = Unet()
            self.unet_ema = Unet()
            self.diffusion = GaussianDiffusion(self.img_size, self.timesteps, self.objective, self.eta, self.beta_schedule)


            """ Finalize model (build) """
            test_images = np.ones([1, self.img_size, self.img_size, 3])
            test_t = self.diffusion.sample_timesteps(n=test_images.shape[0])
            _ = self.unet(test_images, test_t)
            _ = self.unet_ema(test_images, test_t)
            self.unet_ema.set_weights(self.unet.get_weights())


            """ Optimizer """
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, )

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(unet=self.unet,
                                            unet_ema=self.unet_ema,
                                            diffusion=self.diffusion,
                                            optimizer=self.optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            """ Test """
            """ Network """
            self.unet_ema = Unet()
            self.diffusion = GaussianDiffusion(self.img_size, self.timesteps, self.objective, self.eta, self.beta_schedule)


            """ Finalize model (build) """
            test_images = np.ones([1, self.img_size, self.img_size, 3])
            test_t = self.diffusion.sample_timesteps(n=test_images.shape[0])
            _ = self.unet_ema(test_images, test_t)

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(unet_ema=self.unet_ema, diffusion=self.diffusion)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    def train_step(self, real_images):
        with tf.GradientTape() as tape:
            t = self.diffusion.sample_timesteps(n=real_images.shape[0])
            x_t, noise = self.diffusion.noise_images(real_images, t)

            predicted_noise = self.unet(x_t, t)

            loss = tf.square(noise - predicted_noise)
            loss = multi_gpu_loss(loss, global_batch_size=self.batch_size)

        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))

        return loss


    """ Distribute Train """
    @tf.function
    def distribute_train_step(self, real_images):
        loss = self.strategy.run(self.train_step, args=[real_images])

        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

        return loss


    def train(self):
        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # start training
        print('max_steps: {}'.format(self.iteration))
        losses = {'loss': 0.0}
        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            x_real = next(self.dataset_iter)

            if idx == 0:
                params = self.unet.count_params()
                print("network parameters : ", format(params, ','))

            # update
            loss = self.distribute_train_step(x_real)
            losses['loss'] = np.float64(loss)

            # update g_clone
            self.ema.update_model_average(self.unet_ema, self.unet)

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', losses['loss'], step=idx)

            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:
                total_num_samples = min(self.n_samples, self.batch_size)
                partial_size = int(np.floor(np.sqrt(total_num_samples)))

                sampled_images = self.diffusion.sample(self.unet_ema, n=partial_size*partial_size)
                save_images(images=sampled_images[:partial_size * partial_size, :, :, :],
                            size=[partial_size, partial_size],
                            image_path='./{}/sampled_{:06d}.png'.format(self.sample_dir, idx + 1))

                test_t = [50, 100, 150, 200, 300, 600, 700, 999]
                denoised_images = self.diffusion.sample_from_timestep(self.unet_ema, tf.expand_dims(x_real[0], axis=0), t)
                save_images(images=denoised_images,
                            size=[2, len(test_t)],
                            image_path='./{}/denoised_{:06d}.png'.format(self.sample_dir, idx + 1))

            elapsed = time.time() - iter_start_time
            print(self.log_template.format(idx, self.iteration, elapsed, losses['loss']))
        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

        print("Total train time: %4.4f" % (time.time() - start_time))

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)