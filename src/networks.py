from layers import *
from functools import partial
from tqdm import tqdm

class Unet(Model):
    def __init__(self,
                 dim=64,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 resnet_block_groups=8,
                 learned_variance=False,
                 sinusoidal_cond_mlp=True
                 ):
        super(Unet, self).__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp

        if sinusoidal_cond_mlp:
            self.time_mlp = Sequential([
                SinusoidalPosEmb(dim),
                nn.Dense(units=time_dim),
                GELU(),
                nn.Dense(units=time_dim)
            ])
        else:
            self.time_mlp = MLP(time_dim)

        # layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity()
            ])

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity()
            ])

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = Sequential([
            block_klass(dim, dim),
            nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
        ])

    def call(self, x, time=None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x

class GaussianDiffusion(Model):
    def __init__(self, image_size, timesteps=1000, objective='ddpm', eta=1.0, beta_schedule='cosine'):
        super(GaussianDiffusion, self).__init__()

        self.image_size = image_size
        self.timesteps = timesteps
        self.objective = objective

        if beta_schedule == 'linear':
            self.beta = linear_beta_schedule(self.timesteps)
        elif beta_schedule == 'cosine':
            self.beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        self.alpha = 1 - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        self.alpha_hat_prev = tf.pad(self.alpha_hat[:-1], paddings=[[1, 0]], constant_values=1)

        if self.objective == 'ddim':
            self.eta = 0.0
        elif self.objective == 'ddpm':
            self.eta = 1.0
        else: # general form
            self.eta = eta

    def sample_timesteps(self, n):
        return tf.random.uniform(shape=[n], minval=0, maxval=self.timesteps, dtype=tf.int32)

    def noise_images(self, x, t): # forward process q
        sqrt_alpha_hat = tf.sqrt(extract(self.alpha_hat, t))
        sqrt_one_minus_alpha_hat = tf.sqrt(1 - extract(self.alpha_hat, t))

        eps = tf.random.normal(shape=x.shape)

        x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps

        return x, eps

    def sample(self, model, n, start_x=None, start_t=None): # reverse process p
        if start_x is None:
            x = tf.random.normal(shape=[n, self.image_size, self.image_size, 3])
        else:
            x = start_x

        for i in tqdm(reversed(range(1, self.timesteps if start_t is None else start_t)), desc='sampling loop time step', total=self.timesteps):
            t = tf.ones(n, dtype=tf.int32) * i
            predicted_noise = model(x, t, training=False)

            alpha = extract(self.alpha, t)
            alpha_hat = extract(self.alpha_hat, t)
            beta = extract(self.beta, t)

            alpha_hat_prev = extract(self.alpha_hat_prev, t)
            beta_hat = beta * (1 - alpha_hat_prev) / (1 - alpha_hat) # similar to beta

            if i > 1:
                noise = tf.random.normal(shape=x.shape)
            else: # last step
                noise = tf.zeros_like(x)

            if self.objective == 'ddpm':
                direction_point = 1 / tf.sqrt(alpha) * (x - (beta / (tf.sqrt(1 - alpha_hat))) * predicted_noise) # mu
                random_noise = beta_hat * noise # stddev
                x = direction_point + random_noise

            elif self.objective == 'ddim':
                sigma = 0.0
                predict_x0 = alpha_hat_prev * (x - tf.sqrt(1 - alpha_hat) * predicted_noise) / tf.sqrt(alpha_hat)
                direction_point = tf.sqrt(1 - alpha_hat_prev - tf.square(sigma)) * predicted_noise
                random_noise = sigma * noise

                x = predict_x0 + direction_point + random_noise

            else: # general form
                sigma = self.eta * tf.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat)) * tf.sqrt(1 - (alpha_hat / alpha_hat_prev))
                predict_x0 = alpha_hat_prev * (x - tf.sqrt(1 - alpha_hat) * predicted_noise) / tf.sqrt(alpha_hat)
                direction_point = tf.sqrt(1 - alpha_hat_prev - tf.square(sigma)) * predicted_noise
                random_noise = sigma * noise

                x = predict_x0 + direction_point + random_noise

        return x

    def sample_from_timestep(self, model, image, timesteps):
        denoised_images = []
        noised_images = []

        for t in timesteps:
            noised_image, _ = self.noise_images(image, [t])
            denoised_image = self.sample(model, n=noised_image.shape[0], start_x=noised_image, start_t=t)
            denoised_images.append(denoised_image)
            noised_images.append(noised_image)

        x = tf.concat(noised_images + denoised_images, axis=0)
        return x
