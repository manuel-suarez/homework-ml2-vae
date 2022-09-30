import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from zipfile import ZipFile
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

AUTOTUNE = tf.data.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 100
IMG_WIDTH = 64
IMG_HEIGHT = 64

OUTPUT_CHANNELS = 1
R_LOSS_FACTOR = 100
L_LOSS_FACTOR = 1000

# Dimensión de la imagen de entrada (el polinomio) utilizado en el entrenamiento y pruebas
INPUT_DIM     = (IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)
# Dimensión del espacio latente
LATENT_DIM    = 2048
EPOCHS        = 30
INITIAL_EPOCH = 0

use_batch_norm  = True
use_dropout     = True
local_path = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats'
train_dogs_files = glob(os.path.join(local_path, 'train', 'dog.*.jpg'))
train_cats_files = glob(os.path.join(local_path, 'train', 'cat.*.jpg'))
train_dogs_files.sort()
train_cats_files.sort()
train_dogs_files = np.array(train_dogs_files)
train_cats_files = np.array(train_cats_files)
print(len(train_dogs_files), len(train_cats_files))

for dog_file, cat_file in zip(train_dogs_files[:5], train_cats_files[:5]):
  print(dog_file, cat_file)

BUFFER_SIZE      = len(train_cats_files)
steps_per_epoch  = BUFFER_SIZE // BATCH_SIZE
print('num image files : ', BUFFER_SIZE)
print('steps per epoch : ', steps_per_epoch )


def read_and_decode(file):
    '''
    Lee, decodifica y redimensiona la imagen.
    Aplica aumentación
    '''
    # Lectura y decodificación
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    # Normalization
    # img = img/127.5 - 1
    # img = img / 255.0
    # Conversión a escala de grises
    img = tf.image.rgb_to_grayscale(img)
    # Redimensionamiento
    img = tf.image.resize(img, INPUT_DIM[:2],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def load_images(file1, file2, flip=False):
    '''
    Lee el conjunto de imágenes de entrada y las redimensiona al tamaño especificado

    Aumentación: Flip horizontal aleatorio, sincronizado
    '''
    img1 = read_and_decode(file1)
    img2 = read_and_decode(file2)
    # Aumentación (el flip debe aplicarse simultáneamente a las 3 imagenes)
    if flip and tf.random.uniform(()) > 0.5:
        img1 = tf.image.flip_left_right(img1)
        img2 = tf.image.flip_left_right(img2)

    return img1, img2

dogs_imgs = []
cats_imgs = []
# Cargamos 3 imagenes
for i in range(3):
    dog_img, cat_img = load_images(train_dogs_files[i], train_cats_files[i])
    dogs_imgs.append(dog_img)
    cats_imgs.append(cat_img)
# Verificamos la forma de las imagenes cargadas
print(dogs_imgs[0].shape, cats_imgs[0].shape)


def display_images(fname, dogs_imgs=None, cats_imgs=None, rows=3, offset=0):
    '''
    Despliega conjunto de imágenes izquierda y derecha junto a la disparidad
    '''
    # plt.figure(figsize=(20,rows*2.5))
    fig, ax = plt.subplots(rows, 2, figsize=(8, rows * 2.5))
    for i in range(rows):
        ax[i, 0].imshow((dogs_imgs[i + offset] + 1) / 2)
        ax[i, 0].set_title('Left')
        ax[i, 1].imshow((cats_imgs[i + offset] + 1) / 2)
        ax[i, 1].set_title('Right')

    plt.tight_layout()
    plt.savefig(fname)

display_images("figure1.png", dogs_imgs, cats_imgs, rows=3)

train_dogs = tf.data.Dataset.list_files(train_dogs_files, shuffle=False)
train_cats = tf.data.Dataset.list_files(train_cats_files, shuffle=False)
train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

sample_dog, sample_cat = next(iter(train_dataset))
print(sample_dog[0].shape, sample_cat[0].shape)


class Sampler(keras.Model):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, latent_dim, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.model = self.sampler_model()
        self.built = True

    def get_config(self):
        config = super(Sampler, self).get_config()
        config.update({"units": self.units})
        return config

    def sampler_model(self):
        '''
        input_dim is a vector in the latent (codified) space
        '''
        input_data = layers.Input(shape=self.latent_dim)
        z_mean = Dense(self.latent_dim, name="z_mean")(input_data)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(input_data)

        self.batch = tf.shape(z_mean)[0]
        self.dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        model = keras.Model(input_data, [z, z_mean, z_log_var])
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


class Encoder(keras.Model):
    def __init__(self, input_dim, output_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):
        '''
        '''
        super(Encoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.model = self.encoder_model()
        self.built = True

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"units": self.units})
        return config

    def encoder_model(self):
        '''
        '''
        encoder_input = layers.Input(shape=self.input_dim, name='encoder')
        x = layers.Rescaling(1.0 / 255)(encoder_input)

        for i in range(self.n_layers_encoder):
            x = Conv2D(filters=self.encoder_conv_filters[i],
                       kernel_size=self.encoder_conv_kernel_size[i],
                       strides=self.encoder_conv_strides[i],
                       padding='same',
                       name='encoder_conv_' + str(i), )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        self.last_conv_size = x.shape[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.output_dim)(x)
        model = keras.Model(encoder_input, encoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


class Decoder(keras.Model):
    def __init__(self, input_dim, input_conv_dim,
                 decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):

        '''
        '''
        super(Decoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.input_conv_dim = input_conv_dim

        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.model = self.decoder_model()
        self.built = True

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"units": self.units})
        return config

    def decoder_model(self):
        '''
        '''
        decoder_input = layers.Input(shape=self.input_dim, name='decoder')
        x = Dense(np.prod(self.input_conv_dim))(decoder_input)
        x = Reshape(self.input_conv_dim)(x)

        for i in range(self.n_layers_decoder):
            x = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                kernel_size=self.decoder_conv_t_kernel_size[i],
                                strides=self.decoder_conv_t_strides[i],
                                padding='same',
                                name='decoder_conv_t_' + str(i))(x)
            if i < self.n_layers_decoder - 1:

                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)
        decoder_output = x
        model = keras.Model(decoder_input, decoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)

def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

class VAE(keras.Model):
    def __init__(self, r_loss_factor=1, summary=False, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.r_loss_factor = r_loss_factor

        # Architecture
        self.input_dim = INPUT_DIM
        self.latent_dim = LATENT_DIM
        # Utilizamos un número mayor de capas convolucionales para obtener mejor
        # las características del gradiente de entrada
        self.encoder_conv_filters       = [32, 64, 128, 256]
        self.encoder_conv_kernel_size   = [3, 3, 3, 3]
        self.encoder_conv_strides       = [2, 2, 2, 2]
        self.n_layers_encoder = len(self.encoder_conv_filters)

        self.decoder_conv_t_filters = [256, 128, 64, 3]
        self.decoder_conv_t_kernel_size = [3, 3, 3, 3]
        self.decoder_conv_t_strides = [2, 2, 2, 2]
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = True
        self.use_dropout = True

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # Encoder
        self.encoder_model = Encoder(input_dim=self.input_dim,
                                     output_dim=self.latent_dim,
                                     encoder_conv_filters=self.encoder_conv_filters,
                                     encoder_conv_kernel_size=self.encoder_conv_kernel_size,
                                     encoder_conv_strides=self.encoder_conv_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        self.encoder_conv_size = self.encoder_model.last_conv_size
        if summary:
            self.encoder_model.summary()

        # Sampler
        self.sampler_model = Sampler(latent_dim=self.latent_dim)
        if summary:
            self.sampler_model.summary()

        # Decoder
        self.decoder_model = Decoder(input_dim=self.latent_dim,
                                     input_conv_dim=self.encoder_conv_size,
                                     decoder_conv_t_filters=self.decoder_conv_t_filters,
                                     decoder_conv_t_kernel_size=self.decoder_conv_t_kernel_size,
                                     decoder_conv_t_strides=self.decoder_conv_t_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        if summary: self.decoder_model.summary()

        self.built = True

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker, ]

    @tf.function
    def train_step(self, data):
        '''
        '''
        # Desestructuramos data ya que contiene los dos inputs (gradientes, integral)
        dog, cat = data
        with tf.GradientTape() as tape:
            # predict
            x = self.encoder_model(dog)
            z, z_mean, z_log_var = self.sampler_model(x)
            pred = self.decoder_model(z)

            # loss
            r_loss = R_LOSS_FACTOR * replacenan(self.mae(cat, pred))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = L_LOSS_FACTOR * (tf.reduce_mean(tf.reduce_sum(replacenan(kl_loss), axis=1)))
            total_loss = r_loss + kl_loss

        # gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        # train step
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # compute progress
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(), }

    @tf.function
    def generate(self, z_sample):
        '''
        We use the sample of the N(0,I) directly as
        input of the deterministic generator.
        '''
        return self.decoder_model(z_sample)

    @tf.function
    def codify(self, images):
        '''
        For an input image we obtain its particular distribution:
        its mean, its variance (unvertaintly) and a sample z of such distribution.
        '''
        # x = self.encoder_model.predict(images)
        x = self.encoder_model(images)
        z, z_mean, z_log_var = self.sampler_model(x)
        return z, z_mean, z_log_var

    # implement the call method
    @tf.function
    def call(self, inputs, training=False):
        '''
        '''
        tmp1, tmp2 = self.encoder_model.use_dropout, self.decoder_model.use_dropout
        if not training:
            self.encoder_model.use_dropout, self.decoder_model.use_dropout = False, False

        x = self.encoder_model(inputs)
        z, z_mean, z_log_var = self.sampler_model(x)
        pred = self.decoder_model(z)

        self.encoder_model.use_dropout, self.decoder_model.use_dropout = tmp1, tmp2
        return pred

vae_g = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False)
vae_g.summary()
vae_f = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False)
vae_f.summary()

vae_g.compile(optimizer=keras.optimizers.Adam())
from tensorflow.keras.callbacks import ModelCheckpoint
filepath = 'best_weight_model.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
callbacks = [checkpoint]
vae_g.fit(train_dataset,
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        initial_epoch   = INITIAL_EPOCH,
        steps_per_epoch = steps_per_epoch,
        callbacks       = callbacks)

vae_g.save_weights("final_weights_model.h5")
num_vis = 4
fig, ax = plt.subplots(nrows=num_vis, ncols=3, figsize=(12, 12))
n = 0
for data in train_dataset.take(4):
  # Obtenemos la predicción
  x  = vae_g.encoder_model(data[0])
  z, z_mean, z_log_var = vae_g.sampler_model(x)
  x_decoded = vae_g.decoder_model(z)
  #digit = x_decoded[0].reshape(digit_size, digit_size)

  # Desplegamos
  ax[n, 0].imshow(data[0][0])
  ax[n, 0].set_title('Dog')
  ax[n, 1].imshow(x_decoded[0])
  ax[n, 1].set_title('VAE')
  # ax[i, 2].imshow(pred[0,:,:,0])
  ax[n, 2].imshow(data[1][0])
  ax[n, 2].set_title('Cat')
  n += 1
  if n == 4:
    break
fig.tight_layout()
plt.savefig("figure2.png")