# Setup
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

print(tf.__version__)

# Configuración de rutas para la carga de conjuntos de datos
local_path = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train'
train_dogs_files = glob(os.path.join(local_path, 'dog.*.jpg'))
train_cats_files = glob(os.path.join(local_path, 'cat.*.jpg'))
train_dogs_files.sort()
train_cats_files.sort()
train_dogs_files = np.array(train_dogs_files)
train_cats_files = np.array(train_cats_files)

print(len(train_dogs_files), len(train_cats_files))

for dog_file, cat_file in zip(train_dogs_files[:5], train_cats_files[:5]):
  print(dog_file, cat_file)

# Configuration
# Variables de configuración
IMG_WIDTH = 256
IMG_HEIGHT = 256

INPUT_DIM     = (IMG_WIDTH, IMG_HEIGHT, 3)
BATCH_SIZE    = 10
R_LOSS_FACTOR = 10000
EPOCHS        = 100
INITIAL_EPOCH = 0

BUFFER_SIZE      = len(train_cats_files)
steps_per_epoch  = BUFFER_SIZE // BATCH_SIZE
print('num image files : ', BUFFER_SIZE)
print('steps per epoch : ', steps_per_epoch)

# Funciones para apertura y decodificación de los archivos
def read_and_decode(file):
    '''
    Lee, decodifica y redimensiona la imagen.
    Aplica aumentación
    '''
    # Lectura y decodificación
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    # Normalización
    img = img / 127.5 - 1
    # Redimensionamiento
    img = tf.image.resize(img, INPUT_DIM[:2],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def load_image(file, flip=True):
    '''
    Lee el conjunto de imágenes de entrada y las redimensiona al tamaño especificado
    Aumentación: Flip horizontal aleatorio, sincronizado
    '''
    img = read_and_decode(file)
    # Aumentación (el flip debe aplicarse simultáneamente a las 3 imagenes)
    if flip and tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)

    return img

dogs_imgs = []
cats_imgs = []
# Cargamos 3 imagenes
for i in range(3):
    dog_img = load_image(train_dogs_files[i])
    cat_img = load_image(train_cats_files[i])
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

display_images("figure_1.png", dogs_imgs, cats_imgs, rows=3)