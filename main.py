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