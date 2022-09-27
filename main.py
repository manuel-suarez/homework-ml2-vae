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