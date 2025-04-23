import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import preprocessing
from preprocessing import debug_print
from models import *
from GAN import *
from analysis import *

def compute_baselines(models, X, Y):
    for model in models:
        baseline_pred = model.predict(X)
        baseline_loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, baseline_pred))
        debug_print([model.name, 'loss:', baseline_loss.numpy()])