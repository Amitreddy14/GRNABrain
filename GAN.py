import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from tqdm import tqdm

import preprocessing
from preprocessing import debug_print
from models import *

def discriminator_loss(real_output, pred_output, mismatch_output):
    lambda1 = 0.4
    lambda2 = 0.2
    lambda3 = 0.4
    # BC(1, D(g_t, d_t))
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    # BC(0, D(G(d_t), d_t))
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(pred_output), pred_output)
    # BC(0, D(g_t, d_rand))
    mismatch_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(mismatch_output), mismatch_output)
    total_loss = real_loss * lambda1 + fake_loss * lambda2 + mismatch_loss * lambda3
    return total_loss

def generator_loss(pred, real, pred_output, mismatch_output):
    lambda1 = 0.4
    lambda2 = 0.5
    lambda3 = 0.1
    # BC(g_t, G(d_t))
    gen  = tf.keras.losses.CategoricalCrossentropy()(real, pred)
    # BC(1, D(G(d_t), d_t))
    disc = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(pred_output), pred_output)
    # BC(0, D(G(d_t), d_rand))
    disc_mismatch = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(pred_output), mismatch_output)
    return gen * lambda1 + disc * lambda2 + disc_mismatch * lambda3