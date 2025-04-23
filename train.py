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

def train(models, X, Y, epochs, batch_size=64, validation_split=0.2, graph=True, summary=True, loss='categorical_crossentropy'):
    debug_print(['training model'])

    for model in models:
        model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
        model(X)
        if summary: model.summary()
        model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)        