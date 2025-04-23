import numpy as np
import tensorflow as tf

from utils import *

# Layers
class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, ff_dim, dropout=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout_rate = dropout