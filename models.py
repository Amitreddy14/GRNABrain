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

    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(input_shape[-1])
        ])
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)    

    def call(self, inputs, training=True):
        attn_output = self.attention(query=inputs, value=inputs, attention_mask=None, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)   

    # Generators
class ActorMLP(tf.keras.Model):
    def __init__(self, output_shape, name='test_generator', **kwargs):
        super().__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_shape[0] * 4, activation='relu')
        self.reshape = tf.keras.layers.Reshape((output_shape[0], 4))
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape(x)
        x = self.out(x)
        
        return x   

class ActorVAE(tf.keras.Model):
    def __init__(self, input_shape, output_shape, latent_dim=32, num_transformers=3, hidden_size=32, name='actor_vae'):
        super().__init__(name=name)

        self.transformers = tf.keras.Sequential([Transformer(8, 8, hidden_size) for _ in range(num_transformers)])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)

        self.dense_decode1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense_decode2 = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu')
        self.reshape = tf.keras.layers.Reshape(output_shape)
        self.dense_decode3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

        self.sampling_layer = tf.keras.layers.Lambda(self.sampling, output_shape=(latent_dim,))   

    def sampling(self, args):
        mean, log_var = args
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0 * log_var) * epsilon 

    def encode(self, x):
        x = self.transformers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        mean = self.dense_mean(x)
        log_var = self.dense_log_var(x)
        return mean, log_var     
 