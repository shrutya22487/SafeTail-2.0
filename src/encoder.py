# encoder.py
import tensorflow as tf
from tensorflow.keras import layers, models

class Encoder(tf.keras.Model):
    """
    Maps heterogeneous device state arrays into fixed-size embeddings.
    """
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.dense1 = layers.Dense(output_dim * 2, activation="relu")
        self.dense2 = layers.Dense(output_dim, activation="relu")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
