# encoder.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class Encoder(Model):
    """
    Maps heterogeneous device state arrays into fixed-size embeddings.
    
    Encodes variable-length numeric arrays into fixed-size embeddings.
    Works on flattened numeric vectors of arbitrary length.
    """
    def __init__(self, hidden_dim=64, output_dim=32):
        super().__init__()
        self.expand = layers.Dense(hidden_dim, activation='relu')
        self.project = layers.Dense(hidden_dim, activation='relu')
        self.pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(output_dim, activation='relu')

    def call(self, x):
        """
        Expects input x of shape (batch, variable_length, 1)
        """
        x = self.expand(x)
        x = self.project(x)
        x = self.pool(x)          # Aggregate across variable length → (batch, hidden_dim)
        x = self.output_layer(x)  # Map to final fixed embedding
        return x
