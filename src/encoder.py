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
        # First fully connected layer to expand input features.
        # Converts low-dimensional input into a higher-dimensional hidden representation.
        self.expand = layers.Dense(hidden_dim, activation='relu')

        # Another dense layer to further transform the representation.
        # This adds non-linearity and depth to capture more complex relationships.
        self.project = layers.Dense(hidden_dim, activation='relu')

        # Pooling layer that averages over the sequence/time dimension.
        # It compresses variable-length input (e.g., [batch, seq_len, hidden_dim])
        # into a fixed-size vector [batch, hidden_dim].
        self.pool = layers.GlobalAveragePooling1D()

        # Final projection to the target embedding size.
        # This produces a fixed-length output vector per input example.
        self.output_layer = layers.Dense(output_dim, activation='relu')

    def call(self, x):
        """
        Expects input x of shape (batch, variable_length, 1)
        """
        # Step 1: Expand input features → (batch, variable_length, hidden_dim)
        x = self.expand(x)

        # Step 2: Transform features again to add non-linearity and depth
        x = self.project(x)

        # Step 3: Average across the sequence length dimension
        # This removes the variable-length aspect of input.
        # Result → (batch, hidden_dim)
        x = self.pool(x)

        # Step 4: Map to the final embedding space of size output_dim
        # Result → (batch, output_dim)
        x = self.output_layer(x)

        return x
