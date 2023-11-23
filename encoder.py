from tensorflow.keras.layers import Layer, Dropout
from encoder_layer import EncoderLayer
from positional_encoding import PositionalEmbeddingFixedWeightsLayer

class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionalEmbeddingFixedWeightsLayer(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        pos_encoding_output = self.pos_encoding(input_sentence)
        x = self.dropout(pos_encoding_output, training=training)
        for layer in self.encoder_layer:
            x = layer(x, padding_mask, training)
        return x