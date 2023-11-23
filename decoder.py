from tensorflow.keras.layers import Layer, Dropout
from positional_encoding import PositionalEmbeddingFixedWeightsLayer
from decoder_layer import DecoderLayer

class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionalEmbeddingFixedWeightsLayer(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
    
    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        pos_encoding_output = self.pos_encoding(output_target)
        x = self.dropout(pos_encoding_output)
        for layer in self.decoder_layer:
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)
        return x