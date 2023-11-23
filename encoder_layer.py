from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multi_head_attention import MultiHeadAttention

class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        add = x + sublayer_x
        return self.layer_norm(add)
    
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)
        self.fully_connected2 = Dense(d_model)
        self.activation = ReLU()
    
    def call(self, x):
        x_fc1 = self.fully_connected1(x)
        return self.fully_connected2(self.activation(x_fc1))
    
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
    
    def call(self, x, padding_mask, training):
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        multihead_output = self.dropout1(multihead_output, training=training)
        addnorm_output = self.add_norm1(x, multihead_output)
        feedforward_output = self.feed_forward(addnorm_output)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        return self.add_norm2(addnorm_output, feedforward_output)
    