from tensorflow.keras.layers import Layer, Dropout
from multi_head_attention import MultiHeadAttention
from encoder_layer import AddNormalization, FeedForward

class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()
    
    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        multihead_output1 = self.dropout1(multihead_output1, training=training)
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)
        multihead_output2 = self.dropout2(multihead_output2, training=training)
        addnorm_output2 = self.add_norm2(addnorm_output1, multihead_output2)
        feedforward_output = self.feed_forward(addnorm_output2)
        feedforward_output = self.dropout3(feedforward_output, training=training)
        return self.add_norm3(addnorm_output2, feedforward_output)