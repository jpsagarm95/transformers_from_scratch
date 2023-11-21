from scaled_dot_product_attention import DotProductAttention
from tensorflow.keras.layers import Dense, Layer
from tensorflow import reshape, transpose, shape

class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = DotProductAttention()
        self.heads = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = Dense(d_k)
        self.W_k = Dense(d_k)
        self.W_v = Dense(d_v)
        self.W_o = Dense(d_model)
    
    def reshape_tensor(self, x, heads, flag):
        if flag:
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], -1))
        return x
    
    def call(self, queries, keys, values, mask=None):
        q = self.reshape_tensor(self.W_q(queries), self.heads, True)
        k = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v = self.reshape_tensor(self.W_v(values), self.heads, True)
        o = self.attention(q, k, v, self.d_k, mask)
        return self.W_o(self.reshape_tensor(o, self.heads, False))

