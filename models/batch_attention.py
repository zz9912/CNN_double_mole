import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization,Dense,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model, kernel_initializer=initializers.RandomUniform)
        self.wk = Dense(d_model, kernel_initializer=initializers.RandomUniform)
        self.wv = Dense(d_model, kernel_initializer=initializers.RandomUniform)

        self.dense = Dense(d_model, kernel_initializer=initializers.RandomUniform)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, self.num_heads, self.depth))
        return tf.transpose(x, perm=[1,0,2])

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        scaled_attention_logits = matmul_qk

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)


        output = tf.matmul(attention_weights, v)

        return output, scaled_attention_logits

    def call(self, q, mask):
        q = self.wq(q)
        k=q
        v=q

        q = tf.nn.l2_normalize(q, axis=-1)
        k = tf.nn.l2_normalize(k, axis=-1)
        v = tf.nn.l2_normalize(v, axis=-1)
        scaled_attention, att = self.scaled_dot_product_attention(q, k, v, mask)

        output = self.dense(scaled_attention)

        return output,att

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.dropout1 = Dropout(rate)

        self.ffn = Sequential([
            Dense(dff, activation='relu', kernel_initializer=initializers.RandomUniform),
            Dense(d_model, kernel_initializer=initializers.RandomUniform)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        if training==True:
            attn_output,att = self.mha(x, mask)

            attn_output = self.dropout1(attn_output, training=training)

            out1 = self.layernorm1(x + attn_output)

            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)
            return out2,att

        else:

            return x
