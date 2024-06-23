from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Add
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf
from models.batch_attention import EncoderLayer



def basic_block(input,f):
    x = Conv2D(filters=f, kernel_size=3, padding="SAME",kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=f, kernel_size=3, padding="SAME",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input])
    x = ReLU()(x)
    return x

def basic_down(input,f):
    x_add = Conv2D(filters=f, kernel_size=1,strides=2,kernel_initializer='he_normal')(input)
    x = Conv2D(filters=f, kernel_size=3,strides=2, padding="SAME",kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=f, kernel_size=3, padding="SAME",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_add])
    x = ReLU()(x)
    return x

def resnet_34_double_mole_double_batchatt(vec,dim=[3,4,6,3],mode='rec',sub=1):

    if sub==1:
        input_spectra = Input(shape=(vec, 1))
        max = tf.reduce_max(input_spectra, axis=1)
        min = tf.reduce_min(input_spectra, axis=1)
        max = tf.expand_dims(max, axis=1)
        min = tf.expand_dims(min, axis=1)
        if mode == 'rec':
            input_spectra1 = tf.abs(input_spectra - tf.transpose(input_spectra, perm=[0, 2, 1]))
        elif mode == 'rm':
            input_spectra1 = input_spectra - tf.transpose(input_spectra, perm=[0, 2, 1])
        elif mode == 'rpdm':
            mul = tf.abs(input_spectra * tf.transpose(input_spectra, perm=[0, 2, 1]))
            sub = input_spectra - tf.transpose(input_spectra, perm=[0, 2, 1])
            input_spectra1 = sub / (1 + mul)
        elif mode == 'rpsm':
            mul = tf.abs(input_spectra * tf.transpose(input_spectra, perm=[0, 2, 1]))
            sum = input_spectra + tf.transpose(input_spectra, perm=[0, 2, 1])
            input_spectra1 = sum / (1 + mul)
        elif mode == 'ratio':
            sum = input_spectra + tf.transpose(input_spectra, perm=[0, 2, 1])
            sub = tf.abs(input_spectra - tf.transpose(input_spectra, perm=[0, 2, 1]))
            input_spectra1 = tf.sqrt(sum / (sub + 0.01))
        elif mode == 'tan':
            input_spectra1 = (input_spectra - min) / (max - min)
            mul = tf.abs(input_spectra1 * tf.transpose(input_spectra1, perm=[0, 2, 1]))
            sub = input_spectra1 - tf.transpose(input_spectra1, perm=[0, 2, 1])
            input_spectra1 = sub / (1 + mul)
        elif mode == 'sum':
            input_spectra1 = input_spectra + tf.transpose(input_spectra, perm=[0, 2, 1])
        elif mode == 'sum2':
            input_spectra1 = (input_spectra + tf.transpose(input_spectra, perm=[0, 2, 1])) / 2
        elif mode == 'mul':
            input_spectra1 = tf.abs(input_spectra * tf.transpose(input_spectra, perm=[0, 2, 1]))
        elif mode == 'sqrtmul':
            # input_spectra1 = (input_spectra - min) / (max - min)
            input_spectra1 = tf.sqrt(tf.abs(input_spectra * tf.transpose(input_spectra, perm=[0, 2, 1])))
        elif mode == 'glas':
            input_spectra1 = (input_spectra - min) / (max - min)
            input_spectra1 = input_spectra1 * tf.transpose(input_spectra1, perm=[0, 2, 1]) - tf.sqrt(
                1 - input_spectra1 ** 2) * tf.transpose(tf.sqrt(1 - input_spectra1 ** 2), perm=[0, 2, 1])
        elif mode == 'glad':
            input_spectra1 = (input_spectra - min) / (max - min)
            input_spectra1 = tf.sqrt(1 - input_spectra1 ** 2) * tf.transpose(input_spectra1, perm=[0, 2,
                                                                                                   1]) - input_spectra1 * tf.transpose(
                tf.sqrt(1 - input_spectra1 ** 2), perm=[0, 2, 1])

        input_spectra1 = tf.expand_dims(input_spectra1, axis=-1)
        f1 = 64
        f2 = 128
        f3 = 256
        f4 = 512
        x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', kernel_initializer='he_normal')(input_spectra1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        for _ in range(dim[0]):
            x = basic_block(x, f1)

        x = basic_down(x, f2)
        for _ in range(dim[1] - 1):
            x = basic_block(x, f2)

        x = basic_down(x, f3)
        for _ in range(dim[2] - 1):
            x = basic_block(x, f3)

        x = basic_down(x, f4)
        for _ in range(dim[3] - 1):
            x = basic_block(x, f4)

        # MLP head
        x = GlobalAveragePooling2D()(x)

        encoder_layerg1 = EncoderLayer(512, 1, 512*4)
        encoder_layerh1 = EncoderLayer(512, 1, 512*4)
        batch_g,atg = encoder_layerg1(x, training=True, mask=None)
        batch_h ,ath= encoder_layerh1(x, training=True, mask=None)

        answer1 = Dense(1, kernel_initializer=initializers.RandomUniform, name='output1')(x)
        answer2 = Dense(1, kernel_initializer=initializers.RandomUniform, name='output2')(x)
        answer3 = Dense(1, kernel_initializer=initializers.RandomUniform, name='output3')(batch_g)
        answer4 = Dense(1, kernel_initializer=initializers.RandomUniform, name='output4')(batch_h)

        model = Model(inputs=input_spectra, outputs=[answer1, answer2, answer3, answer4,x,batch_g,batch_h])

        return model


