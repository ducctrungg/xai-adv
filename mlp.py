import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization

class MLP(keras.Model):
  def __init__(self, input_shape):
    initializer = tf.keras.initializers.GlorotUniform()
    
    input_layer = keras.Input(shape=input_shape)
    dense1 = Dense(128, activation='relu', kernel_initializer=initializer)(input_layer)
    batch_norm1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.2)(batch_norm1)

    dense2 = Dense(64, activation='relu')(dropout1)
    batch_norm2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.1)(batch_norm2)
    
    dense3 = Dense(32, activation='relu')(dropout2)
    batch_norm3 = BatchNormalization()(dense3)
    
    output_layer = Dense(1, activation='sigmoid')(batch_norm3)
    super(MLP, self).__init__(inputs=input_layer, outputs=output_layer)