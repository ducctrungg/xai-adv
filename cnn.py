import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, MaxPool1D, Flatten, AveragePooling1D, Reshape, Multiply, Dense, BatchNormalization, Dropout, InputLayer

class CNN4(keras.Model):
  def __init__(self, input_shape):
    initializer = tf.keras.initializers.GlorotUniform()

    # Input layer
    input_layer = keras.Input(shape=input_shape)

    # First Dense layer
    dense1 = Dense(1024)(input_layer)
    reshape = Reshape((64, 16))(dense1)
    
    # First Conv1D layer
    batch_norm1 = BatchNormalization()(reshape)
    dropout1 = Dropout(0.1)(batch_norm1)
    conv1 = Conv1D(filters=16, strides=2, kernel_size=5, activation='relu', use_bias=False, padding='SAME', kernel_initializer=initializer)(dropout1)
    avg_pool1 = AveragePooling1D()(conv1)
    
    # Second Conv1D layer
    batch_norm2 = BatchNormalization()(avg_pool1)
    dropout2 = Dropout(0.1)(batch_norm2)
    conv2 = Conv1D(filters=8, kernel_size=5, activation='relu', use_bias=False, padding='SAME', kernel_initializer=initializer)(dropout2)

    # Third Conv1D layer
    batch_norm3 = BatchNormalization()(conv2)
    dropout3 = Dropout(0.1)(batch_norm3)
    conv3 = Conv1D(filters=8, kernel_size=3, activation='relu', use_bias=True, padding='SAME', kernel_initializer=initializer)(dropout3)

    # Fourth Conv1D layer
    batch_norm4 = BatchNormalization()(conv3)
    dropout4 = Dropout(0.1)(batch_norm4)
    conv4 = Conv1D(filters=8, kernel_size=3, activation='relu', use_bias=True, padding='SAME', kernel_initializer=initializer)(dropout4)
    
    # Max pooling and dense layers
    multiply = Multiply()([conv2, conv4])
    max_pool = MaxPool1D(pool_size=4, strides=1)(multiply)

    flatten = Flatten()(max_pool)
    batch_norm5 = BatchNormalization()(flatten)
    dense2 = Dense(100)(batch_norm5)
    output_layer = Dense(1, activation='sigmoid')(dense2)
    super(CNN4, self).__init__(inputs=input_layer, outputs=output_layer)

class CNN2(keras.Model):
  def __init__(self, input_shape):
    super(CNN2, self).__init__()
    initializer = tf.keras.initializers.GlorotUniform()

    # Input layer
    input_layer = keras.Input(shape=input_shape)

    # First Dense layer
    dense1 = Dense(1024)(input_layer)
    reshape = Reshape((64, 16))(dense1)
    
    # First Conv1D layer
    batch_norm1 = BatchNormalization()(reshape)
    dropout1 = Dropout(0.1)(batch_norm1)
    conv1 = Conv1D(filters=16, strides=2, kernel_size=5, activation='relu', use_bias=False, padding='SAME', kernel_initializer=initializer)(dropout1)
    avg_pool1 = AveragePooling1D()(conv1)
    
    # Second Conv1D layer
    batch_norm2 = BatchNormalization()(avg_pool1)
    dropout2 = Dropout(0.1)(batch_norm2)
    conv2 = Conv1D(filters=16, kernel_size=3, activation='relu', use_bias=True, padding='SAME', kernel_initializer=initializer)(dropout2)
    
    # Max pooling and dense layers
    multiply = Multiply()([conv2, avg_pool1])
    max_pool = MaxPool1D(pool_size=4, strides=1)(multiply)
    flatten = Flatten()(max_pool)
    batch_norm3 = BatchNormalization()(flatten)
    dense2 = Dense(100)(batch_norm3)
    output_layer = Dense(1, activation='sigmoid')(dense2)
    super(CNN2, self).__init__(inputs=input_layer, outputs=output_layer)