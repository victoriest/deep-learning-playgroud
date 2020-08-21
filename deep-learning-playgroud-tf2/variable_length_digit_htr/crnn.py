import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Lambda, Permute
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import LSTM, GRU

from variable_length_digit_htr import *


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def crnn_model():
    input = Input(shape=(img_h, None, 1), name='the_input')
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
    m = BatchNormalization(axis=3)(m)

    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)

    m = Bidirectional(GRU(rnn_unit, return_sequences=True, implementation=2, reset_after=True), name='blstm1')(m)
    # m = Bidirectional(LSTM(rnn_unit, return_sequences=True), name='blstm1')(m)
    m = Dense(rnn_unit, name='blstm1_out', activation='linear', )(m)
    # m = Bidirectional(LSTM(rnn_unit, return_sequences=True), name='blstm2')(m)
    m = Bidirectional(GRU(rnn_unit, return_sequences=True, implementation=2, reset_after=True), name='blstm2')(m)
    y_pred = Dense(num_of_classes, name='blstm2_out', activation='softmax')(m)
    base_model = Model(inputs=input, outputs=y_pred)
    # base_model.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    adam = Adam()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

    return base_model, model
