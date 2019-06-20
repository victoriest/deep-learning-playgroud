# CRNN
# Edit:2017-09-14 ~ 2017-09-17
# @sima
# %%
import random

import cv2
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers import Input, Dense, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam, SGD, Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import plot_model
from matplotlib import pyplot as plt

import numpy as np
import os
from PIL import Image
import json
import threading

import tensorflow as tf
import keras.backend.tensorflow_backend as K


def get_session(gpu_fraction=0.6):
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


K.set_session(get_session())


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


char = '0123456789'
# with open('E:\deeplearn\OCR\chi_sim\随机语料\yu\char.txt', encoding='utf-8') as f:
#     for ch in f.readlines():
#         ch = ch.strip('\r\n')
#         char = char + ch
char = char + '^'
print('nclass:', len(char))

char_to_id = {j: i for i, j in enumerate(char)}
id_to_char = {i: j for i, j in enumerate(char)}

maxlabellength = 5
img_h = 42
img_w = 152
nclass = len(char)
rnnunit = 256
batch_size = 64

gen = image.ImageDataGenerator(rescale=1.0 / 255)


def gen1(jsonpath, batchsize=64, maxlabellength=5, imagesize=(42, 152)):
    with open(jsonpath, 'r', encoding='utf-8') as f:
        image_label = json.load(f)
    imagepathlist = [i for i, _ in image_label]
    imagepathlist = np.array(imagepathlist)
    while 1:
        labels = np.ones([batchsize, maxlabellength])
        input_length = np.zeros([batchsize, 1])
        label_length = np.zeros([batchsize, 1])


def gen2(jsonpath, imagepath, batchsize=64, maxlabellength=8, imagesize=(32, 248)):
    with open(jsonpath, 'r', encoding='utf-8') as f:
        image_label = json.load(f)

    print('open json')
    imagelabel = [i['label'] for _, i in image_label.items()]
    _imagefile = [i for i, j in image_label.items()]
    print('--begin gen2')
    v = gen.flow_from_directory(imagepath, target_size=imagesize,
                                color_mode='grayscale', class_mode='sparse', shuffle=True,
                                # save_to_dir=r'E:\deeplearn\OCR\Sample\fixsizetrain',
                                batch_size=batchsize
                                )

    v.classes = np.array([i for i in range(len(imagelabel))])
    v.filenames = _imagefile
    print('end gen2')

    while 1:
        x, l = next(v)
        bz = len(l)
        labels = np.ones([bz, maxlabellength])
        input_length = np.zeros([bz, 1])
        label_length = np.zeros([bz, 1])
        for i in range(bz):
            str = imagelabel[l[i]]
            label_length[i] = len(str)

            input_length[i] = imagesize[1] // 4 + 1
            labels[i, :len(str)] = [char_to_id[i] for i in str]
        #            print(str)
        #        print(labels)
        #        print(_imagefile[l[i]])
        #        print(label_length)
        #        print(input_length)

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        # output = [x,labels,input_length,label_length]
        yield (inputs, outputs)


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)

        else:
            r_n = self.range[self.index:self.index + batchsize]
            self.index = self.index + batchsize
        return r_n


def gen_crnn_batch():
    batchsize = 64
    data_path = []
    g = os.walk('C:/_project/_data_crnn_train')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            # data_path.append(img2vector(d))
            data_path.append((file_name.split("_")[0], d))
    random.shuffle(data_path)

    print(len(data_path))
    r_n = random_uniform_num(len(data_path))

    x = np.zeros((64, 42, 152, 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    idx = 0
    while 1:
        batch_count = 0
        for tag, dir_path in data_path[idx:(idx + batchsize)]:
            if batch_count > batchsize:
                break
            img = cv2.imread(dir_path, 0).reshape(-1, 42, 152, 1).astype('float32') / 255.0
            x[batch_count] = img
            labels[batch_count, :len(tag)] = [char_to_id[i] for i in tag]
            input_length[batch_count] = 152 // 4 + 1
            label_length[batch_count] = len(tag)
            batch_count += 1
            idx += 1
        inputs = {
            'the_input': x,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


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

m = Bidirectional(GRU(rnnunit, return_sequences=True, implementation=2), name='blstm1')(m)
# m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm1')(m)
m = Dense(rnnunit, name='blstm1_out', activation='linear', )(m)
# m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm2')(m)
m = Bidirectional(GRU(rnnunit, return_sequences=True, implementation=2), name='blstm2')(m)
y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)

basemodel = Model(inputs=input, outputs=y_pred)
basemodel.summary()

labels = Input(name='the_labels', shape=[maxlabellength], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

adam = Adam()

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

earlystop = EarlyStopping(patience=10)

# 创建一个权重文件保存文件夹logs
log_dir = "./logs/"
model_dir = './model/'
# 记录所有训练过程，每隔一定步数记录最大值
tensorboard = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(model_dir + "crnn.h5",
                             monitor="loss",
                             mode='min',
                             save_best_only=True,
                             verbose=1,
                             period=1)
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

res = model.fit_generator(gen_crnn_batch(),
                          steps_per_epoch=500000 // batch_size,
                          epochs=100,
                          validation_steps=1000 // batch_size,
                          callbacks=[earlystop, checkpoint, tensorboard, learning_rate_reduction],
                          verbose=1)
