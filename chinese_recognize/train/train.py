# -*- coding:utf-8 -*-
import os
from imp import reload

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Model
from tensorflow.python.ops import ctc_ops as ctc

from chinese_recognize.densenet import densenet

alphabet_en = u"""_ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.<>/?;:'"[]{}!@#$%^&*()-=+\|`"""
character_en = u"""_|abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890#'",.:;-()?*!/&+"""

img_h = 32
img_w = 280
batch_size = 128
maxlabellength = 10


def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


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
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if (len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 1 for k in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def gen_en(image_label, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if (len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 50
            labels[i, :len(str)] = [int(k) - 1 for k in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = tf.to_int32(tf.squeeze(label_length, axis=-1))
    input_length = tf.to_int32(tf.squeeze(input_length, axis=-1))
    sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(y_true, label_length))

    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())

    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length,
                                       ignore_longer_outputs_than_inputs=True), 1)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


def chinese_train():
    char_set = ''.join(densenet.keys.alphabet[1:] + ['卍'])
    nclass = len(char_set)

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = './models/pretrain_model/keras.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen('data_train.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength,
                       imagesize=(img_h, img_w))
    test_loader = gen('data_test.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength,
                      imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=3607567 // batch_size,
                        epochs=10,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=36440 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])


def englist_train():
    char_set = ''.join(alphabet_en[1:] + '卍')
    nclass = len(char_set)

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = './models/weights_densenet_en-03-13.05.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    test_data_dict = {}
    train_data_dict = {}
    idx = 0
    maxlabellength = 0
    g = os.walk("D:/_data/en_txt_2")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            cv_line_arr = []
            s = file_name.split("_", 1)
            maxlabellength = len(s[0]) if maxlabellength < len(s[0]) else maxlabellength
            for c in s[0]:
                cv_line_arr.append(str(alphabet_en.index(c)))
            result = ' '.join(cv_line_arr)
            print(result)
            if idx % 100 == 0:
                test_data_dict[file_name] = cv_line_arr
            else:
                train_data_dict[file_name] = cv_line_arr
            idx += 1

    train_loader = gen_en(train_data_dict, 'D:/_data/en_txt_2', batchsize=batch_size, maxlabellength=maxlabellength,
                          imagesize=(img_h, img_w))
    test_loader = gen_en(test_data_dict, 'D:/_data/en_txt_2', batchsize=batch_size, maxlabellength=maxlabellength,
                         imagesize=(img_h, img_w))

    # print(test_loader.__next__())

    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet_en-{epoch:02d}-{val_loss:.2f}.h5',
                                 monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=100100 // batch_size,
                        epochs=10,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=1000 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])


def english_handwriting_train():
    # rev = ""
    # s = "3,1,18,3,1,19,5,0,13,5,1,20,0,71,0"
    # arr = s.split(",")
    # for a in arr:
    #     idx = int(a)
    #     rev += character_en[idx]
    # print(rev)
    # return

    char_set = ''.join(character_en[1:] + '卍')
    nclass = len(char_set)

    test_data_dict = {}
    train_data_dict = {}
    idx = 0
    max_label_length = 0
    with open('D:/github.com/deep-learning-playgroud/handwriting_english_recognize/sentences.txt', 'r') as f:
        lines = f.readlines()

        for line in lines:
            cv_line_arr = []
            if line.startswith('#'):
                continue
            line_arr = line.split(' ')
            sentences = line_arr[len(line_arr) - 1].strip()
            if len(sentences) > max_label_length:
                max_label_length = len(sentences)

            for c in sentences:
                cv_line_arr.append(str(character_en.index(c)))
            print(cv_line_arr)
            if idx % 100 == 0:
                test_data_dict[line_arr[0] + ".png"] = cv_line_arr
            else:
                train_data_dict[line_arr[0] + ".png"] = cv_line_arr
            idx += 1

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = './models/weights_densenet_hw.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen_en(train_data_dict, 'E:/_dataset/IAM/en_hw', batchsize=batch_size,
                          maxlabellength=max_label_length,
                          imagesize=(img_h, img_w))
    test_loader = gen_en(test_data_dict, 'E:/_dataset/IAM/en_hw', batchsize=batch_size,
                         maxlabellength=max_label_length,
                         imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet_hw-{epoch:02d}-{val_loss:.2f}.h5',
                                 monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=161752 // batch_size,
                        epochs=10,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=1617 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])


if __name__ == '__main__':
    # chinese_train()
    # englist_train()
    english_handwriting_train()
