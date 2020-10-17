import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from utils.random_uniform_number import RandomUniformNumber
from variable_length_digit_htr import *
from variable_length_digit_htr.crnn import crnn_model


def gen_crnn_batch():
    data_path = []
    g = os.walk('E:/_dataset/_digit_crnn_train_32_offset_0925')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            # data_path.append(img2vector(d))
            data_path.append((file_name.split("_")[0], d))
    random.shuffle(data_path)

    print(len(data_path))
    r_n = RandomUniformNumber(len(data_path))

    x = np.zeros((batch_size, img_h, img_w, 1), dtype=np.float)
    labels = np.ones([batch_size, max_label_length]) * 10000
    input_length = np.zeros([batch_size, 1])
    label_length = np.zeros([batch_size, 1])

    idx = 0
    while 1:
        batch_count = 0
        for tag, dir_path in data_path[idx:(idx + batch_size)]:
            if batch_count > batch_size:
                break
            img = cv2.imread(dir_path, 0).reshape(-1, img_h, img_w, 1).astype('float32') / 255.0
            x[batch_count] = img
            labels[batch_count, :len(tag)] = [char_to_id[i] for i in tag]
            input_length[batch_count] = 170 // 4 + 1
            label_length[batch_count] = len(tag)
            batch_count += 1
            idx += 1
        inputs = {
            'the_input': x,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def train_crnn_model(crnn_model):
    early_stop = EarlyStopping(patience=10)

    # 创建一个权重文件保存文件夹logs
    log_dir = "./logs/"
    model_dir = './model/'
    # 记录所有训练过程，每隔一定步数记录最大值
    tensor_board_callback = TensorBoard(log_dir='./logs',  # log 目录
                                        histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                        batch_size=192,  # 用多大量的数据计算直方图
                                        write_graph=True,  # 是否存储网络结构图
                                        write_grads=True,  # 是否可视化梯度直方图
                                        write_images=False,  # 是否可视化参数
                                        embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None)
    checkpoint = ModelCheckpoint(model_dir + "crnn_200925.h5",
                                 monitor="loss",
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1,
                                 period=1)
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    res = crnn_model.fit_generator(gen_crnn_batch(),
                                   steps_per_epoch=50000 // batch_size,
                                   epochs=50,
                                   validation_steps=1000 // batch_size,
                                   callbacks=[early_stop,
                                              checkpoint,
                                              learning_rate_reduction,
                                              tensor_board_callback
                                              ],
                                   verbose=1)


def get_session(gpu_fraction=0.8):
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


tf.compat.v1.keras.backend.set_session(get_session())

char = '0123456789'


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != num_of_classes - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(char[pred_text[i]])
    return u''.join(char_list)


if __name__ == '__main__':
    base_model, model = crnn_model()
    # train_crnn_model(model)

    img = cv2.imread("./20200926111203.jpg", 0)
    print(img.shape)
    height, width = img.shape[0], img.shape[1]
    scale = height * 1.0 / 32
    print(scale)
    width = int(width / scale)
    print((width, img_h))
    img = cv2.resize(img, (width, img_h))
    # cv2.imwrite("xxx.jpg", img)
    img = np.array(img).astype(np.float32) / 255.0

    X = img.reshape([1, img_h, width, 1])

    # model = tf.keras.models.load_model('./model/crnn_200925.h5')
    model.load_weights('./model/crnn_200925.h5')
    y_pred = model.predict(X)
    y_pred = y_pred[:, :, :]

    out = decode(y_pred)
    print(out)
