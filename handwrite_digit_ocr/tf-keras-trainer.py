import base64
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model

from handwrite_digit_ocr.dataset_utils import load_character_dataset, load_digit_dataset
from handwrite_digit_ocr.ocr_model import ModelType, OcrModel


def set_model(self, model):
    self.model = model
    self.sess = K.get_session()
    if self.histogram_freq and self.merged is None:
        for layer in self.model.layers:

            for weight in layer.weights:
                tf.summary.histogram(weight.name, weight)
                if self.write_images:
                    w_img = tf.squeeze(weight)
                    shape = w_img.get_shape()
                    if len(shape) > 1 and shape[0] > shape[1]:
                        w_img = tf.transpose(w_img)
                    if len(shape) == 1:
                        w_img = tf.expand_dims(w_img, 0)
                    w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                    tf.summary.image(weight.name, w_img)

            if hasattr(layer, 'output'):
                tf.summary.histogram('{}_out'.format(layer.name),
                                     layer.output)
    self.merged = tf.summary.merge_all()


def character_ocr_train(output_model_name, model=None):
    x_train, y_train, x_test, y_test = load_character_dataset('E:/_dataset/RCNN/EMNIST-balanced-191127-added.npz', 'ak')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 如果模型不传 则重新训练模型, 否则加载模型
    if model is None:
        model = OcrModel.get_model(ModelType.RCNN, 11)

    # model.summary()

    tensor_board_callback = TensorBoard(log_dir='./logs',  # log 目录
                                        histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                        batch_size=192,  # 用多大量的数据计算直方图
                                        write_graph=True,  # 是否存储网络结构图
                                        write_grads=True,  # 是否可视化梯度直方图
                                        write_images=False,  # 是否可视化参数
                                        embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None)
    tensor_board_callback.set_model(model)

    model_check_file_template = os.path.join("./models/", output_model_name + "-{epoch:02d}-{val_loss:.2f}.h5")

    checkpoint_callback = ModelCheckpoint(
        filepath=model_check_file_template, monitor='val_loss',
        save_best_only=False, save_weights_only=False, period=5)

    # 各种callback的ref: https://keras.io/zh/callbacks/
    # lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    # learning_rate = np.array([lr_schedule(i) for i in range(10)])
    # change_learing_rate_callback = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]), verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    """
    shuffle和validation_split的顺序
    模型的fit函数有两个参数，shuffle用于将数据打乱，validation_split用于在没有提供验证集的时候，
    按一定比例从训练集中取出一部分作为验证集这里有个陷阱是，程序是先执行validation_split，
    再执行shuffle的，所以会出现这种情况：
    假如你的训练集是有序的，比方说正样本在前负样本在后，又设置了validation_split，那么你的验证集中很可能将全部是负样本
    同样的，这个东西不会有任何错误报出来，因为Keras不可能知道你的数据有没有经过shuffle，
    保险起见如果你的数据是没shuffle过的，最好手动shuffle一下
    """
    seed = np.random.randint(1)
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(x_train)
    rand_state.seed(seed)
    rand_state.shuffle(y_train)

    model.fit(x_train, y_train,
              batch_size=192,
              epochs=15,
              verbose=1,
              shuffle=True,
              validation_split=0.15,
              callbacks=[tensor_board_callback,
                         checkpoint_callback,
                         early_stopping_callback])

    model_check_file_template = os.path.join("./models/", output_model_name + ".h5")

    model.save(model_check_file_template)


def character_ocr_evaluate():
    x_train, y_train, x_test, y_test = load_character_dataset('E:/_dataset/RCNN/EMNIST-balanced-191127-added.npz', 'ak')
    model = load_model('./model/model-EMNIST-RCNN-balanced-a-to-k-191127.h5')
    loss_and_metrics1 = model.evaluate(x_test, y_test, verbose=1)
    print(
        "threshold_data_2nd_check -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics1[1] * 100,
                                                                       loss_and_metrics1[0]))


def character_ocr_predict():
    model = load_model('./model/model-EMNIST-RCNN-balanced-a-to-k-191127.h5')
    img_arr = []
    g = os.walk('./test')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            img_arr.append(image_to_base64("./test/" + file_name))
            t_img = cv2.imread("./test/" + file_name, 0)
            t_img = cv2.resize(t_img, (28, 28))
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
            t_img = t_img.reshape(-1, 28, 28, 3)
            predict_result = model.predict(t_img)
            print(file_name, np.argmax(predict_result), chr(int(np.argmax(predict_result)) + 97),
                  predict_result[0][np.argmax(predict_result)])
    # gen_request_json_for_character_ocr(img_arr)
    #
    # for i in range(9):
    #     t_img = cv2.imread("./test/0"+str(i+1)+".jpg", 0)
    #     t_img = cv2.resize(t_img, (28, 28))
    #     t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
    #     t_img = t_img.reshape(-1, 28, 28, 3)
    #     predict_result = model.predict(t_img)
    #     print(np.argmax(predict_result), predict_result[0][np.argmax(predict_result)])


def digit_ocr_train():
    x_train, y_train, x_test, y_test = \
        load_digit_dataset('mnist_with_space.npz',
                           'C:/_project/victoriest_digit_recognizer/dataset_28_28_3/zk.npz',
                           'C:/_project/victoriest_digit_recognizer/dataset_28_28_3/xbk.npz')

    model = OcrModel.get_model(ModelType.RCNN, 11)
    model.summary()
    model.fit(x_train, y_train, batch_size=192, epochs=50, verbose=1, shuffle=True, validation_split=0.85)
    model.save('model-EMNIST-RCNN-balanced-a-to-k-keras.h5')


def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string


def gen_request_json_for_character_ocr(img_arr):
    result = {
        "id": "1111",
        "characterPos": [
        ]
    }
    idx = 0
    for img in img_arr:
        item = {
            "areaId": str(idx),
            "idx": idx,
            "img": img.decode()
        }
        result["characterPos"].append(item)
        idx += 1
    print(result)


if __name__ == '__main__':
    character_ocr_train('model-EMNIST-RCNN-balanced-a-to-k-191129')
