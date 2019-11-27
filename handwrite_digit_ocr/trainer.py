import os
import random
import time

import cv2
import numpy as np
import tensorflow as tf
from deprecated import deprecated
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import np_utils

from handwrite_digit_ocr.config import *
from handwrite_digit_ocr.ocr_model import OcrModel, ModelType


@deprecated
class OcrTrainer:
    def __init__(self):
        self.model = None
        self.model_root_path = './'
        self.model_type = ModelType.SimpleCNN

    def generate_model(self, model_root_path='./', model_type=ModelType.SimpleCNN):
        self.model_root_path = model_root_path
        if not os.path.exists(model_root_path):
            os.makedirs(model_root_path)
        self.model = OcrModel.get_model(model_type)
        self.model_type = model_type

    def mnist_train(self, train_data_path, test_data_path):
        """
        输入为28*28*1的图像

        :param train_data_path:     训练数据目录根路径
        :param test_data_path:      测试数据目录根路径
        :return:
        """
        # 记录所有训练过程，每隔一定步数记录最大值
        log_dir = os.path.join(self.model_root_path, self.model_type.name, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensor_board = TensorBoard(log_dir=log_dir)
        x_train, y_train = OcrTrainer.load_img_to_test_data(train_data_path)
        x_test, y_test = OcrTrainer.load_img_to_test_data(test_data_path)
        history = self.model.fit(x_train,
                                 y_train,
                                 batch_size=128,
                                 epochs=200,
                                 verbose=1,
                                 validation_data=(x_test, y_test),
                                 callbacks=[tensor_board])

        model_path = self.__get_model_path()
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def vgg_train(self, train_data_path, test_data_path):
        """
        因vgg16的input图像尺寸不同, 所以需要这里单独训练方法
        :param train_data_path:
        :param test_data_path:
        :return:
        """
        # 记录所有训练过程，每隔一定步数记录最大值
        log_dir = os.path.join(self.model_root_path, self.model_type.name, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensor_board = TensorBoard(log_dir=log_dir)

        # 保存最好的预测数据
        checkpoint_path = os.path.join(self.model_root_path,
                                       'checkpoint' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.h5')
        checkpoint = ModelCheckpoint(checkpoint_path,
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
        callback_lists = [tensor_board, checkpoint, learning_rate_reduction]

        if test_data_path is None:
            history = self.model.fit_generator(OcrTrainer.__get_img_generator(train_data_path),
                                               steps_per_epoch=444709 // BATCH_SIZE,
                                               epochs=10,
                                               callbacks=callback_lists)
        else:
            x_test, y_test = OcrTrainer.load_img_to_test_data(test_data_path,
                                                              VGG_INPUT_IMG_WIDTH,
                                                              VGG_INPUT_IMG_HEIGHT)
            history = self.model.fit_generator(OcrTrainer.__get_img_generator(train_data_path),
                                               steps_per_epoch=444709 // BATCH_SIZE,
                                               epochs=10,
                                               validation_data=(x_test, y_test),
                                               callbacks=callback_lists)

        model_path = self.__get_model_path()
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def train(self, img_dir_path, test_dir_path, steps_per_epoch=50000, epochs=20):
        # 记录所有训练过程，每隔一定步数记录最大值
        log_dir = os.path.join(self.model_root_path, self.model_type.name, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensor_board = TensorBoard(log_dir=log_dir)

        # 保存最好的预测数据
        checkpoint_path = os.path.join(self.model_root_path,
                                       'checkpoint' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.h5')
        checkpoint = ModelCheckpoint(checkpoint_path,
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
        callback_lists = [tensor_board, checkpoint, learning_rate_reduction]

        if test_dir_path is None:
            history = self.model.fit_generator(OcrTrainer.__load_image_form_path_generator(img_dir_path),
                                               steps_per_epoch=steps_per_epoch,
                                               epochs=epochs,
                                               # validation_data=(x_test, y_test),
                                               callbacks=callback_lists)
        else:
            x_test, y_test = OcrTrainer.load_img_to_test_data(test_dir_path)
            history = self.model.fit_generator(OcrTrainer.__load_image_form_path_generator(img_dir_path),
                                               steps_per_epoch=steps_per_epoch,
                                               epochs=epochs,
                                               validation_data=(x_test, y_test),
                                               callbacks=callback_lists)

        model_path = self.__get_model_path()
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def __get_model_path(self):
        return os.path.join(self.model_root_path, self.model_type.name + '.h5')

    def load_and_train(self, model_root_path):
        # TODO
        pass

    @staticmethod
    def load_img_to_test_data(*dir_paths, img_width=INPUT_IMG_WIDTH, img_height=INPUT_IMG_HEIGHT):
        """
        将指定路径的图片转换为模型所需要的形式, 其中图片文件名形式应为:
        [tag]_[random string].[jpg/png]
        第一个字符为标签数据切以'_'分割,
        如: 0_2lfjrmre.jpg, 其中图片的tag为0
        """
        dir_len = len(dir_paths)
        data_label = []  # 存放类标签
        data = []
        for i_dir in range(dir_len):
            data_path = []
            g = os.walk(dir_paths[i_dir])
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    d = os.path.join(path, file_name)
                    data_path.append((file_name[0], d))

            count = 0
            for path in data_path:
                data_label.append(path[0])
                ig = cv2.imread(path[1], 0)
                ig = cv2.resize(ig, (img_width, img_height))
                data.append(ig / 255)
                count += 1
        x_data = np.array(data).reshape(-1, img_height, img_width, 1).astype('float32')
        y_data = np_utils.to_categorical(data_label, num_classes=10)
        return x_data, y_data

    def __load_model(self, model_root_path):
        self.model_root_path = model_root_path
        self.model = load_model(self.__get_model_path())

    def load_and_evaluate(self, model_root_path, validate_data_dir_path):
        self.__load_model(model_root_path)
        x, y = OcrTrainer.load_img_to_test_data(validate_data_dir_path)
        loss_and_metrics = self.model.evaluate(x, y, verbose=1)
        print(" Accuracy: {0}%, Loss: {1}".format(loss_and_metrics[1] * 100, loss_and_metrics[0]))

    @staticmethod
    def __get_img_generator(img_dir_path):
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            img_dir_path,
            target_size=(48, 48),
            batch_size=BATCH_SIZE)
        return train_generator

    @staticmethod
    def __load_image_form_path_generator(img_dir_path):
        data_path = []
        g = os.walk(img_dir_path)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                d = os.path.join(path, file_name)
                data_path.append((file_name[0], d))
        random.shuffle(data_path)
        count = 0
        while 1:
            for path in data_path:
                x = cv2.imread(path[1], 0).reshape(-1, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 1).astype('float32') / 255.0
                y = np_utils.to_categorical(path[0], num_classes=10).reshape(-1, 10)
                yield (x, y)
                count += 1

    @staticmethod
    def __load_vgg_image_generator(img_dir_path):
        data_path = []
        g = os.walk(img_dir_path)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                d = os.path.join(path, file_name)
                data_path.append((file_name[0], d))
        random.shuffle(data_path)
        count = 0
        while 1:
            for path in data_path:
                ig = cv2.imread(path[1], 0)
                ig = cv2.cvtColor(cv2.resize(ig, (VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT)), cv2.COLOR_GRAY2RGB)
                x = ig.reshape(-1, VGG_INPUT_IMG_HEIGHT, VGG_INPUT_IMG_WIDTH, 3).astype('float32') / 255.0
                y = np_utils.to_categorical(path[0], num_classes=10).reshape(-1, 10)
                yield (x, y)
                count += 1


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


if __name__ == "__main__":
    model = OcrModel.get_model(ModelType.RCNN)
    model.load_weights('./model-DIGIT-WITH-SPACE-RCNN-ex.h5')
    # model = load_model('./model-DIGIT-WITH-SPACE-RCNN-ex.h5')
    model.summary()
    print(model.input)
    print(model.output)
    # # 自定义output_namest
    # frozen_graph = freeze_session(K.get_session())
    # tf.train.write_graph(frozen_graph, "./digit_model", "model.pb", as_text=False)
    # from tensorflow.python.tools import import_pb_to_tensorboard
    #
    # import_pb_to_tensorboard.import_to_tensorboard("./digit_model/model.pb", "./digit_model/log")

    # trainer = OcrTrainer()
    # trainer.generate_model("path/for/model", model_type=ModelType.RCNN)
    # trainer.model.summary()
    # trainer.vgg_train("path/for/train",
    #                   "path/for/test")

    # x, y = OcrTrainer.load_img_to_test_data_to_28_28('D:/_data/_data_2nd_check_threshold_deline_resize/test')
    # model = load_model('D:/_model/simple_cnn_deline_with_mnist_data/SimpleCNN.h5 ')
    # loss_and_metrics1 = model.evaluate(x, y, verbose=1)
    # print(
    #     "threshold_data_2nd_check -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics1[1] * 100,
    #                                                                    loss_and_metrics1[0]))

    # x, y = OcrTrainer.load_img_to_test_data('D:/_data/_data_2nd_check_threshold_deline/test')
    # model = load_model('./threshold_data_2nd_check.h5')
    # loss_and_metrics1 = model.evaluate(x, y, verbose=1)
    #
    # model = load_model('./threshold_with_deline_data_2nd_check.h5')
    # loss_and_metrics2 = model.evaluate(x, y, verbose=1)
    #
    # model = load_model('./threshold_with_deline_data_2nd_check_997.h5')
    # loss_and_metrics3 = model.evaluate(x, y, verbose=1)

    # model = load_model('D:/_model/simple_cnn_deline_with_mnist_30_42_data/SimpleCNN.h5')
    # loss_and_metrics4 = model.evaluate(x, y, verbose=1)

    # print(
    #     "threshold_data_2nd_check -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics1[1] * 100,
    #                                                                    loss_and_metrics1[0]))
    # print(
    #     "threshold_with_deline_data_2nd_check -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics2[1] * 100,
    #                                                                                loss_and_metrics2[0]))
    # print(
    #     "threshold_with_deline_data_2nd_check_997 -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics3[1] * 100,
    #                                                                                    loss_and_metrics3[0]))
    # print(
    #     "simple_cnn_deline_with_mnist_30_42_data -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics4[1] * 100,
    #                                                                                    loss_and_metrics4[0]))
