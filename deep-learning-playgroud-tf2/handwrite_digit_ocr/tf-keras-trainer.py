import base64
import os
from io import BytesIO

import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.python.keras.models import load_model

from handwrite_digit_ocr.dataset_utils import load_character_dataset, load_digit_dataset
from handwrite_digit_ocr.ocr_model import ModelType, OcrModel


def set_model(self, model):
    self.model = model
    # self.sess = K.get_session()
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
    x_train, y_train, x_test, y_test = load_character_dataset('E:/_dataset/RCNN/EMNIST-balanced-191127-added.npz', 'ad')
    # x_train, y_train, x_test, y_test = load_digit_dataset('E:/_dataset/t_f_tf_dataset/t_f_tf_dataset.npz')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 如果模型不传 则重新训练模型, 否则加载模型
    if model is None:
        model = OcrModel.get_model(ModelType.RCNN, 4)

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

    model_check_file_template = os.path.join("./model/", output_model_name + "-{epoch:02d}-{val_loss:.2f}.h5")

    checkpoint_callback = ModelCheckpoint(
        filepath=model_check_file_template, monitor='val_loss',
        save_best_only=False, save_weights_only=False, period=5)

    # 各种callback的ref: https://keras.io/zh/callbacks/
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(50)])
    change_learing_rate_callback = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]), verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

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
              epochs=50,
              verbose=1,
              shuffle=True,
              # validation_split=0.15,
              validation_data=(x_test, y_test),
              callbacks=[tensor_board_callback,
                         checkpoint_callback,
                         change_learing_rate_callback,
                         early_stopping_callback])

    model_check_file_template = os.path.join("./model/", output_model_name + ".h5")

    model.save(model_check_file_template)


def character_ocr_evaluate():
    x_train, y_train, x_test, y_test = load_character_dataset('E:/_dataset/RCNN/EMNIST-balanced-191127-added.npz', 'ak')
    model = load_model('./model/model-EMNIST-RCNN-balanced-a-to-g-200511.h5')
    loss_and_metrics1 = model.evaluate(x_test, y_test, verbose=1)
    print(
        "threshold_data_2nd_check -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics1[1] * 100,
                                                                       loss_and_metrics1[0]))


def character_ocr_predict():
    model = load_model('./model/model-EMNIST-RCNN-balanced.h5')
    img_arr = []
    g = os.walk('./out_test')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            img_arr.append(image_to_base64("./out_test/" + file_name))
            t_img = cv2.imread("./out_test/" + file_name, 0)
            t_img = cv2.resize(t_img, (28, 28))
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
            t_img = t_img.reshape(-1, 28, 28, 3)
            predict_result = model.predict(t_img)
            print(file_name, np.argmax(predict_result), int(np.argmax(predict_result)),
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
    # character_ocr_train('model-EMNIST-RCNN-balanced-a-to-d-200715')
    print(tf.__version__)
    # character_ocr_predict()

    # model_path = 'v7_resnet50_19-0.9068-0.8000.h5'
    # model = tf.keras.models.load_model('./model/model-EMNIST-RCNN-balanced-a-to-d-200715.h5')
    # model.save('./model/RCNN-a-to-d/', save_format='tf')
    import time
    import numpy as np
    import requests
    from tensorflow.keras.preprocessing import image

    input_json = {
        "id": "B44DD4B4E9",
        "characterPos": [{
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/CQBkmvxa/aF/wCDob4nfBD9vbVnb9mzUrr9mnwd4pufBHivxPb2qzXUmtQuxe6glRthUIo2wk/MpJJzgD9A/wDgsv8AtsP/AME//wDgnP8AEb9ojSZtmuwaV/ZvhcbVP/Eyuj5MDYbghCxkI7hDX5Rf8Emf2kf2vf8Agnz+xBB8LPjV/wAEPPil8SfD/jTWpvE2o+J7WOPUDrF3dhHilNk8LeWnkonzEnkZ/ioA/QP4J/8ABf8A+Ef7XOsjS/2Kf2OfjX8S7WK5tItS12z8LR2On2AmlCFpZriUE7EzIQqtkCvvwEkAkY46V4n+wH+0Gv7S37PVj8RR+yn4r+DeLl7X/hDPF+iJY3EIjAAdI0wDERwpwv3TxxXtlAH4w/8AB5d+0Pq/g39mf4T/ALNttp1zNpnjfxq+o6+be0DtJb2CxlYlbPyuXnBA77eo7/Tf7L3/AAcPf8Ea9b+CPh/ToP2rdL8IHRtDtLKTw/4otLm2ubMxwqnk/MhEhXbjcpYHHWvu7xP4F8E+Nvsv/CZ+D9L1b7DcC4sv7T0+Of7PKOkib1O1h6jmvnTxz/wRZ/4JV/En4gXnxR8a/sL+AL7XNQvPtd7etpOwTTZyXZEIQknk/Lyc5zmgDyjQf+Dkv/gmH8QPHulfDb4J+KvHPxA1XV7hYLWHwd8Pb+5AkLhQrFo0xyQc8gDkmvvWCXz4Um8tk3qG2uMEZHQjsa5r4bfBX4P/AAc0WHw78Jvhb4e8NWNuoWG00LR4bVEAUKMCNR2AH4CunoA//9k="
        }, {
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/K/M/wD4K0/8HDXhj9jL4tW37Ff7GnwlPxe+OerSrZx6NZXJa00e6clUjmWIM802RkwAoQMFnWvU/wDg4L/b5+MP/BOP/gnLq/x9+Amp6TaeK7vxHY6Jpdxq1uJhH9oEpd40JAaVUjZhnIG0kg4r8Jf+CW37J3/BeQavq/7TX7H/AOzEkfiH4t26ajYfG7xZbWons7e4lZ7i4tJrlysf2gStvYRs5TO3FAH6ef8ABPT/AILP/wDBR7wP/wAFEPD/APwTr/4LA/BTQ/DniH4naUmqeBL/AESOOA2PmecIreZUd0kV2t5EBDb1cYbOeP17r8kf+Cfn/Bu3+0l4Y/bR8Nf8FFf+Cnf7a9x8UfiF4WlS40TRbWKWaC3mRW8ovczMCVjd2dY44kXcM55Ir9bqAPzt/wCDnv8AYb+Kv7c3/BMW+0H4KaN/afiHwH4pt/FkWlxoWmvbe3tbqGeKEDrJsuC4XBLeXtHJFfN//BIT/g5h/wCCd/wo/Yg8A/syftaeINY+HHi/4Y+FrPw1ewXui3N3BeiyiS3WZGhjZkdgmWjdQVYEZPFftFXz58fP+CUn/BOP9qDxa3j347/sb+BfEOtyO7z6rcaMsdxO7HLNI8W0yEnnLZPJ9TQB8C+Fv+DnTV/2zP8Agoj8L/2SP+Cb37N+qeK/COo+K7eHx/4t1uBopP7MaUxzzwRAHyI4oyJvNlbLY2bF+8f19rj/AILfs+fA39nHwjF4C+Avwk8PeENHh+5p/h/SorWMnjLMEA3E4GSck4rsKAP/2Q=="
        }, {
            "areaId": "DD1256D706",
            "idx": 0,
            "img": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAqAB4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD30ClxQKhvLqOztJbiVgEjUsc+1AHnvjfxPfaB4o01rGdpEb5bi3b7u0kc/WvRVIdFYdxmvIoYp/FFlr2vTpmPyzHbBunBr07QJmufD+nyvje0CbsHvigDSFcZ48uHuYLXR4Ml7qUK+08ha7MdK4WzZdX+Ity+1tlmgz6Z6CgCHS9RawtbnQE0h3trVPLLxnOSR3rX8DXRk0AW7KwaCRkz2IzxStG+m+KZySEt7+LgkfxgUeCtg0y5jQuQty43Mu3PPb2oA6gdK5vw9o1zp2r6tPcKmyaUGFw3JXrz+NdCvSl7/hQBm69pi6jp5AQNPEfMhycfMOlZfg97iWK+a4tXt2E20o4I5A5x6jNdN2pp/rQB/9k="
        },{
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/CQBkmvxa/aF/wCDob4nfBD9vbVnb9mzUrr9mnwd4pufBHivxPb2qzXUmtQuxe6glRthUIo2wk/MpJJzgD9A/wDgsv8AtsP/AME//wDgnP8AEb9ojSZtmuwaV/ZvhcbVP/Eyuj5MDYbghCxkI7hDX5Rf8Emf2kf2vf8Agnz+xBB8LPjV/wAEPPil8SfD/jTWpvE2o+J7WOPUDrF3dhHilNk8LeWnkonzEnkZ/ioA/QP4J/8ABf8A+Ef7XOsjS/2Kf2OfjX8S7WK5tItS12z8LR2On2AmlCFpZriUE7EzIQqtkCvvwEkAkY46V4n+wH+0Gv7S37PVj8RR+yn4r+DeLl7X/hDPF+iJY3EIjAAdI0wDERwpwv3TxxXtlAH4w/8AB5d+0Pq/g39mf4T/ALNttp1zNpnjfxq+o6+be0DtJb2CxlYlbPyuXnBA77eo7/Tf7L3/AAcPf8Ea9b+CPh/ToP2rdL8IHRtDtLKTw/4otLm2ubMxwqnk/MhEhXbjcpYHHWvu7xP4F8E+Nvsv/CZ+D9L1b7DcC4sv7T0+Of7PKOkib1O1h6jmvnTxz/wRZ/4JV/En4gXnxR8a/sL+AL7XNQvPtd7etpOwTTZyXZEIQknk/Lyc5zmgDyjQf+Dkv/gmH8QPHulfDb4J+KvHPxA1XV7hYLWHwd8Pb+5AkLhQrFo0xyQc8gDkmvvWCXz4Um8tk3qG2uMEZHQjsa5r4bfBX4P/AAc0WHw78Jvhb4e8NWNuoWG00LR4bVEAUKMCNR2AH4CunoA//9k="
        }, {
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/K/M/wD4K0/8HDXhj9jL4tW37Ff7GnwlPxe+OerSrZx6NZXJa00e6clUjmWIM802RkwAoQMFnWvU/wDg4L/b5+MP/BOP/gnLq/x9+Amp6TaeK7vxHY6Jpdxq1uJhH9oEpd40JAaVUjZhnIG0kg4r8Jf+CW37J3/BeQavq/7TX7H/AOzEkfiH4t26ajYfG7xZbWons7e4lZ7i4tJrlysf2gStvYRs5TO3FAH6ef8ABPT/AILP/wDBR7wP/wAFEPD/APwTr/4LA/BTQ/DniH4naUmqeBL/AESOOA2PmecIreZUd0kV2t5EBDb1cYbOeP17r8kf+Cfn/Bu3+0l4Y/bR8Nf8FFf+Cnf7a9x8UfiF4WlS40TRbWKWaC3mRW8ovczMCVjd2dY44kXcM55Ir9bqAPzt/wCDnv8AYb+Kv7c3/BMW+0H4KaN/afiHwH4pt/FkWlxoWmvbe3tbqGeKEDrJsuC4XBLeXtHJFfN//BIT/g5h/wCCd/wo/Yg8A/syftaeINY+HHi/4Y+FrPw1ewXui3N3BeiyiS3WZGhjZkdgmWjdQVYEZPFftFXz58fP+CUn/BOP9qDxa3j347/sb+BfEOtyO7z6rcaMsdxO7HLNI8W0yEnnLZPJ9TQB8C+Fv+DnTV/2zP8Agoj8L/2SP+Cb37N+qeK/COo+K7eHx/4t1uBopP7MaUxzzwRAHyI4oyJvNlbLY2bF+8f19rj/AILfs+fA39nHwjF4C+Avwk8PeENHh+5p/h/SorWMnjLMEA3E4GSck4rsKAP/2Q=="
        }, {
            "areaId": "DD1256D706",
            "idx": 0,
            "img": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAqAB4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD30ClxQKhvLqOztJbiVgEjUsc+1AHnvjfxPfaB4o01rGdpEb5bi3b7u0kc/WvRVIdFYdxmvIoYp/FFlr2vTpmPyzHbBunBr07QJmufD+nyvje0CbsHvigDSFcZ48uHuYLXR4Ml7qUK+08ha7MdK4WzZdX+Ity+1tlmgz6Z6CgCHS9RawtbnQE0h3trVPLLxnOSR3rX8DXRk0AW7KwaCRkz2IzxStG+m+KZySEt7+LgkfxgUeCtg0y5jQuQty43Mu3PPb2oA6gdK5vw9o1zp2r6tPcKmyaUGFw3JXrz+NdCvSl7/hQBm69pi6jp5AQNPEfMhycfMOlZfg97iWK+a4tXt2E20o4I5A5x6jNdN2pp/rQB/9k="
        },{
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/CQBkmvxa/aF/wCDob4nfBD9vbVnb9mzUrr9mnwd4pufBHivxPb2qzXUmtQuxe6glRthUIo2wk/MpJJzgD9A/wDgsv8AtsP/AME//wDgnP8AEb9ojSZtmuwaV/ZvhcbVP/Eyuj5MDYbghCxkI7hDX5Rf8Emf2kf2vf8Agnz+xBB8LPjV/wAEPPil8SfD/jTWpvE2o+J7WOPUDrF3dhHilNk8LeWnkonzEnkZ/ioA/QP4J/8ABf8A+Ef7XOsjS/2Kf2OfjX8S7WK5tItS12z8LR2On2AmlCFpZriUE7EzIQqtkCvvwEkAkY46V4n+wH+0Gv7S37PVj8RR+yn4r+DeLl7X/hDPF+iJY3EIjAAdI0wDERwpwv3TxxXtlAH4w/8AB5d+0Pq/g39mf4T/ALNttp1zNpnjfxq+o6+be0DtJb2CxlYlbPyuXnBA77eo7/Tf7L3/AAcPf8Ea9b+CPh/ToP2rdL8IHRtDtLKTw/4otLm2ubMxwqnk/MhEhXbjcpYHHWvu7xP4F8E+Nvsv/CZ+D9L1b7DcC4sv7T0+Of7PKOkib1O1h6jmvnTxz/wRZ/4JV/En4gXnxR8a/sL+AL7XNQvPtd7etpOwTTZyXZEIQknk/Lyc5zmgDyjQf+Dkv/gmH8QPHulfDb4J+KvHPxA1XV7hYLWHwd8Pb+5AkLhQrFo0xyQc8gDkmvvWCXz4Um8tk3qG2uMEZHQjsa5r4bfBX4P/AAc0WHw78Jvhb4e8NWNuoWG00LR4bVEAUKMCNR2AH4CunoA//9k="
        }, {
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/K/M/wD4K0/8HDXhj9jL4tW37Ff7GnwlPxe+OerSrZx6NZXJa00e6clUjmWIM802RkwAoQMFnWvU/wDg4L/b5+MP/BOP/gnLq/x9+Amp6TaeK7vxHY6Jpdxq1uJhH9oEpd40JAaVUjZhnIG0kg4r8Jf+CW37J3/BeQavq/7TX7H/AOzEkfiH4t26ajYfG7xZbWons7e4lZ7i4tJrlysf2gStvYRs5TO3FAH6ef8ABPT/AILP/wDBR7wP/wAFEPD/APwTr/4LA/BTQ/DniH4naUmqeBL/AESOOA2PmecIreZUd0kV2t5EBDb1cYbOeP17r8kf+Cfn/Bu3+0l4Y/bR8Nf8FFf+Cnf7a9x8UfiF4WlS40TRbWKWaC3mRW8ovczMCVjd2dY44kXcM55Ir9bqAPzt/wCDnv8AYb+Kv7c3/BMW+0H4KaN/afiHwH4pt/FkWlxoWmvbe3tbqGeKEDrJsuC4XBLeXtHJFfN//BIT/g5h/wCCd/wo/Yg8A/syftaeINY+HHi/4Y+FrPw1ewXui3N3BeiyiS3WZGhjZkdgmWjdQVYEZPFftFXz58fP+CUn/BOP9qDxa3j347/sb+BfEOtyO7z6rcaMsdxO7HLNI8W0yEnnLZPJ9TQB8C+Fv+DnTV/2zP8Agoj8L/2SP+Cb37N+qeK/COo+K7eHx/4t1uBopP7MaUxzzwRAHyI4oyJvNlbLY2bF+8f19rj/AILfs+fA39nHwjF4C+Avwk8PeENHh+5p/h/SorWMnjLMEA3E4GSck4rsKAP/2Q=="
        }, {
            "areaId": "DD1256D706",
            "idx": 0,
            "img": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAqAB4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD30ClxQKhvLqOztJbiVgEjUsc+1AHnvjfxPfaB4o01rGdpEb5bi3b7u0kc/WvRVIdFYdxmvIoYp/FFlr2vTpmPyzHbBunBr07QJmufD+nyvje0CbsHvigDSFcZ48uHuYLXR4Ml7qUK+08ha7MdK4WzZdX+Ity+1tlmgz6Z6CgCHS9RawtbnQE0h3trVPLLxnOSR3rX8DXRk0AW7KwaCRkz2IzxStG+m+KZySEt7+LgkfxgUeCtg0y5jQuQty43Mu3PPb2oA6gdK5vw9o1zp2r6tPcKmyaUGFw3JXrz+NdCvSl7/hQBm69pi6jp5AQNPEfMhycfMOlZfg97iWK+a4tXt2E20o4I5A5x6jNdN2pp/rQB/9k="
        },{
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/CQBkmvxa/aF/wCDob4nfBD9vbVnb9mzUrr9mnwd4pufBHivxPb2qzXUmtQuxe6glRthUIo2wk/MpJJzgD9A/wDgsv8AtsP/AME//wDgnP8AEb9ojSZtmuwaV/ZvhcbVP/Eyuj5MDYbghCxkI7hDX5Rf8Emf2kf2vf8Agnz+xBB8LPjV/wAEPPil8SfD/jTWpvE2o+J7WOPUDrF3dhHilNk8LeWnkonzEnkZ/ioA/QP4J/8ABf8A+Ef7XOsjS/2Kf2OfjX8S7WK5tItS12z8LR2On2AmlCFpZriUE7EzIQqtkCvvwEkAkY46V4n+wH+0Gv7S37PVj8RR+yn4r+DeLl7X/hDPF+iJY3EIjAAdI0wDERwpwv3TxxXtlAH4w/8AB5d+0Pq/g39mf4T/ALNttp1zNpnjfxq+o6+be0DtJb2CxlYlbPyuXnBA77eo7/Tf7L3/AAcPf8Ea9b+CPh/ToP2rdL8IHRtDtLKTw/4otLm2ubMxwqnk/MhEhXbjcpYHHWvu7xP4F8E+Nvsv/CZ+D9L1b7DcC4sv7T0+Of7PKOkib1O1h6jmvnTxz/wRZ/4JV/En4gXnxR8a/sL+AL7XNQvPtd7etpOwTTZyXZEIQknk/Lyc5zmgDyjQf+Dkv/gmH8QPHulfDb4J+KvHPxA1XV7hYLWHwd8Pb+5AkLhQrFo0xyQc8gDkmvvWCXz4Um8tk3qG2uMEZHQjsa5r4bfBX4P/AAc0WHw78Jvhb4e8NWNuoWG00LR4bVEAUKMCNR2AH4CunoA//9k="
        }, {
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/K/M/wD4K0/8HDXhj9jL4tW37Ff7GnwlPxe+OerSrZx6NZXJa00e6clUjmWIM802RkwAoQMFnWvU/wDg4L/b5+MP/BOP/gnLq/x9+Amp6TaeK7vxHY6Jpdxq1uJhH9oEpd40JAaVUjZhnIG0kg4r8Jf+CW37J3/BeQavq/7TX7H/AOzEkfiH4t26ajYfG7xZbWons7e4lZ7i4tJrlysf2gStvYRs5TO3FAH6ef8ABPT/AILP/wDBR7wP/wAFEPD/APwTr/4LA/BTQ/DniH4naUmqeBL/AESOOA2PmecIreZUd0kV2t5EBDb1cYbOeP17r8kf+Cfn/Bu3+0l4Y/bR8Nf8FFf+Cnf7a9x8UfiF4WlS40TRbWKWaC3mRW8ovczMCVjd2dY44kXcM55Ir9bqAPzt/wCDnv8AYb+Kv7c3/BMW+0H4KaN/afiHwH4pt/FkWlxoWmvbe3tbqGeKEDrJsuC4XBLeXtHJFfN//BIT/g5h/wCCd/wo/Yg8A/syftaeINY+HHi/4Y+FrPw1ewXui3N3BeiyiS3WZGhjZkdgmWjdQVYEZPFftFXz58fP+CUn/BOP9qDxa3j347/sb+BfEOtyO7z6rcaMsdxO7HLNI8W0yEnnLZPJ9TQB8C+Fv+DnTV/2zP8Agoj8L/2SP+Cb37N+qeK/COo+K7eHx/4t1uBopP7MaUxzzwRAHyI4oyJvNlbLY2bF+8f19rj/AILfs+fA39nHwjF4C+Avwk8PeENHh+5p/h/SorWMnjLMEA3E4GSck4rsKAP/2Q=="
        }, {
            "areaId": "DD1256D706",
            "idx": 0,
            "img": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAqAB4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD30ClxQKhvLqOztJbiVgEjUsc+1AHnvjfxPfaB4o01rGdpEb5bi3b7u0kc/WvRVIdFYdxmvIoYp/FFlr2vTpmPyzHbBunBr07QJmufD+nyvje0CbsHvigDSFcZ48uHuYLXR4Ml7qUK+08ha7MdK4WzZdX+Ity+1tlmgz6Z6CgCHS9RawtbnQE0h3trVPLLxnOSR3rX8DXRk0AW7KwaCRkz2IzxStG+m+KZySEt7+LgkfxgUeCtg0y5jQuQty43Mu3PPb2oA6gdK5vw9o1zp2r6tPcKmyaUGFw3JXrz+NdCvSl7/hQBm69pi6jp5AQNPEfMhycfMOlZfg97iWK+a4tXt2E20o4I5A5x6jNdN2pp/rQB/9k="
        },{
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/CQBkmvxa/aF/wCDob4nfBD9vbVnb9mzUrr9mnwd4pufBHivxPb2qzXUmtQuxe6glRthUIo2wk/MpJJzgD9A/wDgsv8AtsP/AME//wDgnP8AEb9ojSZtmuwaV/ZvhcbVP/Eyuj5MDYbghCxkI7hDX5Rf8Emf2kf2vf8Agnz+xBB8LPjV/wAEPPil8SfD/jTWpvE2o+J7WOPUDrF3dhHilNk8LeWnkonzEnkZ/ioA/QP4J/8ABf8A+Ef7XOsjS/2Kf2OfjX8S7WK5tItS12z8LR2On2AmlCFpZriUE7EzIQqtkCvvwEkAkY46V4n+wH+0Gv7S37PVj8RR+yn4r+DeLl7X/hDPF+iJY3EIjAAdI0wDERwpwv3TxxXtlAH4w/8AB5d+0Pq/g39mf4T/ALNttp1zNpnjfxq+o6+be0DtJb2CxlYlbPyuXnBA77eo7/Tf7L3/AAcPf8Ea9b+CPh/ToP2rdL8IHRtDtLKTw/4otLm2ubMxwqnk/MhEhXbjcpYHHWvu7xP4F8E+Nvsv/CZ+D9L1b7DcC4sv7T0+Of7PKOkib1O1h6jmvnTxz/wRZ/4JV/En4gXnxR8a/sL+AL7XNQvPtd7etpOwTTZyXZEIQknk/Lyc5zmgDyjQf+Dkv/gmH8QPHulfDb4J+KvHPxA1XV7hYLWHwd8Pb+5AkLhQrFo0xyQc8gDkmvvWCXz4Um8tk3qG2uMEZHQjsa5r4bfBX4P/AAc0WHw78Jvhb4e8NWNuoWG00LR4bVEAUKMCNR2AH4CunoA//9k="
        }, {
            "areaId": "123123",
            "idx": 111,
            "img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/K/M/wD4K0/8HDXhj9jL4tW37Ff7GnwlPxe+OerSrZx6NZXJa00e6clUjmWIM802RkwAoQMFnWvU/wDg4L/b5+MP/BOP/gnLq/x9+Amp6TaeK7vxHY6Jpdxq1uJhH9oEpd40JAaVUjZhnIG0kg4r8Jf+CW37J3/BeQavq/7TX7H/AOzEkfiH4t26ajYfG7xZbWons7e4lZ7i4tJrlysf2gStvYRs5TO3FAH6ef8ABPT/AILP/wDBR7wP/wAFEPD/APwTr/4LA/BTQ/DniH4naUmqeBL/AESOOA2PmecIreZUd0kV2t5EBDb1cYbOeP17r8kf+Cfn/Bu3+0l4Y/bR8Nf8FFf+Cnf7a9x8UfiF4WlS40TRbWKWaC3mRW8ovczMCVjd2dY44kXcM55Ir9bqAPzt/wCDnv8AYb+Kv7c3/BMW+0H4KaN/afiHwH4pt/FkWlxoWmvbe3tbqGeKEDrJsuC4XBLeXtHJFfN//BIT/g5h/wCCd/wo/Yg8A/syftaeINY+HHi/4Y+FrPw1ewXui3N3BeiyiS3WZGhjZkdgmWjdQVYEZPFftFXz58fP+CUn/BOP9qDxa3j347/sb+BfEOtyO7z6rcaMsdxO7HLNI8W0yEnnLZPJ9TQB8C+Fv+DnTV/2zP8Agoj8L/2SP+Cb37N+qeK/COo+K7eHx/4t1uBopP7MaUxzzwRAHyI4oyJvNlbLY2bF+8f19rj/AILfs+fA39nHwjF4C+Avwk8PeENHh+5p/h/SorWMnjLMEA3E4GSck4rsKAP/2Q=="
        }, {
            "areaId": "DD1256D706",
            "idx": 0,
            "img": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAqAB4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD30ClxQKhvLqOztJbiVgEjUsc+1AHnvjfxPfaB4o01rGdpEb5bi3b7u0kc/WvRVIdFYdxmvIoYp/FFlr2vTpmPyzHbBunBr07QJmufD+nyvje0CbsHvigDSFcZ48uHuYLXR4Ml7qUK+08ha7MdK4WzZdX+Ity+1tlmgz6Z6CgCHS9RawtbnQE0h3trVPLLxnOSR3rX8DXRk0AW7KwaCRkz2IzxStG+m+KZySEt7+LgkfxgUeCtg0y5jQuQty43Mu3PPb2oA6gdK5vw9o1zp2r6tPcKmyaUGFw3JXrz+NdCvSl7/hQBm69pi6jp5AQNPEfMhycfMOlZfg97iWK+a4tXt2E20o4I5A5x6jNdN2pp/rQB/9k="
        }
        ]
    }

    start = time.time()

    instances = []
    iii = None
    payload = None
    for cp in input_json['characterPos']:
        img_base64 = cp['img']
        byte_data = base64.b64decode(img_base64)
        image_data = BytesIO(byte_data)
        origin_img = Image.open(image_data)
        img = origin_img.resize([28, 28], Image.ANTIALIAS)
        img = image.img_to_array(img) / 255.
        img = img.astype('float16')
        instances.append(img.tolist())

    payload = {
        "instances": instances
    }
    print(payload)
    # sending post request to TensorFlow Serving server
    r = requests.post('http://192.168.31.226:8501/v1/models/RCNN-a-to-d:predict', json=payload)
    print(r.content)

    print(time.time() - start)
