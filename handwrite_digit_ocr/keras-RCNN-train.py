import numpy as np

from handwrite_digit_ocr.ocr_model import ModelType, OcrModel


def character_ocr_train():
    mnist_out = np.load('E:/_dataset/EMNIST/EMNIST.npz')
    x_train, y_train, x_test, y_test = mnist_out['x_train'], \
                                       mnist_out['y_train'], \
                                       mnist_out['x_test'], \
                                       mnist_out['y_test']

    model = OcrModel.get_model(ModelType.RCNN)

    model.summary()

    # # model = load_model('model-RCNN.h5')
    model.fit(x_train, y_train, batch_size=192, epochs=20, verbose=1, shuffle=True)

    model.save('model-EMNIST-RCNN.h5')


def digit_ocr_train():
    mnist_out = np.load('C:/_project/victoriest_digit_recognizer/dataset_28_28_3/mnist.npz')
    mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = mnist_out['x_train'], \
                                                               mnist_out['y_train'], \
                                                               mnist_out['x_test'], \
                                                               mnist_out['y_test']

    zk_out = np.load('C:/_project/victoriest_digit_recognizer/dataset_28_28_3/zk.npz')
    zk_x_train, zk_y_train, zk_x_test, zk_y_test = zk_out['x_train'], \
                                                   zk_out['y_train'], \
                                                   zk_out['x_test'], \
                                                   zk_out['y_test']

    xbk_out = np.load('C:/_project/victoriest_digit_recognizer/dataset_28_28_3/xbk.npz')
    xbk_x_train, xbk_y_train, xbk_x_test, xbk_y_test = xbk_out['x_train'], \
                                                       xbk_out['y_train'], \
                                                       xbk_out['x_test'], \
                                                       xbk_out['y_test']

    print(mnist_x_train.shape, zk_x_train.shape, xbk_x_train.shape)
    print(mnist_y_train.shape, zk_y_train.shape, xbk_y_train.shape)
    print(mnist_x_test.shape, zk_x_test.shape, xbk_x_test.shape)
    print(mnist_y_test.shape, zk_y_test.shape, xbk_y_test.shape)

    x_train = np.concatenate((mnist_x_train, zk_x_train, xbk_x_train), axis=0)
    y_train = np.concatenate((mnist_y_train, zk_y_train, xbk_y_train), axis=0)

    x_test = np.concatenate((mnist_x_test, zk_x_test, xbk_x_test), axis=0)
    y_test = np.concatenate((mnist_y_test, zk_y_test, xbk_y_test), axis=0)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


if __name__ == '__main__':
    character_ocr_train()