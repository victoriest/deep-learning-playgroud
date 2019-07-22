import numpy as np

import tensorflow as tf
import cv2

if __name__ == "__main__":
    mnist_model = tf.keras.models.load_model('./handwrite_digit_ocr/model-EMNIST-RCNN.h5')

    mnist_out = np.load('E:/_dataset/EMNIST/EMNIST.npz')
    x_train, y_train, x_test, y_test = mnist_out['x_train'], \
                                       mnist_out['y_train'], \
                                       mnist_out['x_test'], \
                                       mnist_out['y_test']

    # loss_and_metrics = mnist_model.evaluate(x_test, y_test, verbose=2)

    # print("Test Loss: {}".format(loss_and_metrics[0]))
    # print("Test Accuracy: {}%".format(loss_and_metrics[1] * 100))


    img = x_test[123]
    cv2.imshow("est", img)
    cv2.waitKey(0)


    img_data = img.reshape(-1, 28, 28, 3)
    predict_result = mnist_model.predict(img_data)
    print(np.argmax(predict_result), predict_result[0][np.argmax(predict_result)])




    # predicted_classes = mnist_model.predict_classes(x_test)
    #
    # correct_indices = np.nonzero(predicted_classes == y_test)[0]
    # incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
    # print("Classified correctly count: {}".format(len(correct_indices)))
    # print("Classified incorrectly count: {}".format(len(incorrect_indices)))
