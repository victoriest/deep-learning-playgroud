import cv2
import numpy as np

from variable_length_digit_htr import *
from variable_length_digit_htr.crnn import crnn_model

model = None


def decode(pred):
    pred_result_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != num_of_classes - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            pred_result_list.append(char_list[pred_text[i]])
    return u''.join(pred_result_list)


def load_model(weights_path):
    base_model, _ = crnn_model()
    base_model.load_weights(weights_path)
    return base_model


def digit_ocr(img):
    global model
    height, width = img.shape[0], img.shape[1]
    scale = height * 1.0 / img_h
    width = int(width / scale)
    img = cv2.resize(img, (width, img_h))
    img = np.array(img).astype(np.float32) / 255.0

    input_img = img.reshape([1, img_h, width, 1])

    if model is None:
        model = load_model('./model/crnn_200820.h5')
    y_pred = model.predict(input_img)
    y_pred = y_pred[:, :, :]

    out = decode(y_pred)

    return out


if __name__ == '__main__':
    img = cv2.imread("./6.jpg", 0)
    print(digit_ocr(cv2.imread("./6.jpg", 0)))