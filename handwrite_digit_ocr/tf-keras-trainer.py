import base64

from handwrite_digit_ocr.dataset_utils import load_character_dataset, load_digit_dataset
from handwrite_digit_ocr.ocr_model import ModelType, OcrModel, DataType


def character_ocr_train():
    x_train, y_train, x_test, y_test = load_character_dataset('E:/_dataset/EMNIST/EMNIST-balanced.npz', 'ak')
    print(x_train.shape, y_train.shape, y_train)
    model = OcrModel.get_model(ModelType.RCNN, 11)
    model.summary()
    model.fit(x_train, y_train, batch_size=192, epochs=50, verbose=1, shuffle=True, validation_split=0.85)
    model.save('model-EMNIST-RCNN-balanced-a-to-k-keras.h5')


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
    character_ocr_train()

    # img_arr = []
    # g = os.walk('./test')
    # for path, dir_list, file_list in g:
    #     for file_name in file_list:
    #         img_arr.append(image_to_base64("./test/" + file_name))
    #         # t_img = cv2.imread("./test/" + file_name, 0)
    #         # t_img = cv2.resize(t_img, (28, 28))
    #         # t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
    #         # t_img = t_img.reshape(-1, 28, 28, 3)
    #         # predict_result = model.predict(t_img)
    #         # print(file_name, np.argmax(predict_result), chr(int(np.argmax(predict_result)) + 97), predict_result[0][np.argmax(predict_result)])
    #
    # gen_request_json_for_character_ocr(img_arr)
    #
    # for i in range(9):
    #     t_img = cv2.imread("./test/0"+str(i+1)+".jpg", 0)
    #     t_img = cv2.resize(t_img, (28, 28))
    #     t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
    #     t_img = t_img.reshape(-1, 28, 28, 3)
    #     predict_result = model.predict(t_img)
    #     print(np.argmax(predict_result), predict_result[0][np.argmax(predict_result)])

    # loss_and_metrics1 = model.evaluate(x_test_a_to_k, y_test_a_to_k, verbose=1)
    #
    # print(
    #     "threshold_data_2nd_check -- Accuracy: {0}%, Loss: {1}".format(loss_and_metrics1[1] * 100,
    #                                                                    loss_and_metrics1[0]))
