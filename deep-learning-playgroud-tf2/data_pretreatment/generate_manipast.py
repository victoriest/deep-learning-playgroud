import os

import cv2


def generate_yolov4_manipast():
    f = open('train.txt', 'w')
    g = os.walk("E:/_dataset/zyb_mark_qnsc_data/mixed")
    lines = []

    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.splitext(file_name)[1] == '.jpg' or os.path.splitext(file_name)[1] == '.png':
                print(os.path.join(path, file_name), os.path.splitext(file_name))
                lines.append("data/imgs/" + file_name + "\n")
    f.writelines(lines)
    f.close()


def generate_crnn_manipast():
    f = open('E:/_dataset/annotation.txt', 'w')
    g = os.walk("E:/_dataset/zyb_opt_mark_data")
    lines = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.splitext(file_name)[1] == '.jpg':
                label = os.path.splitext(file_name)[0].split("_")[0]
                print(os.path.join(path, file_name), os.path.splitext(file_name),
                      "img/" + file_name + " " + label + "\n")
                lines.append("img/" + file_name + " " + label + "\n")
    f.writelines(lines)
    f.close()


from concurrent.futures import ThreadPoolExecutor

count = 0


def conver_to_32_util(in_path, out_path):
    global count
    count += 1
    print(count, in_path, out_path)
    img = cv2.imread(in_path)
    resized = cv2.resize(img, (115, 32))
    cv2.imwrite(out_path, resized)


def convert_to_32():
    g = os.walk("E:/_dataset/_data_crnn_train")
    # with ThreadPoolExecutor(8) as executor:
    for path, dir_list, file_list in g:
        for file_name in file_list:
            p = os.path.join(path, file_name)
            img = cv2.imread(p)
            resized = cv2.resize(img, (115, 32))
            cv2.imwrite(os.path.join("E:/_dataset/_data_crnn_train_32", file_name), resized)
            global count
            count += 1
            print(count, p)


def convert_to_gray():
    g = os.walk("E:/_dataset/zyb_tuotuo_data/gray")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.splitext(file_name)[1] != '.jpg':
                continue
            p = os.path.join(path, file_name)
            img = cv2.imread(p, 0)
            cv2.imwrite(os.path.join("E:/_dataset/zyb_tuotuo_data/gray", file_name), img)
            global count
            count += 1
            print(count, p)


if __name__ == '__main__':
    generate_yolov4_manipast()
    # convert_to_gray()
