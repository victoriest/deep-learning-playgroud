import os

from shutil import copyfile

import cv2
import numpy as np

if __name__ == '__main__':
    word_tag_map = {}
    max_label = 0
    with open('./sentences.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            line_arr = line.split(' ')
            l = len(line_arr)
            sentences = line_arr[l - 1]
            print(line_arr[0])
            tmp = len(sentences)
            if tmp > max_label:
                max_label = tmp

            # sentences_arr = sentences.split('|')
            # tmp = 0
            # for word in sentences_arr:
            #     tmp += 1
            #     for c in word:
            #         if c in word_tag_map:
            #             word_tag_map[c] += 1
            #         else:
            #             word_tag_map[c] = 1
            # if tmp > max_label:
            #     max_label = tmp


    for k, v in word_tag_map.items():
        print(k, v)
    print(max_label)

    # g = os.walk("E:/_dataset/IAM/sentences")
    # for path, dir_list, file_list in g:
    #     for file_name in file_list:
    #         print(os.path.join(path, file_name))
    #         dest_path = "E:/_dataset/IAM/en_hw"
    #         # copyfile(os.path.join(path, file_name),
    #         #          os.path.join(dest_path, file_name))
    #
    #         bg_img = np.ones((32, 280), np.uint8) * 255
    #
    #         img_path = os.path.join(path, file_name)
    #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         height, width = img.shape
    #
    #         tmp_img = None
    #         if width < height * 3:
    #             print(1)
    #             # 以高为准
    #             dest_h = 32
    #             dest_w = int(width * 32 / height)
    #             tmp_img = cv2.resize(img, (dest_w, dest_h))
    #             bg_img_roi = bg_img[0:dest_h, 0:dest_w]
    #             bg_img_roi = cv2.bitwise_and(tmp_img, bg_img_roi)
    #             bg_img[0:dest_h, 0:dest_w] = bg_img_roi
    #         elif width > height * 10:
    #             print(2)
    #             bg_img = cv2.resize(img, (280, 32))
    #         else:
    #             print(3)
    #             bg_img = cv2.resize(img, (280, 32))
    #             # # 以宽为准
    #             # dest_h = int(height * 280 / width)
    #             # dest_w = 280
    #             # d_h = int((32 - dest_h)/2)
    #             #
    #             # tmp_img = cv2.resize(img, (dest_w, dest_h))
    #             # bg_img_roi = bg_img[d_h:(d_h + dest_h), 0:dest_w]
    #             # print(dest_h, d_h, bg_img_roi.shape, tmp_img.shape)
    #             # bg_img_roi = cv2.bitwise_and(tmp_img, bg_img_roi)
    #             # bg_img[d_h:d_h + dest_h, 0:dest_w] = bg_img_roi
    #
    #         # cv2.imshow("est", bg_img)
    #         # cv2.waitKey(0)
    #         cv2.imwrite(os.path.join(dest_path, file_name), bg_img)
