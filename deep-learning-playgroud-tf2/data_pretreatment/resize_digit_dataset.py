import os
import cv2
import numpy as np

DEST_SIZE = 28

SRC_PATH = 'E:\\_dataset\\digit_and_character\\zyb_digit_1104'
DEST_PATH = 'E:\\_dataset\\digit_and_character\\zyb_digit_1104_resized'

def add_image(img2, img1):
    # ret是阈值（175）mask是二值化图像
    ret, mask = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 获取把logo的区域取反 按位运算
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1, img1, mask=mask)  # 在img1上面，将logo区域和mask取与使值为0
    # 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0 。
    # 把logo放到图片当中 获取logo的像素信息
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    # cv2.imshow('mask', mask_inv)
    # cv2.imshow('img1_bg', img1_bg)
    # cv2.imshow('img2_fg', img2_fg)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst

g = os.walk(SRC_PATH)
for path, dir_list, file_list in g:
    for file_name in file_list:
        d = os.path.join(path, file_name)
        src_img = cv2.imread(d)
        dest_img = None
        # if src_img.shape[0] > DEST_SIZE and src_img.shape[1] > DEST_SIZE:
        #     dest_img = cv2.resize(src_img, (DEST_SIZE, DEST_SIZE))
        if src_img.shape[0] > src_img.shape[1]:
            dest_height = int(DEST_SIZE / src_img.shape[0] * src_img.shape[1])
            bg_img = np.ones((DEST_SIZE, DEST_SIZE, 3), np.uint8) * 255
            dest_img = cv2.resize(src_img, (dest_height, DEST_SIZE))
            d_h = int((DEST_SIZE - dest_height) / 2)
            bg_img[0:DEST_SIZE, d_h:(d_h + dest_height)] = dest_img
            dest_img = bg_img
            # cv2.imshow('dst1', bg_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        elif src_img.shape[1] > src_img.shape[0]:
            dest_width = int(DEST_SIZE / src_img.shape[1] * src_img.shape[0])
            bg_img = np.ones((DEST_SIZE, DEST_SIZE, 3), np.uint8) * 255
            dest_img = cv2.resize(src_img, (DEST_SIZE, dest_width))
            d_w = int((DEST_SIZE - dest_width) / 2)
            bg_img[d_w:(d_w + dest_width), 0:DEST_SIZE] = dest_img
            dest_img = bg_img
            # # cv2.imshow('dst2', bg_img)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()

        else:
            dest_img = cv2.resize(src_img, (DEST_SIZE, DEST_SIZE))


        label = os.path.splitext(file_name)[0].split("_")[0]
        dst_path = os.path.join(DEST_PATH, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        cv2.imwrite(os.path.join(dst_path, file_name), dest_img)

