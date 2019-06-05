"""
通过文档图片, 以及背景图片, 生成含有文档元素的场景图片作为训练数据
"""
import os
import random
import shutil
import string

import cv2
import numpy as np


def make_image_height_greater_than_width(img_path):
    g = os.walk(img_path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            print(path + '/' + file_name, os.path.exists(d))
            im = cv2.imread(path + '/' + file_name, cv2.IMREAD_COLOR)
            (h, w, _) = im.shape
            if h > w:
                continue
            im = np.rot90(im)
            cv2.imwrite(path + '/' + file_name, im)


def resize_image_to_normal(img_path):
    g = os.walk(img_path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            print(path + '/' + file_name, os.path.exists(d))
            im = cv2.imread(path + '/' + file_name, cv2.IMREAD_COLOR)
            (h, w, _) = im.shape
            if h < 2000 and w < 2000:
                continue
            im = cv2.resize(im, (int(w * 0.25), int(h * 0.25)))
            cv2.imwrite(path + '/' + file_name, im)


def random_transform(bg_img_path, t_img_path, target_img_path, gt_img_path):
    bg_img = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)
    t_img = cv2.imread(t_img_path, cv2.IMREAD_COLOR)
    (bg_h, bg_w, _) = bg_img.shape
    t_img = cv2.resize(t_img, (bg_w, bg_h))
    pts1 = np.float32([[0, 0], [bg_w, 0], [0, bg_h], [bg_w, bg_h]])
    x, y = bg_w / 2, bg_h / 2
    # 左上角
    x1, y1 = random.randint(int(x / 3), int(2 * x / 3)), random.randint(int(y / 4), int(y / 2))
    # 右上角
    x2, y2 = bg_w - random.randint(int(x / 3), int(2 * x / 3)), random.randint(int(y / 4), int(y / 2))
    # 左下角
    x3, y3 = random.randint(int(x / 10), int(x / 2)), bg_h - random.randint(int(x / 10), int(y / 3))
    # 右下角
    x4, y4 = bg_w - random.randint(int(x / 10), int(x / 2)), bg_h - random.randint(int(x / 10), int(y / 3))

    pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    t_img = cv2.warpPerspective(t_img, M, (bg_w, bg_h))
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)  # 这个254很重要
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
    # Take only region of logo from logo image.
    img_fg = cv2.bitwise_and(t_img, t_img, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img_fg, img_bg)
    cv2.imwrite(target_img_path, dst)

    gt_img = np.zeros((bg_h, bg_w, 1), np.uint8)
    # gt_img = cv2.line(gt_img, (x1, y1), (x2, y2), 255, 2)
    # gt_img = cv2.line(gt_img, (x2, y2), (x4, y4), 255, 2)
    # gt_img = cv2.line(gt_img, (x3, y3), (x4, y4), 255, 2)
    # gt_img = cv2.line(gt_img, (x3, y3), (x1, y1), 255, 2)

    rect = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])
    cv2.fillConvexPoly(gt_img, rect, (255, 255, 255))
    cv2.imwrite(gt_img_path, gt_img)


def gen_train_data(bg_dir, t_dir, dest_dir, gt_dir):
    bg_path = []
    g = os.walk(bg_dir)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            bg_path.append((file_name[0], d))

    t_path = []
    g = os.walk(t_dir)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            t_path.append((file_name[0], d))

    bg_num = len(bg_path)
    t_num = len(t_path)

    for i in range(10000):
        bg_idx = random.randint(0, bg_num - 1)
        t_idx = random.randint(0, t_num - 1)

        bp = bg_path[bg_idx][1]
        tp = t_path[t_idx][1]
        dest_file_name = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.jpg'
        dest_path = os.path.join(dest_dir, dest_file_name)
        gt_img_path = os.path.join(gt_dir, dest_file_name)
        print(i, bp, tp, dest_path, gt_img_path)
        random_transform(bp, tp, dest_path, gt_img_path)


def gen_train_pair_lst(data_path):
    lst_file = os.path.join(data_path, 'train_pair.lst')
    with open(lst_file, 'w+') as f:
        g = os.walk(os.path.join(data_path, 'data'))
        for path, dir_list, file_list in g:
            for file_name in file_list:
                d1 = os.path.join('data', file_name)
                d2 = os.path.join('gt', file_name)
                print(d1, d2)
                f.writelines(d1 + ' ' + d2 + '\n')


def zoom_out_train_data(data_path):
    g = os.walk(os.path.join(data_path, 'data'))
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d1 = os.path.join(data_path, 'data', file_name)
            d2 = os.path.join(data_path, 'gt', file_name)
            im = cv2.imread(d1, cv2.IMREAD_COLOR)
            (h, w, _) = im.shape
            im = cv2.resize(im, (int(w / 2), int(h / 2)))
            cv2.imwrite(d1, im)

            im = cv2.imread(d2, cv2.IMREAD_COLOR)
            (h, w, _) = im.shape
            im = cv2.resize(im, (int(w / 2), int(h / 2)))
            cv2.imwrite(d2, im)
            print(d1, d2)


def move_full_data_to_simple_data(src_path, dst_path):
    simple_data_arr = []
    g = os.walk(os.path.join(src_path, 'data'))
    for path, dir_list, file_list in g:
        for file_name in file_list:
            simple_data_arr.append(file_name)

    random.shuffle(simple_data_arr)
    for i in range(10000):
        src1 = os.path.join(src_path, 'data', simple_data_arr[i])
        src2 = os.path.join(src_path, 'gt', simple_data_arr[i])
        dst1 = os.path.join(dst_path, 'data', simple_data_arr[i])
        dst2 = os.path.join(dst_path, 'gt', simple_data_arr[i])
        shutil.copy(src1, dst1)
        shutil.copy(src2, dst2)


def resize_to_480(data_path):
    g = os.walk(os.path.join(data_path, 'data'))
    i = 0
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d1 = os.path.join(data_path, 'data', file_name)
            d2 = os.path.join(data_path, 'gt', file_name)
            im = cv2.imread(d1, cv2.IMREAD_COLOR)
            im = cv2.resize(im, (480, 480))
            cv2.imwrite(d1, im)

            im = cv2.imread(d2, cv2.IMREAD_COLOR)
            im = cv2.resize(im, (480, 480))
            cv2.imwrite(d2, im)
            print(i, d1, d2)
            i += 1


if __name__ == "__main__":
    # gen_train_data('D:/_data/_hed/_pre/bg', 'D:/_data/_hed/_pre/target', 'D:/_data/_hed/_pre/data', 'D:/_data/_hed/_pre/gt')
    # gen_train_pair_lst('D:/_data/_hed/_pre')
    # zoom_out_train_data('D:/_data/_hed/_pre')
    # move_full_data_to_simple_data('D:/_data/_hed/train', 'D:/_data/_hed/simple_train')
    resize_to_480('D:/_data/_hed/simple_train')
