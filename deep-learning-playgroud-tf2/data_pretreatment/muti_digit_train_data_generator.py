"""
根据mnist生成手写数据集中生成2-5个数字的训练样本
"""
import os
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageFilter
from trdg import background_generator, distorsion_generator

from utils.image_converter import cv2_to_pil, pil_to_cv2


def add_noise(img, percetage=0.02):
    noise_num = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        if np.random.randint(0, 1) == 0:
            img[rand_x, rand_y] = 255
        else:
            img[rand_x, rand_y] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.random.randint(1, 2), np.random.randint(1, 16)))
    img = cv2.dilate(img, kernel)
    return img


def add_skewing(img, mask, skewing_angle=10, random_skew=True):
    random_angle = random.randint(0 - skewing_angle, skewing_angle)
    rotated_img = img.rotate(
        skewing_angle if not random_skew else random_angle, expand=1,
        fillcolor=(255, 255, 255)
    )
    rotated_mask = mask.rotate(
        skewing_angle if not random_skew else random_angle, expand=1
    )
    return rotated_img, rotated_mask


def add_distorsion(img, mask):
    distorsion_type = random.randint(0, 3)
    distorsion_orientation = random.randint(0, 2)
    if distorsion_type == 0:
        distorted_img = img  # Mind = blown
        distorted_mask = mask
    elif distorsion_type == 1:
        distorted_img, distorted_mask = distorsion_generator.sin(
            img,
            mask,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
        )
    elif distorsion_type == 2:
        distorted_img, distorted_mask = distorsion_generator.cos(
            img,
            mask,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
        )
    else:
        distorted_img, distorted_mask = distorsion_generator.random(
            img,
            mask,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
        )
    return distorted_img, distorted_mask


def add_blurring(img, mask, blur=2):
    gaussian_filter = ImageFilter.GaussianBlur(
        radius=random.randint(0, blur)
    )
    img = img.filter(gaussian_filter)
    mask = mask.filter(gaussian_filter)
    return img, mask


def get_random_file_suffix():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))


def gen_train_data():
    cell_w = 30
    cell_h = 42
    data_path = []
    g = os.walk('C:/_project/_data_threshold_train_2nd_check')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            data_path.append((file_name[0], d))
    data_len = len(data_path)
    print(data_len)

    for num in range(500000):
        # 生成随机2-5个空白格子
        cell_count = 5
        img = np.ones((cell_h, cell_w * cell_count + 2, 1), np.uint8) * 255
        # for i in range(cell_count):
        #     img = cv2.rectangle(img, (i * cell_w, 0), ((i + 1) * cell_w, cell_h), (0, 0, 0), 2)
        # 将格子随机腐蚀
        img = add_noise(img)
        # 随机将tag中的数字丢入格子中
        tag_str = ""
        for i in range(cell_count):
            t_tag, t_path = data_path[np.random.randint(0, data_len)]
            print(t_tag, t_path)
            tag_str += t_tag
            img_roi = img[0:cell_h, i * cell_w:((i + 1) * cell_w)]
            add_img = cv2.imread(t_path, 0)
            img_roi = cv2.bitwise_and(add_img, img_roi)
            img[0:cell_h, i * cell_w:((i + 1) * cell_w)] = img_roi

        suffix = get_random_file_suffix()
        t_file_name = tag_str + "_" + suffix + ".jpg"
        print(t_file_name)
        # 随机震动产生偏斜
        img = cv2.bitwise_not(img)
        rows, cols = img.shape
        M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
        img = cv2.warpAffine(img, M, (cols, rows))
        img = cv2.bitwise_not(img)
        cv2.imwrite("C:/_project/_data_crnn_train/" + t_file_name, img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)


def gen_space_img_data():
    img = np.ones((28, 28, 3), np.uint8) * 255
    img = cv2.rectangle(img, (0, 0), (26, 26), (0, 0, 0), 2)
    # 将格子随机腐蚀
    img = add_noise(img)
    # 随机震动产生偏斜
    img = cv2.bitwise_not(img)
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
    img = cv2.warpAffine(img, M, (cols, rows))
    img = cv2.bitwise_not(img)
    return img


def gen_variable_length_digit(gen_num=100000,
                              cell_count=5,
                              enable_random_digit=True,
                              enable_noise=True,
                              enable_skewing=True,
                              enable_distortion=True,
                              enable_blurring=True,
                              dataset_path='E:/_dataset/digit_and_character/mnist_with_space_train.npz',
                              output_path='e:/_dataset/_digit_crnn_train_32_offset/'):
    mnist_out = np.load(dataset_path)
    mnist_x_train, mnist_y_train = mnist_out['x_train'], mnist_out['y_train']
    print(mnist_x_train.shape)
    print(mnist_y_train.shape)

    # 过滤空格
    filter_indices = np.where(np.argmax(mnist_y_train, axis=1) != 10)
    filted = mnist_x_train[filter_indices]
    filted_tags = mnist_y_train[filter_indices]
    cell_w = 32
    cell_h = 32

    data_len = len(filted)
    print(data_len)

    for num in range(gen_num):
        img = np.ones((cell_h, cell_w * cell_count + 2 * cell_count, 1), np.uint8) * 255

        # 生成随机2-5个空白格子
        digit_count = cell_count
        if enable_random_digit:
            digit_count = np.random.randint(2, cell_count + 1)

        # 将格子随机腐蚀
        if enable_noise:
            img = add_noise(img)

        # 随机将tag中的数字丢入格子中
        tag_str = ""
        offset_w = 0
        for i in range(digit_count):
            random_idx = np.random.randint(0, data_len)
            print(random_idx, filted_tags[random_idx])
            d_img = filted[random_idx]
            d_tag = np.argmax(filted_tags[random_idx])
            # 对比实验，将最后一维变为单通道
            d_img = d_img[:, :, 0]
            d_img = np.expand_dims(d_img, axis=2)
            d_img = cv2.resize(d_img, (32, 32))
            tag_str += str(d_tag)
            d_w = np.random.randint(-12, 2)
            if i == 0:
                d_w = int(d_w / 3)
            d_h = np.random.randint(-4, 4)
            M = np.float32([[1, 0, 0], [0, 1, d_h]])
            d_img = cv2.warpAffine(d_img, M, (32, 32), borderValue=255)
            start_w = offset_w + d_w
            if start_w < 0:
                start_w = 0
            bg = np.ones((cell_h, cell_w * cell_count + 2 * 5), np.uint8) * 255
            bg[0:cell_h, start_w:(start_w + 32)] = d_img
            # bg = cv2.warpAffine(bg, M, (32, 32), borderValue=255)
            # d_img = bg
            # d_img = cv2.warpAffine(d_img, M, (32, 32), borderValue=255)
            # img[0:cell_h, start_w:(start_w + 32)] = d_img
            img = add_image(img, bg)
            offset_w = offset_w + d_w + 32

        suffix = get_random_file_suffix()
        t_file_name = tag_str + "_" + suffix + ".jpg"
        print(t_file_name)
        # 随机震动产生偏斜
        img = cv2.bitwise_not(img)
        rows, cols = img.shape
        M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
        img = cv2.warpAffine(img, M, (cols, rows))
        img = cv2.bitwise_not(img)

        img = cv2_to_pil(img)
        mask = Image.new("RGB", (cols, rows), (0, 0, 0))

        random_s = random.randint(0, 3)
        if enable_skewing and random_s == 1:
            img, mask = add_skewing(img, mask)

        if enable_distortion and random_s == 2:
            img, mask = add_distorsion(img, mask)

        if enable_blurring and random_s == 3:
            img, mask = add_blurring(img, mask)

        img = img.resize((cols, rows), Image.ANTIALIAS)
        mask = mask.resize((cols, rows), Image.ANTIALIAS)
        background_type = random.randint(0, 2)

        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                rows, cols,
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                rows, cols,
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                rows, cols,
            )
        background_mask = Image.new(
            "RGB", (cols, rows), (0, 0, 0)
        )
        img = transparent_back(img)
        print(random_s, background_type, img.width, img.height, background_img.width, background_img.height)
        background_img.paste(img, (0, 0), img)
        background_mask.paste(mask, (0, 0))

        img = pil_to_cv2(background_img)
        mask = pil_to_cv2(background_mask)
        cv2.imwrite(output_path + t_file_name, img)
        # show_image(tag_str, img)


# 以第一个像素为准，相同色改为透明
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0, 0))
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot, color_1)
    return img


def show_image(txt, img):
    cv2.imshow(txt, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


if __name__ == "__main__":
    # i1 = cv2.imread("1.jpg", 0)
    # i2 = cv2.imread("2.jpg", 0)
    # added_img = add_image(i1, i2)
    #
    # cv2.imshow('res', added_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gen_variable_length_digit(enable_random_digit=False,
                              output_path='e:/_dataset/_digit_crnn_train_32_offset_0925/')
