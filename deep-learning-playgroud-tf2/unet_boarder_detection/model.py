import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# 重新创建完全相同的模型，包括其权重和优化程序
model = tf.keras.models.load_model('unet_frame.h5')

# 显示网络结构
model.summary()


def load_tensor_from_file(img_file):
    img = Image.open(img_file)
    img_np = np.array(img)
    # 样本图片规格不一致，需要做通道转换，否则抛异常
    if len(img_np.shape) != 3 or img_np.shape[2] == 4:
        img = img.convert("RGB")
        img_np = np.array(img)
    print(img_np.shape)
    img_np = tf.image.resize(img_np, [512, 512], method='nearest')
    img_np = tf.squeeze(img_np)
    return img_np


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


img_path = './1.jpg'
for i in range(12):
    img_path = './' + str(i + 1) + '.jpg'

    img = load_tensor_from_file(img_path)
    img = tf.cast(img, tf.float32) / 255.0
    xxx = model.predict(img[tf.newaxis, ...])
    xxx = create_mask(xxx)
    # pred_img = tf.keras.preprocessing.image.array_to_img(xxx)
    pred_img = xxx * 255


    print(np.unique(xxx))
    pred_img = np.array(pred_img, dtype=np.uint8)

    # cv2.imshow('dst1', pred_img)

    kernel = np.ones((5, 5), np.uint8)

    pred_img = cv2.erode(pred_img, kernel, iterations=4)
    pred_img = cv2.dilate(pred_img, kernel, iterations=3)

    pred_img = cv2.resize(pred_img, (1600, 2100))
    # cv2.imshow('dst', pred_img)
    # cv2.waitKey(0)

    # img_show = cv2.cvtColor(pred_img.copy(), cv2.COLOR_GRAY2BGR)

    contours, hier = cv2.findContours(pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_show = Image.open(img_path)
    img_show = np.array(img_show)

    for c in contours:

        # 轮廓绘制方法一
        # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
        x, y, w, h = cv2.boundingRect(c)

        if w * h < 100000:
            continue

        # 在img图像画出矩形，(x, y), (x + w, y + h)是矩形坐标，(0, 255, 0)设置通道颜色，2是设置线条粗度
        cv2.rectangle(img_show, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        print(w * h)

        #
        # # 轮廓绘制方法二
        # # 查找最小区域
        # rect = cv2.minAreaRect(c)
        # # 计算最小面积矩形的坐标
        # box = cv2.boxPoints(rect)
        # # 将坐标规范化为整数
        # box = np.int0(box)
        # # 绘制矩形
        # cv2.drawContours(img_show, [box], 0, (0, 0, 255), 3)

        # # 轮廓绘制方法三
        # # 圆心坐标和半径的计算
        # (x, y), radius = cv2.minEnclosingCircle(c)
        # # 规范化为整数
        # center = (int(x), int(y))
        # radius = int(radius)
        # # 勾画圆形区域
        # pred_img = cv2.circle(pred_img, center, radius, (0, 255, 0), 2)

    # # 轮廓绘制方法四
    # 围绕图形勾画蓝色线条
    # cv2.drawContours(img_show, contours, -1, (255, 0, 0), 2)

    # 显示图像
    img_show = cv2.resize(img_show, None, fx=0.5, fy=0.5)
    cv2.imshow("mark", cv2.resize(pred_img, None, fx=0.5, fy=0.5))
    cv2.imshow("contours", img_show)
    cv2.waitKey()
    cv2.destroyAllWindows()
