# coding:utf8
"""
传统机器视觉方式寻找文档中的文本区域
"""
import cv2
import numpy as np


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 7. 存储中间图片
    # cv2.imwrite("binary.png", binary)
    # cv2.imwrite("dilation.png", dilation)
    # cv2.imwrite("erosion.png", erosion)
    # cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 1000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print
        "rect is: "
        print
        rect

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2):
            continue

        region.append(box)

    return region


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)

    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    # 带轮廓的图片
    cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_line(img_path):
    img_o = cv2.imread(img_path)
    (bg_h, bg_w, _) = img_o.shape
    img_l = np.zeros((bg_h, bg_w, 1), np.uint8)
    img_p = np.zeros((bg_h, bg_w, 1), np.uint8)
    img = cv2.imread(img_path, 0)
    minLineLength = 100
    maxLineGap = 50

    # 1
    kernel = np.ones((3, 3), np.uint8)
    t_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    lines = cv2.HoughLinesP(t_img, 1, np.pi / 180, 200, minLineLength, maxLineGap)
    lines1 = lines[:, 0, :]
    for x1, y1, x2, y2 in lines1[:]:
        # cv2.line(img_o, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(img_p, (x1, y1), (x2, y2), 255, 2)
    cv2.imshow("est", img_p)
    cv2.waitKey(0)

    # 2
    kernel = np.ones((3, 3), np.uint8)
    t_img = cv2.morphologyEx(img_p, cv2.MORPH_OPEN, kernel, iterations=5)
    lines = cv2.HoughLinesP(t_img, 1, np.pi / 180, 200, minLineLength, maxLineGap)
    img_p = np.zeros((bg_h, bg_w, 1), np.uint8)
    lines1 = lines[:, 0, :]
    for x1, y1, x2, y2 in lines1[:]:
        # cv2.line(img_o, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(img_p, (x1, y1), (x2, y2), 255, 2)
    cv2.imshow("est", img_p)
    cv2.waitKey(0)

    t_img = cv2.GaussianBlur(img_p, (3, 3), 0)
    edges = cv2.Canny(t_img, 50, 200)
    cv2.imshow("est", edges)
    cv2.waitKey(0)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    lines1 = lines[:, 0, :]
    for rho, theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(img_o, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(img_l, (x1, y1), (x2, y2), 255, 2)

    cv2.imshow("est", img_l)
    cv2.waitKey(0)

    kernel = np.ones((5, 5), np.uint8)
    img_l = cv2.morphologyEx(img_l, cv2.MORPH_CLOSE, kernel, iterations=5)
    cv2.imshow("est", img_l)
    cv2.waitKey(0)

    # contours, hier = cv2.findContours(img_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for c in contours:
    #     # find minimum area
    #     rect = cv2.minAreaRect(c)
    #     # calculate coordinates of the minimum area rectangle
    #     box = cv2.boxPoints(rect)
    #     # normalize coordinates to integers
    #     box = np.int0(box)
    #     # draw contours
    #     draw_img3 = cv2.drawContours(img_o.copy(), [box], 0, (0, 0, 255), 3)
    #     cv2.imshow("est", draw_img3)
    #     cv2.waitKey(0)

    # draw_img0 = cv2.drawContours(img_o.copy(), contours, 0, (0, 255, 255), 1)
    # cv2.imshow("est", draw_img0)
    # cv2.waitKey(0)
    # draw_img1 = cv2.drawContours(img_o.copy(), contours, 1, (255, 0, 255), 1)
    # cv2.imshow("est", draw_img1)
    # cv2.waitKey(0)
    # draw_img2 = cv2.drawContours(img_o.copy(), contours, 2, (255, 255, 0), 3)
    # cv2.imshow("est", draw_img2)
    # cv2.waitKey(0)
    # draw_img3 = cv2.drawContours(img_o.copy(), contours, -1, (0, 0, 255), 1)
    # cv2.imshow("est", draw_img3)
    cv2.waitKey(0)
    # cv2.imwrite('111.jpg', draw_img3)


if __name__ == '__main__':
    # find_line('./hed/out/bJW2weiuLJ_ORIGINAL.jpg')

    # 读取文件
    img = cv2.imread('3BA9E58C76.jpg')
    detect(img)
