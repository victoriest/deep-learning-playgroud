import os

import cv2


def inverse_color(image):
    height, width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i, j] = (255 - image[i, j])
    return img2


# path_name = os.getcwd() + "/pre_pic"
# print(path_name)
# dix = 0
# dest_path = os.path.join(path_name, "pre_item")
# for item in os.listdir(path_name):
#     # print(os.path.join(path_name, ('f_' + str(i) + '.png')))
#     sss = item.split("_")
#     d = os.path.join(path_name, item)
#     if not d.endswith(".jpg"):
#         continue
#     img = Image.open(d)
#     w = img.size[0] / 4
#     h = img.size[1] / 8
#     for i in range(4):
#         for j in range(8):
#             offset_x = i * w
#             offset_y = j * h
#             cropped = img.crop((offset_x + 50, offset_y + 50, offset_x + w - 50, offset_y + h - 50))
#             dest_img = "./pre_pic/pre_item/" + sss[0] + "_" + str(dix) + ".jpg"
#             cropped.save(dest_img)
#             dix += 1

# img = Image.open("./f_01.jpg")
# print(img.size)
# w = img.size[0] / 4
# h = img.size[1] / 8
#
# for i in range(4):
#     print(i)
#     for j in range(8):
#         offset_x = i * w
#         offset_y = j * h
#         cropped = img.crop((offset_x + 50, offset_y + 50, offset_x + w - 50, offset_y + h - 50))
#         cropped.save("./c_" + str(i) + "_" + str(j) + ".jpg")


for item in os.listdir("./pre_pic/pre_item"):
    if not item.endswith(".jpg"):
        continue

    img = cv2.imread('./pre_pic/pre_item/' + item)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色转灰度
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # 进行二值化
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(inverse_color(thresh))
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roiImg = img[y:(y + h), x:(x + w)]
    cv2.imwrite('./pre_pic/pre_item/' + item, roiImg)
    # cv2.imshow('img', roiImg)  # 显示原始图像
    # cv2.waitKey()
