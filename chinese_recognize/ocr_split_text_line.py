"""
传统机器视觉方式寻找文档中的文本区域
by deng
"""
import cv2


def contour_points(contours):
    lst = []
    current = None
    for i in range(len(contours)):
        if i == 0:
            lst.append(contours[i][0])
        else:
            if current[0] == contours[i][0][0]:
                for j in range(min(current[1], contours[i][0][1]) + 1, max(current[1], contours[i][0][1])):
                    lst.append([current[0], j])
            if current[1] == contours[i][0][1]:
                for j in range(min(current[0], contours[i][0][0]) + 1, max(current[0], contours[i][0][0])):
                    lst.append([j, current[1]])
            lst.append((contours[i][0][0], contours[i][0][1]))

        if i == len(contours) - 1:
            lst.append([contours[i][0][0], contours[i][0][1]])
            if contours[0][0][0] == contours[i][0][0]:
                for j in range(min(contours[0][0][1], contours[i][0][1]) + 1,
                               max(contours[0][0][1], contours[i][0][1])):
                    lst.append([contours[0][0][0], j])
            if contours[0][0][1] == contours[i][0][1]:
                for j in range(min(contours[0][0][0], contours[i][0][0]) + 1,
                               max(contours[0][0][0], contours[i][0][0])):
                    lst.append([j, contours[0][0][1]])

        current = contours[i][0]

    return lst


def get_rect(points):
    minx = 10000
    maxx = -1
    miny = 10000
    maxy = -1
    for p in points:
        minx = min(p[0][0], minx)
        miny = min(p[0][1], miny)
        maxx = max(p[0][0], maxx)
        maxy = max(p[0][1], maxy)
    return [minx, miny, maxx - minx, maxy - miny]


def get_index(lst, point):
    k = 0
    index = point[0] * 10000 + point[1]
    for l in lst:
        if l[0] * 10000 + l[0] < index:
            k += 1

    return k


def get_black_steps_list(param):
    lst = []
    if len(param) is 0:
        return lst
    black = False
    for i in range(len(param) - 1):
        if not black:
            if param[i] > 0 and param[i - 1] == 0:
                lst.append(i)
                black = True
        else:
            if param[i] == 0 and param[i - 1] > 0:
                lst.append(i - 1)
                black = False
    if len(lst) % 2 == 1:
        lst.append(len(param) - 1)
    return lst


def get_blank_y(all_point, start_x, end_x, start_y, end_y):
    tmp = [0 for i in range(len(all_point))]
    for i in range(len(all_point)):
        if start_y < all_point[i][1] and end_y > all_point[i][1] and \
                start_x < all_point[i][0] and end_x > all_point[i][0]:
            tmp[int(all_point[i][1])] = int(tmp[int(all_point[i][1])] + 1)
    black_y = get_black_steps_list(tmp)

    return black_y


def get_white_steps_list(black_y, min_y, max_y):
    lst = []
    if len(black_y) == 0:
        return lst
    start = min_y
    for i in range(len(black_y)):
        if i % 2 == 0:
            tmp = int((start + black_y[i]) / 2)
            lst.append(tmp)
        else:
            start = black_y[i]
    lst.append(int((start + max_y - 1) / 2))
    return lst


def get_min_row(white_y):
    min_row = white_y[1] - white_y[0]
    for i in range(len(white_y) - 1):
        tmp = int(white_y[i + 1]) - int(white_y[i])
        if min_row > tmp:
            min_row = tmp
    return int(min_row)


def get_max_x_blank(all_point, input, start_y, end_y):
    tmp = [0 for i in range(input.shape[1])]
    for i in range(len(all_point)):
        if all_point[i][1] > start_y and all_point[i][1] <= end_y:
            tmp[int(all_point[i][0])] = int(tmp[int(all_point[i][0])]) + 1

    black_x = get_black_steps_list(tmp)
    col = 0
    index = -1
    for i in range(2, len(black_x), 2):
        if col < black_x[i] - black_x[i - 1]:
            col = black_x[i] - black_x[i - 1]
            index = i
    if index > -1:
        return int(black_x[index] + black_x[index - 1]) / 2

    return -1


def find_text_area(img):

    # 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary, binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

    dilation = cv2.dilate(binary, kernel, anchor=(-1, -1), iterations=1)
    cv2.imwrite('o1.jpg', dilation)

    lst = []
    black_y = []
    white_y = []
    rowWhiteMaxXMap = {}
    rowWhiteYMap = {}
    region = []
    all_point = []
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))
    for i in range(len(contours)):
        rect = get_rect(contours[i])
        if rect[2] > 2 or rect[3] > 2:
            con_list = contour_points(contours[i])
            all_point += con_list
            index = get_index(lst, (rect[0], rect[1]))
            lst.insert(index, rect)
    black_y = get_blank_y(all_point, 0, img.shape[1], 0, img.shape[0])
    print(black_y)
    white_y = get_white_steps_list(black_y, 0, img.shape[0])
    print(white_y)
    min_row = -1
    if len(white_y) > 1:
        min_row = get_min_row(white_y)
        rate = min_row * 2

        for i in range(len(white_y) - 1):
            tmp = white_y[i + 1] - white_y[i]
            # code_sub = img[int(white_y[i]):int(tmp), 0:int(img.shape[0] - 1)]
            # face = src[200:300, 200:400]  # 选择200:300行、200:400列区域作为截取对象

            # 含有跨行的大图 取最大间距
            if rate >= tmp:
                continue
            print(i)
            max_x_blank = get_max_x_blank(all_point, img, white_y[i], white_y[i + 1])
            if max_x_blank < img.shape[1] * 0.3 or max_x_blank > img.shape[1] * 0.7:
                print("整行图")
            elif max_x_blank >= img.shape[1] * 0.3 and max_x_blank <= img.shape[1] * 0.7:
                row_black_left_y = get_blank_y(all_point, 0, max_x_blank, white_y[i], white_y[i + 1])
                row_black_right_y = get_blank_y(all_point, max_x_blank, img.shape[1], white_y[i], white_y[i + 1])

                if len(row_black_right_y) < len(row_black_left_y):
                    print("右侧图")
                    rowWhiteMaxXMap[white_y[i]] = max_x_blank
                    row_white_left_y = get_white_steps_list(row_black_left_y, row_black_left_y[0],
                                                            row_black_left_y[len(row_black_left_y) - 1])
                    # 去掉头尾
                    if len(row_white_left_y) > 2:
                        del row_white_left_y[len(row_white_left_y) - 1]
                        del row_white_left_y[0]
                    row_white_left_y.insert(0, white_y[i])
                    row_white_left_y.append(white_y[i + 1])
                    rowWhiteYMap[white_y[i]] = row_white_left_y
                else:
                    print("左侧图")
                    rowWhiteMaxXMap[white_y[i]] = -1 * max_x_blank
                    row_white_right_y = get_white_steps_list(row_black_right_y, row_black_left_y[0],
                                                             row_black_left_y[len(row_black_left_y) - 1])
                    # 去掉头尾
                    if len(row_white_right_y) > 2:
                        del row_white_right_y[len(row_white_right_y) - 1]
                        del row_white_right_y[0]
                    row_white_right_y.insert(0, white_y[i])
                    row_white_right_y.append(white_y[i + 1])
                    rowWhiteYMap[white_y[i]] = row_white_right_y
    return lst, white_y, rowWhiteMaxXMap, rowWhiteYMap


def text_detect(img):
    result_arr, white_y, rowWhiteMaxXMap, rowWhiteYMap = find_text_area(img)
    rect = []
    for i in range(len(white_y) - 1):
        max_x = int(rowWhiteMaxXMap[white_y[i]]) if white_y[i] in rowWhiteMaxXMap else None

        row_white_y = rowWhiteYMap[white_y[i]] if white_y[i] in rowWhiteYMap else None
        if max_x is not None and row_white_y is not None:
            if max_x > 0:
                for j in range(len(row_white_y) - 1):
                    rect.append([0, row_white_y[j],
                                 max_x, row_white_y[j],
                                 0, row_white_y[j + 1],
                                 max_x, row_white_y[j + 1]])
                    cv2.rectangle(img, (0, row_white_y[j]), (max_x, row_white_y[j + 1]), (0, 255, 0))
            else:
                max_x = img.shape[1] - (img.shape[1] + max_x)
                for j in range(len(row_white_y) - 1):
                    rect.append([max_x, row_white_y[j],
                                 img.shape[1] - 1, row_white_y[j],
                                 max_x, row_white_y[j + 1],
                                 img.shape[1] - 1, row_white_y[j + 1]])
                    cv2.rectangle(img, (max_x, row_white_y[j]), (img.shape[1] - 1, row_white_y[j + 1]), (0, 255, 0))

        if row_white_y is not None and white_y[i] == row_white_y[0] and white_y[i + 1] == row_white_y[-1]:
            continue
        rect.append([0, white_y[i],
                     img.shape[1] - 1, white_y[i],
                     0, white_y[i + 1],
                     img.shape[1] - 1, white_y[i + 1]])
        cv2.rectangle(img, (0, white_y[i]), (img.shape[1] - 1, white_y[i + 1]), (0, 0, 255))

    return rect, img


if __name__ == '__main__':
    img = cv2.imread('6.jpg')
    result_arr, white_y, rowWhiteMaxXMap, rowWhiteYMap = find_text_area(img)
    rect = []
    for i in range(len(white_y) - 1):
        max_x = int(rowWhiteMaxXMap[white_y[i]]) if white_y[i] in rowWhiteMaxXMap else None
        # if max_x is not None and i < len(white_y) - 1:
        #     cv2.line(img, (max_x, white_y[i]), (max_x, white_y[i + 1]), (255, 0, 0))

        row_white_y = rowWhiteYMap[white_y[i]] if white_y[i] in rowWhiteYMap else None
        if max_x is not None and row_white_y is not None:
            if max_x > 0:
                for j in range(len(row_white_y) - 1):
                    # cv2.line(img, (0, row_white_y[j]), (max_x, row_white_y[j]), (0, 255, 0))
                    rect.append([0, row_white_y[j], max_x, row_white_y[j + 1]])
                    cv2.rectangle(img, (0, row_white_y[j]), (max_x, row_white_y[j + 1]), (0, 255, 0))
            else:
                max_x = img.shape[1] - (img.shape[1] + max_x)
                for j in range(len(row_white_y) - 1):
                    # cv2.line(img, (max_x, row_white_y[j]), (img.shape[1], row_white_y[j]), (0, 255, 0))
                    rect.append([max_x, row_white_y[j], img.shape[1]-1, row_white_y[j + 1]])
                    cv2.rectangle(img, (max_x, row_white_y[j]), (img.shape[1]-1, row_white_y[j + 1]), (0, 255, 0))

        # cv2.line(img, (0, white_y[i]), (img.shape[1], white_y[i]), (0, 0, 255))
        if row_white_y is not None and white_y[i] == row_white_y[0] and white_y[i+1] == row_white_y[-1]:
            continue
        rect.append([0, white_y[i], img.shape[1]-1, white_y[i+1]])
        cv2.rectangle(img, (0, white_y[i]), (img.shape[1]-1, white_y[i+1]), (0, 0, 255))

    cv2.imshow('est', img)
    cv2.waitKey(0)
