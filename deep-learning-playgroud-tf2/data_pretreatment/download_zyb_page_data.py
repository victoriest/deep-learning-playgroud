"""
下载作业本
"""

import csv
import os
import urllib
import urllib.request
from concurrent.futures.thread import ThreadPoolExecutor

URL_PREFIX = "http://axb-img-prod.oss-cn-shanghai.aliyuncs.com"


def load_csv(file_path):
    csv_file = open(file_path, "r")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result.append(item)
    csv_file.close()
    return result


def down_load_image(url, dest_path, dest_file_name):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    filename = '{}{}{}'.format(dest_path, os.sep, dest_file_name)
    # 下载图片，并保存到文件夹中
    try:
        urllib.request.urlretrieve(url, filename=filename)
    except Exception as e:
        print("Exception", e)


if __name__ == '__main__':
    result = load_csv("E:\\_dataset\\zyb_data_10_27_yy.csv")
    count = 0
    size = len(result)
    pool = ThreadPoolExecutor(4)
    for item in result:
        f_name_arr = os.path.basename(item[2]).split(".")
        dest_file_name = f_name_arr[0] + "_" + item[1] + "." + f_name_arr[1]
        img_url = URL_PREFIX + item[2]
        print("Doloading {}... ({}/{})".format(dest_file_name, count, size))
        pool.submit(down_load_image, img_url, "E:\\_dataset\\zyb_data_1027_yy\\" + item[3], dest_file_name)
        # down_load_image(img_url, "E:\\_dataset\\zyb_data_1009\\" + item[3], dest_file_name)
        count += 1
