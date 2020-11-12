import os
import traceback
import urllib
import urllib.request

import pymysql

db = pymysql.connect(host='192.168.31.227',
                     port=3306,
                     user='root',
                     password='root',
                     db='htr_db')

cursor = db.cursor()

def down_load_image(url, dest_path, dest_file_name):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    filename = '{}{}{}'.format(dest_path, os.sep, dest_file_name)
    # 下载图片，并保存到文件夹中
    try:
        urllib.request.urlretrieve(url, filename=filename)
    except Exception as e:
        print("Exception", e)


select_sql = """
SELECT img_url, real_result FROM htr_db.t_htr_data where id > 1130440 and is_verificated=1
"""

try:
    # 执行SQL语句
    cursor.execute(select_sql)
    r = cursor.fetchone()
    while r is not None:
        print(r, cursor.rownumber)
        dest_file_name = r[1] + "_" + str(cursor.rownumber) + ".jpg"
        print("Doloading {}... ({}/{})".format(dest_file_name, cursor.rownumber, cursor.arraysize))
        down_load_image(r[0], "E:\\_dataset\\digit_and_character\\zyb_digit_1104\\" + r[1], dest_file_name)
        r = cursor.fetchone()
except Exception as e:
    traceback.print_exc()

db.close()
