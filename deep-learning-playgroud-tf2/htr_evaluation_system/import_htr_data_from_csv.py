import csv

import pymysql

db = pymysql.connect(host='192.168.31.227',
                     port=3306,
                     user='root',
                     password='root',
                     db='htr_db')

cursor = db.cursor()


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


result = load_csv("E:\\_dataset\\zyb_digit_data__sch1001_1101_1103.csv")

insert_sql = """
INSERT INTO `htr_db`.`t_htr_data`(
`img_url`,
`img_cache_path`,
`pred_result`,
`real_result`,
`data_time`,
`is_verificated`)
VALUES(%s, %s, %s, %s, %s, %s)
"""

vals = []

num = 0
for item in result:
    print(item)
    vals.append(
        # (item[5], '', item[6] if item[6] is not None else '', item[7] if item[7] is not None else '', item[9], 0)
        (item[1], '', item[2] if item[2] is not None else '', '', '2020-11-04', 0)
    )
    num += 1
    if num > 1000:
        cursor.executemany(insert_sql, vals)
        db.commit()
        num = 0
        vals = []

if len(vals) > 0:
    cursor.executemany(insert_sql, vals)
    db.commit()
