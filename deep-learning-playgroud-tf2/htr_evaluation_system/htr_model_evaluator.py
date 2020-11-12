import traceback
import urllib

import cv2
import numpy as np
import pymysql
import tensorflow as tf

MODEL_PATH_TO_BE_EVALUATED = 'static/models/mnist_plus_online_1105.h5'
MODEL_ID_TO_BE_EVALUATED = 7
MODEL_TYPE_TO_BE_EVALUATED = '01'

model_tf = tf.compat.v1.keras.models.load_model(MODEL_PATH_TO_BE_EVALUATED)


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def recognize_character(img):
    try:
        # img = cv2.imdecode(img, 0)
        img_data = cv2.resize(img, (28, 28))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        img_data = img_data.reshape(-1, 28, 28, 3)

        # img_data = cv2.resize(img, (32, 32))
        # # img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        # img_data = img_data.reshape(-1, 32, 32, 3)
        # # predict_result = model_a_to_g.predict(img_data)

        predict_result = model_tf.predict(img_data)
        return chr(int(np.argmax(predict_result)) + 65), predict_result[0][np.argmax(predict_result)]
    except Exception as e1:
        traceback.print_exc()


def recognize_digit(img):
    try:
        # img = cv2.resize(img, (28, 28))
        img = resize_cv2_img(img)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except Exception as e1:
            # logger.warning(e1)
            pass
        img = img.reshape(-1, 28, 28, 3)
        predict_result = model_tf.predict(img)
        return str(int(np.argmax(predict_result))), predict_result[0][np.argmax(predict_result)]
    except Exception as e1:
        traceback.print_exc()


db = pymysql.connect(host='192.168.31.227',
                     port=3306,
                     user='root',
                     password='root',
                     db='htr_db')
cursor = db.cursor()

# 按照模型取出验证过的数据集
sql = """select * from t_htr_data a where a.is_verificated=1"""
if MODEL_TYPE_TO_BE_EVALUATED == 'ad':
    sql = """select * from t_htr_data a where a.is_verificated=1 and a.real_result in ('A', 'B', 'C', 'D')"""
elif MODEL_TYPE_TO_BE_EVALUATED == '01':
    sql = """select * from t_htr_data a where a.is_verificated=1 and a.real_result in (
    '0', '1', '2', '3','4', '5', '6', '7', '8', '9')"""

db_result = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    r = cursor.fetchone()
    while r is not None:
        db_result.append(r)
        r = cursor.fetchone()
except Exception as e:
    traceback.print_exc()

# 开始验证

select_sql = """
SELECT * FROM htr_db.t_htr_model_pred_result a where a.t_htr_data_id=%s and a.t_htr_model_id=%s
"""

insert_sql = """
INSERT INTO `htr_db`.`t_htr_model_pred_result`
(`t_htr_data_id`,
`t_htr_model_id`,
`pred_result`,
`real_result`)
VALUES(%s, %s, %s, %s)
"""

DEST_SIZE = 28


def resize_cv2_img(src_img):
    dest_img = None
    if src_img.shape[0] > src_img.shape[1]:
        dest_height = int(DEST_SIZE / src_img.shape[0] * src_img.shape[1])
        bg_img = np.ones((DEST_SIZE, DEST_SIZE, 3), np.uint8) * 255
        dest_img = cv2.resize(src_img, (dest_height, DEST_SIZE))
        d_h = int((DEST_SIZE - dest_height) / 2)
        bg_img[0:DEST_SIZE, d_h:(d_h + dest_height)] = dest_img
        dest_img = bg_img
    elif src_img.shape[1] > src_img.shape[0]:
        dest_width = int(DEST_SIZE / src_img.shape[1] * src_img.shape[0])
        bg_img = np.ones((DEST_SIZE, DEST_SIZE, 3), np.uint8) * 255
        dest_img = cv2.resize(src_img, (DEST_SIZE, dest_width))
        d_w = int((DEST_SIZE - dest_width) / 2)
        bg_img[d_w:(d_w + dest_width), 0:DEST_SIZE] = dest_img
        dest_img = bg_img
    else:
        dest_img = cv2.resize(src_img, (DEST_SIZE, DEST_SIZE))
    return dest_img


vals = []
num = 0
count = 0
failed_count = 0
for item in db_result:
    # # 执行模型预测
    #     # img = url_to_image(item[1])
    #     # pred_result, _ = recognize_character(img)
    #     # # 可以执行插入
    #     # print("pred: " + pred_result + "   real: " + item[4])
    #     # if pred_result != item[4]:
    #     #     failed_count += 1
    #     # count += 1

    try:
        # 执行SQL语句
        cursor.execute(select_sql, (item[0], MODEL_ID_TO_BE_EVALUATED))
        r = cursor.fetchone()
        if r is None:
            # 执行模型预测
            img = url_to_image(item[1])
            # pred_result, _ = recognize_character(img)
            pred_result, _ = recognize_digit(img)
            # 可以执行插入
            print("pred: " + pred_result + "   real: " + item[4])
            vals.append((item[0], MODEL_ID_TO_BE_EVALUATED, pred_result, item[4]))
            num += 1
            if num > 10:
                cursor.executemany(insert_sql, vals)
                db.commit()
                num = 0
                vals = []
            print(r, item[0], "insert")
        else:
            print(item[0], "pass")
    except Exception as e:
        traceback.print_exc()

print(count, failed_count)
db.close()

# SET @A=5;
# select (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A) as total_count,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result) as failed_count,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='A') as a_failed,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='B') as b_failed,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='C') as c_failed,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='D') as d_failed,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='E') as e_failed,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='F') as f_failed,
# (select count(id) from  htr_db.t_htr_model_pred_result a where t_htr_model_id=@A and
# pred_result != real_result and real_result='G') as g_failed;


# select (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='A' and a.is_verificated=1) as a,
# (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='B' and a.is_verificated=1) as b,
# (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='C' and a.is_verificated=1) as c,
# (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='D' and a.is_verificated=1) as d,
# (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='E' and a.is_verificated=1) as e,
# (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='F' and a.is_verificated=1) as f,
# (SELECT count(id) FROM htr_db.t_htr_data a where a.pred_result='G' and a.is_verificated=1) as g;
