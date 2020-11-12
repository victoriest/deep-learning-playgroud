# import traceback
#
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
#
# # 连接数据库
# engine = create_engine('mysql+pymysql://root:root@192.168.31.227:3307/axb_hb', echo=False)
# session_factory = sessionmaker(bind=engine)
# session = session_factory()
#
#
# def to_dict(data):
#     my_column = ["STU_ID", "stu_ans", "LXB_STU_QUE_ID", "LXB_STU_IMG_ID", "AUTO_OCR_JSON", "img_url", "v", "o", "r"]
#     return dict(zip(my_column, data))
#
#
# sql = """
#  SELECT t1.`STU_ID`,t2.stu_ans,t2.LXB_STU_QUE_ID,t3.LXB_STU_IMG_ID,t3.`AUTO_OCR_JSON`,
#         concat('http://axb-img-prod.oss-cn-shanghai.aliyuncs.com',t4.DISPOSE_PATH,'?x-oss-process=image/crop,x_',
#         t3.`TOP_X`,',y_',t3.`TOP_Y`,',w_',t3.`AREA_W`,',h_',t3.`AREA_H`),
#         getJsonStringValue('v',t3.`AUTO_OCR_JSON`) AS v,
#         getJsonStringValue('o',t3.`AUTO_OCR_JSON`) AS o,
#         getJsonStringValue('r',t3.`AUTO_OCR_JSON`) AS r FROM `t_hb_lxb_submit_stu` t1
# 		INNER JOIN `t_hb_lxb_stu_que` t2 ON t1.`LXB_SUBMIT_STU_ID`=t2.`LXB_SUBMIT_STU_ID`
# 		INNER JOIN `t_hb_lxb_stu_img` t3 ON t2.`LXB_STU_QUE_ID`=t3.`LXB_STU_QUE_ID`
#         inner join t_hb_lxb_page t4 on t4.LXB_PAGE_ID=t3.LXB_PAGE_ID
# 		WHERE t1.`LXB_GROUP_ID`='2lasHXV6Lo' AND t2.`OBJ_QUE_FLAG`='1' AND  t2.SCH_ID='0000001001'
# 		AND t2.GRADE_YEAR='2020' AND t3.GRADE_YEAR=2020
#         ORDER BY  t1.`STU_ID`,t2.LXB_STU_QUE_ID,t2.stu_ans;
#     """
#
# try:
#     resultproxy = session.execute(
#         sql
#     )
#     session.commit()
# except Exception as e:
#     traceback.print_exc()
#     results = []
# else:
#     results = resultproxy.fetchall()
#
# for result in results:
#     print(to_dict(result))


import traceback

import pymysql

db = pymysql.connect(host='192.168.31.237',
                     port=3307,
                     user='root',
                     password='Axb123456&',
                     db='axb_hb')
cursor = db.cursor()

# sql = """
#  SELECT t1.`STU_ID`,t2.stu_ans,t2.LXB_STU_QUE_ID,t3.LXB_STU_IMG_ID,t3.`AUTO_OCR_JSON`,
#         concat('http://axb-img-prod.oss-cn-shanghai.aliyuncs.com',t4.DISPOSE_PATH,'?x-oss-process=image/crop,x_',
#         t3.`TOP_X`,',y_',t3.`TOP_Y`,',w_',t3.`AREA_W`,',h_',t3.`AREA_H`),
#         getJsonStringValue('v',t3.`AUTO_OCR_JSON`) AS v,
#         getJsonStringValue('o',t3.`AUTO_OCR_JSON`) AS o,
#         getJsonStringValue('r',t3.`AUTO_OCR_JSON`) AS r,
#         t1.CREATE_TIME as data_time  FROM `t_hb_lxb_submit_stu` t1
# 		INNER JOIN `t_hb_lxb_stu_que` t2 ON t1.`LXB_SUBMIT_STU_ID`=t2.`LXB_SUBMIT_STU_ID`
# 		INNER JOIN `t_hb_lxb_stu_img` t3 ON t2.`LXB_STU_QUE_ID`=t3.`LXB_STU_QUE_ID`
#         inner join t_hb_lxb_page t4 on t4.LXB_PAGE_ID=t3.LXB_PAGE_ID
# 		WHERE t1.`LXB_GROUP_ID`='2lasHXV6Lo' AND t2.`OBJ_QUE_FLAG`='1' AND  t2.SCH_ID='0000001001'
# 		AND t2.GRADE_YEAR='2020' AND t3.GRADE_YEAR=2020
#         ORDER BY  t1.`STU_ID`,t2.LXB_STU_QUE_ID,t2.stu_ans;
#     """

sql = """
select concat('http://axb-img-prod.oss-cn-shanghai.aliyuncs.com',t2.`DISPOSE_PATH`,
    '?x-oss-process=image/crop,x_',t1.`TOP_X`,',y_',t1.`TOP_Y`,',w_',t1.`AREA_W`,',h_',t1.`AREA_H`),
    t1.`ANS_CONTENT`, t1.`CREATE_TIME` from `t_hb_lxb_page_ans` t1
    inner join `t_hb_lxb_page` t2 on t1.`LXB_PAGE_ID`=t2.`LXB_PAGE_ID` and t1.create_time > '2020-09-15'
    where `ANS_TYPE` = '13' 
"""

db_result = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    r = cursor.fetchone()
    while r is not None:
        # print(r, cursor.rownumber)
        db_result.append(r)
        r = cursor.fetchone()
except Exception as e:
    traceback.print_exc()

db.close()

db = pymysql.connect(host='192.168.31.227',
                     port=3306,
                     user='root',
                     password='root',
                     db='htr_db')
cursor = db.cursor()

count = 0
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
print(len(db_result))
vals = []
num = 0
for item in db_result:
    vals.append(
        # (item[5], '', item[6] if item[6] is not None else '', item[7] if item[7] is not None else '', item[9], 0)
        (item[0], '', item[1] if item[1] is not None else '', '', item[2], 0)
    )
    num += 1
    if num > 1000:
        cursor.executemany(insert_sql, vals)
        db.commit()
        num = 0
        vals = []
        print(count)
    count += 1
