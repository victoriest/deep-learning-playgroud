import csv
import os
from concurrent.futures.thread import ThreadPoolExecutor
import requests
import sqlite3

JSON_CSV_FILE_DIR_PATH = 'F:\\pixiv'
PATH_PREFIX = 'F:\\pixiv_imgs'
COOKIE = '__cfduid=d07a82e5b37c4f553156719a9f12b3e051606206690; first_visit_datetime_pc=2020-11-24+17%3A31%3A30; p_ab_id=9; p_ab_id_2=6; p_ab_d_id=1406419122; yuid_b=hWRzMBA; __utmz=235335808.1606206729.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); _ga=GA1.2.943748102.1606206729; PHPSESSID=1533882_rki3Utd4oXtIWO9jXj0AQOSCIJAKhOwq; device_token=137e8f6650bd9b61bcc172d7a1192741; c_type=36; privacy_policy_agreement=2; a_type=1; b_type=1; _fbp=fb.1.1606210078887.689299955; login_ever=yes; __utmv=235335808.|2=login%20ever=yes=1^3=plan=normal=1^5=gender=male=1^6=user_id=1533882=1^9=p_ab_id=9=1^10=p_ab_id_2=6=1^11=lang=zh=1; ki_s=211476%3A0.0.0.0.0; __utmc=235335808; __cf_bm=41c5b07619f288a1b8e8f0109c8e0e2ab54a8da0-1606888736-1800-ARaIogHM335f3x0UZmf/g1irkqjGtpEsJNtccHaQIjeAXL5ViKmomNfzll15SlRz2GSsJ6/fPFFjix88pLakffH8gpK1GYt3G9CI4gvfkirjvBI0V+GCcqbPwIR7XdUdwlk1hYlPco/bVIF2Nc5FbezHubdKMEWqPuUC8q09yuy7Op+ezRku30Tflr7E1WnvnQ==; __utma=235335808.943748102.1606206729.1606878690.1606888736.5; __utmt=1; ki_t=1606210090168%3B1606878695994%3B1606888739152%3B3%3B27; __utmb=235335808.14.9.1606888788446'


class PixivDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('pixiv.db')
        self.__create_db(self.conn)
        self.cursor = self.conn.cursor()

    @staticmethod
    def __create_db(conn):
        try:
            create_tb_cmd = '''
                CREATE TABLE IF NOT EXISTS T_PIXIV
                (id TEXT,
                tags TEXT,
                tags_transl TEXT,
                user TEXT,
                user_id TEXT,
                title TEXT,
                type TEXT,
                page INT,
                bookmark TEXT,
                bookmarked TEXT,
                width INT,
                height INT, 
                date TEXT, 
                original TEXT, 
                file_name TEXT,
                is_downloaded TEXT);
                '''
            conn.execute(create_tb_cmd)
        except:
            print('Create table failed')

    def add_pixiv_img(self, item):
        cursor = self.conn.cursor()
        idx = 0
        for it in item:
            it = it.replace("'", "")
            item[idx] = it
            idx += 1
        sql = f"insert into T_PIXIV (id, tags, tags_transl, user, user_id, title, type, page, bookmark, bookmarked, width, height, date, original, file_name, is_downloaded) values ('{item[0]}', '{item[1]}','{item[2]}','{item[3]}','{item[4]}','{item[5]}','{item[6]}',{item[7]}, '{item[8]}','{item[9]}',{item[10]},{item[11]},'{item[12]}','{item[13]}','{item[14]}','0')"
        print(sql)
        cursor.execute(sql)
        cursor.close()
        self.conn.commit()

    def is_pixiv_img_exists(self, original):
        cursor = self.conn.cursor()
        self.cursor.execute(f'select * from T_PIXIV where original={original}')
        values = cursor.fetchall()
        cursor.close()
        if values is None or len(values) < 1:
            return False
        return True

    def pixiv_img_downloaded(self, original):
        cursor = self.conn.cursor()
        cursor.execute(f"update T_PIXIV SET is_downloaded = '1' WHERE original = {original}")
        cursor.close()
        self.conn.commit()


db = PixivDatabase()


def load_csv(file_path):
    csv_file = open(file_path, "r", encoding="utf8")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result.append(item)
        db.add_pixiv_img(item)
    csv_file.close()
    return result


def down_load_image(url, dest_path, dest_file_name, count, size):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
        'referer': 'https://accounts.pixiv.net/login?lang=zh&source=pc&view_type=page&ref=wwwtop_accounts_index',
        'cookie': COOKIE
    }

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if os.path.exists(os.path.join(dest_path, dest_file_name)):
        print(f'图片{dest_file_name}已存在， 跳过 ({count}/{size})')
        return
    if db.is_pixiv_img_exists(url):
        print(f'图片{dest_file_name}已在数据库中， 跳过 ({count}/{size})')
        return
    filename = '{}{}{}'.format(dest_path, os.sep, dest_file_name)
    # 下载图片，并保存到文件夹中
    try:
        pixiv_img = requests.get(url, headers=headers)
        print(f'图片{dest_file_name}正在保存...({count}/{size})')
        # urllib.request.urlretrieve(url, filename=filename)
        with open(filename, 'wb') as f:
            f.write(pixiv_img.content)
    except Exception as e:
        print("Exception", e)


if __name__ == '__main__':
    g = os.walk(JSON_CSV_FILE_DIR_PATH)
    result = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.splitext(file_name)[1] != '.csv':
                continue
            print(os.path.join(path, file_name))
            result.extend(load_csv(os.path.join(path, file_name)))

    # idx 0 - id
    # idx 3 - user
    # idx 4 - user_id
    # idx 5 - title
    # idx 13 - url
    count = 0
    size = len(result)
    pool = ThreadPoolExecutor(4)
    if not os.path.exists(PATH_PREFIX):  # 创建文件夹
        os.mkdir(PATH_PREFIX)

    for item in result:
        dir_name = f'{item[4]}'
        dir_name = os.path.join(PATH_PREFIX, dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        f_name = item[13].replace('https://i.pximg.net/img-original/img/', '').replace('/', '_')

        print(os.path.join(dir_name, f_name))

        pool.submit(down_load_image, item[13], dir_name, f_name, count, size)
        count += 1
