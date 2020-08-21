import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


##PIL读取、保存图片方法
# img = Image.open(img_path)
# img.save(img_path2)

##cv2读取、保存图片方法
# img = cv2.imread(img_path)
# cv2.imwrite(img_path2, img)


def img_path_to_base64(img_path):
    """
    图片文件打开为base64
    :param img_path:
    :return:
    """
    with open(img_path, "rb") as f:
        base64_str = base64.b64encode(f.read())
    return base64_str


def cv2_to_base64(img):
    base64_str = cv2.imencode('.jpg', img)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


def base64_to_cv2(base64_str):
    byte_data = base64.b64decode(base64_str)
    np_arr = np.fromstring(byte_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def base64_to_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image


def pil_to_base64(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def pil_to_cv2(img_src):
    img = cv2.cvtColor(np.asarray(img_src), cv2.COLOR_RGB2BGR)
    return img


def cv2_to_pil(img_src):
    img = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
    return img
