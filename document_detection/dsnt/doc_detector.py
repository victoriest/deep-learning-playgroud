import base64
from datetime import datetime
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

from .dsnt import dsnt

graph = None
sess = None
inputs, hm1, kp1, hm2, kp2, hm3, kp3, hm4, kp4 = None, None, None, None, None, None, None, None, None
DOC_DETECT_MODEL_PATH = 'frozen_model.pb'


def _load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def _get_graph_and_session():
    global graph, sess, inputs, hm1, kp1, hm2, kp2, hm3, kp3, hm4, kp4
    if graph is None or sess is None:
        graph = _load_graph(DOC_DETECT_MODEL_PATH)
        sess = tf.Session(graph=graph)

        inputs = graph.get_tensor_by_name('input:0')
        activation_map = graph.get_tensor_by_name("heats_map_regression/pred_keypoints/BiasAdd:0")

        hm1, kp1, = dsnt(activation_map[..., 0])
        hm2, kp2, = dsnt(activation_map[..., 1])
        hm3, kp3, = dsnt(activation_map[..., 2])
        hm4, kp4, = dsnt(activation_map[..., 3])

    return graph, sess


def doc_detect_base64(data):
    img_base64 = data['img']
    byte_data = base64.b64decode(img_base64)
    image_data = BytesIO(byte_data)
    origin_img = Image.open(image_data)
    return doc_detect_util(data['id'], origin_img)


def doc_detect(data):
    img_path = data['imgUrl']
    origin_img = Image.open(img_path)
    return doc_detect_util(data['id'], origin_img)


def doc_detect_util(data_id, origin_img):
    global graph, sess, inputs, hm1, kp1, hm2, kp2, hm3, kp3, hm4, kp4
    a = datetime.now()
    result = {
        'id': data_id,
        'recognizeResult': {}
    }
    graph, sess = _get_graph_and_session()

    width, height = origin_img.size
    img_nd = np.array(origin_img.resize((600, 800)))

    hm1_nd, hm2_nd, hm3_nd, hm4_nd, kp1_nd, kp2_nd, kp3_nd, kp4_nd = sess.run(
        [hm1, hm2, hm3, hm4, kp1, kp2, kp3, kp4], feed_dict={inputs: np.expand_dims(img_nd, 0)})

    keypoints_nd = np.array([kp1_nd[0], kp2_nd[0], kp3_nd[0], kp4_nd[0]])
    keypoints_nd = ((keypoints_nd + 1) / 2 * np.array([600, 800])).astype('int')

    i = 0
    for x, y in keypoints_nd:
        keypoints_nd[i][0] = round(x * width / 600)
        keypoints_nd[i][1] = round(y * height / 800)
        i += 1

    b = datetime.now()

    print((b - a), keypoints_nd)

    recognize_result = {'topLeftX': str(keypoints_nd[0][0]), 'topLeftY': str(keypoints_nd[0][1]),
                        'topRightX': str(keypoints_nd[1][0]), 'topRightY': str(keypoints_nd[1][1]),
                        'bottomLeftX': str(keypoints_nd[2][0]), 'bottomLeftY': str(keypoints_nd[2][1]),
                        'bottomRightX': str(keypoints_nd[3][0]), 'bottomRightY': str(keypoints_nd[3][1])}

    result['recognizeResult'] = recognize_result

    return result


if __name__ == "__main__":
    data = {
        'id': '111',
        'imgUrl': '1.jpg'
    }
    doc_detect(data)
