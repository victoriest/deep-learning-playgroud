"""
使用darknet训练好的yolov4模型, 进行物体检测的api
"""
import cv2

import pyyolo
from utils.image_converter import base64_to_cv2

# 训练模型加载地址
YOLOV4_TINY_WEIGHTS = 'static/models/yolov4-tiny-tftf.weights'
YOLOV4_TINY_CFG = 'static/cfg/yolov4-tiny-tftf.cfg'
YOLOV4_TINY_NAMES = 'static/cfg/tftf.data'

YOLOV4_WEIGHTS = 'static/models/yolov4-tftf.weights'
YOLOV4_CFG = 'static/cfg/yolov4-tftf.cfg'
YOLOV4_NAMES = 'static/cfg/tftf.data'

YOLOV4_TINY_TFTFQNSC_WEIGHTS = 'static/models/yolov4-tiny-tftfqnsc.weights'
YOLOV4_TINY_TFTFQNSC_CFG = 'static/cfg/yolov4-tiny-tftfqnsc.cfg'
YOLOV4_TINY_TFTFQNSC_NAMES = 'static/cfg/tftfqnsc.data'

# 识别阈值
DETECTION_THRESHOLD = 0.3

yolov4_tiny_tftf = pyyolo.YOLO(YOLOV4_TINY_TFTFQNSC_CFG,
                               YOLOV4_TINY_TFTFQNSC_WEIGHTS,
                               YOLOV4_TINY_TFTFQNSC_NAMES,
                               detection_threshold=DETECTION_THRESHOLD,
                               hier_threshold=0.5,
                               nms_threshold=0.45)

yolov4_tftf = pyyolo.YOLO(YOLOV4_CFG,
                          YOLOV4_WEIGHTS,
                          YOLOV4_NAMES,
                          detection_threshold=DETECTION_THRESHOLD,
                          hier_threshold=0.5,
                          nms_threshold=0.45)


def detect_by_yolov4_tftf(data, is_tiny=True):
    image = base64_to_cv2(data['img'])
    if is_tiny:
        boxes = yolov4_tiny_tftf.detect(image, rgb=False)
    else:
        boxes = yolov4_tftf.detect(image, rgb=False)
    pos = []
    for _, det in enumerate(boxes):
        item = {"x": int(det.x), "y": int(det.y), "w": int(det.w), "h": int(det.h), "probability": float(det.prob),
                "name": det.name}
        pos.append(item)
    result = {"pos": pos}
    return result


def detect(path="./0.jpg"):
    frame = cv2.imread(path)
    dets = yolov4_tiny_tftf.detect(frame, rgb=False)
    for i, det in enumerate(dets):
        print(f'Detection: {i}, {det}')
        xmin, ymin, xmax, ymax = det.to_xyxy()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255))
    # cv2.imshow('cvwindow', frame)
    cv2.imwrite("est.jpg")


if __name__ == '__main__':
    detect()
