from pyyolo.yolov4_util import yolov4_tiny_tftf
from utils.image_converter import base64_to_cv2
from variable_length_digit_htr.recognizer import digit_ocr


def detect_by_zyb_ans_htr(data):
    image = base64_to_cv2(data['img'])
    boxes = yolov4_tiny_tftf.detect(image, rgb=False)
    pos = []
    for _, det in enumerate(boxes):
        x, y, w, h = int(det.x), int(det.y), int(det.w), int(det.h)
        item = {"x": int(det.x), "y": int(det.y), "w": int(det.w), "h": int(det.h), "probability": float(det.prob),
                "name": det.name}
        if det.name == 'true':
            item['objType'] = 'answer_mark'
            item['objValue'] = '9'
        elif det.name == 'false' or det.name == 'half-true':
            item['objType'] = 'answer_mark'
            item['objValue'] = '0'
        elif det.name == 'half-true':
            item['objType'] = 'answer_mark'
            item['objValue'] = '5'
        elif det.name == 'question_num':
            im_crop = image[y: y + h, x:x + w]
            item['objType'] = 'question_num'
            item['objValue'] = digit_ocr(im_crop)
        pos.append(item)

    result = {"pos": pos}
    return result
