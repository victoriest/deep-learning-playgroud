from flask import jsonify, request

from . import web

common_api_headers = {
    'content-type': 'application/json'
}


@web.route('/axb-ocr/yolov4/tftf', methods={'POST', 'GET'})
def yolov4_tftf():
    """
    利用yolov4模型识别给定图片base64数据(对错半对)
    :return:
    """
    result = detect_by_yolov4_tftf(request.get_json(), is_tiny=False)
    return jsonify(bizCode=0, message='success', data=result, params=None), 200, common_api_headers
