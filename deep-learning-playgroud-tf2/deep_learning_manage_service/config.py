import os
import socket

# 这里的配置文件属性名称必须为全大写, 否则app.add_url_rule 不会加载属性

DEBUG = False

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 获取IP地址
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
SERVER_HOST = s.getsockname()[0]
s.close()

SERVER_PORT = 8200
SERVICE_URL = 'http://' + SERVER_HOST + ':' + str(SERVER_PORT) + '/axb-yolov4'
