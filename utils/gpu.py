"""
获取可用GPU设备信息
"""
import os

from tensorflow.python.client import device_lib

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'


def get_available_gpus():
    # gpu_device_name = tf.test.gpu_device_name()
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print(get_available_gpus())
