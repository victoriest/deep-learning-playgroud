import os

from .hed import hed
from .hed_data_parser import DataParser

from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import glob
from PIL import Image
import cv2

if __name__ == "__main__":
    # environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # if not os.path.isdir(model_dir): os.makedirs(model_dir)
    # model
    model = hed()
    # plot_model(model, to_file=os.path.join(model_dir, 'model.pdf'), show_shapes=True)

    # training
    # call backs
    model.load_weights('./checkpoint.03-0.01.hdf5')

    data_path = []
    g = os.walk('./in')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            data_path.append((file_name[0], d))
            print(d, file_name)

            # count = 0
            # for path in data_path:
            #     data_label.append(path[0])
            #     data.append(cv2.imread(path[1], 0) / 255)
            #     count += 1

            x_batch = []
            im = Image.open(d)
            (h, w) = im.size
            print(h, w)
            im = im.resize((480, 480))
            im = np.array(im, dtype=np.float32)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            x_batch.append(im)
            x_batch = np.array(x_batch, np.float32)
            prediction = model.predict(x_batch)
            mask = np.zeros_like(im[:, :, 0])
            for i in range(len(prediction)):
                mask += np.reshape(prediction[i], (480, 480))
            ret, mask = cv2.threshold(mask, np.mean(mask) + 1.2 * np.std(mask), 255, cv2.THRESH_BINARY)
            out_mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_CUBIC)
            # out_mask = mask.resize((h,w))
            out_d = os.path.join('./out', file_name)
            cv2.imwrite(out_d, out_mask)
