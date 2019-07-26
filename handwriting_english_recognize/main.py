from __future__ import division
from __future__ import print_function

import argparse

import cv2
from DataLoader import Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    fnCharList = './model/charList.txt'


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])


def main():
    model = Model(open(FilePaths.fnCharList).read(), DecoderType.BestPath, mustRestore=True)
    infer(model, './model/test8.jpg')


if __name__ == '__main__':
    # from pattern.en import suggest
    # print(suggest("it iss a gooood fooodd"))
    main()
