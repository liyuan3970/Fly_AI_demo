# -*- coding: utf-8 -*
import cv2
import numpy
import numpy as np
from flyai.processor.base import Base
from flyai.processor.download import check_download

from path import DATA_PATH


class Processor(Base):

    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        # path 为图片的真实路径
        path = check_download(image_path, DATA_PATH)
        image = cv2.imread(path)
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_CUBIC)
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = numpy.multiply(x_data, 1.0 / 255.0)
        return x_data

    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''

        input_y = label
        return input_y

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels = np.array(data)
        labels = labels.astype(np.float32)

        return np.argmax(labels)