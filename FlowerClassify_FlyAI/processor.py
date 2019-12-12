# -*- coding: utf-8 -*
import numpy as np
from flyai.processor.base import Base
from flyai.processor.download import check_download
from skimage import io, transform

from path import DATA_PATH


class Processor(Base):

    def input_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        image = io.imread(path)
        image = transform.resize(image, (100, 100))
        image = np.array(image)
        image = image.astype(np.float32)
        return image

    def input_y(self, label):
        return np.asarray(label, np.float32)

    def output_y(self, data):
        labels = np.array(data)
        labels = labels.astype(np.float32)
        return labels