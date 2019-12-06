'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(image_path='images/00022/01055_00001.jpg')
print(p)