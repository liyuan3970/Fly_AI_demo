# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(image_path='images/125.Lincoln_Sparrow/Lincoln_Sparrow_0084_117492.jpg')
#source = model.predict_all(data)
#print("666666666:",source)
print(p)
