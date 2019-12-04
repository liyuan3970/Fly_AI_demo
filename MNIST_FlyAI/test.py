import os
from flyai.dataset import Dataset

from model import Model

data = Dataset(epochs=10, batch=32)
model = Model(data)
x_val, y_val = data.next_validation_batch()
#val = {y_val: x_val}
# x_train, y_train, x_test, y_test = data.next_batch(32)
# feed_dict={x: x_train}
print(x_val.shape)
#print(y_val)
print(x_val[0].shape)
val={x_val[0]:'0'}
p = model.predict_all(val)


