from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(image_path="images/daisy/15207766_fc2f1d692c_n.jpg")
print(p)