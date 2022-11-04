import torch
from transformers import DeformableDetrModel

print(torch.cuda.is_available())
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("驱动为：",device)
print("GPU型号： ",torch.cuda.get_device_name(0))
