from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2

url ="https://www.ourchinastory.com/images/cover/thats-day/2021/03/normal/%E7%95%B6%E4%BB%A3%E4%B8%AD%E5%9C%8B-%E5%AD%B8%E6%87%82%E4%B8%AD%E5%9C%8B-%E7%95%B6%E5%B9%B4%E4%BB%8A%E6%97%A5-%E5%A7%9A%E6%98%8ENBA%E5%90%8D%E4%BA%BA%E5%A0%82COVER_x3.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(type(outputs))

# convert outputs (bounding boxes and class logits) to COCO API
# target_sizes = torch.tensor([image.size[::-1]])
# results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
#
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     # let's only keep detections with score > 0.9
#     if score > 0.9 :
#         print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#         )
#
