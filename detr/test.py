from transformers import DetrFeatureExtractor, DetrForObjectDetection,DetrModel, DetrConfig
import torch
import io
import matplotlib as plt
from PIL import Image
import requests
import cv2
import numpy
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import  COCO
url ="https://www.ourchinastory.com/images/cover/thats-day/2021/03/normal/%E7%95%B6%E4%BB%A3%E4%B8%AD%E5%9C%8B-%E5%AD%B8%E6%87%82%E4%B8%AD%E5%9C%8B-%E7%95%B6%E5%B9%B4%E4%BB%8A%E6%97%A5-%E5%A7%9A%E6%98%8ENBA%E5%90%8D%E4%BA%BA%E5%A0%82COVER_x3.jpg"
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

coco = COCO('instances_val2017.json')

coco_predicted = coco.loadRes('my_result.json')
cocoEval = COCOeval(coco,coco_predicted,'bbox')
cocoEval.params.imgIds =[2]

z = cocoEval.evaluate()
x = cocoEval.accumulate()
y = cocoEval.summarize()

a = torch.tensor([[1,2.3,4]])
x = [int(x.item()) for x in a[0]]




model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
config = DetrConfig( num_queries=70,
                     use_pretrained_backbone=False,
                    )

model = DetrForObjectDetection(config)
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
path = '../data/18680715_1-0002.jpg'
# z = cv2.imread(path)
im=Image.open(path, mode='r')
print(im.size)
x= im.getdata()
y = numpy.array(im)
im2 = Image.open(requests.get(url, stream=True).raw)
z = numpy.array(im2)
config = DetrConfig(name_or_path='facebook/detr-resnet-50', num_queries=70)

feature_extractor = DetrFeatureExtractor(config)
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
# inputs = feature_extractor(images=im, return_tensors="pt")

im2 = Image.open(requests.get(url, stream=True).raw)
print(im2.size)
inputs = feature_extractor(images=im, return_tensors="pt")


url ="https://www.ourchinastory.com/images/cover/thats-day/2021/03/normal/%E7%95%B6%E4%BB%A3%E4%B8%AD%E5%9C%8B-%E5%AD%B8%E6%87%82%E4%B8%AD%E5%9C%8B-%E7%95%B6%E5%B9%B4%E4%BB%8A%E6%97%A5-%E5%A7%9A%E6%98%8ENBA%E5%90%8D%E4%BA%BA%E5%A0%82COVER_x3.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(requests.get(url, stream=True).raw)
id2label = {4:'N/A', 0:'pic', 1:'caption', 2:'paragraph', 3:'heading'}
config = DetrConfig(name_or_path='facebook/detr-resnet-50', num_queries=70,id2label=id2label)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# feature_extractor = DetrFeatureExtractor(config)
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection(config)
inputs = feature_extractor(images=image, return_tensors="pt")
print(inputs)
outputs = model(**inputs)
print(type(outputs))

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process_object_detection(outputs, threshold=0,target_sizes=target_sizes)[0]
print(results)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    # let's only keep detections with score > 0.9
    if score > 0.9 :
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

