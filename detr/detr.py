from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrModel, DetrConfig
from PIL import Image
import cv2
from transformers import AdamW
import data_class
import xml_load
import torch
import numpy as np

id2label = { 0:'pic', 1:'caption', 2:'paragraph', 3:'heading'}
epoch = 100
data_loader = data_class.get_data_loader(2)

# feature_extractor = DetrFeatureExtractor(feature_extractor_type='DetrFeatureExtractor',id2label=id2label )
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
# from pre-trained
detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
# diy
config = DetrConfig(name_or_path='facebook/detr-resnet-50', num_channels=3,num_queries=70, id2label=id2label)
detr_model_diy = DetrForObjectDetection(config=config)

detr_model.zero_grad()
detr_model.train()
optimizer = AdamW(params=detr_model.parameters(), lr=0.01, weight_decay=3)
# feature_extractor.train()

print('start to train')
for num_epoch in range(epoch):
    for batch_index, file_path_list in enumerate(data_loader):
        images = []
        annotations = []
        target_size_for_post = []
        for index, file_path in enumerate(file_path_list):
            im = cv2.imread(file_path+'.jpg')
            target_size_for_post.append([im.shape[0], im.shape[1]])
            images.append(im)
            annotations.append({'annotations':xml_load.xml_reader(file_path+'.xml').annotations,
                                'image_id':index})

        inputs = feature_extractor(images=images, annotations=annotations, return_tensors='pt')
        outputs = detr_model_diy(**inputs)

        results = feature_extractor.post_process_object_detection(outputs, threshold=0, target_sizes=target_size_for_post)
        loss = outputs[0]
        print(num_epoch)
        print(loss)
        for index, info in enumerate(results):
            print(results[index]['labels'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

