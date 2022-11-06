from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrModel, DetrConfig
from PIL import Image
import cv2
from transformers import AdamW
import data_class
import xml_load

id2label = {4:'N/A', 0:'pic', 1:'caption', 2:'paragraph', 3:'heading'}
data_loader = data_class.get_data_loader(2)
feature_extractor = DetrFeatureExtractor(feature_extractor_type='DetrFeatureExtractor')
# from pre-trained
detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
# diy
config = DetrConfig(num_channels = 3, num_queries=70, id2label=id2label)
detr_model_diy = DetrForObjectDetection(config=config)
epoch = 10
learning_rate = 3e-5
adam_epsilon = 1e-8
weight_decay = 0
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in detr_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in detr_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
detr_model.zero_grad()
detr_model.train()

print('start to train')
for _ in range(epoch):
    for batch_index, file_path_list in enumerate(data_loader):
        images = []
        annotations = []
        target_size_for_post = []
        for index, file_path in enumerate(file_path_list):
            im = Image.open(file_path+'.jpg')
            height = im.height
            width = im.width
            target_size_for_post.append([height, width])
            im = cv2.imread(file_path+'.jpg').reshape(3, height, width)
            images.append(im)
            annotations.append({'annotations':xml_load.xml_reader(file_path+'.xml').annotations,
                                'image_id':index})
        inputs = feature_extractor(images=images, annotations=annotations, return_tensors='pt')

        outputs = detr_model_diy(**inputs)

        results = feature_extractor.post_process_object_detection(outputs,  target_sizes=target_size_for_post)
        loss = outputs[0]
        print(loss)
        print(results)
        loss.backward()
        optimizer.step()

        detr_model.zero_grad()

