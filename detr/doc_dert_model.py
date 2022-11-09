import json

from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrConfig
import cv2
from transformers import AdamW
import data_class
import xml_load
import torch
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# data_loader, loader_type = data_class.get_data_loader(8)
data_loader, loader_type = data_class.get_coco_data_loader(8)
if loader_type == 'coco':
    annotations_coco = COCO('../coco_annotations/annotations.json')
x = annotations_coco.imgToAnns[0]
class doc_detr(torch.nn.Module):

    def __init__(self):
        super(doc_detr, self).__init__()
        self.epoch = 100
        self.id2label = { 0:'pic', 1:'caption', 2:'paragraph', 3:'heading'}
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.config = DetrConfig(
                                use_pretrained_backbone = False,
                                 num_channels=3,
                                 num_queries=100,
                                 id2label=self.id2label)
        self.detr_model = DetrForObjectDetection(config=self.config)

    def forward(self, images, annotations, target_size_for_post_process):
        inputs = self.feature_extractor(images=images, annotations=annotations, return_tensors='pt')
        outputs = self.detr_model(**inputs)
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0, target_sizes=target_size_for_post_process)
        loss = outputs[0]

        return loss, results

def train_model():
    doc_model = doc_detr()
    if os.path.exists('doc_model.pkl'):
        doc_model.load_state_dict(torch.load('doc_model.pkl'))
    doc_model.train()
    optimizer = AdamW(params=doc_model.parameters(), lr=0.001)

    for num_epoch in range(doc_model.epoch):
        result = []
        for batch_index, content in enumerate(data_loader):
            print(num_epoch)
            images, annotations, target_size_for_post_process, image_ids = get_annotation(content)
            loss, output_model = doc_model(images, annotations, target_size_for_post_process)

            print(loss)

            for index, info in enumerate(output_model):
                for cell_index in range(len(info['scores'])):
                    result.append({
                        "image_id": index,
                        "category_id": info['labels'][cell_index].item(),
                        "bbox": [info['boxes'][cell_index][0].item(),
                                 info['boxes'][cell_index][1].item(),
                                 info['boxes'][cell_index][2].item()-info['boxes'][cell_index][0].item(),
                                 info['boxes'][cell_index][3].item()-info['boxes'][cell_index][1].item()],
                        "score": info['scores'][cell_index].item(),
                    })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with open('train_log.txt', 'a') as file:
                file.write(str(loss.item()) + '\n')


        torch.save(doc_model.state_dict(), 'doc_model.pkl')
        with open('batch_result.json', 'w') as file:
            file.write(json.dumps(result))
        coco_predict = annotations_coco.loadRes('batch_result.json')
        coco_eval = COCOeval(annotations_coco, coco_predict, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        try:
            with open('train_log.txt', 'a') as file:
                file.write(str(coco_eval.stats) + '\n')
        except:
            continue

def get_annotation(output_loader):
    images = []
    annotations = []
    target_size_for_post_process = []

    if loader_type == 'no_coco':
        for index, file_path in enumerate(output_loader):
            print(file_path + '.jpg')
            im = cv2.imread(file_path + '.jpg')
            target_size_for_post_process.append([im.shape[0], im.shape[1]])
            images.append(im)
            annotations.append({'annotations': xml_load.xml_reader(file_path + '.xml').annotations,
                                'image_id': index})

    if loader_type == 'coco':
        file_path = '../coco_annotations/train_image/'
        for index, image_index in enumerate(output_loader):
            print(file_path + annotations_coco.imgs[image_index.item()]['file_name'])
            im = cv2.imread(file_path + annotations_coco.imgs[image_index.item()]['file_name'])
            target_size_for_post_process.append([im.shape[0], im.shape[1]])
            images.append(im)
            annotations.append({'annotations': annotations_coco.imgToAnns[image_index.item()],
                                'image_id': index})

    image_ids = [x['annotations'][0]['image_id'] for x in annotations]

    return images, annotations, target_size_for_post_process, image_ids


if __name__ == '__main__':
    train_model()