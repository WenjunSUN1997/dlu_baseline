from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrModel, DetrConfig
from PIL import Image
import cv2
from transformers import AdamW
import data_class
import xml_load
import torch

data_loader = data_class.get_data_loader(8)

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

    def forward(self, file_path_list):
        print(file_path_list)
        images = []
        annotations = []
        target_size_for_post = []

        for index, file_path in enumerate(file_path_list):
            im = cv2.imread(file_path+'.jpg')
            target_size_for_post.append([im.shape[0], im.shape[1]])
            images.append(im)
            annotations.append({'annotations':xml_load.xml_reader(file_path+'.xml').annotations,
                                'image_id':index})

        inputs = self.feature_extractor(images=images, annotations=annotations, return_tensors='pt')
        outputs = self.detr_model(**inputs)
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0, target_sizes=target_size_for_post)
        loss = outputs[0]

        return loss, results

def train_model():
    doc_model = doc_detr()
    doc_model.train()
    optimizer = AdamW(params=doc_model.parameters(), lr=0.001)

    for num_epoch in range(doc_model.epoch):
        for batch_index, file_path_list in enumerate(data_loader):
            print(num_epoch)

            loss, results = doc_model(file_path_list)
            print(loss)
            # for index, info in enumerate(results):
            #     print(results[index]['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(doc_model.state_dict(), 'doc_model.pkl')

if __name__ == '__main__':
    train_model()