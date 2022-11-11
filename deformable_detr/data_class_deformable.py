import torchvision
import os
import matplotlib.pyplot as plt
from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "annotations.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

def get_dataloader():
    train_dataset = CocoDetection(img_folder='../coco_annotations/train_image/', feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder='../coco_annotations/test_image/', feature_extractor=feature_extractor,
                                train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

    return train_dataloader, val_dataloader

# if __name__ == "__main__":
#     a, b = get_dataloader()
#     batch = next(iter(a))
#     print(batch)