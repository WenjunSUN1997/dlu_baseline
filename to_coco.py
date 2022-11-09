import os
import xml_load
from PIL import Image
import json
import random
from shutil import copyfile

path = 'E:/chrome/AS_TrainingSet_BnF_NewsEye_v2/'
file_list = os.listdir(path)
jpg_file = [x for x in file_list if x.split('.')[1]=='jpg']
random.shuffle(jpg_file)

train_file = jpg_file[:int(0.8*len(jpg_file))]
test_file = jpg_file[int(0.8*len(jpg_file)):]

for file in train_file:
    copyfile(path+file, 'coco_annotations/train_image/'+file)

for file in test_file:
    copyfile(path+file, 'coco_annotations/test_image/'+file)

coco_annotations = {'images':[], 'annotations': [], 'categories': [],}

for index, file_name in enumerate(jpg_file):
    dict_ann = {}
    im = Image.open(path+file_name)
    dict_ann['file_name'] = file_name
    dict_ann['height'] = im.height
    dict_ann['width'] = im.width
    dict_ann['id'] = index
    coco_annotations['images'].append(dict_ann)

coco_annotations['categories'] = [{'supercategory': 'pic', 'id': 0, 'name': 'pic'},
                                  {'supercategory': 'caption', 'id': 1, 'name': 'caption'},
                                  {'supercategory': 'paragraph', 'id': 2, 'name': 'paragraph'},
                                  {'supercategory': 'heading', 'id': 3, 'name':'heading'}]

for index, fig in enumerate(jpg_file):
    file_path = (path + fig).replace('jpg', 'xml')
    annotations_info = xml_load.xml_reader(file_path).annotations

    for index_cell, info_cell in enumerate(annotations_info):
        info_cell['image_id'] = index
        info_cell['id'] = index_cell
        coco_annotations['annotations'].append(info_cell)

final_file = json.dumps(coco_annotations)
j_file = open('coco_annotations/annotations.json', 'w')
j_file.write(final_file)
j_file.close()
