import layoutparser as lp
import cv2

im = cv2.imread('../coco_annotations/train_image/18680715_1-0001.jpg')
image = im[..., ::-1]

model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

layout = model.detect(image)

lp.draw_box(image, layout, box_width=3)
