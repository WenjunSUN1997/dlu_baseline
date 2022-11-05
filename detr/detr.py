from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2

class detr_doc():
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor()