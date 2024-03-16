import torch

import torchvision
from torchvision import models
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, KeypointRCNN
import torch

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

class Net(KeypointRCNN):
    def __init__(self):
        super(Net, self).__init__()