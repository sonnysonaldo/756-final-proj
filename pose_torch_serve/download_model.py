#scripted mode
import torchvision
from torchvision import models
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
import torch

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
sm = torch.jit.script(model)
sm.save("keypoint_rcnn.pt")