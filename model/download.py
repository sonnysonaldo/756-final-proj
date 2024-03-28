import torch
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
model_name = "keypoint_rcnn"
model_file = f"./{model_name}.pt"

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
sm = torch.jit.script(model)
sm.save(model_file)

# torch-model-archiver --model-name model --version 1.0  --serialized-file keypoint_rcnn.pt --handler handler.py