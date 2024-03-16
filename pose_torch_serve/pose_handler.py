from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler
from torch.profiler import ProfilerActivity


class PoseHandler(VisionHandler):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    def __init__(self):
        super(PoseHandler, self).__init__()
        self.profiler_args = {
            "activities" : [ProfilerActivity.CPU],
            "record_shapes": True,
        }        