# Section 0: Importing necessary libraries
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

import numpy as np
import os, sys
import cv2
from PIL import Image
import random



# Section 1: Obtain keypointrcnn_resnet50_fpn from torchvision and load the pre-trained weights
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
# print(model)

# Section 2: Prepare the image for inference
def output_keypoints_from_fn(im_path):
    image1 = Image.open(im_path)
    transform = transforms.Compose([
                transforms.ToTensor()
            ])
    image1 = transform(image1)[:3].unsqueeze(0)
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    preds = model(image1)
    preds = preds[0]
    return preds


def output_keypoints_from_img(image1):
    transform = transforms.Compose([
                transforms.ToTensor()
            ])
    image1 = transform(image1)[:3].unsqueeze(0)
    # print(image1.shape)
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Section 3: Perform inference and visualize the results on a1.png and a2.png
    model.eval()
    preds = model(image1)
    preds = preds[0]
    return preds


def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = np.arctan2(abs(x2 - x1), abs(y2 - y1))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Section 3: Perform inference and visualize the results on a1.png and a2.png

def visualize_kp_results(cv_image, preds, det_obj_thresh=0.7, show=True):
    for obj in range(len(preds['scores'])):
        keypoints = preds['keypoints'].detach()
        kp_scores = preds['keypoints_scores'].detach()
        score = preds['scores'][obj].detach()

        if score < det_obj_thresh:  
            continue

        for i in range(len(keypoints[obj])):
            x, y, conf = keypoints[obj][i]
            x, y = int(x), int(y)
            # score = kp_scores[obj][i]
            # print("x {} y {} conf {}".format(x, y, conf))

            cv2.circle(cv_image, (int(x), int(y)), 5, (0, 255, 0), -1)
            # Add the index number near the keypoint
            cv2.putText(cv_image, str(i), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        shoulder_center = np.mean([keypoints[obj][5][:-1].numpy(), keypoints[obj][6][:-1].numpy()], axis=0)
        hip_center = np.mean([keypoints[obj][11][:-1].numpy(), keypoints[obj][12][:-1].numpy()], axis=0)

        shoulder_x , shoulder_y = int(shoulder_center[0]), int(shoulder_center[1])
        hip_x , hip_y = int(hip_center[0]), int(hip_center[1])

        cv2.line(cv_image, (shoulder_x, shoulder_y), (hip_x, hip_y), (0, 0, 255), 5)
        angle = calculate_angle(shoulder_center, hip_center)

        if angle > 45:
            cv2.putText(cv_image, "Fall", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            cv2.putText(cv_image, "No Fall", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    return cv_image


# im_path = sys.argv[1]
# print("getting output")
# preds = output_keypoints_from_fn(im_path)
# print("visualizing output")
# visualize_kp_results(cv2.imread(im_path), preds)

# cv2.imshow('Frame', visualize_kp_results(cv2.imread(im_path), preds))
# cv2.waitKey(0)
# #     if cv2.waitKey(2) & 0xFF == ord('q'):
# #         break

# Section 4: Writing the Fall Detection Logic

# video_path = sys.argv[1]  # Update with your video file path
# cap = cv2.VideoCapture(video_path)
# frame_count = 0

# append_frames = []

# while True:
#     # Read a frame
#     ret, frame = cap.read()
#     if not ret:
#         break  # Break the loop when the video ends
#     preds = output_keypoints_from_img(frame) 
#     res = visualize_kp_results(frame, preds, show=False)
#     append_frames.append(res)

# print("done predicting")

# def save_video(frames, output_video_path='res.mp4'):
#     # Get the height and width of the frames (assuming all frames have the same size)
#     height, width, _ = frames[0].shape

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
#     fps = 30  # Frames per second
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#     # Write each frame to the video
#     for frame in frames:
#         video_writer.write(frame)
#     # Release the VideoWriter object
#     video_writer.release()

# save_video(append_frames)


# # for res in append_frames:
# #     # Show the frame (optional, for visualization)
# #     cv2.imshow('Frame', res)
# #     if cv2.waitKey(2) & 0xFF == ord('q'):
# #         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


# Section 5: Visualize the Fall Detection on a Single Image (a1.png and a2.png)

# Section 6: Make it run on a video and test it on v1.mp4