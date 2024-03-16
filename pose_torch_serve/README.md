# Run this after running download_model.py to get the .mar file GCP is expecting:
```
torch-model-archiver --model-name model --version 1.0 --model-file model.py --serialized-file keypoint_rcn
n.pt --handler pose_handler.py 
```