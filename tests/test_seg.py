'''
使用当前环境的ultralytics模块，测试yolo11n-seg.pt模型
pip install -e . 安装当前目录下的ultralytics模块
'''

import sys
import os
# Add the parent directory to sys.path to ensure we use the local ultralytics module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# Load a model
model = YOLO("tests/weights/yolo11n-seg.pt")  # load an official model

# Predict with the model
results = model("tests/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    result.save(filename="tests/results/result.jpg")