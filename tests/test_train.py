from ultralytics import YOLO

# Load a model
model = YOLO("tests/weights/yolo11n-seg.pt")  # load an official model

# Train the model
results = model.train(data="ultralytics/cfg/datasets/coco128-seg.yaml", epochs=100, imgsz=640)