from ultralytics import YOLO

# Load a model
model = YOLO("tests/weights/yolo11n-seg.pt")  # load an official model

# Train the model
results = model.train(data="tests/datasets/emergency_lane_seg.yaml", epochs=100, imgsz=640)