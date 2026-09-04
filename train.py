from ultralytics import YOLO

# # Load a model
model = YOLO("./yolo11m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./cfg/.yaml", epochs=50, imgsz=960, device=3)


