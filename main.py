from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch


# Use the model
model.train(data="config.yaml", epochs=500, optimizer='Adam', lr0=0.01)  # train the model with the specified number of epochs