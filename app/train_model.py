from ultralytics import YOLO
def model():
    model_ = YOLO("yolo11n-cls.pt")
    model_.train(data='../dataset/data1a/training', epochs=5, verbose=True, device='cuda', workers=0)
    return model_
model()
