from ultralytics import YOLO
from PIL import Image

model = YOLO('../runs/classify/train5/weights/best.pt')

def predict(image: Image):
    results = model(image, task='classify')
    if results[0].probs is not None:
        probs = results[0].probs
        return {
            'class': results[0].names[probs.top1],
            'confidence': float(probs.top1conf),
            'all_probs': probs.data.tolist()
        }
    return {'error': 'No predictions'}
