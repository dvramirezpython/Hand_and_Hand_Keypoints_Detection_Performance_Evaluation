# Python Script to train and validate the YOLO algorithms for hand keypoint detection.

from ultralytics import YOLO
from ultralytics import RTDETR

SOURCE_DIR = 'dataset/hand-keypoints/data.yaml'
MODEL = 'rtdetr-l'  # You can choose a different pre-trained model if needed

def train_yolo_model(source_dir):
    # Load the YOLO model
    # model = YOLO(f'{MODEL}.pt', task='pose')  # Use a pre-trained YOLO model as a starting point
    model = RTDETR(f'{MODEL}.pt')  # Use a pre-trained YOLO model as a starting point

    # Train the model on the provided dataset
    model.train(data=source_dir, 
                epochs=200, 
                batch=8, 
                device='cuda',
                name=f'{MODEL}')  # Adjust epochs and batch size as needed

if __name__ == "__main__":
    train_yolo_model(SOURCE_DIR)