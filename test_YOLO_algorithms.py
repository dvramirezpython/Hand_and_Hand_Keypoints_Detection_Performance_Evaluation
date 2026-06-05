# This file will be used to test the performance of the YOLO algorithms
# on the hand keypoint detection task.

from ultralytics import YOLO, RTDETR

def test_yolo_model(model_path, source_dir, model_type, hand_type):
    # Load the YOLO model
    if 'yolo' in model_type:
        model = YOLO(model_path)
    elif model_type == 'rtdetr': 
        model = RTDETR(model_path)
    # Validate the model on the source directory
    results = model.val(data=source_dir, device='cuda', name=f'{model_type}_{hand_type}')
    source_images = source_dir.replace('data.yaml', 'images')
    model.predict(source_images, device='cuda', name=f'{model_type}_{hand_type}_predictions', save=True)
    return results

if __name__ == "__main__":
    for model_type in ['yolov8', 'yolo11', 'yolo12', 'yolo26']:
        for hand_type in ['plain', 'deformed']:
           
            SOURCE_DIR = f'dataset/Validation_Hand_keypoints_by_morphology/{hand_type}/data.yaml'  # Path to the validation dataset
            MODEL_PATH = f'runs/pose/{model_type}s-pose/weights/best.pt'  # Path to the trained YOLO model
            
            results = test_yolo_model(MODEL_PATH, SOURCE_DIR, model_type, hand_type)
            # Save results in a txt file
            with open(f'results_{model_type}_{hand_type}.txt', 'w') as f:
                f.write(str(results))

            print(f'Evaluation successfully completed {model_type}_{hand_type}. \n Results saved to CSV.')

    # Evaluate RTDETR model
    # model_type = 'rtdetr'
    # hand_type = 'plain'
    # SOURCE_DIR = f'dataset/Validation_Hand_keypoints_by_morphology/{hand_type}/data.yaml'  # Path to the validation dataset
    # MODEL_PATH = f'runs/detect/{model_type}-l/weights/best.pt'  # Path to the trained YOLO model
    
    # results = test_yolo_model(MODEL_PATH, SOURCE_DIR, model_type, hand_type)
    # # Save results in a txt file
    # with open(f'results_{model_type}_{hand_type}.txt', 'w') as f:
    #     f.write(str(results))

    # print(f'Evaluation successfully completed {model_type}_{hand_type}. \n Results saved to CSV.')
