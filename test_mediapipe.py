# Python Script to validate the MediaPipe hand keypoint detection algorithm on the provided dataset.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(boxA, boxB):
    # Coordinate layout: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[0])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[0])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def parse_yolo_line(line, img_w, img_h):
    """Parses a YOLO keypoint line into absolute coordinates."""
    data = list(map(float, line.strip().split()))
    if len(data) < 5:
        return None
    
    # Parse Bounding Box (Normalized -> Absolute)
    _, cx, cy, nw, nh = data[:5]
    x1 = int((cx - nw / 2) * img_w)
    y1 = int((cy - nh / 2) * img_h)
    x2 = int((cx + nw / 2) * img_w)
    y2 = int((cy + nh / 2) * img_h)
    bbox = [x1, y1, x2, y2]
    
    # Parse 21 Keypoints (Each is x, y, visibility)
    kp_data = data[5:]
    gt_kps = []
    for i in range(0, len(kp_data), 3):
        kx = int(kp_data[i] * img_w)
        ky = int(kp_data[i+1] * img_h)
        gt_kps.append([kx, ky])
        
    return {"bbox": bbox, "kps": np.array(gt_kps), "diag": np.sqrt((x2-x1)**2 + (y2-y1)**2)}


def calculate_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using all-points area integration.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

def calculate_oks(gt_kps, pred_kps, box_area):
    """
    Computes Object Keypoint Similarity (OKS) between matched ground truth and predicted keypoints.
    """
    if box_area <= 0 or len(gt_kps) == 0 or len(pred_kps) == 0:
        return 0.0
    
    distances = np.linalg.norm(gt_kps - pred_kps, axis=1)
    scale = np.sqrt(box_area)
    k = 0.05  # Standard deviation factor for tightly-bound hand landmarks
    
    oks_vals = np.exp(-(distances ** 2) / (2 * (scale ** 2) * (k ** 2)))
    return np.mean(oks_vals)

def evaluate_dataset(image_dir, label_dir, detector):
    """
    Compares MediaPipe Landmarker outputs with YOLO ground truth files.
    Computes Box and Pose metrics, including Pose Precision and Pose Recall.
    """
    total_gt_hands = 0
    total_mp_hands = 0
    box_true_positives = 0
    pose_true_positives = 0  # Track true positives for keypoints (OKS >= 0.50)
    
    all_ious = []
    all_maes = []
    pck_hits = 0
    total_kp_count = 0
    
    pck_threshold_ratio = 0.05 

    img_with_mult_hands = []
    img_with_no_hand = []

    box_predictions_registry = []
    kps_predictions_registry = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        base_name = os.path.splitext(filename)[0]
        lbl_path = os.path.join(label_dir, base_name + '.txt')
        img_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(lbl_path):
            continue
            
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w, _ = frame.shape
        
        # 1. Parse Ground Truth
        gt_hands = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parsed = parse_yolo_line(line, w, h)
                if parsed:
                    gt_hands.append(parsed)
        
        # 2. Run MediaPipe Inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        mp_hands = []
        if results.hand_landmarks:
            if len(results.hand_landmarks) > 1:
                print(f"Multiple hands detected in {filename}. Evaluating all detected hands.")
                img_with_mult_hands.append(filename)
                annotated_image = draw_landmarks_on_image(frame, results)
                cv2.imwrite(f"annotated_{filename}", annotated_image)
            
            scores = getattr(results, 'handedness', None)
            
            for idx, landmarks in enumerate(results.hand_landmarks):
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                mx1, my1, mx2, my2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                kps = np.array([[x, y] for x, y in zip(xs, ys)])
                
                score = 1.0
                if scores and idx < len(scores):
                    score = scores[idx][0].score
                    
                mp_hands.append({"bbox": [mx1, my1, mx2, my2], "kps": kps, "score": score})
        else:
            print(f"No hands detected in {filename}.")
            img_with_no_hand.append(filename)

        total_gt_hands += len(gt_hands)
        total_mp_hands += len(mp_hands)
        
        if len(gt_hands) == 0:
            for mp_h in mp_hands:
                box_predictions_registry.append({"score": mp_h["score"], "max_match_val": 0.0})
                kps_predictions_registry.append({"score": mp_h["score"], "max_match_val": 0.0})
            continue

        if len(mp_hands) == 0:
            continue

        # 3. Match GT Hands with MP Predicted Hands
        cost_matrix = np.zeros((len(gt_hands), len(mp_hands)))
        for g_idx, gt in enumerate(gt_hands):
            for m_idx, mp_h in enumerate(mp_hands):
                cost_matrix[g_idx, m_idx] = 1.0 - calculate_iou(gt["bbox"], mp_h["bbox"])
                
        gt_ind, mp_ind = linear_sum_assignment(cost_matrix)
        
        # 4. Evaluate Matched Hand Targets
        for g_i, m_i in zip(gt_ind, mp_ind):
            iou = calculate_iou(gt_hands[g_i]["bbox"], mp_hands[m_i]["bbox"])          
            
            gt_bbox = gt_hands[g_i]["bbox"]
            gt_box_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            oks = calculate_oks(gt_hands[g_i]["kps"], mp_hands[m_i]["kps"], gt_box_area)
            
            box_predictions_registry.append({"score": mp_hands[m_i]["score"], "max_match_val": iou})
            kps_predictions_registry.append({"score": mp_hands[m_i]["score"], "max_match_val": oks})

            # Check Box True Positive (IoU >= 0.45)
            if iou >= 0.45:
                box_true_positives += 1
                all_ious.append(iou)
                
                gt_kps = gt_hands[g_i]["kps"]
                pred_kps = mp_hands[m_i]["kps"]
                
                distances = np.linalg.norm(gt_kps - pred_kps, axis=1)
                all_maes.extend(distances)
                
                pck_limit = pck_threshold_ratio * gt_hands[g_i]["diag"]
                pck_hits += np.sum(distances <= pck_limit)
                total_kp_count += len(distances)

            # Check Pose True Positive (OKS >= 0.50)
            if oks >= 0.50:
                pose_true_positives += 1
            else:
                # If not a true positive for pose, we can still log it as a false positive 
                # for pose evaluation
                annotated_image = draw_landmarks_on_image(frame, results)
                cv2.imwrite(f"annotated_pose_fail_{filename}", annotated_image)

    # --- mAP Calculation Helper ---
    def compute_map_metrics(registry, total_gts):
        if not registry or total_gts == 0:
            return 0.0, 0.0
        
        registry.sort(key=lambda x: x["score"], reverse=True)
        thresholds = np.linspace(0.50, 0.95, 10)
        ap_scores = []
        
        for th in thresholds:
            tp_array = np.zeros(len(registry))
            fp_array = np.zeros(len(registry))
            for idx, pred in enumerate(registry):
                if pred["max_match_val"] >= th:
                    tp_array[idx] = 1
                else:
                    fp_array[idx] = 1
            tp_cum = np.cumsum(tp_array)
            fp_cum = np.cumsum(fp_array)
            precisions = tp_cum / (tp_cum + fp_cum)
            recalls = tp_cum / total_gts
            ap_scores.append(calculate_ap(recalls, precisions))
            
        return ap_scores[0], np.mean(ap_scores)

    # Compute Final mAP Metrics
    box_mAP50, box_mAP50_95 = compute_map_metrics(box_predictions_registry, total_gt_hands)
    pose_mAP50, pose_mAP50_95 = compute_map_metrics(kps_predictions_registry, total_gt_hands)

    # --- Box-Level Evaluation Metrics ---
    box_precision = box_true_positives / total_mp_hands if total_mp_hands > 0 else 0
    box_recall = box_true_positives / total_gt_hands if total_gt_hands > 0 else 0
    box_f1_score = 2 * (box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) > 0 else 0
    
    # --- Pose-Level Evaluation Metrics (OKS >= 0.50) ---
    pose_precision = pose_true_positives / total_mp_hands if total_mp_hands > 0 else 0
    pose_recall = pose_true_positives / total_gt_hands if total_gt_hands > 0 else 0
    pose_f1_score = 2 * (pose_precision * pose_recall) / (pose_precision + pose_recall) if (pose_precision + pose_recall) > 0 else 0

    mean_iou = np.mean(all_ious) if all_ious else 0
    mean_mae_pixels = np.mean(all_maes) if all_maes else 0
    pck_score = (pck_hits / total_kp_count) * 100 if total_kp_count > 0 else 0
    
    return {
        "multiple_hands_images": img_with_mult_hands,
        "no_hand_images": img_with_no_hand,
        "total_gt_hands": total_gt_hands,
        "total_mp_hands": total_mp_hands,
        "box_precision": box_precision, 
        "box_recall": box_recall, 
        "box_f1_score": box_f1_score,
        "pose_precision": pose_precision, 
        "pose_recall": pose_recall, 
        "pose_f1_score": pose_f1_score,
        "mean_iou": mean_iou,
        "mean_mae_pixels": mean_mae_pixels,
        "pck_score": pck_score,
        "box_mAP50": box_mAP50,
        "box_mAP50-95": box_mAP50_95,
        "pose_mAP50": pose_mAP50,
        "pose_mAP50-95": pose_mAP50_95
    }

def validate_mediapipe_on_dataset(source_dir):

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    for i, img_name in enumerate(os.listdir(source_dir)):
        img_path = os.path.join(source_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print('No image found')
            continue
        # frame = cv2.resize(frame, (640, 640))  # Resize the image to a fixed size for better performance
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = detector.detect(image)
        
        if results.hand_landmarks:
            annotated_image = draw_landmarks_on_image(frame, results)
        else: 
            annotated_image = frame
        
        # Display the image with detected hand keypoints
        cv2.imshow('MediaPipe Hand Keypoint Detection', annotated_image)
        if cv2.waitKey(700) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (54, 205, 88)

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  mp_hands = mp.tasks.vision.HandLandmarksConnections
  mp_drawing = mp.tasks.vision.drawing_utils
  mp_drawing_styles = mp.tasks.vision.drawing_styles

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def test_mediapipe_on_camera():
    cap = cv2.VideoCapture(0)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow('MediaPipe Hand Keypoint Detection', annotated_image)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # # Run visualization on the dataset
    # SOURCE_DIR = 'dataset/Validation_Hand_keypoints_by_morphology/deformed/images'  # Adjust the path to your dataset images
    # validate_mediapipe_on_dataset(SOURCE_DIR)    
    
    # Initialize MediaPipe Task
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Run valuation on the dataset
    dataset = 'deformed'
    # dataset = 'plain'
    
    metrics = evaluate_dataset(
        image_dir=f"dataset/Validation_Hand_keypoints_by_morphology/{dataset}/images", 
        label_dir=f"dataset/Validation_Hand_keypoints_by_morphology/{dataset}/labels", 
        detector=detector
    )

    with open(f'mediapipe_evaluation_results_{dataset}.txt', 'w') as f:
        print("====== HAND DETECTION METRICS ======")
        print(f"Images with multiple hands detected: {metrics['multiple_hands_images']}")
        print(f"Images with no hands detected: {metrics['no_hand_images']}")
        print(f"Ground Truth Hands Count : {metrics['total_gt_hands']}")
        print(f"MediaPipe Detected Hands : {metrics['total_mp_hands']}")
        print(f"Precision                : {metrics['box_precision']:.4f}")
        print(f"Recall (Sensitivity)     : {metrics['box_recall']:.4f}")
        print(f"F1-Score                 : {metrics['box_f1_score']:.4f}")
        print(f"Mean Box IoU             : {metrics['mean_iou']:.4f}")
        print(f"Box mAP50                : {metrics['box_mAP50']:.4f}")
        print(f"Box mAP50-95             : {metrics['box_mAP50-95']:.4f}")
        print("\n====== KEYPOINT METRICS (On Correctly Detected Hands) ======")
        print(f"Mean Absolute Error (MAE): {metrics['mean_mae_pixels']:.2f} pixels")
        print(f"PCK @ 0.05 Threshold     : {metrics['pck_score']:.2f}%")
        print(f"mAP50                    : {metrics['pose_mAP50']:.4f}")
        print(f"mAP50-95                 : {metrics['pose_mAP50-95']:.4f}")
        print(f"Pose Precision           : {metrics['pose_precision']:.4f}")
        print(f"Pose Recall              : {metrics['pose_recall']:.4f}")
        print(f"Pose F1-Score            : {metrics['pose_f1_score']:.4f}")
        f.write("====== HAND DETECTION METRICS ======\n")
        f.write(f"Images with multiple hands detected: {metrics['multiple_hands_images']}\n")
        f.write(f"Images with no hands detected      : {metrics['no_hand_images']}\n")
        f.write(f"Ground Truth Hands Count           : {metrics['total_gt_hands']}\n")
        f.write(f"MediaPipe Detected Hands           : {metrics['total_mp_hands']}\n")
        f.write(f"Precision                          : {metrics['box_precision']:.4f}\n")
        f.write(f"Recall (Sensitivity)               : {metrics['box_recall']:.4f}\n")
        f.write(f"F1-Score                           : {metrics['box_f1_score']:.4f}\n")
        f.write(f"Mean Box IoU                       : {metrics['mean_iou']:.4f}\n")
        f.write(f"mAP50                              : {metrics['box_mAP50']:.4f}\n")
        f.write(f"mAP50-95                           : {metrics['box_mAP50-95']:.4f}\n" )
        f.write("\n====== KEYPOINT METRICS (On Correctly Detected Hands) ======\n")
        f.write(f"Mean Absolute Error (MAE)          : {metrics['mean_mae_pixels']:.2f} pixels\n")
        f.write(f"PCK @ 0.05 Threshold               : {metrics['pck_score']:.2f}%\n")
        f.write(f"Pose mAP50                         : {metrics['pose_mAP50']:.4f}\n")
        f.write(f"Pose mAP50-95                      : {metrics['pose_mAP50-95']:.4f}\n" )
        f.write(f"Pose Precision                     : {metrics['pose_precision']:.4f}\n")
        f.write(f"Pose Recall                        : {metrics['pose_recall']:.4f}\n")
        f.write(f"Pose F1-Score                      : {metrics['pose_f1_score']:.4f}\n")