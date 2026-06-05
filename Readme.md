# Hand and Hand Keypoints Detection Performance Evaluation

Benchmark and evaluate hand keypoint detection models using consistent metrics, datasets, and reporting.

## Overview

This project provides a lightweight workflow to:

- Run inference for hand keypoint detectors
- Compare predictions against ground truth annotations
- Compute standard keypoint metrics
- Export clear, reproducible evaluation reports

## Features

- Unified evaluation pipeline for multiple models
- Common keypoint metrics (e.g., PCK, OKS-style matching, precision/recall)
- Per-keypoint and per-image analysis
- Aggregate summaries and optional visual diagnostics
- Configurable experiment settings

## Project Structure

```text
.
├── data/                    # Datasets, annotations (download Hand Keypoints database from Kaggle)
├── weight/                  # Fine-tuned weights for hand and hand keypoint detection
├── statistical_analysis.py  # Statistical test to compare algorithms' performance
├── train_YOLO_algorithms.py # Train YOLO-family, and RT-DETR algorithms using Ultralytics
├── test_YOLO_algorithms.py  # Test YOLO-family, and RT-DETR algorithms using Ultralytics
├── test_mediapipe.py        # Test MediaPipe framework with the testing datasets
├── Filenames_of_irregularly # Filenames of the irregularly shaped hands from Kaggle val test set
├── Filenames_of_regularly   # Filenames of the regularly shaped hands from Kaggle val test set
└── Readme.md
```

## Requirements

- Python 3.9+
- pip or conda
- Common scientific stack (NumPy, OpenCV, pandas, matplotlib)
- Deep learning framework used by your models (PyTorch and/or TensorFlow)
- Ultralytics
- Mediapipe



## Data Preparation

1. Place images in `data/images/`.
2. Place annotation files in `data/annotations/`.
3. Ensure annotation format matches the expected schema (image id, keypoints, visibility, bbox if required).
4. Define train/val/test splits in `data/splits/` (or via config).

Example:

```yaml
experiment_name: baseline_eval
dataset:
	images: data/images
	annotations: data/annotations/val.json
model:
	name: hand_pose_model
	checkpoint: models/checkpoints/model.pt
evaluation:
	pck_threshold: 0.05
	conf_threshold: 0.2
output_dir: outputs/baseline_eval
```

## Run Evaluation

``` test_YOLO_algorithms.py ```

## Output

Outputs include:

- `metrics_summary.txt`
- Per-keypoint performance tables
- Error distribution plots
- Optional visual overlays for qualitative inspection
- Statistical analysis output

## Metrics

- **mAP50**
- **mAP50-95**
- **Precision**
- **Recall**

## Contributing

Contributions are welcome through focused pull requests:

1. Create a feature branch
2. Add or update tests for changes
3. Submit PR with a clear summary of impact on metrics

## License

`Licence CC BY 4.0`

## Acknowledgments

This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101034371
