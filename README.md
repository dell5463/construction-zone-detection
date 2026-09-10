# Real-Time Construction Site Object Detection

YOLO-based construction-zone object detection for autonomous-driving imagery. This project trains and evaluates a lightweight YOLOv11s detector that identifies common roadway construction objects from dashcam-style images.

The README is based on the accompanying research paper, "Real-Time Construction Site Object Detection for Autonomous Vehicles," and on the files in this repository.

## Project Overview

Construction zones are difficult for autonomous vehicles because object layouts, colors, occlusions, and object scales vary heavily from scene to scene. This project explores whether a camera-only, real-time YOLO detector can recognize construction-zone elements without relying on LiDAR, radar, or additional sensors.

The model is trained to detect six classes:

| ID | Class |
| --- | --- |
| 0 | Cone |
| 1 | Construction_Cylinder |
| 2 | Signs |
| 3 | Fences |
| 4 | Barriers |
| 5 | Vehicles |

## Dataset

The research dataset contains 1,000 forward-facing dashcam images manually selected from the nuScenes dataset. The images were annotated in CVAT and exported in YOLO format.

The dataset uses an 80/20 split:

| Split | Images | Labels |
| --- | ---: | ---: |
| Train | 800 | 800 |
| Validation | 200 | 200 |

The final dataset contains 6,433 labeled object instances:

| Class | Instances |
| --- | ---: |
| Cone | 3,519 |
| Construction Cylinder | 268 |
| Signs | 593 |
| Fences | 135 |
| Barriers | 1,629 |
| Vehicles | 289 |

The original annotation process briefly included a human/construction-worker class, but it was removed to avoid confusion with general pedestrians.

## Repository Layout

```text
.
|-- ConstructionSiteObjectDetection/
|   |-- data.yaml                  # YOLO dataset config
|   |-- train.py                   # YOLOv11s training script
|   |-- images/                    # train/val images
|   |-- labels/                    # train/val YOLO labels
|   |-- runs/                      # training outputs from runs started in this folder
|   |-- yolo11s.pt                 # YOLOv11s base weights
|   |-- yolo11n.pt                 # YOLOv11n base weights
|   `-- yolov8n.pt                 # YOLOv8n base weights
|-- runs/detect/train3/weights/
|   `-- best.pt                    # trained detector used by inference scripts
|-- randimg/                       # sample images for batch inference
|-- inference_output/              # saved prediction images
|-- drivingvid.mp4                 # sample video input
|-- test.py                        # batch image inference
`-- vid.py                         # video inference preview
```

## Setup

Create a Python environment, then install the runtime dependencies:

```bash
pip install ultralytics opencv-python
```

If you plan to train on a GPU, install a PyTorch build that matches your CUDA version before running training.

## Training

The main training script is:

```bash
cd ConstructionSiteObjectDetection
python train.py
```

The script trains `yolo11s.pt` on `data.yaml` for 60 epochs with SGD, batch size 4, mosaic augmentation, mixup, and copy-paste augmentation.

Current script settings:

```python
epochs=60
imgsz=1200
batch=4
optimizer="SGD"
mosaic=1.0
mixup=0.15
copy_paste=0.2
```

The paper and saved `train3` run artifacts report an image size of `1024`. The current `train.py` uses `1200`, which may improve small-object visibility at the cost of higher GPU memory use.

Note: `data.yaml` currently contains absolute Windows paths. If the project is moved, update the `train` and `val` paths in `ConstructionSiteObjectDetection/data.yaml`.

## Batch Image Inference

Run prediction on every image in `randimg/`:

```bash
python test.py
```

Predicted images are saved to:

```text
inference_output/
```

`test.py` currently loads:

```text
runs/detect/train3/weights/best.pt
```

Update `MODEL_PATH` in `test.py` if you want to use a different training run.

## Video Inference

Run real-time preview inference on `drivingvid.mp4`:

```bash
python vid.py
```

Press `q` to close the OpenCV preview window.

`vid.py` also uses the trained `best.pt` checkpoint from `runs/detect/train3/weights/`.

## Reported Results

The paper reports the following validation performance:

| Class | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cone | 0.862 | 0.839 | 0.850 | 0.895 | 0.505 |
| Construction Cylinder | 0.827 | 0.797 | 0.811 | 0.862 | 0.508 |
| Signs | 0.861 | 0.866 | 0.863 | 0.881 | 0.595 |
| Fences | 0.687 | 0.564 | 0.620 | 0.611 | 0.336 |
| Barriers | 0.664 | 0.651 | 0.657 | 0.703 | 0.440 |
| Vehicles | 0.723 | 0.673 | 0.697 | 0.722 | 0.383 |
| Overall | 0.771 | 0.731 | 0.751 | 0.779 | 0.461 |

The best overall F1 score was approximately `0.75` at a confidence threshold of `0.343`.

The paper also reports real-time inference performance of about `7 ms` total latency per frame, including preprocessing, model inference, and postprocessing. This corresponds to more than 140 FPS on comparable hardware.

## Findings From the Paper

The model performs best on cones, construction cylinders, and signs because they are visually distinctive and appear more consistently in the dataset.

Fences and vehicles perform worse because they have fewer examples and more visual variation. Barriers are common, but their bounding-box annotations were sometimes split into multiple smaller boxes, which likely caused over-detection and reduced precision.

## Future Work

Potential improvements include:

- Expanding the dataset with more construction scenes, regions, weather conditions, and lighting conditions.
- Adding more fence, vehicle, and construction-worker examples.
- Replacing segmented barrier bounding boxes with segmentation masks.
- Training an Ultralytics YOLO segmentation model for objects where box annotations include too much background.
- Making model paths and dataset paths configurable through command-line arguments.
- Adding a `requirements.txt` file for reproducible setup.

## Acknowledgments

The research dataset was derived from nuScenes dashcam imagery and annotated using CVAT. The detection model uses the Ultralytics YOLO framework.
