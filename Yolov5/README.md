# YOLOv5

PyTorch implementation of YOLOv5: Real-Time Facial Mask Detection. Forked from: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## Custom setting

We saved 5 classes name into data/data.yaml.

```python
yaml_text = """train: data/train/images
val: data/train/images

nc: 5
names: ['no mask', 'improper', 'proper surgical', 'proper N95', 'proper cloth']"""
```

We modified models/custom_yolov5s.yaml to configurate the classes number. Remaining configurations are the same as default

```python
nc: 5  # number of classes
```

## Usage

### Train:

`python train.py --img 640 --batch 32--epochs 50 --data data/data.yaml --cfg models/custom_yolov5s.yaml --weights yolov5s.pt --name yolov5s_results  --cache`

### Detect: 

Detect some images: `python detect.py --source data/test --weight runs/train/yolov5s_results/weights/best.pt --name expTestImage --conf 0.4`

Detect using camera: `python detect.py --source 0 --weight runs/train/yolov5s_results/weights/best.pt --name expTestImage --conf 0.4`

## DataSet link
https://drive.google.com/drive/folders/1erKN4l5_LU3mZrY0T8nV-r01RyHVGWgY?usp=sharing
Download the dataset, copy the 3 folds(train, test, val) and paste to yolov5/data.
