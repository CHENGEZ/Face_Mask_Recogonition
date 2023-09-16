# VGG16 Faster-RCNN with degree based ROI selection

Implementation of Faster-RCNN.
Based on https://github.com/kbardool/keras-frcnn

This branch is the training and testing for the VGG16 model as feature extractor with the degree function as the region proposal selector as mentioned in the report. Note that when calculating the mAP we still use the conventional IOU formula.

### Train

To train, open the Jupyter Notebook `FRCNN_new_degree_iou.ipynb` using Google Colab and execute the command lines in its code cells. The training outputs were reserved in the file.

The pre-trained weights can be accessed at https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
The pre-trained weights need to be put under the current directory directly.

### Test

To test, open the Jupyter Notebook `TestVGG16NewIOU.ipynb` using Google Colab and execute the command lines in its code cells. The testing outputs were reserved in the file. Note that the due to my mistakes, the output FPS is not valid, the actual FPS can be calculated using 1/average elapsed time.

Due to the inconsistency of our manual labeling process, when testing, the following modification needs to be done in line 25 of `simple_parser.py`:

```python
# During Training,
(filename,class_name,x1,y1,x2,y2) = line_split
# During Testing
(class_name,x1,y1,x2,y2,filename) = line_split
```

Also in line 43 of `simple_parser.py`:

```python
# During Training
img = cv2.imread("trainImgs/"+filename)
# During Testing
img = cv2.imread("testdataset/"+filename)
```

### Degree IoU:
Here is the code of the degree IOU: (details are in keras_frcnn/data_generator.py)
```python
def iou(a, b):
	# a and b should be tuples in the format of (x1,y1,x2,y2)
	# a is prediction, b is GT

	left_x_of_intersection = max(a[0],b[0])
	right_x_of_intersection = min(a[2],b[2])
	left_y_of_intersection = max(a[1],b[1])
	right_y_of_intersection = min(a[3],b[3])
	A = np.sqrt((right_x_of_intersection - left_x_of_intersection)**2+(right_y_of_intersection - left_y_of_intersection)**2)
	if intersection(a,b) == 0:
		A = - np.inf

	center1 = ((a[0]+a[2])/2, (a[1]+a[3])/2)
	center2 = ((b[0]+b[2])/2, (b[1]+b[3])/2)

	B = np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)

	epislon = np.sqrt((b[0]-b[2])**2+(b[1]-b[3])**2)

	degree_iou = np.exp(0.1*(A-B-epislon))

	return degree_iou
```

### DataSet:
Dataset can be accessed from: https://drive.google.com/drive/folders/1x-dTkgSeUCE29mpxbOvpYbs6F6XEsy1p?usp=sharing. Directly put the folder `trainImgs` and `testdataset` under current directory.

