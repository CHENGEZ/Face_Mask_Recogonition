# Modifed ResNet50 Faster-RCNN

Implementation of Faster-RCNN.
Based on https://github.com/kbardool/keras-frcnn

This branch is the training and testing for the modified ResNet50 model as feature extractor, which had additional skip connections between different identity blocks, the detailed sructure is shown in the report.

### Train

To train, open the Jupyter Notebook `frcnn_resnet50.ipynb` using Google Colab and execute the command lines in its code cells. The training outputs were reserved in the file.

The pre-trained weights can be accessed at https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
The pre-trained weights need to be put under the current directory directly.

### Test

To test, open the Jupyter Notebook `TestModifiedResnet50.ipynb` using Google Colab and execute the command lines in its code cells. The testing outputs were reserved in the file. Note that the due to my mistakes, the output FPS is not valid, the actual FPS can be calculated using 1/average elapsed time.

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


### Implementation of the modified resnet50:
details are in keras_frcnn/resnet.py
```python
def modifiedResNet50(x, trainable)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = trainable)
    x_1 = x
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable = trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable = trainable)
    x = add([x, x_1])

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable = trainable)
    x_2 = x
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = trainable)
    x_3 = x
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = trainable)
    x = add([x, x_2])
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = trainable)
    x = add([x, x_3])

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
    x_4 = x
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
    x_5 = x
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
    x_6 = x
    x = add([x, x_4])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
    x_7 = x
    x = add([x, x_5])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
    x = add([x, x_6])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)
    x = add([x, x_7])
    
    return x
```

### DataSet
The dataset can be accessed from: https://drive.google.com/drive/folders/1x-dTkgSeUCE29mpxbOvpYbs6F6XEsy1p?usp=sharing. Directly put the folder `trainImgs` and `testdataset` under current directory.
