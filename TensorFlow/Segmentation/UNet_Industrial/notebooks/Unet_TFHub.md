# Module NVIDIA/Unet/1.0

<!-- module-type:  -->
<!-- network-architecture: Unet -->
<!-- dataset: DAGM2007 -->
<!-- language:  -->
<!-- fine-tunable:  -->
<!-- format:  -->

**Module URL:** [https://tfhub.dev/<publisher>/<model-handle>/<version>](https://tfhub.dev/<publisher>/<model-handle>/<version>)

[![Open Colab notebook]](https://colab.research.google.com/github/vinhngx/DeepLearningExamples/blob/vinhn_unet_industrial_demo/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_inference_demo.ipynb#scrollTo=fW0OKDzvmTbt)

## Qualitative Information

This U-Net model is adapted from the original version of the [U-Net model](https://arxiv.org/abs/1505.04597) which is
a convolutional auto-encoder for 2D image segmentation. U-Net was first introduced by
Olaf Ronneberger, Philip Fischer, and Thomas Brox in the paper:
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

#### Model Details
This module is based on a modified version of U-Net, called `TinyUNet` which performs efficiently and with very high accuracy
on the industrial anomaly dataset [DAGM2007](https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html).
*TinyUNet*, like the original *U-Net* is composed of two parts:
- an encoding sub-network (left-side)
- a decoding sub-network (right-side).

It repeatedly applies 3 downsampling blocks composed of two 2D convolutions followed by a 2D max pooling
layer in the encoding sub-network. In the decoding sub-network, 3 upsampling blocks are composed of a upsample2D
layer followed by a 2D convolution, a concatenation operation with the residual connection and two 2D convolutions.

`TinyUNet` has been introduced to reduce the model capacity which was leading to a high degree of over-fitting on a
small dataset like DAGM2007. 

#### Suitable Use(s), Limitations, and Tradeoffs.
This model is suitable for prediction on test data similar to the [DAGM2007](https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html) training data for the application of industrial defect detection.

#### Example Use

```
import tensorflow_hub as hub
module = hub.Module("NVIDIA/Unet/Class_1", trainable=False)

# Load a test image
import numpy as np
import matplotlib.image as mpimg

img = mpimg.imread('./data/raw_images/public/Class1_def/1.png')

# Image preprocessing
img =  np.expand_dims(img, axis=2)
img =  np.expand_dims(img, axis=0)
img = (img-0.5)/0.5

output = module(img)

import tensorflow as tf

# Start a session for inference 
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    pred = sess.run(output)
      
```

See example Colab notebooks on NVIDIA Unet TF-Hub module [creation](https://colab.research.google.com/github/vinhngx/DeepLearningExamples/blob/vinhn_unet_industrial_demo/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_export.ipynb#scrollTo=HRQiqCSMAOZS) and [inference](https://colab.research.google.com/github/vinhngx/DeepLearningExamples/blob/vinhn_unet_industrial_demo/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_inference_demo.ipynb#scrollTo=Gwt7z7qdmTbW). 

#### Training Data

This UNet model was trained on the [Weakly Supervised Learning for Industrial Optical Inspection (DAGM 2007)](https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html) dataset. 

> The competition is inspired by problems from industrial image processing. In order to satisfy their customers' needs, companies have to guarantee the quality of their products, which can often be achieved only by inspection of the finished product. Automatic visual defect detection has the potential to reduce the cost of quality assurance significantly.
>
> The competitors have to design a stand-alone algorithm which is able to detect miscellaneous defects on various background textures.
>
> The particular challenge of this contest is that the algorithm must learn, without human intervention, to discern defects automatically from a weakly labeled (i.e., labels are not exact to the pixel level) training set, the exact characteristics of which are unknown at development time. During the competition, the programs have to be trained on new data without any human guidance.

**Source:** https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html

## Other Information

#### License

Copyright 2019 NVIDIA Corporation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
