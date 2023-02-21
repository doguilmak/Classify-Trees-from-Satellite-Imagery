# -*- coding: utf-8 -*-
"""3_0_pretrained_models_vgg16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12H3vccZ7hqzHe6fjOlTGSDS9G6zj4Awi

<h1 align=center><font size = 6>Classify Trees in Satellite Imagery</font></h1>

<img src="https://eoimages.gsfc.nasa.gov/images/imagerecords/40000/40228/moorhead_tm5_2009253.jpg" width=1000 height=400 alt="esto.nasa.gov"/>

<small>Picture Source: <a href="https://eoimages.gsfc.nasa.gov/">NASA</a></small>

<br>

<h2>Objective</h2>

<p>Tree detection can be used for applications such as vegetation management, forestry, urban planning, etc. Tree identifications are very important in terms of impending famine and forest fires. In this project, we build classification model with <i>VGG-16 architecture</i> to predict; an area <b>have trees</b> or <b>not</b>.</p>

<br>

<h2>About Dataset</h2>

<p>This dataset is being used for classifying the land with class of trees or not in geospatial images.</p>

<p>Satellite: <a href='https://sentinel.esa.int/web/sentinel/missions/sentinel-2'>Sentinel - 2</a></p>

<h3>Context</h3>

<p>The content architecture is simple. Each datum has 64x64 resolution and located under <i>tree</i> and <i>notree</i> folders.
Each folder (class) has 5200 images. So the total dataset has 10400 images.</p>

<p>To download the dataset, you need to have a Kaggle account.</p>

<ul>
  <li>Dataset download link: <a href='https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery/download?datasetVersionNumber=1'>Kaggle</a></li>
  <li>Dataset website: <a href='https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery'>Kaggle Trees in Satellite Imagery</a></li>
</ul>

<br><a href=''></a>

<h2>Table of Contents</h2>

<div class="alert alert-block alert-info" style="margin-top: 20px">
  <ul>
    <li><a href="https://#unzip_data">Unzip Data</a></li>
    <li><a href="https://#libraries">Import Libraries and Packages </a></li>
    <li><a href="https://#data_preparation">Dataset Preparation</a></li>
    <li><a href="https://#compile_fit">Compile and Fit VGG-16 Model</a></li>
    <li><a href="https://#train_model">Train the VGG16 Model</a></li>
  </ul>

  <br>

  <p>Estimated Time Needed: <strong>60 min</strong></p>

</div>

<br>

<h2 align=center id="unzip_data">Unzip Data</h2>

<p>After downloading the dataset, we can unzip the file.</p>
"""

!unzip -q archive.zip

"""<br>

<h2 align=center id="libraries">Import Libraries and Packages</h2>

<p>Before we proceed, let's import the libraries and packages that we will need to complete the rest of this lab.</p>
"""

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings("ignore")

num_classes = 1  # Our classification is binary. That's why we wrote as 1.

image_resize = (224, 224) # Our images are 64 x 64 but VGG16 was originally trained on 224 × 224

batch_size_training = 100
batch_size_validation = 100
directory = "/content/Trees in Satellite Imagery"

"""<br>

<h2 align=center id="data_preparation">Dataset Preparation</h2>

<p>We are going to separate our dataset as <i>7800</i> files for training and <i>2600</i> files for validation.</p>
"""

train_dataset = tf.keras.utils.image_dataset_from_directory(
  directory,
  validation_split=0.25,
  subset="training",
  seed=123,
  image_size=image_resize,
  batch_size=batch_size_training)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  directory,
  validation_split=0.25,
  subset="validation",
  seed=123,
  image_size=image_resize,
  batch_size=batch_size_validation)

valdation_batches = tf.data.experimental.cardinality(validation_dataset)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))

test_batches = valdation_batches // 5
test_dataset = validation_dataset.take(test_batches)
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

"""<br>

<h2 align=center id="compile_fit">Compile and Fit VGG-16 Model</h2>

<p><i>VGG-16</i> is a <i>convolution neural net (CNN)</i> architecture which was used to win <i>ILSVR (Imagenet)</i> competition in 2014. It is considered to be one of the excellent <i>vision model architecture</i> till date. In this section, we will start building our model. We will use the <i>Sequential model class</i> from <i>Keras</i>.</p></p>
"""

model_vgg16 = Sequential()

"""<p>Then, we will define our output layer as a <b>Dense</b> layer.</p>"""

model_vgg16.add(VGG16(include_top=False, pooling="avg", weights="imagenet"))
model_vgg16.add(Dense(num_classes, activation="softmax"))

model_vgg16.layers[0].trainable = False

"""<p>You can access the model's layers using the <i>layers</i> attribute of our model object.</p>"""

model_vgg16.layers

"""<p>And now using the <i>summary</i> attribute of the model.</p>"""

model_vgg16.summary()

"""<p>Next, we compile our model using the <b>adam</b> optimizer.</p>"""

model_vgg16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""<br>

<h2 align=center id="train_model">Train the VGG16 Model</h2>

<p>We will need to define how many steps compose an epoch. Typically, that is the number of images divided by the batch size. Therefore, we define our steps per epoch as follows:</p>
"""

num_epochs = 2
steps_per_epoch_training = len(train_dataset)
steps_per_epoch_validation = len(validation_dataset)

import time
start = time.time()

history = model_vgg16.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=num_epochs
)

end = time.time()

elapsed_time = end - start

print(f"Elapsed Time:{elapsed_time}s")

model_vgg16.save('classifier_vgg16_model.h5')

# Function taken from https://www.kaggle.com/code/lucasarielsaavedra/satellite-images-classification-94-accuracy
import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(acc)), acc, label='Training Accuracy')
    plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss)), loss, label='Training Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

num_parameters_time = {}
num_parameters_accuracy = {}
num_parameters = {}

num_parameters["Base"] = 14715201
num_parameters_time["Base"] = elapsed_time
num_parameters_accuracy["Base"] = history.history["val_accuracy"][-1]

plot_history(history)

"""<h1>Contact Me</h1>
<p>If you have something to say to me please contact me:</p>

<ul>
  <li>Twitter: <a href="https://twitter.com/Doguilmak">Doguilmak</a></li>
  <li>Mail address: doguilmak@gmail.com</li>
</ul>
"""

from datetime import datetime
print(f"Changes have been made to the project on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")