**<h1 align=center><font size = 5>Classify Trees in Satellite Imagery </font></h1>**

<img src="https://eoimages.gsfc.nasa.gov/images/imagerecords/40000/40228/moorhead_tm5_2009253.jpg" width=1000 height=400 alt="esto.nasa.gov"/>

<small>Fig.1 -  esto.nasa.gov</small>

<br>

## Project Statement

In this project, satellite photographs taken by the *Sentinel-2* satellite were classified with pre-trained *ResNet-50* and *VGG16* models. You can find the [dataset on Kaggle](https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery). Tree detection can be used for applications such as vegetation management, forestry, urban planning, etc. Tree identifications are very important in terms of impending famine and forest fires.

<br>

## Keywords

 - Satellite Imagery
 - Computer Science
 - ResNet-50
 - VGG-16
 - Classification
 - Trees

<br>

## About Dataset

<img src="https://www.umvoto.com/wp-content/uploads/2021/06/Sentinel-2-infographic.jpg" width=1000 height=500 alt="esto.nasa.gov"/>

<small>Fig.2 -  sentinel.esa.int</small>

<br>

**The Copernicus Sentinel-2** mission comprises a constellation of two polar-orbiting satellites placed in the same sun-synchronous orbit, phased at $180°$ to each other. It aims at monitoring variability in land surface conditions, and its wide swath width (290 km) and high revisit time (10 days at the equator with one satellite, and 5 days with 2 satellites under cloud-free conditions which results in 2-3 days at mid-latitudes) will support monitoring of Earth's surface changes.

### Context

This dataset is being used for classifying the land with class of trees or not in geospatial images.

### Content

The content architecture is simple. Each datum has *64x64* resolution and located under *tree* and *notree* folders.  
Each folder (class) has *5200* images. So the total dataset has *10.400* images.

### Cite

M.Ç.Aksoy (2022). Trees in Satellite Imagery

And you can also cite the source of this data *EUROSAT:  
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217-2226.*

<br>

## About Models

### ResNet-50

<img src="https://miro.medium.com/max/1400/0*9LqUp7XyEx1QNc6A.png" width=1000 height=300 alt="miro.medium.com"/>

<small>Fig.3 -  medium.com</small>

*ResNet-50* is a convolutional neural network that is *50* layers deep. The network can take the input image having height, width as multiples of *32* and *3* as channel width. For the sake of explanation, we will consider the input size as *224x224x3*. The *ResNet-50* model consists of 5 stages each with a convolution and Identity block. Each convolution block has 3 convolution layers and each identity block also has *3* convolution layers. The *ResNet-50* has over *23 million* trainable parameters.

<br>

### VGG-16

<img src="https://miro.medium.com/max/1400/1*NNifzsJ7tD2kAfBXt3AzEg.png" width=1000 height=300 alt="medium.com"/>

<small>Fig.4 -  medium.com</small>

*VGG16* is a convolution neural net (CNN) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. In this section, we will start building our model. We will use the Sequential model class from Keras.

<br>

## Notebooks

On below, there are informations about the notebooks created respectively.

 1. In this sections, visualizations of the data set were made. <br> Notebooks: 
		 
	1. [1_0_data_preparation.ipynb](https://github.com/doguilmak/Classify-Trees-in-Satellite-Imagery/blob/main/1_0_data_preparation.ipynb)

	2.  [1_0_load_and_display_data.ipynb](https://github.com/doguilmak/Classify-Trees-in-Satellite-Imagery/blob/main/1_0_load_and_display_data.ipynb)
		 
 2. In this section, I build *ResNet-15* model for classification. <br> Notebook: 
 
	1. [2_0_pretrained_models_resnet50.ipynb](https://github.com/doguilmak/Classify-Trees-in-Satellite-Imagery/blob/main/2_0_pretrained_models_resnet50.ipynb) - Model has 98% accuracy.
 3. In this section, I build *VGG16* model for classification. <br> Notebook:
 
 	1. [3_0_comparing_models_vgg16.ipynb](https://github.com/doguilmak/Classify-Trees-in-Satellite-Imagery/blob/main/3_0_comparing_models_vgg16.ipynb) - Model has 50% accuracy.

 4.  In this section, I build classification model with Class Activation Map (CAM).
	  1. [4_0_model_with_cam.ipynb](https://github.com/doguilmak/Classify-Trees-in-Satellite-Imagery/blob/main/4_0_model_with_cam.ipynb) - Model has 93% accuracy.

<br>

## References

- [ESA](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [Kaggle](https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery/code)
- [Towards Data Science](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16%20is%20a%20convolution%20neural,vision%20model%20architecture%20till%20date.)
- [MathWorks](https://www.mathworks.com/help/deeplearning/ref/resnet50.html#:~:text=ResNet-50%20is%20a%20convolutional,,%20pencil,%20and%20many%20animals.)
- [NASA](https://earthobservatory.nasa.gov/)

<br>

## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak)  
 - Mail address: doguilmak@gmail.com
