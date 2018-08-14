# ImageProject
There're some interesting projects about image. 
If you're keen to explore the beauty of image processing or computer vision, I think you might like this.  
__<i>The following content may help you with read all.</i>__

## Image in Deep Learning --ImageDL
[There's some python codes in the field of image, including image classification and other interesting aspects.](https://github.com/YibaYan/ImageProjects/tree/master/ImageDL)  
The [Pekoman_dataset](https://github.com/YibaYan/ImageProjects/tree/master/Pokeman_dataset) is collected by [Adrian Rosebrock(such a genius guy in image)](https://www.pyimagesearch.com/).  

### Pokeman classification -- 5 kinds.  
  Something so strange happened here when i was trainning the cnn for pokeman classification. My cnn consists of three conv layers, 
2 fully connected layers and 1 softmax layer. The loss function is cross-entropy. Activated by [relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
 activation function, the
 net preduced a 'Nan' loss at the beginning(oh my goodness and so neverous i was). I search for why this happened in the internet and i got a 
solution from a blog website, replacing the relu activate function with [tanh](https://www.tensorflow.org/api_docs/python/tf/tanh)
 activate funtion to the fully connected layer. So, 
i replaced all the activate function(please forgive my ignorance...). Ha, stopping dissing me, the loss was going down.

<div align="left">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/pokeman_classify.png" width="65%" height="65%" alt="train" >
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/pokeman_test.png" width="65%" height="65%" alt="classification" >
</div>  
* [preprocess_pokemon.py](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/preprocess_pokemon.py)  
The file is usd to prepare the pokeman dataset, including image resizing , label to one-hot encoding and so on.
* [build_network.py](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/build_network.py)  
CNN structure has been accomplished in this file, consisting of building net, training net and classifying net functions.
* [classify_pokeman.py](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/classify_pokeman.py)  
Some test data of pokeman have been employed to test the performance of the CNN model. Finally the highest score produced by 
the model was marked in the picture and the corresponding true and predict label was given. 

### MNIST Loss and Accuracy, train by simple [CNN](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/train_net_mnist.py).  
  I employ [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) with interactive ploting mode to
show the classification loss and accuracy dynamically. 
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/loss_accur.gif" width="65%" height="65%"> 
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/mnist_classify.png" width="70%" height="70%"> 

## KMeansImageCompression
[There's simple code in python about compressing image with scikit-learn's KMeans.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/compress_image.py)  
- One example result listed in the following. The number of cluster centroids is 3.  
<div align="left">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu.png" alt="initial image" >
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu_compress.png" alt="compressed image" >
</div>

## PCA for dimensionality reduction
[There's simple code in python about PCA for dimensionality reduction with scikit-learn's PCA.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/PCA.py)  
- We simulate 4 data clusters and process with PCA into 2 dimensions. The center result is caculated by sklearn.decompsition.PCA, and the right one is caculated through numpy.linalg.svd, choosing 2 dimensions for reduction with svd decompsition. But there's difference between two results which also confuses me.     
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/data.png" width="65%" height="65%">  
