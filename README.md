# ImageProject
There're some interesting projects about image. 
If you're keen to explore the beauty of image processing or computer vision, I think you might like this.  
__<i>The following content may help you with read all.</i>__

## ImageDL --Minst and Pokeman Classification
[There's some python codes in the field of image, including image classification and other interesting aspects.](https://github.com/YibaYan/ImageProjects/tree/master/ImageDL)  
The [Pekoman_dataset](https://github.com/YibaYan/ImageProjects/tree/master/Pokeman_dataset) is collected by [Adrian Rosebrock(such a genius guy in image)](https://www.pyimagesearch.com/).  

### Image Generation  
[There're some python codes of image generation.](https://github.com/YibaYan/ImageProjects/tree/master/ImageGenerator) The genetors
 includes [convolutional auto encoder](https://github.com/YibaYan/ImageProjects/tree/master/ImageGenerator/cnn_autoencoder) and so on.

### Pokeman classification -- 5 kinds.  
  Something so strange happened here when i was trainning the cnn for pokeman classification. My cnn consists of three conv layers, 
2 fully connected layers and 1 softmax layer. The loss function is cross-entropy. Activated by [relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
 activation function, the
 net preduced a 'Nan' loss at the beginning(oh my goodness and so neverous i was). I search for why this happened in the internet and i got a 
solution from a blog website, replacing the relu activate function with [tanh](https://www.tensorflow.org/api_docs/python/tf/tanh)
 activate funtion to the fully connected layer. So, 
i replaced all the activate function(please forgive my ignorance...). Ha, stopping dissing me, the loss was going down.

<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/pokeman_test.png" width="75%" height="75%" alt="classification" >
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/pokeman_classify.png" width="75%" height="75%" alt="train" >
</div>  

- [preprocess_pokemon.py](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/preprocess_pokemon.py)  
The file is usd to prepare the pokeman dataset, including image resizing , label to one-hot encoding and so on.

- [build_network.py](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/build_network.py)  
CNN structure has been accomplished in this file, consisting of building net, training net and classifying net functions.

- [classify_pokeman.py](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/classify_pokeman.py)  
Some test data of pokeman have been employed to test the performance of the CNN model. Finally the highest score produced by 
the model was marked in the picture and the corresponding true and predict label was given.  

**【Awesome】**  
<p>
<code>
loss = tf.reduce_mean(-tf.reduce_sum(input_labels * tf.log(y_hat + tf.constant(1e-5, 'float32'))))
</code>
</p>
If you'd rather train the pokeman cnn model with relu activate function, and i got the truth.
The relu would produce 0 values and propagate this zero values to the softmax layer, then somthing happened!
In the softmax layer, log(0) generated by the zero values propagated before is positive infinite. Then python shows ‘Nan’.
We can handle this issue by plusing a very small value, like 0.000001, with the predict value before propagating to the log function.
Then log(0) would be replaced with log(0.000001). However, the loss curve is less smooth than the model trained with tanh activate function, 
even oscillated. The code shown above makes sense.  

### MNIST Loss and Accuracy, train by simple [CNN](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/train_net_mnist.py).  
  I employ [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) with interactive ploting mode to
show the classification loss and accuracy dynamically. 

<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/loss_accur.gif" width="70%" height="70%"> 
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/mnist_classify.png" width="75%" height="75%"> 
</div>

## KMeansImageCompression
[There's simple code in python about compressing image with scikit-learn's KMeans.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/compress_image.py)  
- One example result listed in the following. The number of cluster centroids is 3.  
<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu.png"  alt="initial image" >
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu_compress.png" alt="compressed image" >
</div>

## PCA for dimensionality reduction
[There's simple code in python about PCA for dimensionality reduction with scikit-learn's PCA.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/PCA.py)  
- We simulate 4 data clusters and process with PCA into 2 dimensions. The center result is caculated by sklearn.decompsition.PCA, and the right one is caculated through numpy.linalg.svd, choosing 2 dimensions for reduction with svd decompsition. But there's difference between two results which also confuses me.     
<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/data.png" width="75%" height="75%">  
</div>