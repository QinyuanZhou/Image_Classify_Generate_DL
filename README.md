# ImageProject
There're some interesting projects about image. 
If you're keen to explore the beauty of image processing or computer vision, I think you might like this.  
__<i>The following content may help you with read all.</i>__

## 1. Image Application in Deep Learning
[There's some python codes in the field of image, including image classification and other interesting aspects.](https://github.com/YibaYan/ImageProjects/tree/master/ImageDL)  
The [Pekoman_dataset](https://github.com/YibaYan/ImageProjects/tree/master/Pokeman_dataset) is collected by [Adrian Rosebrock(such a genius guy in image)](https://www.pyimagesearch.com/).  

### 1.1. Image Generation  
[There're some python codes of image generation.](https://github.com/YibaYan/ImageProjects/tree/master/ImageGenerator) The genetors
 includes [convolutional auto encoder](https://github.com/YibaYan/ImageProjects/tree/master/ImageGenerator/cnn_autoencoder) and so on.
#### 1.1.1 Convolutional Auto Encoder
 - __<i>Must construct net model firstly before initialize your tensorflow's variables.</i>__
- __<i>Don't make a mistake that initialize your net model over every batch, be careful.</i>__
- __<i>Warning! Normalizing your image data, you like doing it or not anyway.</i>__
- __<i>Remember! Specify the GPU device to train/retrain the model, otherwise may stucked into GPU Memory Error.</i>__

<p align="center"><strong>Good Reconstruction  </strong></p>
<div align="center">
<img src="https://github.com/YibaYan/Image_Classify_Generate_DL/blob/master/ImageGenerator/cnn_autoencoder/Image_Reconstruct_good.png" width="75%" height="75%" >
</div>

<p align="center"><strong>Bad Reconstruction  </strong></p>
<div align="center">
<img src="https://github.com/YibaYan/Image_Classify_Generate_DL/blob/master/ImageGenerator/cnn_autoencoder/Image_Reconstruct_bad.png" width="75%" height="75%">
</div> 

	import tensorflow as tf  
	import numpy as np  

	BATCH_SIZE = 10
	BATCH = 9
	EPOCH = 2
	BUFFER_SIZE = 10

	x = np.arange(1, 91)

	dataset = tf.data.Dataset.from_tensor_slices(x)
	dataset = dataset.shuffle(BUFFER_SIZE) 
	# shuffle() must be befor batch()
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.repeat(EPOCH)

	iter = dataset.make_one_shot_iterator()
	next_batch = iter.get_next()
	with tf.Session() as sess:
		for epoch in range(EPOCH):
			for batch in range(BATCH):
				x_batch = sess.run(next_batch)
				print('epoch/batch_x_batch: {}/{}  {}'.format(epoch+1, batch+1, x_batch))

Which yields:
<pre><code>
If Shuffled:  
epoch/batch_x_batch: 1/1  [ 9  6 11  2 10  5 15 17  3  8]
epoch/batch_x_batch: 1/2  [20  1 21  7 16  4 24 22 18 27]
epoch/batch_x_batch: 1/3  [23 28 30 29 32 13 14 26 34 39]
epoch/batch_x_batch: 1/4  [31 33 37 36 40 12 19 35 42 38]
epoch/batch_x_batch: 1/5  [49 50 25 48 53 52 43 51 44 56]
epoch/batch_x_batch: 1/6  [54 47 55 62 58 45 61 60 46 65]
epoch/batch_x_batch: 1/7  [64 41 66 59 57 74 67 69 78 79]
epoch/batch_x_batch: 1/8  [75 72 71 70 83 63 81 82 76 85]
epoch/batch_x_batch: 1/9  [86 90 89 77 80 73 84 88 68 87]
epoch/batch_x_batch: 2/1  [ 7  8  6 13  1 12 14 17  5 18]
epoch/batch_x_batch: 2/2  [ 4 21 11 23 15 16 22  3 19 29]
epoch/batch_x_batch: 2/3  [10 30 20 31 27 35 34 28 32  2]
epoch/batch_x_batch: 2/4  [38 36 42 43 25 45 39 40 26 37]
epoch/batch_x_batch: 2/5  [24 41 48 47 49 52 44 54 51 33]
epoch/batch_x_batch: 2/6  [57  9 56 61 64 65 62 59 67 55]
epoch/batch_x_batch: 2/7  [68 66 46 53 72 69 74 58 77 71]
epoch/batch_x_batch: 2/8  [73 80 75 82 76 85 84 79 87 50]    
If Not Shuffled:  
epoch/batch_x_batch: 1/1  [ 1  2  3  4  5  6  7  8  9 10]
epoch/batch_x_batch: 1/2  [11 12 13 14 15 16 17 18 19 20]
epoch/batch_x_batch: 1/3  [21 22 23 24 25 26 27 28 29 30]
epoch/batch_x_batch: 1/4  [31 32 33 34 35 36 37 38 39 40]
epoch/batch_x_batch: 1/5  [41 42 43 44 45 46 47 48 49 50]
epoch/batch_x_batch: 1/6  [51 52 53 54 55 56 57 58 59 60]
epoch/batch_x_batch: 1/7  [61 62 63 64 65 66 67 68 69 70]
epoch/batch_x_batch: 1/8  [71 72 73 74 75 76 77 78 79 80]
epoch/batch_x_batch: 1/9  [81 82 83 84 85 86 87 88 89 90]
epoch/batch_x_batch: 2/1  [ 1  2  3  4  5  6  7  8  9 10]
epoch/batch_x_batch: 2/2  [11 12 13 14 15 16 17 18 19 20]
epoch/batch_x_batch: 2/3  [21 22 23 24 25 26 27 28 29 30]
epoch/batch_x_batch: 2/4  [31 32 33 34 35 36 37 38 39 40]
epoch/batch_x_batch: 2/5  [41 42 43 44 45 46 47 48 49 50]
epoch/batch_x_batch: 2/6  [51 52 53 54 55 56 57 58 59 60]
epoch/batch_x_batch: 2/7  [61 62 63 64 65 66 67 68 69 70]
epoch/batch_x_batch: 2/8  [71 72 73 74 75 76 77 78 79 80]
epoch/batch_x_batch: 2/9  [81 82 83 84 85 86 87 88 89 90] 
</code></pre>
 
### 1.2. Pokeman classification -- 5 kinds.  
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

### 1.3. MNIST Loss and Accuracy, train by simple [CNN](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/train_net_mnist.py).  
  I employ [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) with interactive ploting mode to
show the classification loss and accuracy dynamically. 

<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/loss_accur.gif" width="70%" height="70%"> 
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/mnist_classify.png" width="75%" height="75%"> 
</div>

## 2. KMeansImageCompression
[There's simple code in python about compressing image with scikit-learn's KMeans.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/compress_image.py)  
- One example result listed in the following. The number of cluster centroids is 3.  
<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu.png"  alt="initial image" >
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu_compress.png" alt="compressed image" >
</div>

## 3. PCA for dimensionality reduction
[There's simple code in python about PCA for dimensionality reduction with scikit-learn's PCA.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/PCA.py)  
- We simulate 4 data clusters and process with PCA into 2 dimensions. The center result is caculated by sklearn.decompsition.PCA, and the right one is caculated through numpy.linalg.svd, choosing 2 dimensions for reduction with svd decompsition. But there's difference between two results which also confuses me.     
<div align="center">
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/data.png" width="75%" height="75%">  
</div>