# ImageProject
There're some interesting projects about image. 
If you're keen to explore the beauty of image processing or computer vision, I think you might like this.  
__<i>The following content may help you with read all.</i>__

## Image in Deep Learning --ImageDL
[There's some python codes in the field of image, including image classification and other interesting aspects.](https://github.com/YibaYan/ImageProjects/tree/master/ImageDL)  
The [Pekoman_dataset](https://github.com/YibaYan/ImageProjects/tree/master/Pokeman_dataset) is collected by [Adrian Rosebrock(such a genius guy in image)](https://www.pyimagesearch.com/).  
- MNIST Loss and Accuracy, train by simple [CNN](https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/train_net_mnist.py).
<img src="https://github.com/YibaYan/ImageProjects/blob/master/ImageDL/loss_accur.gif" width="65%" height="65%"> 

## KMeansImageCompression
[There's simple code in python about compressing image with scikit-learn's KMeans.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/compress_image.py)  
- One example result listed in the following. The number of cluster centroids is 3.  
![initial image](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu.png)
![compressed image](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/pikachu_compress.png)  

## PCA for dimensionality reduction
[There's simple code in python about PCA for dimensionality reduction with scikit-learn's PCA.](https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/PCA.py)  
- We simulate 4 data clusters and process with PCA into 2 dimensions. The center result is caculated by sklearn.decompsition.PCA, and the right one is caculated through numpy.linalg.svd, choosing 2 dimensions for reduction with svd decompsition. But there's difference between two results which also confuses me.     
<img src="https://github.com/YibaYan/ImageProjects/blob/master/KMeansImageCompression/data.png" width="65%" height="65%">  
