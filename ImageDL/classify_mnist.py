import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from train_net_mnist import Class_Mnist_Net
import matplotlib.pyplot as plt

if __name__ == '__main__':

    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_x, test_y =  mnist_data.test.images, mnist_data.test.labels
    test_x = np.reshape(test_x, [test_x.shape[0], 28, 28, 1])

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Input_Images')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='Input_Labels')
    net = Class_Mnist_Net(10)
    y_hat, accur = net.classify_test(x, y, image_channels=1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './model_mnist/model_mnist.ckpt')
        y_hat1, accur1  = sess.run([y_hat, accur], feed_dict={x:test_x, y:test_y})
        y_predict = np.argmax(y_hat1, axis=1)
        y_real = np.argmax(test_y, axis=1)
        print(y_predict, y_real, accur1)

        test_images = test_x[0:15]
        fig = plt.figure()
        plt.axis('off')
        plt.title('Average accuracy: {}'.format(accur1), fontsize=18)
        for i, image in enumerate(test_images):
            image = np.reshape(image, [28, 28])
            axi = fig.add_subplot(3, 5, i+1)
            axi.imshow(image)
            axi.text(0, 5, 'y_predict:{}\ny_real:{}'.format(y_predict[i], y_real[i]),
                     fontdict={'size': 16, 'color': 'w'})
            plt.axis('off')
        plt.show()


