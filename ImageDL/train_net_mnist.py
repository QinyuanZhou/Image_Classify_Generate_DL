import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

class Class_Pekoman_Net(object):
    def __init__(self, output_classes = 5):
        self.output_classes = output_classes
        self.learning_rate = 0.0001
        self.batch_size = 256
        self.drop = 0.5

    def build_net(self, input_images, image_channels=3):
        with tf.name_scope('Conv_Layer'):
            with tf.name_scope('conv1'):
                conv1 = tf.nn.conv2d(input_images, filter=tf.Variable(
                    tf.truncated_normal([5, 5, image_channels, 32], stddev=0.1, dtype=tf.float32), name='weight1'),
                                     strides=[1, 1, 1, 1], padding='SAME')
                relu1 = tf.nn.relu(conv1 + tf.Variable(tf.constant(0.1, shape=[32]), name='bias1'))
                pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')
            with tf.name_scope('conv2'):
                conv2 = tf.nn.conv2d(pool1, filter=tf.Variable(
                    tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32), name='weight2'),
                                     strides=[1, 1, 1, 1], padding='SAME')
                relu2 = tf.nn.relu(conv2 + tf.Variable(tf.constant(0.1, shape=[64]), name='bias2'))
                pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')
                image_shape = input_images.shape.as_list()
                flat_size = (image_shape[1]) * (image_shape[2]) * 4
                print(image_shape)
            with tf.name_scope('dense1'):
                pool2_flat = tf.reshape(pool2, [-1, flat_size])
                fc1 = tf.matmul(pool2_flat, tf.Variable(
                    tf.truncated_normal([flat_size, 1024], stddev=0.1, dtype=tf.float32, name='weight4')))
                fc1_relu = tf.nn.relu(fc1 + tf.Variable(tf.constant(0.1, shape=[1024], name='bias4')))
                fc1_relu = tf.nn.dropout(fc1_relu, self.drop )
            with tf.name_scope('dense2'):
                fc3 = tf.matmul(fc1_relu, tf.Variable(
                    tf.truncated_normal([1024, self.output_classes], stddev=0.1, dtype=tf.float32), name='weight6'))
                model = tf.nn.softmax(fc3 + tf.Variable(tf.constant(0.1, shape=[self.output_classes]), name='bias6'))
        return model

    def train_model(self, input_images, input_labels, image_channels):
        y_hat = self.build_net(input_images, image_channels)
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(-tf.reduce_sum(input_labels * tf.log(y_hat)))
            tf.summary.scalar('loss', loss)  # tensorflow >= 0.12
        correct_prediction_conv = tf.equal(tf.argmax(y_hat, 1), tf.argmax(input_labels, 1))
        accuracy_conv = tf.reduce_mean(tf.cast(correct_prediction_conv, dtype=tf.float32))
        with tf.name_scope('Train'):
            train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return y_hat, loss, accuracy_conv, train

def next_batch(train_data, train_labels, batch_size):
    nums = np.shape(train_data)[0]
    index = np.arange(0, nums)
    np.random.shuffle(index)
    batch_index = index[0:batch_size]
    return train_data[batch_index], train_labels[batch_index]

if __name__ == '__main__':
    figure = plt.figure()
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.set_title('Accuracy')
    ax2 = figure.add_subplot(1, 2, 2)
    ax2.set_title('Loss')
    plt.ion() # open interactive mode

    input_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Input_Images')
    input_labels = tf.placeholder(tf.float32, shape=[None, 10], name='Input_Labels')
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_x, train_y, test_x, test_y= mnist_data.train.images, mnist_data.train.labels, mnist_data.test.images, mnist_data.test.labels
    train_x = np.reshape(train_x, [train_x.shape[0], 28, 28, 1])

    net = Class_Pekoman_Net(10)
    y_hat, loss, accur, train = net.train_model(input_images, input_labels, image_channels=1)
    warn = net.build_net(input_images, image_channels=1)
    batch_size = net.batch_size

    init_var = tf.global_variables_initializer()
    saver = tf.train.Saver() # save visualize graph
    hist_accur, hist_loss = [], []
    with tf.Session() as sess:
        sess.run(init_var)
        for i in range(500):
            print('[INFO] itera {} training, wait...'.format(i))
            X, Y = next_batch(train_x, train_y, batch_size)
            # pool3 = sess.run(warn, feed_dict={input_images: X, input_labels: Y})
            # print(pool3.shape)
            accur1, train1, loss1 = sess.run([accur, train, loss], feed_dict={input_images:X, input_labels:Y})
            hist_accur.append(accur1)
            ax1.plot(hist_accur)
            hist_loss.append(loss1)
            ax2.plot(hist_loss)
            plt.pause(0.1) # pause 0.1 seconds
            if i % 5 == 0:
                print("[INFO] step {}, accuracy {}, loss {}".format(i, accur1, loss1))
        saver.save(sess, "./model/model_mnist.ckpt")
        plt.ioff() # close interactive mode
        plt.show()