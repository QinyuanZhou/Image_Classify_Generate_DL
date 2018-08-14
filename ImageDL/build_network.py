import tensorflow as tf

class Class_Pekoman_Net(object):
    def __init__(self, output_classes = 5):
        self.output_classes = output_classes
        self.learning_rate = 0.001
        self.batch_size = 512
        self.drop = 0.5

    def build_net(self, input_images, image_channels=3):
        with tf.name_scope('Conv_Layer'):
            with tf.name_scope('conv1'):
                conv1 = tf.nn.conv2d(input_images, filter=tf.Variable(
                    tf.truncated_normal([5, 5, image_channels, 32], stddev=0.1, dtype=tf.float32), name='weight1'),
                                     strides=[1, 1, 1, 1], padding='SAME')
                relu1 = tf.nn.tanh(conv1 + tf.Variable(tf.constant(0.1, shape=[32]), name='bias1'))
                pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')
            with tf.name_scope('conv2'):
                conv2 = tf.nn.conv2d(pool1, filter=tf.Variable(
                    tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32), name='weight2'),
                                     strides=[1, 1, 1, 1], padding='SAME')
                relu2 = tf.nn.tanh(conv2 + tf.Variable(tf.constant(0.1, shape=[64]), name='bias2'))
                pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')
            with tf.name_scope('conv3'):
                conv3 = tf.nn.conv2d(pool2, filter=tf.Variable(
                    tf.truncated_normal([5, 5, 64, 128], stddev=0.1, dtype=tf.float32), name='weight3'),
                                     strides=[1, 1, 1, 1], padding='SAME')
                relu3 = tf.nn.tanh(conv3 + tf.Variable(tf.constant(0.1, shape=[128]), name='bias3'))
                pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')
                image_shape = input_images.shape.as_list()
                flat_size = (image_shape[1]) * (image_shape[2]) * 2
                # print(image_shape)
            with tf.name_scope('dense1'):
                pool3_flat = tf.reshape(pool3, [-1, flat_size])
                fc1 = tf.matmul(pool3_flat, tf.Variable(
                    tf.truncated_normal([flat_size, 1024], stddev=0.1, dtype=tf.float32, name='weight4')))
                fc1_relu = tf.nn.tanh(fc1 + tf.Variable(tf.constant(0.1, shape=[1024], name='bias4')))
                fc1_relu = tf.nn.dropout(fc1_relu, self.drop)
            with tf.name_scope('dense2'):
                fc2 = tf.matmul(fc1_relu, tf.Variable(
                    tf.truncated_normal([1024, 256], stddev=0.1, dtype=tf.float32), name='weight5'))
                fc2_relu = tf.nn.tanh(fc2 + tf.Variable(tf.constant(0.1, shape=[256]), name='bias5'))
                fc2_relu = tf.nn.dropout(fc2_relu, self.drop)
            with tf.name_scope('dense3'):
                fc3 = tf.matmul(fc2_relu, tf.Variable(
                    tf.truncated_normal([256, self.output_classes], stddev=0.1, dtype=tf.float32), name='weight6'))
                output = tf.nn.softmax(fc3 + tf.Variable(tf.constant(0.1, shape=[self.output_classes]), name='bias6'))
        return output

    def train_model(self, input_images, input_labels, image_channels):
        y_hat, accuracy_conv = self.classify_test(input_images, input_labels, image_channels)
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(-tf.reduce_sum(input_labels * tf.log(y_hat)))
            tf.summary.scalar('loss', loss)  # tensorflow >= 0.12
        with tf.name_scope('Train'):
            train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return y_hat, loss, accuracy_conv, train

    def classify_test(self, input_images, input_labels, image_channels):
        y_hat = self.build_net(input_images, image_channels)
        correct_prediction_conv = tf.equal(tf.argmax(y_hat, 1), tf.argmax(input_labels, 1))
        accuracy_conv = tf.reduce_mean(tf.cast(correct_prediction_conv, dtype=tf.float32))
        return y_hat, accuracy_conv


if __name__ == '__main__':
    net = Class_Pekoman_Net()
    print(net)
