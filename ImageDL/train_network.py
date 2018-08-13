import preprocess_pokemon
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from build_network import Class_Pekoman_Net
'''
train_data = preprocess.train_data
train_labels = preprocess.train_labels
train_bin_labels = np.array(LabelBinarizer().fit_transform(train_labels))
'''

def shuffle_inputdata(train_data, train_labels):
    nums = np.shape(train_data)[0]
    index = np.arange(0, nums)
    np.random.shuffle(index)
    train_data = train_data[index]
    train_labels = train_labels[index]
    return train_data, train_labels

train_data, train_labels = shuffle_inputdata(np.array(preprocess_pokemon.train_data), np.array(preprocess_pokemon.train_labels))
train_data = train_data.astype('float32')/255.0
train_bin_labels = np.array(LabelBinarizer().fit_transform(train_labels)).astype('float32')

def next_batch(train_data, train_labels, batch_size):
    nums = np.shape(train_data)[0]
    index = np.arange(0, nums)
    np.random.shuffle(index)
    batch_index = index[0:batch_size]
    return train_data[batch_index], train_labels[batch_index]

input_images = tf.placeholder(tf.float32, shape=[None, 96, 96, 3], name='Input_Images')
input_labels = tf.placeholder(tf.float32, shape=[None, 5], name='Input_Labels')
'''
conv1 = tf.nn.conv2d(input_images, filter=tf.Variable(
            tf.truncated_normal([5, 5, 3, 32], stddev=0.1, dtype=tf.float32), name='weight1'),
                             strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(conv1 + tf.Variable(tf.constant(0.1, shape=[32]), name='bias1'))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

conv2 = tf.nn.conv2d(pool1, filter=tf.Variable(
    tf.truncated_normal([7, 7, 32, 64], stddev=0.1, dtype=tf.float32), name='weight2'),
                     strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(conv2 + tf.Variable(tf.constant(0.1, shape=[64]), name='bias2'))
pool2= tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

conv3 = tf.nn.conv2d(pool2, filter=tf.Variable(
    tf.truncated_normal([7, 7, 64, 128], stddev=0.1, dtype=tf.float32), name='weight3'),
                     strides=[1, 1, 1, 1], padding='SAME')
relu3 = tf.nn.relu(conv3 + tf.Variable(tf.constant(0.1, shape=[128]), name='bias3'))
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

pool3_flat = tf.reshape(pool3, [-1, 12*12*128])
fc1 = tf.matmul(pool3_flat, tf.Variable(
    tf.truncated_normal([12*12*128, 1024], stddev=0.1, dtype=tf.float32, name='weight4')))
fc1_relu = tf.nn.relu(fc1 + tf.Variable(tf.constant(0.1, shape=[1024], name='bias4')))
fc1_relu = tf.nn.dropout(fc1_relu, 0.25)

fc2= tf.matmul(fc1_relu, tf.Variable(
    tf.truncated_normal([1024, 256], stddev=0.1, dtype=tf.float32), name='weight5'))
fc2_relu = tf.nn.relu(fc2 + tf.Variable(tf.constant(0.1, shape=[256]), name='bias5'))
fc2_relu = tf.nn.dropout(fc2_relu, 0.25)

fc3 = tf.matmul(fc2_relu, tf.Variable(
    tf.truncated_normal([256, 5], stddev=0.1, dtype=tf.float32), name='weight6'))
output = tf.nn.softmax(fc3 + tf.Variable(tf.constant(0.1, shape=[5]),  name='bias6'))
'''

net = Class_Pekoman_Net()
y_hat, loss, accur, train = net.train_model(input_images, input_labels, image_channels=3)

'''
loss_conv = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)
train_conv = tf.train.AdamOptimizer(1e-2).minimize(loss_conv)
correct_prediction_conv = tf.equal(tf.argmax(output, 1), tf.argmax(input_labels, 1))
accuracy_conv = tf.reduce_mean(tf.cast(correct_prediction_conv, dtype=tf.float32))
batch_size = 100
'''
init_var = tf.global_variables_initializer()
batch_size = net.batch_size
saver = tf.train.Saver() # save visualize graph
with tf.Session() as sess:
    merged = tf.summary.merge_all()  # merge all chart together
    writer = tf.summary.FileWriter("logs/", sess.graph)  # save visualize graph
    sess.run(init_var)
    for i in range(100):
        print('[INFO]...itera {} training, wait...'.format(i))
        X, Y = next_batch(train_data, train_bin_labels, batch_size)
        accur1, train1 = sess.run([accur, train],
                                             feed_dict={input_images:X, input_labels:Y})
        if i % 10 == 0:
            # rs = sess.run(merged,
            #          feed_dict={input_images: X, input_labels: Y})
            # writer.add_summary(rs, i)
            print("[INFO] step %d, training accuracy %g" % (i, accur1))
        '''
        sess.run(train_conv, feed_dict={input_images: X, input_labels: Y})
        train_accuracy = sess.run(accuracy_conv, feed_dict={input_images:X, input_labels:Y})
        if i % 10 == 0:
            print("step %d, training accuracy %g" % (i, train_accuracy))
        '''
    saver.save(sess, "../model/model.ckpt")