import preprocess_pokemon
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from build_network import Class_Pekoman_Net
import matplotlib.pyplot as plt

def shuffle_inputdata(train_data, train_labels):
    nums = np.shape(train_data)[0]
    index = np.arange(0, nums)
    np.random.shuffle(index)
    train_data = train_data[index]
    train_labels = train_labels[index]
    return train_data, train_labels

def next_batch(train_data, train_labels, batch_size):
    nums = np.shape(train_data)[0]
    batch = int(np.ceil(nums / batch_size))
    train_data_batch, train_labels_batch = [], []
    for i in range(0, batch):
        if i == batch-1:
            train_data_batch.append(train_data[i * batch_size:])
            train_labels_batch.append(train_labels[i * batch_size:])
        else:
            train_data_batch.append(train_data[i*batch_size : (i+1)*batch_size])
            train_labels_batch.append(train_labels[i * batch_size: (i + 1) * batch_size])
    return train_data_batch, train_labels_batch

train_data, train_labels = shuffle_inputdata(np.array(preprocess_pokemon.train_data), np.array(preprocess_pokemon.train_labels))
train_data = train_data / 255.0
lb = LabelBinarizer()
train_bin_labels = np.array(lb.fit_transform(train_labels)).astype('float32')
pokrman_genus = lb.classes_

if __name__ == '__main__':

    input_images = tf.placeholder(tf.float32, shape=[None, 96, 96, 3], name='Input_Images')
    input_labels = tf.placeholder(tf.float32, shape=[None, 5], name='Input_Labels')

    hist = [[], []]
    figure = plt.figure()
    plt.axis('off')
    plt.title('Training Accuracy and Loss\n\n')
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.set_title('Accuracy')
    ax2 = figure.add_subplot(1, 2, 2)
    ax2.set_title('Loss')
    plt.ion()  # open interactive mode

    net = Class_Pekoman_Net(5)
    y_hat, loss, accur, train = net.train_model(input_images, input_labels, image_channels=3)
    init_var = tf.global_variables_initializer()
    batch_size = net.batch_size
    saver = tf.train.Saver() # save visualize graph
    with tf.Session() as sess:
        merged = tf.summary.merge_all()  # merge all chart together
        writer = tf.summary.FileWriter("logs/", sess.graph)  # save visualize graph
        sess.run(init_var)
        for epoch in range(200):
            print('[INFO] epoch {} training, wait...'.format(epoch+1))
            train_data, train_bin_labels = shuffle_inputdata(train_data, train_bin_labels)
            X, Y = next_batch(train_data, train_bin_labels, batch_size)
            all_batch = len(X)
            for k, xbatch in enumerate(X):
                if k == all_batch - 1:
                    break
                X_batch, Y_batch = xbatch, Y[k]
                accur1, train1, loss1 = sess.run([accur, train, loss], feed_dict={input_images:X_batch, input_labels:Y_batch})
                hist[1].append(accur1)
                hist[0].append(loss1)
                ax1.plot(hist[1])
                ax2.plot(hist[0])
                plt.pause(0.05)  # pause 0.05 seconds
                print("[INFO] epoch/batch: {}/{}, accuracy {:.5f}, loss {:.5f}".format(epoch + 1, k + 1, accur1, loss1))
        saver.save(sess, "./model_poke/model_pokeman.ckpt")
    plt.ioff()  # close interactive mode
    plt.show()