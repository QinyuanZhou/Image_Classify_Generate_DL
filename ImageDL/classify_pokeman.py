import preprocess_pokemon
import tensorflow as tf
import numpy as np
from build_network import Class_Pekoman_Net
from train_network import lb
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    test_images = preprocess_pokemon.test_data
    test_labels = preprocess_pokemon.test_labels

    test_images = np.reshape(test_images, [-1, 96, 96, 3])
    test_bin_labels = np.array(lb.fit_transform(test_labels)).astype('float32')
    test_bin_labels = np.reshape(test_bin_labels, [-1, 5])

    net = Class_Pekoman_Net(5)
    x = tf.placeholder(tf.float32, shape=[None, 96, 96, 3], name='Input_Images')
    y = tf.placeholder(tf.float32, shape=[None, 5], name='Input_Labels')
    y_hat, accur = net.classify_test(x, y, image_channels=3)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './model_poke/model_pokeman.ckpt')
        y_hat1, accur1  = sess.run([y_hat, accur], feed_dict={x:test_images, y:test_bin_labels})
        print(accur1)
        fig = plt.figure()
        plt.axis('off')
        plt.title('Pokeman Classification\n', fontsize=18)
        for i in range(0, 15, 1):
            image = np.reshape(test_images[i], [96, 96, 3])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            y_p, y_r = lb.classes_[np.argmax(y_hat1[i])], test_labels[i]
            print(y_p, y_r)
            ax = fig.add_subplot(3, 5, i + 1)
            ax.imshow(image)
            if y_p == y_r:
                color = 'r'
            else:
                color = 'g'
            ax.text(0, 25, 'predict:{}\nscore: {:.2%}\nreal:{}'.format(y_p, np.max(y_hat1[i]), y_r),
                     fontdict={'size': 12, 'color': color})
            ax.axis('off')
        plt.show()

