from cnn_autoencoder.construct_cnn_ae_model import Conv_AE
from prepare_image_data.prepare_data import face_images
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def next_batch(train_data, batch_size):
    nums = np.shape(train_data)[0]
    index = np.arange(0, nums)
    np.random.shuffle(index)
    train_data = train_data[index]

    nums = np.shape(train_data)[0]
    batch = int(np.ceil(nums / batch_size))
    train_data_batch, train_labels_batch = [], []
    for i in range(0, batch):
        if i == batch-1:
            train_data_batch.append(train_data[i * batch_size:])
        else:
            train_data_batch.append(train_data[i*batch_size : (i+1)*batch_size])
    return train_data_batch

if __name__ == '__main__':

    print('[INFO] loading images')
    source_image = face_images[:256] # (51223, 96, 96, 3)
    print('[INFO] Complete loading images')

    batch_size = 64
    epoch = 500
    COVN_AE = Conv_AE()

    hist_loss = []
    fig1 = plt.figure('Loss')
    fig2 = plt.figure('Image_Reconstruct')
    plt.ion()

    with tf.Session() as sess:
        for e in range(epoch):
            print('[INFO] epoch {} training, wait...'.format(e + 1))
            X_batches = next_batch(source_image, batch_size)
            all_batch = len(X_batches)
            for k, X in enumerate(X_batches):
                if k == all_batch - 1:
                    break
                recon_image1, loss1, train1, saver = COVN_AE.train_net(sess, X)
                hist_loss.append(loss1)
                plt.figure('Loss')
                plt.plot(hist_loss)
                print("[INFO] epoch/batch: {}/{}, loss {:.5f}".format(e + 1, k + 1, loss1))

                for i in range(5):
                    ax = fig2.add_subplot(2, 5, i+1)
                    ax.imshow(np.reshape(X[i], [96, 96, 3]))
                    ax.axis('off')
                    ax = fig2.add_subplot(2, 5, i + 5)
                    ax.imshow(np.reshape(recon_image1[i], [96, 96, 3]))
                    ax.axis('off')

                plt.pause(0.1)
            saver.save(sess, "../model/COVN_AE.ckpt")
        saver.save(sess, "../model/COVN_AE.ckpt")
    plt.ioff()  # close interactive mode
    plt.show()
