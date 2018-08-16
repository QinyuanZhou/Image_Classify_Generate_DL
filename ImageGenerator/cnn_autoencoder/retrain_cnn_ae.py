from cnn_autoencoder.construct_cnn_ae_model import Conv_AE
from prepare_image_data.prepare_data import face_images
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    source_image = face_images[:1024] # (51223, 96, 96, 3)
    source_image = np.array((source_image / 255)) # (512, 96, 96, 3)
    print(source_image.shape)
    print('[INFO] Complete loading images')

    batch_size = 256
    epoch = 20
    # COVN_AE = Conv_AE()
    COVN_AE = Conv_AE(learning_rate=0.00)
    COVN_AE.encoder()
    COVN_AE.middle_code()
    recon_image = COVN_AE.decoder(COVN_AE.m_code)
    loss = tf.reduce_mean(tf.square(COVN_AE.X - recon_image))
    train = tf.train.AdamOptimizer(COVN_AE.learning_rate).minimize(loss)
    model_saver = tf.train.Saver()
    # init_var = tf.global_variables_initializer()

    hist_loss = []
    fig1 = plt.figure('Loss')
    fig2 = plt.figure('Image_Reconstruct')
    plt.ion()

    with tf.Session() as sess:
        # sess.run(init_var) # first train
        model_saver.restore(sess, '../model_r/COVN_AE.ckpt')

        for e in range(epoch):
            print('[INFO] epoch {} training, wait...'.format(e + 1))
            X_batches = next_batch(source_image, batch_size)
            all_batch = len(X_batches)
            for k, X in enumerate(X_batches):
                # if k == all_batch-1:
                #     break
                # with tf.device('/gpu:0'):
                feed_dict = {COVN_AE.X: X}
                recon_image1, loss1, train1 = sess.run([recon_image, loss, train], feed_dict=feed_dict)
                hist_loss.append(loss1)
                plt.figure('Loss')
                plt.plot(hist_loss)
                print("[INFO] epoch/all_batch/batch: {}/{}/{}, loss {:.5f}".format(e + 1, all_batch, k + 1, loss1))

                for i in range(5):
                    # print(i, X.shape)
                    ax = fig2.add_subplot(2, 5, i+1)
                    init_image = (X[i] * 255).astype(np.uint8)
                    # print(init_image.shape)
                    init_image = cv2.cvtColor(np.reshape(init_image, [96, 96, 3]), cv2.COLOR_BGR2RGB)
                    ax.imshow(init_image)
                    ax.axis('off')
                    ax = fig2.add_subplot(2, 5, i + 6)
                    re_image = (recon_image1[i] * 255).astype(np.uint8)
                    re_image = cv2.cvtColor(np.reshape(re_image, [96, 96, 3]), cv2.COLOR_BGR2RGB)
                    ax.imshow(re_image)
                    ax.axis('off')

                plt.pause(0.1)
            if e%5 == 0:
                model_saver.save(sess, "../model_r/COVN_AE.ckpt")
        model_saver.save(sess, "../model_r/COVN_AE.ckpt")
    plt.ioff()  # close interactive mode
    plt.show()