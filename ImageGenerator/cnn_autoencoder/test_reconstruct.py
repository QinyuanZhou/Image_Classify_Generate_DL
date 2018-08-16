from cnn_autoencoder.construct_cnn_ae_model import Conv_AE
from prepare_image_data.prepare_data import face_images
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    print('[INFO] loading images')
    source_image = face_images[-256:] # collect last 256 images for testing
    source_image = np.array((source_image / 255)) # (512, 96, 96, 3)
    print(source_image.shape)
    print('[INFO] Complete loading images')

    COVN_AE = Conv_AE()
    COVN_AE.encoder()
    COVN_AE.middle_code()
    recon_image = COVN_AE.decoder(COVN_AE.m_code)
    loss = tf.reduce_mean(tf.square(COVN_AE.X - recon_image))
    model_saver = tf.train.Saver()

    hist_loss = []
    hist_re_image = []
    fig1 = plt.figure('Loss')
    fig2 = plt.figure('Image_Reconstruct')
    plt.ion()

    with tf.Session() as sess:
        model_saver.restore(sess, '../model_r/COVN_AE.ckpt')
        feed_dict = {COVN_AE.X: source_image}
        for i in range(256):
            recon_image1, loss1 = sess.run([recon_image, loss], feed_dict=feed_dict)
            hist_loss.append(loss1)
            hist_re_image.append(recon_image1)
        plt.figure('Loss')
        plt.plot(hist_loss)
        for i in range(5):
            ax = fig2.add_subplot(2, 5, i+1)
            init_image = (source_image[i] * 255).astype(np.uint8)

            init_image = cv2.cvtColor(np.reshape(init_image, [96, 96, 3]), cv2.COLOR_BGR2RGB)
            ax.imshow(init_image)
            ax.axis('off')
            ax = fig2.add_subplot(2, 5, i + 6)
            re_image = (hist_re_image[i] * 255).astype(np.uint8)
            re_image = cv2.cvtColor(np.reshape(re_image, [96, 96, 3]), cv2.COLOR_BGR2RGB)
            ax.imshow(re_image)
            ax.axis('off')
        plt.pause(0.1)
    plt.ioff()  # close interactive mode
    plt.show()