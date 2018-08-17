from cnn_autoencoder.construct_cnn_ae_model import Conv_AE
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    poke_image = cv2.imread('D:\\Documents\\GitHub\\Image_Classify_Generate_DL\\Pokeman_dataset\\train\\bulbasaur\\00000004.png')
    poke_image = np.reshape(poke_image, [-1, 96, 96, 3]) / 255
    print(poke_image.shape)
    print('[INFO] Complete loading images')

    COVN_AE = Conv_AE()
    COVN_AE.encoder()
    COVN_AE.middle_code()
    recon_image = COVN_AE.decoder(COVN_AE.m_code)
    loss = tf.reduce_mean(tf.square(COVN_AE.X - recon_image))
    model_saver = tf.train.Saver()

    fig = plt.figure('Image_Reconstruct')

    with tf.Session() as sess:
        model_saver.restore(sess, '../model_r/COVN_AE.ckpt')
        feed_dict = {COVN_AE.X: poke_image}
        recon_image1, loss1 = sess.run([recon_image, loss], feed_dict=feed_dict)
        recon_image1 = (recon_image1 * 255).astype(np.uint8)
        ax1 = fig.add_subplot(2,1,1)
        ax1.imshow(cv2.cvtColor(np.reshape(recon_image1, [96, 96, 3]), cv2.COLOR_BGR2RGB))
        ax1.axis('off')

        ax1 = fig.add_subplot(2, 1, 2)
        init_image1 = (poke_image * 255).astype(np.uint8)
        ax1.imshow(cv2.cvtColor(np.reshape(init_image1, [96, 96, 3]), cv2.COLOR_BGR2RGB))
        ax1.axis('off')