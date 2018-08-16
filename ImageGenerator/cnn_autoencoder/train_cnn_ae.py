from cnn_autoencoder.construct_cnn_ae_model import Conv_AE
from prepare_image_data.prepare_data import face_images
import tensorflow as tf
import cv2
import numpy as np

if __name__ == '__main__':

    source_image = face_images
    # source_image = cv2.imread('1.jpg')
    # source_image = np.reshape(np.array(source_image).astype('float32'), [-1, 96, 96, 3])

    print('[INFO] Complete loading images')
    COVN_AE = Conv_AE()

    with tf.Session() as sess:
        recon_image1, loss1, train1 = COVN_AE.train_net(sess, source_image)

