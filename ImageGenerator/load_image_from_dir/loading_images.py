import cv2
import os
import numpy as np

'''
load_oneClass_images(folder_path, resize_shape, resize_image=False): 
    load images from a specified folder which just contains images.
    parameter:
        folder_path: an absolute path, like D:\images\ish 
        resize_shape: the image will be converted into the specified shape
        resize_image = False: whether resize the image
    return:
        the image array, has been converted into np.array()
    
load_multiClass_images(dir, resize_shape, resize_image=False):
    load images from a specified dir which just contains several folders, which folders including one kind of image.
    parameter:
        dir: an absolute path, like D:\images\ish 
        resize_shape: the image will be converted into the specified shape
        resize_image = False: whether resize the image
    return:
        : the image array, converted into np.array()
        : the label array, the corresponding label of the image, also converted into np.array()s
'''

def load_oneClass_images(folder_path, resize_shape=[], resize_image=False):
    image_array = []
    image_files = os.listdir(folder_path)
    image_path_list = [os.path.join(folder_path, x) for x in image_files]
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        if resize_image:
            image = cv2.resize(image, resize_shape)
        image_array.append(image)
    return np.array(image_array)

def load_multiClass_images(dir, resize_shape=[], resize_image=False):
    image_array, label_array = np.array([]), []
    class_files = os.listdir(dir)
    class_path = [os.path.join(dir, x) for x in class_files]
    index_class = 0
    for i, folder in enumerate(class_path):
        len_image = len(os.listdir(folder))
        label_array[index_class:] = [class_files[i] for i in range(len_image)]
        index_class += len_image
        if i == 0:
            image_array = load_oneClass_images(folder, resize_shape, resize_image=resize_image)
            continue
        image_array = np.concatenate((image_array, load_oneClass_images(folder, resize_shape, resize_image=resize_image)), axis=0)
    return np.array(image_array), np.array(label_array)
