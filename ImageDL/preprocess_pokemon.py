'''
Thanks for Adrian Rosebrock, the Pokeman data was collected by him.
Thanks for Adrian Rosebrock's kindness, and the following is his website.
https://www.pyimagesearch.com
There're many fantastic blogs and i think you'll like it as i do, and thanks for Adrian Rosebrock again.
'''
import os
import cv2
import numpy as np
import pickle

if False:
    IMAGE_DIMS = (96, 96, 3)
    PATH = '../Pokeman_dataset/' # there're two files named train and test
    TRAIN_PATH = '../Pokeman_dataset/train/'
    TEST_PATH = '../Pokeman_dataset/test/'

    train_file = os.listdir(TRAIN_PATH) # ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
    train_class = train_file # ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
    # print(train_file, train_labels, test_file, test_labels)

    train_pic_path = [os.path.join(TRAIN_PATH, x) for x in train_file]
    for i, genus in enumerate(train_pic_path):
        for j, item in enumerate(os.listdir(genus)):
            image_path = genus+ '/' + item
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            # cv2.imshow('1', image)
            # cv2.waitKey(0)
            # print(j, genus, image.shape, type(image))
            train_data.append(image)
            train_labels.append(train_class[i])

    test_file = os.listdir(TEST_PATH) # ['bulbasaur_plush.png', 'charmander_counter.png', 'charmander_hidden.png', 'mewtwo_toy.png', 'pikachu_toy.png', 'squirtle_plush.png']
    test_labels = [x.split('_')[0] for x in test_file] # ['bulbasaur', 'charmander', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
    test_data = []
    test_pics_path = [os.path.join(TEST_PATH, x) for x in test_file]
    for i in test_pics_path:
        image = cv2.imread(i)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        # cv2.imshow('1', image)
        # cv2.waitKey(0)
        # print(j, genus, image.shape, type(image))
        test_data.append(image)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    with open('train_data.pickle', 'wb') as f:
        pickle.dump(train_data, f)
    with open('train_labels.pickle', 'wb') as f:
        pickle.dump(train_labels, f)
else:
    with open('train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open('train_labels.pickle', 'rb') as f:
        train_labels = np.array(pickle.load(f))
    # print(type(train_data), type(train_labels), train_data.shape, train_labels.shape)
