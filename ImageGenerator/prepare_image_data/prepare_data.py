import load_image_from_dir.loading_images as ld_image
import pickle
import os

file = './face_image.pickle'

if not os.path.exists(file):
    path = 'D:\\Documents\\GitHub\\ImageProjects\\image_data_for_ImageGenerator\\faces'
    face_images = ld_image.load_oneClass_images(path)
    with open(file, 'wb') as f:
        pickle.dump(face_images, f)
else:
    with open(file, 'rb') as f:
        face_images = pickle.load(f)

if __name__ == '__main__':
    import cv2
    cv2.imshow(face_images[1])
    cv2.waitKey(0)
