import cv2
from sklearn.cluster import KMeans

def compress_image(filename, outputPath, kcluster=4):
    image = cv2.imread(filename)
    imgr = image.reshape(-1, 3)
    print(image.shape, imgr.shape)

    k_colrs = KMeans(n_clusters=kcluster).fit(imgr)
    imgCompress = k_colrs.cluster_centers_[k_colrs.labels_]
    imgCompress = imgCompress.reshape(image.shape).astype('uint8')
    cv2.imwrite(outputPath, imgCompress)
    # cv2.imshow('initial image', image)
    # cv2.imshow('compress image', imgCompress)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    filename = ['pikachu.png', 'tortoise.jpg']
    for i in filename:
        outPath = i.split('.')[0] + '_compress.png'
        compress_image(i, outPath, kcluster=4)