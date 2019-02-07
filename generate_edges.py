import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
import math

if __name__ == '__main__':
    img_dir = 'data/images/celebA/images'
    edges_dir = 'data/edges/celebA'
    max_side = 748

    img_filenames = glob.glob(os.path.join(img_dir, '*.*'))

    for filename in tqdm(img_filenames):

        # print(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        image = cv2.imread(filename)

        original_shape = image.shape
        scaling_factor = float(max_side / max(image.shape[0:2]))

        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

        min_side = min(image.shape[0:2])
        k_size = float(min_side) / 40
        k_size = int(max(3.0, k_size))
        k_size += int(math.floor((k_size - 1) % 2))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((k_size, k_size), np.uint8)
        erosion = cv2.erode(thresh1, kernel, iterations=1)
        blur = cv2.medianBlur(image, k_size)
        edges = cv2.Canny(blur, 100, 200)

        laplacian = cv2.Laplacian(edges, cv2.CV_8UC1)
        sobely = cv2.Sobel(laplacian, cv2.CV_8UC1, 0, 1, ksize=5)
        im2, contours, hierarchy = cv2.findContours(sobely, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        edges = cv2.drawContours(im2, contours, -1, (255, 0, 0), 3)
        edges = cv2.resize(edges, (original_shape[1], original_shape[0]), cv2.INTER_LINEAR)
        # cv2.imshow('image', image)
        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(edges_dir, basename + '.png'), edges)