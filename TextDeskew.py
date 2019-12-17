import os
import cv2
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter


def text_deskew(image, filename, debug_path):
    image = cv2.copyMakeBorder(image,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])
    img = im.fromarray(image)
    wd,ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    # correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(debug_path, filename + "_3_page__after_deskew.jpg"), rotated)
    return rotated, best_angle
