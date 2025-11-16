import cv2
import numpy as np

def augment_image(img):
    imgs = [img]
    h,w = img.shape[:2]

    imgs.append(cv2.flip(img, 1))

    for a in [0.8,1.2]:
        imgs.append(cv2.convertScaleAbs(img, alpha=a, beta=10))

    M = cv2.getRotationMatrix2D((w//2,h//2), 5, 1.0)
    imgs.append(cv2.warpAffine(img, M, (w,h)))

    return imgs
