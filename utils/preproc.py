import cv2
import numpy as np

def preprocess(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return enhanced
