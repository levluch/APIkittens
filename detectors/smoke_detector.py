import cv2
import numpy as np

def detect_smoke(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 80])
    upper = np.array([180, 60, 200])
    mask = cv2.inRange(hsv, lower, upper)

    blur = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = blur.var()

    detected = (mask.mean() > 20) and (sharpness < 200)

    percent = (mask > 0).mean() * 100
    return detected, percent, mask
