import cv2
import numpy as np

def detect_oil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    percent = (mask > 0).mean() * 100

    detected = percent > 3.5

    return detected, percent, mask
