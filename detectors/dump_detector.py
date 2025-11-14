import cv2
import numpy as np

def detect_dump(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 40, 40])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    percent = (mask > 0).mean() * 100

    detected = percent > 5  # порог
    return detected, percent, mask
