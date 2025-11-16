import cv2
import numpy as np

def detect_smoke(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 60])
    upper = np.array([180, 70, 220])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = lap.var()

    percent = (mask > 0).mean() * 100
    detected = (percent > 2.0) and (sharpness < 250)

    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500: continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x+w, y+h, area))

    return detected, mask, boxes, percent / 100
