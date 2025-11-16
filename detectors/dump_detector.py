import cv2
import numpy as np

def detect_dump(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 30])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), 1)

    percent = (mask > 0).mean() * 100
    detected = percent > 3.5

    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500: continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x+w, y+h, area))

    return detected, mask, boxes, percent / 100
