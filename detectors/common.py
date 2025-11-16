import cv2
import numpy as np

def contours_to_boxes(mask, min_area=300):
    boxes = []
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = mask.shape[:2]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        x,y,wc,hc = cv2.boundingRect(c)
        score = min(1.0, area / (w*h*0.05))
        boxes.append((x, y, x+wc, y+hc, score))

    return boxes
