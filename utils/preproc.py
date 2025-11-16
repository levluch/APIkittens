import cv2
import numpy as np

def white_balance_gray(img):
    b, g, r = cv2.split(img)
    kb = np.mean(b)
    kg = np.mean(g)
    kr = np.mean(r)
    k = (kb + kg + kr) / 3

    b = np.clip((b * (k/kb)), 0, 255).astype(np.uint8)
    g = np.clip((g * (k/kg)), 0, 255).astype(np.uint8)
    r = np.clip((r * (k/kr)), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def shadow_reduction(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def preprocess(img, target_width=1000):
    h,w = img.shape[:2]
    if w != target_width:
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h*scale)))

    img = white_balance_gray(img)
    img = shadow_reduction(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img
