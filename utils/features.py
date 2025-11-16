import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy

def extract_haralick(gray):
    gcm = graycomatrix(gray, [2], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(gcm, 'contrast')[0,0]
    homogeneity = graycoprops(gcm, 'homogeneity')[0,0]
    energy = graycoprops(gcm, 'energy')[0,0]
    correlation = graycoprops(gcm, 'correlation')[0,0]
    return contrast, homogeneity, energy, correlation

def extract_features(img, mask):
    ys, xs = np.where(mask>0)
    if len(xs)==0: return None

    x1,x2 = xs.min(), xs.max()
    y1,y2 = ys.min(), ys.max()

    roi = img[y1:y2+1, x1:x2+1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask_roi = mask[y1:y2+1, x1:x2+1] > 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0][mask_roi], hsv[:,:,1][mask_roi], hsv[:,:,2][mask_roi]
    h_mean, s_mean, v_mean = H.mean(), S.mean(), V.mean()
    h_std,  s_std,  v_std = H.std(),  S.std(),  V.std()

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_val = np.mean(lbp[mask_roi])

    contrast, homogeneity, energy, correlation = extract_haralick(gray)

    edges = cv2.Canny(gray, 40, 140)
    edge_density = edges[mask_roi].mean()

    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()

    entropy = shannon_entropy(mask_roi.astype(np.uint8))

    area = mask_roi.sum()
    aspect = (x2-x1+1) / (y2-y1+1 + 1e-6)

    feats = np.array([
        h_mean,s_mean,v_mean, h_std,s_std,v_std,
        lbp_val,
        contrast,homogeneity,energy,correlation,
        edge_density,
        sharp,
        entropy,
        area, aspect
    ], dtype=np.float32)

    return feats
