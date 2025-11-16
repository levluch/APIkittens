import os
import cv2
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE

from utils.preproc import preprocess
from utils.features import extract_features
from detectors.dump_detector import detect_dump
from detectors.smoke_detector import detect_smoke
from detectors.oil_detector import detect_oil
from utils.augment import augment_image

CLASSES = ["dump", "smoke", "oil", "background"]

DATA_DIR = "data"
X, y = [], []

def get_mask_for_training(img_p, cls):
    if cls=="dump": _,mask,_,_=detect_dump(img_p)
    elif cls=="smoke": _,mask,_,_=detect_smoke(img_p)
    elif cls=="oil": _,mask,_,_=detect_oil(img_p)
    else:
        h,w = img_p.shape[:2]
        mask = np.zeros((h,w), np.uint8)
        mask[h//4:3*h//4, w//4:3*w//4] = 255

    if (mask>0).sum()==0:
        mask = np.ones(img_p.shape[:2], np.uint8)*255
    return mask

for cls_id, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg','png','jpeg'))]

    for fname in files:
        img = cv2.imread(os.path.join(folder,fname))
        img_p = preprocess(img)

        for aug in augment_image(img_p):
            mask = get_mask_for_training(aug, cls)
            feats = extract_features(aug, mask)
            if feats is None: continue
            X.append(feats)
            y.append(cls_id)

X, y = np.array(X), np.array(y)

sm = SMOTE()
X, y = sm.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(300, max_depth=18, n_jobs=-1)
cv = StratifiedKFold(5)
scores = cross_val_score(clf, X_scaled, y, cv=cv)

print("CV:", scores.mean())
clf.fit(X_scaled, y)

os.makedirs("models", exist_ok=True)
dump(clf, "models/classifier.joblib")
dump(scaler, "models/scaler.joblib")
print("Saved models/")
