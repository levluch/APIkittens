import cv2
import os
import sys
import time
import csv
from joblib import load
from utils.preproc import preprocess
from utils.features import extract_features
from utils.drawing import draw_boxes
from detectors.dump_detector import detect_dump
from detectors.smoke_detector import detect_smoke
from detectors.oil_detector import detect_oil

MODEL_PATH = "models/classifier.joblib"
SCALER_PATH = "models/scaler.joblib"
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "detections_log.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "image_path", "class", "proba",
            "x1", "y1", "x2", "y2", "area_px", "status", "saved_image"
        ])

try:
    clf = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("Модели загружены")
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    sys.exit(1)

CLASSES = ["dump", "smoke", "oil", "background"]

def classify_mask(img, mask):
    feats = extract_features(img, mask)
    if feats is None:
        return "background", 0.0

    try:
        feats_scaled = scaler.transform([feats])
        pred = clf.predict(feats_scaled)[0]
        proba = clf.predict_proba(feats_scaled).max()
        return CLASSES[pred], float(proba)
    except Exception:
        return "background", 0.0

def analyze_image(image_path, show=True):
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return

    print(f"Обработка: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print(f"Ошибка чтения изображения: {image_path}")
        return

    img_p = preprocess(img)
    out = img.copy()
    detections = []

    MIN_MASK_AREA = 300
    MIN_BOX_AREA = 400
    MIN_PROBA = 0.55

    detectors = [
        ("dump",  detect_dump,  (0, 255, 255)),
        ("smoke", detect_smoke, (255, 255, 0)),
        ("oil",   detect_oil,   (0, 0, 255)),
    ]

    for name, det, base_color in detectors:
        detected, mask, boxes, _ = det(img_p)

        if not detected:
            continue

        for (x1, y1, x2, y2, score) in boxes:
            area_px = (x2 - x1) * (y2 - y1)

            if area_px < MIN_BOX_AREA:
                continue

            roi_mask = mask[y1:y2, x1:x2]

            if roi_mask.size == 0 or roi_mask.sum() < MIN_MASK_AREA:
                continue

            cls, proba = classify_mask(img_p, roi_mask)

            if proba < MIN_PROBA or cls == "background":
                continue

            if cls in ["oil", "smoke"]:
                status = "threat"
                color = (0, 0, 255)
                label = cls

            elif cls == "dump":
                if area_px >= 5000:
                    status = "threat"
                    color = (0, 0, 255)
                else:
                    status = "warning"
                    color = (0, 255, 255)
                label = "dump"

            out = draw_boxes(out, [(x1, y1, x2, y2, proba)], color, label)
            detections.append((cls, proba, x1, y1, x2, y2, area_px, status))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)

    save_path = os.path.join(SAVE_DIR, f"{name}_detected_{timestamp}{ext}")
    cv2.imwrite(save_path, out)

    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for cls, proba, x1, y1, x2, y2, area_px, status in detections:
            writer.writerow([
                time.time(),
                image_path,
                cls,
                proba,
                x1, y1, x2, y2,
                area_px,
                status,
                save_path
            ])

    print(f"Найдено объектов: {len(detections)} → {save_path}")

    if show:
        display = out
        h, w = display.shape[:2]
        if w > 1200:
            scale = 1200 / w
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        cv2.imshow(f"Result: {base}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python main.py image1.jpg image2.png")
        print("  python main.py папка/")
        return

    arg = sys.argv[1]

    if os.path.isdir(arg):
        print(f"Обработка папки: {arg}")
        for file in os.listdir(arg):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                analyze_image(os.path.join(arg, file), show=True)
    else:
        for path in sys.argv[1:]:
            analyze_image(path, show=True)

    print(f"Результаты сохранены в: {SAVE_DIR}")

if __name__ == "__main__":
    main()