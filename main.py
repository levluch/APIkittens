import cv2
from utils.preproc import preprocess
from utils.drawing import draw_mask
from detectors.dump_detector import detect_dump
from detectors.smoke_detector import detect_smoke
from detectors.oil_detector import detect_oil

def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return "Картинка не найдена", None

    img_prep = preprocess(img)
    output = img.copy()

    results = []

    dump_det, dump_p, mask_dump = detect_dump(img_prep)
    if dump_det:
        results.append(f"Свалка мусора ({dump_p:.1f}%)")
        output = draw_mask(output, mask_dump, color=(0, 255, 255))

    smoke_det, smoke_p, mask_smoke = detect_smoke(img_prep)
    if smoke_det:
        results.append(f"Дым / выбросы ({smoke_p:.1f}%)")
        output = draw_mask(output, mask_smoke, color=(255, 255, 0))

    oil_det, oil_p, mask_oil = detect_oil(img_prep)
    if oil_det:
        results.append(f"Утечка топлива ({oil_p:.1f}%)")
        output = draw_mask(output, mask_oil, color=(0, 0, 255))

    if not results:
        results.append("Загрязнений не обнаружено")

    return "\n".join(results), output


if __name__ == "__main__":
    result, frame = analyze_image("test.jpg")
    print(result)

    if frame is not None:
        cv2.imshow("Result", frame)
        cv2.waitKey(0)
