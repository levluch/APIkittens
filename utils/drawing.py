import cv2

def draw_boxes(img, boxes, color=(0,255,0), label="obj"):

    out = img.copy()
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    return out