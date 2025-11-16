import os
from main import analyze_image

INPUT_DIR = "test_images"
for f in os.listdir(INPUT_DIR):
    if f.lower().endswith(('.jpg','.png','.jpeg')):
        path = os.path.join(INPUT_DIR, f)
        print("===", f)
        analyze_image(path, show=False)
