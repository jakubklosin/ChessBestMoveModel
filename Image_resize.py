import cv2
import os
import numpy as np

def resize_with_padding(img, size=512, pad_color=255):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    h_resized, w_resized = resized.shape[:2]

    top = (size - h_resized) // 2
    bottom = size - h_resized - top
    left = (size - w_resized) // 2
    right = size - w_resized - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[pad_color]*3)
    return padded

def process_all_images(input_dir='Images', output_dir='resized_images'):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Błąd wczytywania: {fname}")
            continue

        resized_img = resize_with_padding(img, 512)
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, resized_img)
        print(f"Zapisano: {output_path}")

def process_single_image(input_path='Images/new_chess10.png', output_dir='resized_images'):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Błąd wczytywania: {input_path}")
        return

    resized_img = resize_with_padding(img, 512)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    cv2.imwrite(output_path, resized_img)
    print(f"Zapisano: {output_path}")

process_single_image()

#process_all_images()
