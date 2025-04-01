from PIL import Image, ImageEnhance, ImageOps
import os
import random

# Klasy, które chcemy augmentować
TARGET_CLASSES = ['bp', 'br', 'bq', 'wp', 'bn', 'empty']
SOURCE_DIRS = ['dataset', 'eval_pola']
OUTPUT_DIR = 'augmented_dataset'
AUGS_PER_IMAGE = 4  # ile augmentacji na obraz

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definicje prostych augmentacji
def augment_image(img):
    augmentations = []

    # 1. Odbicie lustrzane
    augmentations.append(ImageOps.mirror(img))

    # 2. Obrót losowy (90, 180, 270)
    angle = random.choice([90, 180, 270])
    augmentations.append(img.rotate(angle))

    # 3. Jasność
    enhancer = ImageEnhance.Brightness(img)
    augmentations.append(enhancer.enhance(random.uniform(0.6, 1.4)))

    # 4. Kontrast
    enhancer = ImageEnhance.Contrast(img)
    augmentations.append(enhancer.enhance(random.uniform(0.6, 1.4)))

    return augmentations[:AUGS_PER_IMAGE]

# Główna pętla
for class_name in TARGET_CLASSES:
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    all_images = []

    for source in SOURCE_DIRS:
        input_class_dir = os.path.join(source, class_name)
        if os.path.exists(input_class_dir):
            for fname in os.listdir(input_class_dir):
                if fname.endswith('.png'):
                    all_images.append(os.path.join(input_class_dir, fname))

    for img_path in all_images:
        img = Image.open(img_path).convert('RGB')
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        augmented = augment_image(img)
        for i, aug in enumerate(augmented):
            save_path = os.path.join(output_class_dir, f"{base_name}_aug{i+1}.png")
            aug.save(save_path)

"Augmentacja zakończona – obrazy zapisane w 'augmented_dataset/'"
