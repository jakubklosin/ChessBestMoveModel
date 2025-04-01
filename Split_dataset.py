import os
from sklearn.model_selection import train_test_split
import shutil

# Zakładamy, że dane są w katalogu "dataset/class_name/*.png"
# Cele: podzielić dane na: train/val/test (np. 70/15/15) w strukturze:
# dataset_split/train/class_name/*.png
# dataset_split/val/class_name/*.png
# dataset_split/test/class_name/*.png

def split_dataset(original_dir='dataset_extended', output_dir='dataset_split',
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Podziały muszą się sumować do 1.0"
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=seed)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

        for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_name in split_imgs:
                src_path = os.path.join(class_dir, img_name)
                dst_path = os.path.join(split_class_dir, img_name)
                shutil.copyfile(src_path, dst_path)

    return "Podział zakończony"

split_dataset()