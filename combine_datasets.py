import shutil
import os

# Połącz dane z dataset + augmented_dataset do dataset_extended
SOURCE_DIRS = ['dataset', 'augmented_dataset']
TARGET_DIR = 'dataset_extended'

# Tworzymy strukturę dataset_extended/
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)

# Przenoszenie danych z dataset i augmented_dataset
for source in SOURCE_DIRS:
    for class_name in os.listdir(source):
        class_src = os.path.join(source, class_name)
        class_dst = os.path.join(TARGET_DIR, class_name)
        os.makedirs(class_dst, exist_ok=True)

        for fname in os.listdir(class_src):
            src_file = os.path.join(class_src, fname)
            dst_file = os.path.join(class_dst, fname)
            shutil.copyfile(src_file, dst_file)

# Usuwanie starych katalogów
shutil.rmtree('dataset')
shutil.rmtree('augmented_dataset')

"Połączono dane do dataset_extended/ i usunięto dataset oraz augmented_dataset."
