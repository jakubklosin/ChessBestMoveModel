import os

def delete_images_with_prefix(prefix='chess5', dataset_dir='dataset'):
    deleted = 0
    for label_dir in os.listdir(dataset_dir):
        full_label_path = os.path.join(dataset_dir, label_dir)
        if not os.path.isdir(full_label_path):
            continue

        for fname in os.listdir(full_label_path):
            if fname.startswith(prefix):
                file_path = os.path.join(full_label_path, fname)
                os.remove(file_path)
                deleted += 1
                print(f"Usunięto: {file_path}")

    print(f"Usunięto {deleted} plików z prefixem '{prefix}'.")

delete_images_with_prefix()
