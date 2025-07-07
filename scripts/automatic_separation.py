import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

# настройки
PROJECT_PATH = "E:/Project"
IMAGES_DIR = os.path.join(PROJECT_PATH, "data/frames")
ANNOTATIONS_DIR = os.path.join(PROJECT_PATH, "data/labels")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "dataset")

# создание структуры папок
for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", subset), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", subset), exist_ok=True)

# подготовка данных
print("Подготовка данных...")

# Собираем все изображения
image_files = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(glob(os.path.join(IMAGES_DIR, "**", ext), recursive=True))

if not image_files:
    raise ValueError("Не найдены изображения в папке frames!")

# Создаем список пар изображение-аннотация
valid_pairs = []
for img_path in image_files:
    img_name = os.path.basename(img_path)
    ann_path = os.path.join(ANNOTATIONS_DIR, os.path.splitext(img_name)[0] + ".txt")

    if os.path.exists(ann_path):
        valid_pairs.append((img_path, ann_path))

if not valid_pairs:
    raise ValueError("Не найдено ни одной пары изображение-аннотация!")

# Разделяем данные
image_paths = [pair[0] for pair in valid_pairs]
train, temp = train_test_split(image_paths, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)


# копирование файлов
def copy_dataset(files, subset):
    copied = 0
    for img_path in files:
        img_name = os.path.basename(img_path)
        ann_path = os.path.join(ANNOTATIONS_DIR, os.path.splitext(img_name)[0] + ".txt")

        # Копируем изображение
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", subset, img_name))
        # Копируем аннотацию
        shutil.copy(ann_path, os.path.join(OUTPUT_DIR, "labels", subset, os.path.basename(ann_path)))
        copied += 1

    print(f"Скопировано в {subset}: {copied} файлов")


print("\nКопирование данных...")
copy_dataset(train, "train")
copy_dataset(val, "val")
copy_dataset(test, "test")

print("\nГотово! Разделение завершено.")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")