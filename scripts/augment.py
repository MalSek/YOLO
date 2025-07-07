import cv2
import os
import albumentations as A
import random
from glob import glob

# Пути к данным
DATASET_DIR = "E:/Project/dataset"
TRAIN_IMAGES = os.path.join(DATASET_DIR, "images/train")
TRAIN_LABELS = os.path.join(DATASET_DIR, "labels/train")

# Оптимизированные аугментации для Albumentations 2.0.8
transform = A.Compose([
    # Геометрические преобразования
    A.HorizontalFlip(p=0.5),
    A.Affine(
        rotate=(-15, 15),
        translate_percent=(0, 0.05),
        scale=(0.9, 1.1),
        shear=(-5, 5),
        p=0.5
    ),

    # Цветовые преобразования
    A.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.05,
        p=0.7
    ),

    # Альтернативные эффекты
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.ChannelShuffle(p=0.1),

    # Безопасные трансформации для версии 2.0.8
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.ToGray(p=0.1)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

# Создаем папки для аугментированных данных
os.makedirs(os.path.join(TRAIN_IMAGES, "augmented"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_LABELS, "augmented"), exist_ok=True)


def process_image(image_path, label_path):
    try:
        # Чтение изображения и аннотаций
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_ids = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, xc, yc, w, h = map(float, parts)
                    bboxes.append([xc, yc, w, h])
                    class_ids.append(int(class_id))

        # Применение аугментаций
        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_ids=class_ids
        )

        if transformed['bboxes']:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            aug_name = f"aug_{base_name}_{random.randint(1000, 9999)}"

            # Сохранение изображения
            cv2.imwrite(
                os.path.join(TRAIN_IMAGES, "augmented", f"{aug_name}.jpg"),
                cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            )

            # Сохранение аннотаций
            with open(os.path.join(TRAIN_LABELS, "augmented", f"{aug_name}.txt"), 'w') as f:
                for class_id, bbox in zip(transformed['class_ids'], transformed['bboxes']):
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")


# Обработка всех изображений
image_files = glob(os.path.join(TRAIN_IMAGES, "*.jpg"))
print(f"Найдено {len(image_files)} изображений для обработки")

for img_path in image_files:
    label_path = os.path.join(TRAIN_LABELS, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    if os.path.exists(label_path):
        process_image(img_path, label_path)
    else:
        print(f"Предупреждение: Не найден файл разметки для {img_path}")

augmented_count = len(glob(os.path.join(TRAIN_IMAGES, "augmented", "*.jpg")))
print(f" Аугментация завершена! Создано {augmented_count} новых изображений")


def show_random_augmented_samples(n=5):
    "Визуализация случайных аугментированных образцов"
    aug_images = glob(os.path.join(TRAIN_IMAGES, "augmented", "*.jpg"))
    if not aug_images:
        print("Нет аугментированных изображений для отображения")
        return

    class_names = {0: "Чай", 1: "Лепешки", 2: "Мясо", 3: "Суп", 4: "Салат"}
    class_colors = {
        0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
        3: (255, 255, 0), 4: (255, 0, 255)
    }

    for img_path in random.sample(aug_images, min(n, len(aug_images))):
        label_path = os.path.join(TRAIN_LABELS, "augmented", os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, xc, yc, bw, bh = map(float, parts)
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    color = class_colors.get(int(class_id), (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, class_names.get(int(class_id), "Unknown"),
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow(f"Augmented: {os.path.basename(img_path)}", img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


show_random_augmented_samples(10)