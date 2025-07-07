# YOLO
# Детекция пищевых продуктов с YOLOv11

Проект по обнаружению 5 классов продуктов: чай, лепешки, мясо, суп, салат.

## Как воспроизвести результаты

### 1. Системные требования
- ОС: Ubuntu 20.04+ / Windows 10+ (с WSL2)
- GPU: NVIDIA (минимум 8GB VRAM)
- Python 3.8-3.10
- CUDA 11.7+ (для GPU)

### 2. Установка зависимостей
```bash
# Клонировать репозиторий
git clone https://github.com/ваш-логин/food-detection-yolo.git
cd food-detection-yolo

# Установка Python-пакетов
pip install -r requirements.txt  # если есть файл
# Или вручную:
pip install torch==2.0.1 torchvision==0.15.2
pip install albumentations opencv-python ultralytics
```

### 3. Подготовка данных
1. Скачайте датасет: (https://disk.yandex.ru/d/-VhiX2BOWdw-rg)
2. Поместите в папку `data/videos`:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/

## Структура проекта
```

├── dataset/          # Данные
├── runs/             # Результаты обучения
├── sriptc/              # Основной код
│   ├── augments.py    # Аугментация
│   ├── automatic_separation    # Автоматическое разделение
│   └── extract_frames.py        # Выделение кадров
├── README.md         # Этот файл
```

## Результаты
| Метрика       | Значение |
|---------------|----------|
| mAP50         | 0.958    |
| mAP50-95      | 0.636    |
| Precision     | 0.966    |
| Recall        | 0.966    |


Все файлы, которые также должны быть в папке project, но появились проблемы с загрузкой из-за их веса находятся по ссылке - 
