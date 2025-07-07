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
├── Project/
    ├── dataset.yaml
    ├── dsa.py
    ├── hyp.finetune.yaml
    ├── project_tree.txt
    ├── yolo11n.pt
    └── ... (6 файлов)
    ├── .idea/
        ├── .gitignore
        ├── misc.xml
        ├── modules.xml
        ├── project.iml
        ├── workspace.xml
        ├── inspectionProfiles/
            ├── profiles_settings.xml
    ├── .venv/
        ├── .gitignore
        ├── CACHEDIR.TAG
        ├── pyvenv.cfg
        ├── Lib/
            ├── site-packages.rar
            ├── site-packages/
                ├── 30fcd23745efe32ce681__mypyc.cp310-win_amd64.pyd
                ├── appdirs.py
                ├── attr.py
                ├── distutils-precedence.pth
                ├── dry_attr.py
                └── ... (25 файлов)
        ├── Scripts/
            ├── activate
            ├── activate-global-python-argcomplete.exe
            ├── activate.bat
            ├── activate.fish
            ├── activate.nu
            └── ... (90 файлов)
            ├── __pycache__/
                ├── jp.cpython-310.pyc
        ├── share/
            ├── man/
    ├── data/
        ├── exported/
        ├── frames/
            ├── 1_frame_0000.jpg
            ├── 1_frame_0001.jpg
            ├── 1_frame_0002.jpg
            ├── 2_1_frame_0000.jpg
            ├── 2_1_frame_0001.jpg
            └── ... (162 файлов)
        ├── labels/
            ├── 1_frame_0002.txt
            ├── 2_1_frame_0000.txt
            ├── 2_1_frame_0001.txt
            ├── 2_1_frame_0002.txt
            ├── 2_1_frame_0003.txt
            └── ... (160 файлов)
        ├── videos/
            ├── 1.mp4
            ├── 2_1.mp4
            ├── 3_1.mp4
            ├── 3_2.mp4
            ├── 4.mp4
            └── ... (6 файлов)
    ├── dataset/
        ├── images/
            ├── test/
                ├── 2_1_frame_0014.jpg
                ├── 2_1_frame_0015.jpg
                ├── 3_1_frame_0000.jpg
                ├── 3_1_frame_0004.jpg
                ├── 3_1_frame_0005.jpg
                └── ... (24 файлов)
            ├── train/
                ├── 1_frame_0002.jpg
                ├── 2_1_frame_0000.jpg
                ├── 2_1_frame_0002.jpg
                ├── 2_1_frame_0003.jpg
                ├── 2_1_frame_0004.jpg
                └── ... (224 файлов)
            ├── val/
                ├── 2_1_frame_0001.jpg
                ├── 2_1_frame_0008.jpg
                ├── 2_1_frame_0011.jpg
                ├── 2_1_frame_0017.jpg
                ├── 2_1_frame_0018.jpg
                └── ... (24 файлов)
        ├── labels/
            ├── test.cache
            ├── train.cache
            ├── val.cache
            ├── test/
                ├── 2_1_frame_0014.txt
                ├── 2_1_frame_0015.txt
                ├── 3_1_frame_0000.txt
                ├── 3_1_frame_0004.txt
                ├── 3_1_frame_0005.txt
                └── ... (24 файлов)
            ├── train/
                ├── 1_frame_0002.txt
                ├── 2_1_frame_0000.txt
                ├── 2_1_frame_0002.txt
                ├── 2_1_frame_0003.txt
                ├── 2_1_frame_0004.txt
                └── ... (224 файлов)
            ├── val/
                ├── 2_1_frame_0001.txt
                ├── 2_1_frame_0008.txt
                ├── 2_1_frame_0011.txt
                ├── 2_1_frame_0017.txt
                ├── 2_1_frame_0018.txt
                └── ... (24 файлов)
    ├── my_project/
        ├── food_config.xml
    ├── runs/
        ├── detect/
            ├── food_detection/
                ├── args.yaml
                ├── BoxF1_curve.png
                ├── BoxPR_curve.png
                ├── BoxP_curve.png
                ├── BoxR_curve.png
                └── ... (18 файлов)
            ├── tune_v12/
                ├── args.yaml
                ├── BoxF1_curve.png
                ├── BoxPR_curve.png
                ├── BoxP_curve.png
                ├── BoxR_curve.png
                └── ... (18 файлов)
            ├── val/
                ├── BoxF1_curve.png
                ├── BoxPR_curve.png
                ├── BoxP_curve.png
                ├── BoxR_curve.png
                ├── confusion_matrix.png
                └── ... (10 файлов)
    ├── scripts/
        ├── augment.py
        ├── automatic_separation.py
        ├── compare.py
        ├── comparison_plot.png
        ├── extract_frames.py
        └── ... (6 файлов)

## Результаты
| Метрика       | Значение |
|---------------|----------|
| mAP50         | 0.958    |
| mAP50-95      | 0.636    |
| Precision     | 0.966    |
| Recall        | 0.966    |


Все файлы, которые также должны быть в папке project, но появились проблемы с загрузкой из-за их веса находятся по ссылке - https://drive.google.com/drive/folders/1faVWwDwNa60raa3Xem90_t5XwSxGwwgK?usp=sharing
[Document.pdf](https://github.com/user-attachments/files/21094541/Document.pdf)
[Document (1).pdf](https://github.com/user-attachments/files/21094805/Document.1.pdf)
