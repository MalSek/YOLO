import cv2
import os
from tqdm import tqdm

# Конфигурация
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'videos')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'frames')
EXTRACT_FRAME_EVERY_SEC = 2
SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')

def extract_frames(video_path, output_dir, interval_sec=2):
    """Извлекает кадры из видео с заданным интервалом."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps * interval_sec))

        saved_count = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        os.makedirs(output_dir, exist_ok=True)

        for frame_count in tqdm(range(total_frames), desc=f"Обработка {video_name}"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
                if not cv2.imwrite(frame_path, frame):
                    print(f"Ошибка: Не удалось сохранить кадр {frame_path}")
                saved_count += 1

        cap.release()
        return saved_count

    except Exception as e:
        print(f"Ошибка при обработке {video_path}: {str(e)}")
        return 0

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Исходные видео: {INPUT_DIR}")
    print(f"Выходные кадры: {OUTPUT_DIR}")

    for video_file in os.listdir(INPUT_DIR):
        if video_file.lower().endswith(SUPPORTED_FORMATS):
            video_path = os.path.join(INPUT_DIR, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(OUTPUT_DIR, video_name)

            print(f"\nОбработка видео: {video_file}")
            saved = extract_frames(video_path, video_output_dir, EXTRACT_FRAME_EVERY_SEC)
            print(f"Сохранено кадров: {saved}")