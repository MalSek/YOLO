import pandas as pd
import matplotlib.pyplot as plt
import os


def load_and_compare(path1, path2):
    # Проверка существования файлов
    if not os.path.exists(path1):
        raise FileNotFoundError(f"Файл {path1} не найден! Проверьте путь.")
    if not os.path.exists(path2):
        raise FileNotFoundError(f"Файл {path2} не найден! Проверьте путь.")

    # Загрузка данных с обработкой возможных ошибок
    try:
        df1 = pd.read_csv(path1, sep='[,;\t|]', engine='python')
        df2 = pd.read_csv(path2, sep='[,;\t|]', engine='python')
    except Exception as e:
        print(f"Ошибка при чтении CSV: {str(e)}")
        print("\nПодсказка: Попробуйте открыть файл вручную и проверить формат (разделители, кодировку).")
        return

    # Функция для поиска столбцов по шаблонам
    def find_metric(df, patterns):
        for col in df.columns:
            col_lower = col.lower()
            if any(p.lower() in col_lower for p in patterns):
                return col
        return None

    # Ищем метрики
    metrics = {
        'mAP50': ['mAP50', 'map50'],
        'mAP50-95': ['mAP50-95', 'map_5095'],
        'precision': ['precision', 'prec'],
        'recall': ['recall', 'rec']
    }

    results = {}
    for name, patterns in metrics.items():
        col1 = find_metric(df1, patterns)
        col2 = find_metric(df2, patterns)

        if col1 and col2:
            results[name] = {
                'Tune_v12': df1[col1].iloc[-1],
                'Food_detection': df2[col2].iloc[-1],
                'Difference': df1[col1].iloc[-1] - df2[col2].iloc[-1]
            }

    # Вывод результатов
    if results:
        print("\nСравнение метрик (последняя эпоха):")
        result_df = pd.DataFrame(results).T
        print(result_df)

        # Сохранение результатов в CSV
        result_df.to_csv("metrics_comparison.csv")
        print("\nРезультаты сохранены в metrics_comparison.csv")
    else:
        print("\nНе удалось найти совпадающие метрики в файлах.")
        print("Доступные столбцы в tune_v12:", df1.columns.tolist())
        print("Доступные столбцы в food_detection:", df2.columns.tolist())

    # Построение графиков
    if 'mAP50-95' in results:
        plt.figure(figsize=(12, 5))

        # Поиск столбца с эпохами
        epoch_col1 = find_metric(df1, ['epoch', 'epoche', 'ep'])
        epoch_col2 = find_metric(df2, ['epoch', 'epoche', 'ep'])

        if epoch_col1 and epoch_col2:
            # График mAP
            plt.subplot(121)
            plt.plot(df1[epoch_col1], df1[find_metric(df1, ['mAP50-95'])], label="Tune_v12")
            plt.plot(df2[epoch_col2], df2[find_metric(df2, ['mAP50-95'])], label="Food_detection")
            plt.xlabel("Эпохи")
            plt.ylabel("mAP50-95")
            plt.title("Сравнение mAP50-95")
            plt.legend()

            # График потерь
            plt.subplot(122)
            loss_col1 = find_metric(df1, ['train/loss', 'loss', 'box_loss'])
            loss_col2 = find_metric(df2, ['train/loss', 'loss', 'box_loss'])
            if loss_col1 and loss_col2:
                plt.plot(df1[epoch_col1], df1[loss_col1], label="Tune_v12 train")
                plt.plot(df2[epoch_col2], df2[loss_col2], label="Food_detection train")
                plt.xlabel("Эпохи")
                plt.ylabel("Loss")
                plt.title("Сравнение потерь")
                plt.legend()

            plt.tight_layout()
            plt.savefig("comparison_plot.png", dpi=300)
            print("\nГрафики сохранены в comparison_plot.png")
        else:
            print("\nНе удалось найти данные об эпохах для построения графиков.")


# Указываем пути к файлам
tune_path = "E:/Project/runs/detect/tune_v12/results.csv"
food_path = "E:/Project/runs/detect/food_detection/results.csv"

# Проверяем существование файлов перед запуском
print("Проверка файлов:")
print(f"- Tune_v12: {'найден' if os.path.exists(tune_path) else 'не найден'}")
print(f"- Food_detection: {'найден' if os.path.exists(food_path) else 'не найден'}")

if os.path.exists(tune_path) and os.path.exists(food_path):
    load_and_compare(tune_path, food_path)
else:
    print("\nОшибка: Один или оба файла не найдены.")
    print("Проверьте пути:")
    print(f"1. {tune_path}")
    print(f"2. {food_path}")
    print("\nСовет: Запустите detect.py для генерации отсутствующих файлов.")