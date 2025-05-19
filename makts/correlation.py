import pandas as pd
from scipy.stats import pearsonr
import io

# --- Конфигурация ---
file_path = 'сalculation.xlsx'

# --- Имена столбцов ---
# !!! ИЗМЕНЕНО: Указываем точные имена из вашего файла !!!
human_score_col = 'DA_score'     # Столбец с оценками человека
metric1_col = 'chrF_score'     # Столбец с метрикой chrF
metric2_col = 'MAKTS_score'  # Столбец с вашей метрикой Makts

# --- Загрузка данных ---
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"Файл '{file_path}' успешно загружен.")
    # print(f"Найденные столбцы: {df.columns.tolist()}") # Можно раскомментировать для доп. проверки

except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден. Убедитесь, что он находится в нужной директории и имя указано верно.")
    exit()
except ImportError:
    print(f"Ошибка: Библиотека 'openpyxl' не найдена. Пожалуйста, установите ее: pip install openpyxl")
    exit()
except Exception as e:
    print(f"Ошибка при чтении файла '{file_path}': {e}")
    exit()

# --- Проверка наличия необходимых столбцов ---
required_cols = [human_score_col, metric1_col, metric2_col]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"Ошибка: В файле отсутствуют необходимые столбцы: {', '.join(missing_cols)}")
    print(f"Доступные столбцы: {df.columns.tolist()}")
    exit()

# --- Очистка данных (удаление строк с NaN и преобразование в число) ---
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_cleaned = df[required_cols].dropna()

if len(df_cleaned) < len(df):
    rows_dropped = len(df) - len(df_cleaned)
    print(f"\nПредупреждение: Удалено {rows_dropped} строк из-за пропущенных или нечисловых значений в столбцах {required_cols}.")

if len(df_cleaned) < 2:
    print("Ошибка: Недостаточно данных для расчета корреляции после очистки.")
    exit()
else:
    print(f"Используется {len(df_cleaned)} строк для расчета корреляции.")


# --- Расчет корреляции Пирсона ---

print("\n--- Результаты корреляции Пирсона ---")

correlation_chrf = None
correlation_makts = None

# 1. Корреляция между DA и chrF
try:
    correlation_chrf, p_value_chrf = pearsonr(df_cleaned[human_score_col], df_cleaned[metric1_col])
    print(f"\nКорреляция между '{human_score_col}' и '{metric1_col}':")
    print(f"  Коэффициент Пирсона (r): {correlation_chrf:.4f}")
    print(f"  p-значение: {p_value_chrf:.4g}")
    significant_chrf = p_value_chrf < 0.05
    print(f"  Статистически значима (p < 0.05): {'Да' if significant_chrf else 'Нет'}")

except ValueError as e:
    print(f"\nНе удалось рассчитать корреляцию для '{metric1_col}': {e}")


# 2. Корреляция между DA и Makts
try:
    correlation_makts, p_value_makts = pearsonr(df_cleaned[human_score_col], df_cleaned[metric2_col])
    print(f"\nКорреляция между '{human_score_col}' и '{metric2_col}':")
    print(f"  Коэффициент Пирсона (r): {correlation_makts:.4f}")
    print(f"  p-значение: {p_value_makts:.4g}")
    significant_makts = p_value_makts < 0.05
    print(f"  Статистически значима (p < 0.05): {'Да' if significant_makts else 'Нет'}")

except ValueError as e:
    print(f"\nНе удалось рассчитать корреляцию для '{metric2_col}': {e}")


# --- Сравнение метрик ---
print("\n--- Сравнение корреляций с DA ---")

if correlation_chrf is not None and correlation_makts is not None:
    abs_corr_chrf = abs(correlation_chrf)
    abs_corr_makts = abs(correlation_makts)

    print(f"Абсолютная корреляция |{human_score_col} vs {metric1_col}|: {abs_corr_chrf:.4f}")
    print(f"Абсолютная корреляция |{human_score_col} vs {metric2_col}|: {abs_corr_makts:.4f}")

    if abs_corr_makts > abs_corr_chrf:
        print(f"\nВывод: Метрика '{metric2_col}' показывает БОЛЕЕ СИЛЬНУЮ корреляцию с '{human_score_col}', чем '{metric1_col}'.")
    elif abs_corr_chrf > abs_corr_makts:
         print(f"\nВывод: Метрика '{metric1_col}' показывает БОЛЕЕ СИЛЬНУЮ корреляцию с '{human_score_col}', чем '{metric2_col}'.")
    else:
        print(f"\nВывод: Метрики '{metric1_col}' и '{metric2_col}' показывают ОДИНАКОВУЮ силу корреляции с '{human_score_col}'.")
else:
    print("Не удалось рассчитать обе корреляции для сравнения.")

print("\n--- Завершено ---")