# utils.py

import pandas as pd
import plotly.graph_objects as go
import numpy as np # Добавлен для обработки числовых данных

def validate_input(text):
    """Проверяет, что введенный текст является непустой строкой."""
    if not isinstance(text, str):
        return False
    # .strip() удаляет пробелы по краям, проверка что строка не пустая после этого
    if not text.strip():
        return False
    return True

def create_metrics_chart(scores):
    """
    Создает радарную диаграмму для визуализации метрик.
    Ограничивает значения диапазоном [0, 1] для корректного отображения.
    TER больше не обрабатывается отдельно, т.к. метрика удалена.
    """
    metrics_for_chart = []
    values_for_chart = []

    for metric, value in scores.items():
        # Пропускаем None или нечисловые значения
        if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
            continue

        # Обрезаем все значения до диапазона [0, 1]
        # Так как TER удален, специальная обработка для него не нужна
        chart_value = max(0.0, min(1.0, value))
        metrics_for_chart.append(metric)
        values_for_chart.append(chart_value)

    # Если нет данных для построения графика
    if not metrics_for_chart:
        return go.Figure().update_layout(title_text="Нет данных для отображения диаграммы")

    # Добавляем первую метрику в конец, чтобы замкнуть радар
    if len(metrics_for_chart) > 1:
        values_for_chart.append(values_for_chart[0])
        metrics_for_chart.append(metrics_for_chart[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_for_chart,
        theta=metrics_for_chart,
        fill='toself',
        name='Оценки',
        mode='lines+markers' # Отображаем линии и маркеры
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1], # Диапазон от 0 до 1
                tickvals=[0, 0.25, 0.5, 0.75, 1.0], # Деления на оси
                ticktext=['0', '0.25', '0.5', '0.75', '1.0'],
                angle=90,
                tickangle = 90
            ),
            angularaxis=dict(
                 tickfont_size=10
             )
        ),
        showlegend=False,
        title=dict(
            text='Обзор Оценок Качества Перевода',
            x=0.5
        ),
        margin=dict(l=60, r=60, t=80, b=40)
    )

    return fig


def create_results_df(scores):
    """Создает DataFrame Pandas с результатами (метрика и оценка)."""
    formatted_scores = []
    for metric, score in scores.items():
        if isinstance(score, (int, float)) and np.isfinite(score):
            formatted_scores.append(f"{score:.4f}") # Форматируем до 4 знаков
        elif score is None:
            formatted_scores.append("N/A")
        else:
            formatted_scores.append(str(score))

    return pd.DataFrame({
        'Метрика': list(scores.keys()),
        'Оценка': formatted_scores
    })


def get_metric_descriptions():
    """Возвращает словарь с описаниями оставшихся метрик (использует Markdown)."""
    return {
        'BLEU': """
        **BLEU (Bilingual Evaluation Understudy)** измеряет *точность* (precision) n-грамм (последовательностей из n слов) в переводе-кандидате по сравнению с эталонным(и) переводом(ами).
        * **Диапазон:** 0 - 1 (или 0 - 100).
        * **Чем выше, тем лучше.**
        * Хорошо коррелирует с оценками людей в среднем, но может штрафовать за синтаксические вариации и предпочитает более короткие переводы.
        """,

        'chrF': """
        **chrF (character n-gram F-score)** измеряет перекрытие символьных n-грамм (последовательностей из n символов). Вычисляет точность и полноту на основе этих n-грамм и затем их F-меру (гармоническое среднее).
        * **Диапазон:** 0 - 1 (или 0 - 100).
        * **Чем выше, тем лучше.**
        * Менее чувствителен к ошибкам токенизации, чем метрики на словах (BLEU), и часто лучше коррелирует с оценками людей для морфологически богатых языков.
        """,

        # Удалены описания для TER, METEOR, BEER

        'MAKTS': """
        **MAKTS v2 (Root-Weighted chrF)**: Эта версия MAKTS основана на метрике **chrF**, но улучшена с использованием морфологической информации от **Apertium**.
        * Она вычисляет перекрытие символьных n-грамм, как и chrF.
        * **Ключевая особенность:** Идентифицирует основы (корни) слов с помощью Apertium и присваивает **повышенный вес** (например, 2x) тем совпадающим n-граммам, которые полностью попадают в эти корневые зоны в *обоих* текстах (эталоне и кандидате).
        * N-граммы в суффиксальных частях или неразобранных словах получают стандартный вес (1x).
        * **Цель:** Лучше отражать семантическую близость, подчеркивая совпадения в основных частях слов, несущих смысл.
        * **Диапазон:** 0 - 1.
        * **Чем выше, тем лучше.**
        * **Зависимость:** Требует корректно установленных **Apertium** и языкового пакета `apertium-kaz`. Если Apertium недоступен, метрика вернет стандартное значение **chrF**.
        """
    }