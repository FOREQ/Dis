# main.py

import streamlit as st
import pandas as pd
# Импортируем только нужные метрики
from metrics import (
    calculate_bleu,
    calculate_chrf,
    calculate_makts # НОВАЯ реализация MAKTS v2
)
# Импортируем утилиты
from utils import (
    validate_input,
    # create_metrics_chart, # Больше не используется
    create_results_df,
    get_metric_descriptions # Должен содержать описания только для BLEU, chrF, MAKTS
)

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="Оценка Качества Перевода",
    page_icon="⚖️",
    layout="wide"
)

# --- Заголовок и Описание ---
st.title("Инструмент Оценки Качества Перевода")
st.markdown("""
Этот инструмент помогает оценить качество перевода, используя метрики:
**BLEU**, **chrF** и **MAKTS v2** (взвешенный chrF с использованием морфологии).

**MAKTS v2** требует установленного **Apertium** и языкового пакета `apertium-kaz`.
Если Apertium недоступен, MAKTS вернет стандартное значение **chrF**.
""")

# --- Выбор Способа Ввода ---
input_method = st.radio(
    "Выберите способ ввода текста",
    ["Прямой ввод текста", "Загрузка файла"]
)

reference_text = ""
candidate_text = ""
upload_error = False # Флаг ошибки загрузки файла

if input_method == "Прямой ввод текста":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Эталонный перевод (Reference)")
        reference_text = st.text_area("Введите эталонный перевод", height=150, key="ref_text")
    with col2:
        st.subheader("Перевод-кандидат (Candidate)")
        candidate_text = st.text_area("Введите перевод-кандидат", height=150, key="cand_text")
else: # Загрузка файла
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Файл эталонного перевода (.txt)")
        reference_file = st.file_uploader("Загрузите эталонный перевод", type=['txt'], key="ref_file")
        if reference_file:
            try:
                reference_text = reference_file.getvalue().decode("utf-8")
                st.text_area("Содержимое эталонного файла:", reference_text, height=100, disabled=True, key="ref_file_content")
            except Exception as e:
                st.error(f"Ошибка чтения файла эталона: {e}")
                upload_error = True
    with col2:
        st.subheader("Файл перевода-кандидата (.txt)")
        candidate_file = st.file_uploader("Загрузите перевод-кандидат", type=['txt'], key="cand_file")
        if candidate_file:
            try:
                candidate_text = candidate_file.getvalue().decode("utf-8")
                st.text_area("Содержимое файла кандидата:", candidate_text, height=100, disabled=True, key="cand_file_content")
            except Exception as e:
                st.error(f"Ошибка чтения файла кандидата: {e}")
                upload_error = True

# --- Кнопка Оценки и Расчеты ---
if st.button("Оценить перевод", disabled=upload_error):
    is_ref_valid = validate_input(reference_text)
    is_cand_valid = validate_input(candidate_text)

    if not is_ref_valid or not is_cand_valid:
        st.error("Пожалуйста, предоставьте и эталонный перевод, и перевод-кандидат (через текст или файл).")
    else:
        with st.spinner("Вычисление метрик... Это может занять некоторое время, особенно для MAKTS."):
            try:
                # Вычисляем стандартные метрики
                bleu_score = calculate_bleu(reference_text, candidate_text)
                chrf_score = calculate_chrf(reference_text, candidate_text)

                # --- Вычисляем MAKTS с разными весами ---
                makts_weights = [1.5, 2.0, 2.5, 3.0] # Задаем веса для тестирования
                makts_scores = {}
                #st.write("--- DEBUG: Начинаю расчет MAKTS с разными весами ---") # Отладка
                for weight in makts_weights:
                    #st.write(f"--- DEBUG: Расчет MAKTS для root_weight={weight} ---") # Отладка
                    makts_scores[f'MAKTS (w={weight})'] = calculate_makts(
                        reference_text,
                        candidate_text,
                        root_weight=weight
                    )
                    #st.write(f"--- DEBUG: Результат MAKTS (w={weight}): {makts_scores[f'MAKTS (w={weight})']:.4f} ---") # Отладка
                # --- Конец расчета MAKTS с разными весами ---

                # Собираем основные результаты в словарь
                scores = {
                    'BLEU': bleu_score,
                    'chrF': chrf_score,
                    'MAKTS (w=2.0)': makts_scores.get('MAKTS (w=2.0)', None) # Берем значение по умолчанию из рассчитанных
                }

                # Убираем None, если какая-то метрика вернула None
                scores_filtered = {k: v for k, v in scores.items() if v is not None}

                # --- Отображение Результатов ---
                st.subheader("Результаты Оценки")

                # Используем две колонки, как и раньше
                col_res1, col_res2 = st.columns([2, 3]) # Можно настроить ширину

                with col_res1:
                    st.markdown("##### Основные метрики:") # Заголовок для первой таблицы
                    # Таблица с основными результатами (BLEU, chrF, MAKTS с w=2.0)
                    if scores_filtered:
                        results_df = create_results_df(scores_filtered)
                        st.dataframe(results_df, hide_index=True, use_container_width=True)

                        # Кнопка скачивания CSV для основных метрик
                        try:
                            csv = results_df.to_csv(index=False).encode('utf-8-sig') # Используем utf-8-sig для Excel
                            st.download_button(
                                label="Скачать основные результаты (CSV)",
                                data=csv,
                                file_name="translation_metrics_main.csv",
                                mime="text/csv",
                                key='download-csv-main'
                            )
                        except Exception as e:
                            st.error(f"Не удалось создать CSV: {e}")
                    else:
                        st.warning("Нет данных для отображения в таблице.")

                with col_res2:
                    # --- Убрали Радарную Диаграмму ---
                    # if scores_for_chart:
                    #     fig = create_metrics_chart(scores_for_chart) # create_metrics_chart больше не нужна
                    #     st.plotly_chart(fig, use_container_width=True)
                    # else:
                    #     st.warning("Нет данных для отображения на диаграмме.")

                    # --- Добавили Таблицу Сравнения Весов MAKTS ---
                    st.markdown("##### Сравнение MAKTS с разными весами корня (`root_weight`):") # Заголовок для второй таблицы
                    if makts_scores:
                        # Создаем DataFrame для сравнения весов MAKTS
                        makts_compare_df = pd.DataFrame({
                            'Вес корня (root_weight)': [f"{w}" for w in makts_weights],
                            'Оценка MAKTS': [f"{makts_scores.get(f'MAKTS (w={w})', 'N/A'):.4f}" if isinstance(makts_scores.get(f'MAKTS (w={w})'), float) else 'N/A' for w in makts_weights]
                        })
                        st.dataframe(makts_compare_df, hide_index=True, use_container_width=True)

                        # Кнопка скачивания CSV для сравнения весов
                        try:
                            csv_makts = makts_compare_df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="Скачать сравнение весов MAKTS (CSV)",
                                data=csv_makts,
                                file_name="translation_metrics_makts_weights.csv",
                                mime="text/csv",
                                key='download-csv-makts'
                            )
                        except Exception as e:
                            st.error(f"Не удалось создать CSV для MAKTS: {e}")
                    else:
                        st.warning("Не удалось рассчитать MAKTS для сравнения весов.")
                    # --- Конец Таблицы Сравнения Весов MAKTS ---


                # --- Описания Метрик (Остаются под таблицами) ---
                st.subheader("Описание Метрик")
                descriptions = get_metric_descriptions() # Должен содержать только BLEU, chrF, MAKTS
                displayed_metrics = list(scores_filtered.keys())

                for metric in displayed_metrics:
                     # Специально показываем общее описание MAKTS
                     metric_key = 'MAKTS' if metric.startswith('MAKTS') else metric
                     if metric_key in descriptions:
                         with st.expander(f"Подробнее о {metric_key}"):
                             st.markdown(descriptions[metric_key], unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Произошла ошибка при вычислении метрик: {e}")
                st.exception(e)


# --- Подвал (Footer) ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey;'>
    <p>Инструмент Оценки Качества Перевода - 2025</p>
</div>
""", unsafe_allow_html=True)
