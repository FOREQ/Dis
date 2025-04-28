# main.py

import streamlit as st
import pandas as pd
# Импортируем только оставшиеся метрики
from metrics import (
    calculate_bleu,
    calculate_chrf,
    calculate_makts # НОВАЯ реализация MAKTS v2
)
# Импортируем утилиты
from utils import (
    validate_input,
    create_metrics_chart,
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
# Обновленное описание с оставшимися метриками
st.markdown("""
Этот инструмент помогает оценить качество перевода, используя метрики:
**BLEU**, **chrF** и **MAKTS v2** (взвешенный chrF с использованием морфологии).

**MAKTS v2** требует установленного **Apertium** и языкового пакета `apertium-kaz`.
Если Apertium недоступен, MAKTS вернет стандартное значение **chrF**.
""") # Убрали TER, METEOR, BEER и выбор языка

# --- Выбор Языковой Пары (УДАЛЕНО) ---
# language_pairs = [...]
# selected_pair = st.selectbox(...)
# source_language = selected_pair.split(...)[0].lower() # Больше не нужно

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
                upload_error = True # Устанавливаем флаг ошибки
    with col2:
        st.subheader("Файл перевода-кандидата (.txt)")
        candidate_file = st.file_uploader("Загрузите перевод-кандидат", type=['txt'], key="cand_file")
        if candidate_file:
            try:
                candidate_text = candidate_file.getvalue().decode("utf-8")
                st.text_area("Содержимое файла кандидата:", candidate_text, height=100, disabled=True, key="cand_file_content")
            except Exception as e:
                st.error(f"Ошибка чтения файла кандидата: {e}")
                upload_error = True # Устанавливаем флаг ошибки

# --- Кнопка Оценки и Расчеты ---
if st.button("Оценить перевод", disabled=upload_error):
    is_ref_valid = validate_input(reference_text)
    is_cand_valid = validate_input(candidate_text)

    if not is_ref_valid or not is_cand_valid:
        st.error("Пожалуйста, предоставьте и эталонный перевод, и перевод-кандидат (через текст или файл).")
    else:
        with st.spinner("Вычисление метрик... Это может занять некоторое время, особенно для MAKTS."):
            try:
                # Вычисляем только оставшиеся метрики
                # Убрали аргумент language из calculate_bleu, т.к. он больше не используется
                bleu_score = calculate_bleu(reference_text, candidate_text)
                chrf_score = calculate_chrf(reference_text, candidate_text)

                # Вычисляем новый MAKTS v2
                # Используем значение по умолчанию root_weight=2.0
                makts_score = calculate_makts(reference_text, candidate_text, root_weight=2.0)

                # Собираем результаты в словарь (только оставшиеся метрики)
                scores = {
                    'BLEU': bleu_score,
                    'chrF': chrf_score,
                    'MAKTS': makts_score
                }

                scores_filtered = {k: v for k, v in scores.items() if v is not None}
                scores_for_chart = scores_filtered.copy()

                # --- Отображение Результатов ---
                st.subheader("Результаты Оценки")
                col_res1, col_res2 = st.columns([2, 3])

                with col_res1:
                    # Таблица с результатами
                    if scores_filtered:
                        results_df = create_results_df(scores_filtered)
                        st.dataframe(results_df, hide_index=True, use_container_width=True)
                        # Кнопка скачивания CSV
                        try:
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Скачать результаты (CSV)",
                                data=csv,
                                file_name="translation_metrics.csv",
                                mime="text/csv",
                                key='download-csv'
                            )
                        except Exception as e:
                            st.error(f"Не удалось создать CSV: {e}")
                    else:
                        st.warning("Нет данных для отображения в таблице.")

                with col_res2:
                    # Радарная диаграмма
                    if scores_for_chart:
                        fig = create_metrics_chart(scores_for_chart)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Нет данных для отображения на диаграмме.")

                # --- Описания Метрик ---
                st.subheader("Описание Метрик")
                descriptions = get_metric_descriptions()
                displayed_metrics = list(scores_filtered.keys())

                for metric in displayed_metrics:
                     if metric in descriptions:
                         with st.expander(f"Подробнее о {metric}"):
                             st.markdown(descriptions[metric], unsafe_allow_html=True)
                     # else: # Описание должно быть в utils.py для оставшихся метрик
                         # st.warning(f"Описание для метрики {metric} не найдено.")

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