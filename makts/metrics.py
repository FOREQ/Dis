# metrics.py

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize # Можно удалить, если не используется в других функциях
import numpy as np
import re
from collections import Counter
import sacrebleu # Используется для стандартного chrF как fallback

# --- НОВЫЕ ИМПОРТЫ ---
try:
    import apertium
    APERTIUM_AVAILABLE = True
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ: Библиотека 'apertium-python' не найдена. MAKTS будет недоступен.")
    print("Для установки: pip install apertium-python (также требуется системная установка Apertium)")
    APERTIUM_AVAILABLE = False
# --- КОНЕЦ НОВЫХ ИМПОРТОВ ---


# --- Примечание об установке ---
# Этот код теперь требует библиотеку 'sacrebleu' для расчета chrF.
# Для нового MAKTS также требуется 'apertium-python' и системная установка Apertium + apertium-kaz.
# Вы можете установить их с помощью pip:
# pip install sacrebleu
# pip install apertium-python
# Инструкции по установке Apertium: https://wiki.apertium.org/wiki/Installation

# Download required NLTK data (оставляем для BLEU, если он нужен)
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download issue: {e}")

# --- Существующие функции (calculate_bleu, calculate_ter, calculate_meteor, calculate_beer, calculate_chrf) ---
# Оставляем без изменений, как в предыдущих ответах.
# Убедитесь, что у вас есть функция calculate_chrf, использующая sacrebleu.

# Пример функции calculate_chrf (если ее нет):
def calculate_chrf(reference, candidate):
    """
    Calculate chrF score between reference and candidate translations using sacrebleu.
    chrF measures character n-gram overlaps.
    """
    if not reference or not candidate:
        return 0.0
    try:
        chrf_score_obj = sacrebleu.sentence_chrf(candidate, [reference])
        return chrf_score_obj.score / 100.0
    except Exception as e:
        print(f"Error calculating chrF: {e}")
        return 0.0

# --- НОВЫЕ ФУНКЦИИ ДЛЯ MAKTS ---

# Глобальная переменная для анализатора, чтобы инициализировать один раз
kaz_analyzer_instance = None

def initialize_apertium_analyzer():
    """Инициализирует анализатор Apertium для казахского языка."""
    global kaz_analyzer_instance
    if not APERTIUM_AVAILABLE or kaz_analyzer_instance is not None:
        return kaz_analyzer_instance

    try:
        # Убедитесь, что 'kaz' - правильный идентификатор пакета в вашей системе Apertium
        kaz_analyzer_instance = apertium.Analyzer('kaz')
        print("Apertium Analyzer для 'kaz' успешно инициализирован.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Apertium Analyzer для 'kaz': {e}")
        print("Убедитесь, что Apertium и языковой пакет 'apertium-kaz' установлены и доступны.")
        kaz_analyzer_instance = None
    return kaz_analyzer_instance

def get_stems_with_spans(text, analyzer):
    """
    Анализирует текст с помощью Apertium и возвращает список кортежей:
    (лемма, начальный_индекс_в_тексте, конечный_индекс_в_тексте)
    Использует первый анализ и гипотезу префикса.
    """
    if not analyzer or not text:
        return []

    stems_info = []
    current_pos = 0
    # Используем re.finditer для поиска слов и пробелов/пунктуации между ними
    # Простое разделение по пробелу теряет информацию о позициях
    words_and_boundaries = re.finditer(r'(\S+)|(\s+)', text)

    for match in words_and_boundaries:
        token = match.group(0)
        start_index = match.start()
        end_index = match.end()

        # Обрабатываем только слова (\S+), пропускаем пробелы
        if match.group(1): # Если это слово (не пробел)
            surface_word = token
            try:
                analyses = analyzer.analyze(surface_word)
                # Выбираем первый анализ, если он есть
                analysis_to_parse = analyses[0] if isinstance(analyses, list) and analyses else str(analyses)

                lemma = None
                # Простой парсинг: ищем текст между первым '/' и первым '<'
                match_lemma = re.search(r'/([^<]+)', analysis_to_parse)
                if match_lemma:
                    lemma = match_lemma.group(1)

                if lemma:
                    # Применяем гипотезу префикса
                    stem_len = len(lemma)
                    # Проверка, что лемма действительно префикс и не длиннее слова
                    if surface_word.startswith(lemma) and stem_len <= len(surface_word):
                        stem_start_in_text = start_index
                        stem_end_in_text = start_index + stem_len
                        stems_info.append((lemma, stem_start_in_text, stem_end_in_text))
                    else:
                         # Лемма не является префиксом - обрабатываем как слово без основы?
                         # print(f"Предупреждение: Лемма '{lemma}' не является префиксом слова '{surface_word}'.")
                         pass # Не добавляем в stems_info, n-граммы из этого слова получат вес 1.0
                else:
                    # Не удалось извлечь лемму
                    # print(f"Предупреждение: Не удалось извлечь лемму из анализа '{analysis_to_parse}' для слова '{surface_word}'.")
                    pass # N-граммы из этого слова получат вес 1.0

            except Exception as e:
                print(f"Ошибка при анализе слова '{surface_word}' с Apertium: {e}")
                # Пропускаем слово, n-граммы получат вес 1.0
                pass
        # Обновляем позицию (это не требуется с finditer, start/end дают позиции)
        # current_pos = end_index

    return stems_info


def get_char_ngrams_with_pos(text, n):
    """Генерирует n-граммы с их начальными и конечными позициями в тексте."""
    if len(text) < n:
        return []
    return [(text[i:i+n], i, i+n) for i in range(len(text) - n + 1)]

def calculate_weighted_chrf(reference, candidate, ref_stems_info, cand_stems_info,
                            root_weight=2.0, n=6, beta=1.0):
    """
    Вычисляет взвешенный chrF.
    ref_stems_info, cand_stems_info: списки (lemma, start, end) от get_stems_with_spans.
    n: длина n-грамм.
    beta: параметр для F-меры (beta=1 для F1).
    """
    if not reference or not candidate:
        return 0.0

    # 1. Получить n-граммы с позициями
    ref_ngrams_pos = get_char_ngrams_with_pos(reference, n)
    cand_ngrams_pos = get_char_ngrams_with_pos(candidate, n)

    # 2. Определить вес каждой n-граммы (1.0 или root_weight)
    ref_ngram_weights = {} # ngram_str -> weight
    cand_ngram_weights = {} # ngram_str -> weight

    ref_stem_spans = set((s, e) for _, s, e in ref_stems_info) # Для быстрой проверки
    cand_stem_spans = set((s, e) for _, s, e in cand_stems_info) # Для быстрой проверки

    weighted_total_ref = 0.0
    for ngram, start, end in ref_ngrams_pos:
        is_root = False
        # Проверяем, попадает ли n-грамма ПОЛНОСТЬЮ в какой-либо корневой диапазон
        for stem_start, stem_end in ref_stem_spans:
            if start >= stem_start and end <= stem_end:
                is_root = True
                break
        weight = root_weight if is_root else 1.0
        ref_ngram_weights[ngram] = ref_ngram_weights.get(ngram, 0) + weight # Суммируем веса для одинаковых n-грамм? Или просто запоминаем? Лучше Counter.
        weighted_total_ref += weight

    weighted_total_cand = 0.0
    for ngram, start, end in cand_ngrams_pos:
        is_root = False
        for stem_start, stem_end in cand_stem_spans:
            if start >= stem_start and end <= stem_end:
                is_root = True
                break
        weight = root_weight if is_root else 1.0
        cand_ngram_weights[ngram] = cand_ngram_weights.get(ngram, 0) + weight
        weighted_total_cand += weight

    # Используем Counter для правильного подсчета (как в sacrebleu)
    ref_ngram_counts = Counter(ngram for ngram, _, _ in ref_ngrams_pos)
    cand_ngram_counts = Counter(ngram for ngram, _, _ in cand_ngrams_pos)

    # 3. Считаем взвешенные совпадения
    weighted_matches = 0.0
    # Итерируем по общим n-граммам
    common_ngrams = ref_ngram_counts.keys() & cand_ngram_counts.keys()

    for ngram in common_ngrams:
         # Определяем, считать ли ЭТО совпадение корневым
         # Нужен более сложный способ, чем просто проверка наличия в ref_ngram_weights/cand_ngram_weights
         # Нужно знать позицию КАЖДОГО вхождения n-граммы.
         # --- УПРОЩЕНИЕ: Применим средний вес для n-граммы ---
         # (Это не совсем точно отражает идею, но проще в реализации)
        avg_ref_weight = ref_ngram_weights.get(ngram, 1.0 * ref_ngram_counts[ngram]) / ref_ngram_counts[ngram]
        avg_cand_weight = cand_ngram_weights.get(ngram, 1.0 * cand_ngram_counts[ngram]) / cand_ngram_counts[ngram]

        # Если n-грамма в среднем корневая в обоих, применяем root_weight?
        # Или просто берем минимальное количество совпадений и умножаем на усредненный вес?
        # Логика chrF использует min(count_ref, count_cand)
        matches_count = min(ref_ngram_counts[ngram], cand_ngram_counts[ngram])

        # Какой вес применить к этим совпадениям?
        # Вариант 1: Усредненный вес (приблизительно)
        # match_weight = (avg_ref_weight + avg_cand_weight) / 2.0
        # Вариант 2: Минимальный вес (более консервативно)
        match_weight = min(avg_ref_weight, avg_cand_weight)
        # Вариант 3: Максимальный? Или вес=root_weight если оба > 1?

        # --- Используем Вариант 1 (усредненный) ---
        # weighted_matches += matches_count * match_weight
        # --- Используем Вариант 2 (минимальный) ---
        weighted_matches += matches_count * match_weight

    # 4. Считаем взвешенные P, R, F
    if weighted_total_cand == 0 or weighted_total_ref == 0:
        return 0.0

    P_w = weighted_matches / weighted_total_cand
    R_w = weighted_matches / weighted_total_ref

    if P_w == 0 and R_w == 0:
        return 0.0

    # F-beta score
    score = ((1 + beta**2) * P_w * R_w) / ((beta**2 * P_w) + R_w)
    return score


# --- НОВАЯ РЕАЛИЗАЦИЯ calculate_makts ---
def calculate_makts(reference, candidate, root_weight=2.0, n=6, beta=1.0):
    """
    Вычисляет Morpheme-Aware Kazakh Translation Score (MAKTS) v2.
    Основан на chrF с взвешиванием n-грамм в корневых зонах слов.
    Требует установленного Apertium и языкового пакета 'kaz'.
    """
    analyzer = initialize_apertium_analyzer()

    if not analyzer:
        print("ПРЕДУПРЕЖДЕНИЕ: Apertium недоступен. Возвращаем стандартный chrF.")
        # Возвращаем стандартный chrF как запасной вариант
        return calculate_chrf(reference, candidate)

    # Получаем информацию об основах и их позициях
    ref_stems_info = get_stems_with_spans(reference, analyzer)
    cand_stems_info = get_stems_with_spans(candidate, analyzer)

    # Вычисляем взвешенный chrF
    score = calculate_weighted_chrf(reference, candidate, ref_stems_info, cand_stems_info,
                                    root_weight=root_weight, n=n, beta=beta)

    return score

# --- КОНЕЦ НОВЫХ ФУНКЦИЙ ---

# --- Старые функции MAKTS (segment_kazakh_word_all, select_best_variant, etc.) ---
# Можно удалить или закомментировать, так как calculate_makts теперь использует новую логику.
# def segment_kazakh_word_all(word, suffixes=None): ...
# def select_best_variant(variants): ...
# def calculate_metric(stem, suffixes, weights): ...
# def get_morph_score(word, weights): ...
# def calculate_makts_old(reference, candidate, weights=None): ... # Переименовал старую