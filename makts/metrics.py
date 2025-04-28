# metrics.py

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import re
from collections import Counter
import sacrebleu # Для chrF и как fallback для MAKTS

# --- Импорты для MAKTS v2 ---
try:
    import apertium
    # streamparser используется для объекта LexicalUnit, который возвращает apertium.Analyzer
    from streamparser import LexicalUnit, SReading
    APERTIUM_AVAILABLE = True
except ImportError:
    print("="*80)
    # Исправленный текст предупреждения
    print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать 'apertium' или 'streamparser'.")
    print("Возможно, пакет 'apertium'/'apertium-streamparser' для Python не установлен")
    print("(pip install apertium apertium-streamparser)")
    print("или отсутствуют необходимые системные компоненты Apertium.")
    print("Новая метрика MAKTS (взвешенный chrF) будет недоступна и вернет стандартный chrF.")
    print("ВАЖНО: Требуется системная установка Apertium core, apertium-kaz и python3-apertium-core.")
    print("См. инструкции по установке Apertium: https://wiki.apertium.org/wiki/Installation")
    print("="*80)
    APERTIUM_AVAILABLE = False
# --- КОНЕЦ ИМПОРТОВ для MAKTS v2 ---


# --- Примечание об установке ---
# Этот код требует:
# - nltk: pip install nltk
# - numpy: pip install numpy
# - sacrebleu: pip install sacrebleu
# - apertium: pip install apertium (Python-обертка)
# - apertium-streamparser: pip install apertium-streamparser
# - Системная установка: Apertium core, apertium-kaz, python3-apertium-core

# Download required NLTK data (оставляем для BLEU)
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download issue: {e}")

# ==============================================================================
# Вспомогательные функции (Только нужные для оставшихся метрик)
# ==============================================================================

def get_char_ngrams_with_pos(text, n):
    """Генерирует n-граммы с их начальными и конечными позициями в тексте."""
    if not isinstance(text, str) or len(text) < n:
        return []
    text_lower = text.lower() # Сравнение без учета регистра
    return [(text_lower[i:i+n], i, i+n) for i in range(len(text_lower) - n + 1)]

# ==============================================================================
# Функции расчета метрик (Оставлены BLEU, chrF, MAKTS v2)
# ==============================================================================

def calculate_bleu(reference, candidate, language='english'):
    """
    Calculate BLEU score between reference and candidate translations.
    Uses simple whitespace tokenization for robustness.
    """
    if not reference or not candidate: return 0.0
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    if not candidate_tokens: return 0.0
    if not reference_tokens: return 0.0 # Return 0 if reference is empty
    reference_list = [reference_tokens]
    smoothing = SmoothingFunction().method1 # Standard smoothing for sentence BLEU
    try:
        # Ensure score is within [0, 1] range
        bleu_score = sentence_bleu(reference_list, candidate_tokens, smoothing_function=smoothing, auto_reweigh=True) # auto_reweigh helps with short sentences
        return max(0.0, min(bleu_score, 1.0))
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0.0

def calculate_chrf(reference, candidate):
    """
    Calculate chrF score between reference and candidate translations using sacrebleu.
    """
    if not reference or not candidate: return 0.0
    try:
        # sentence_chrf uses default parameters (char_order=6, word_order=0, beta=1)
        chrf_score_obj = sacrebleu.sentence_chrf(candidate, [reference])
        # Score is 0-100, scale to 0-1
        return chrf_score_obj.score / 100.0
    except Exception as e:
        print(f"Error calculating chrF: {e}")
        return 0.0

# ==============================================================================
# НОВЫЙ MAKTS v2 (Root-Weighted chrF) с зависимостью от Apertium
# ==============================================================================

# Глобальная переменная для анализатора
kaz_analyzer_instance = None
apertium_init_attempted = False

def initialize_apertium_analyzer():
    """Инициализирует анализатор Apertium для казахского языка (если доступен)."""
    global kaz_analyzer_instance, apertium_init_attempted
    if not APERTIUM_AVAILABLE or apertium_init_attempted:
        return kaz_analyzer_instance

    apertium_init_attempted = True
    try:
        kaz_analyzer_instance = apertium.Analyzer('kaz')
        print("Info: Apertium Analyzer для 'kaz' успешно инициализирован.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Apertium Analyzer для 'kaz': {e}")
        kaz_analyzer_instance = None
    return kaz_analyzer_instance

def get_stems_with_spans(text, analyzer):
    """
    Анализирует текст с помощью Apertium и возвращает список кортежей:
    (лемма, начальный_индекс_в_тексте, конечный_индекс_в_тексте).
    Использует эвристику для выбора лучшей леммы из ВСЕХ анализов Apertium.
    Версия с исправленной обработкой LexicalUnit и без проверки префикса startswith.
    """
    if not analyzer or not text: return []
    stems_info = []

    for match in re.finditer(r'(\S+)|(\s+)', text):
        token = match.group(0)
        start_index = match.start()

        if match.group(1): # Process only non-whitespace tokens
            word_token_from_regex = token
            # print(f"\n+++ Processing Word: '{word_token_from_regex}' at index {start_index}") # DEBUG WORD
            try:
                analyses = analyzer.analyze(word_token_from_regex) # Returns List[LexicalUnit]

                extracted_pairs = [] # Collect all possible (lemma, surface) pairs

                if analyses and isinstance(analyses, list):
                    # Assume the list contains ONE main LexicalUnit for the word form
                    lu = analyses[0] # Take the first LexicalUnit object
                    # Ensure it's the correct type using the imported class
                    if not isinstance(lu, LexicalUnit): continue

                    surface = lu.wordform if hasattr(lu, 'wordform') else word_token_from_regex

                    if hasattr(lu, 'readings') and lu.readings:
                        # Iterate through all analysis paths provided
                        for analysis_path in lu.readings:
                            # Check if path is valid and get the first segment (SReading)
                            if isinstance(analysis_path, list) and analysis_path and isinstance(analysis_path[0], SReading):
                                first_segment = analysis_path[0]
                                if hasattr(first_segment, 'baseform'):
                                    lemma = first_segment.baseform
                                    # Add valid string pairs
                                    if lemma and surface and isinstance(lemma, str) and isinstance(surface, str):
                                        extracted_pairs.append((lemma, surface))

                # --- Apply heuristic to choose the best lemma ---
                best_lemma = None
                best_surface = None
                unique_extracted_pairs = list(set(extracted_pairs))

                if not unique_extracted_pairs: continue # Skip if no valid pairs found

                # Find lemmas shorter than the surface form
                shorter_lemmas = [(lem, surf) for lem, surf in unique_extracted_pairs if len(lem) < len(surf)]

                if shorter_lemmas:
                    # Choose the pair with the longest lemma among the shorter ones
                    best_lemma, best_surface = max(shorter_lemmas, key=lambda item: len(item[0]))
                else:
                    # Fallback: choose the first extracted pair (might be same as surface)
                    best_lemma, best_surface = unique_extracted_pairs[0]
                # --- End of heuristic ---

                # Apply length check only (prefix check removed)
                stem_len = len(best_lemma)
                if stem_len <= len(best_surface): # Lemma should not be longer than surface
                    stem_start_in_text = start_index
                    stem_end_in_text = start_index + stem_len # Span determined by lemma length
                    stems_info.append((best_lemma, stem_start_in_text, stem_end_in_text))
                    # print(f"    >>> SUCCESS (Prefix check ignored): Stem added: ('{best_lemma}', {stem_start_in_text}, {stem_end_in_text})") # DEBUG SUCCESS
                # else:
                    # print(f"    !!! Length check failed for lemma '{best_lemma}' and surface '{best_surface}'") # DEBUG LEN FAIL

            except Exception as e:
                print(f"  ERROR processing word '{word_token_from_regex}': {e}")
                pass

    # Final output of extracted stems (optional control print)
    if stems_info:
        print(f"--- Extracted stems for: '{text[:50]}...'")
        for lem, s, e in stems_info:
            print(f"  Lemma: '{lem}', Span: ({s}, {e}), Text: '{text[s:e]}'")
    else:
        print(f"--- *** No stems extracted for text: '{text[:50]}...' ***")
    return stems_info


def calculate_weighted_chrf(reference, candidate, ref_stems_info, cand_stems_info,
                            root_weight=2.0, n=6, beta=1.0):
    """
    Вычисляет взвешенный chrF.
    Версия с "агрессивной" логикой взвешивания совпадений (индивидуальная проверка).
    """
    if not reference or not candidate: return 0.0

    ref_ngrams_pos = get_char_ngrams_with_pos(reference, n)
    cand_ngrams_pos = get_char_ngrams_with_pos(candidate, n)
    if not ref_ngrams_pos or not cand_ngrams_pos: return 0.0

    # --- Определение корневых зон ---
    ref_is_root_ngram_pos = {} # Словарь: (start_pos, end_pos) -> bool
    cand_is_root_ngram_pos = {} # Словарь: (start_pos, end_pos) -> bool
    ref_stem_spans = set((s, e) for _, s, e in ref_stems_info)
    cand_stem_spans = set((s, e) for _, s, e in cand_stems_info)

    for _, start, end in ref_ngrams_pos:
        ref_is_root_ngram_pos[(start, end)] = any(start >= s and end <= e for s, e in ref_stem_spans)
    for _, start, end in cand_ngrams_pos:
        cand_is_root_ngram_pos[(start, end)] = any(start >= s and end <= e for s, e in cand_stem_spans)

    # --- Подсчет взвешенных сумм (знаменатели) ---
    weighted_total_ref = sum(root_weight if ref_is_root_ngram_pos.get((s, e), False) else 1.0 for _, s, e in ref_ngrams_pos)
    weighted_total_cand = sum(root_weight if cand_is_root_ngram_pos.get((s, e), False) else 1.0 for _, s, e in cand_ngrams_pos)
    if weighted_total_cand == 0 or weighted_total_ref == 0: return 0.0

    # --- Подсчет взвешенных совпадений (числитель) ---
    ref_ngram_positions = {}
    for ngram, start, end in ref_ngrams_pos:
        if ngram not in ref_ngram_positions: ref_ngram_positions[ngram] = []
        ref_ngram_positions[ngram].append((start, end))

    cand_ngram_positions = {}
    for ngram, start, end in cand_ngrams_pos:
        if ngram not in cand_ngram_positions: cand_ngram_positions[ngram] = []
        cand_ngram_positions[ngram].append((start, end))

    weighted_matches = 0.0
    processed_cand_indices = {} # {ngram: set(indices_used)}

    # Итерируем по позициям N-грамм в ЭТАЛОНЕ
    for ngram, r_start, r_end in ref_ngrams_pos:
        # Если такая n-грамма есть в кандидате И еще не все ее вхождения там использованы
        if ngram in cand_ngram_positions:
            # Ищем неиспользованную позицию этой же n-граммы в кандидате
            found_match_pos = -1
            if ngram not in processed_cand_indices:
                processed_cand_indices[ngram] = set()

            for c_idx, (c_start, c_end) in enumerate(cand_ngram_positions[ngram]):
                if c_idx not in processed_cand_indices[ngram]:
                    found_match_pos = c_idx
                    break # Нашли первое свободное совпадение

            if found_match_pos != -1:
                 # Есть совпадение! Определяем его вес
                c_start, c_end = cand_ngram_positions[ngram][found_match_pos]
                is_ref_ngram_root = ref_is_root_ngram_pos.get((r_start, r_end), False)
                is_cand_ngram_root = cand_is_root_ngram_pos.get((c_start, c_end), False)

                # Применяем повышенный вес, если ОБА вхождения корневые
                match_weight = root_weight if (is_ref_ngram_root and is_cand_ngram_root) else 1.0
                weighted_matches += match_weight
                # Помечаем позицию кандидата как использованную
                processed_cand_indices[ngram].add(found_match_pos)

    # --- Расчет P, R, F ---
    P_w = weighted_matches / weighted_total_cand
    R_w = weighted_matches / weighted_total_ref
    if P_w == 0 and R_w == 0: return 0.0
    denominator = (beta**2 * P_w) + R_w
    score = ((1 + beta**2) * P_w * R_w) / denominator if denominator > 0 else 0.0
    # Ограничиваем результат диапазоном [0, 1] на всякий случай
    return max(0.0, min(score, 1.0))


# --- Финальная функция calculate_makts ---
def calculate_makts(reference, candidate, root_weight=2.0, n=6, beta=1.0):
    """
    Вычисляет Morpheme-Aware Kazakh Translation Score (MAKTS) v2.
    Основан на chrF с взвешиванием n-грамм в корневых зонах слов (Агрессивный вариант).
    Требует установленного Apertium и языкового пакета 'kaz'.
    """
    if not APERTIUM_AVAILABLE:
        print("ПРЕДУПРЕЖДЕНИЕ: Apertium недоступен. Возвращаем стандартный chrF.")
        return calculate_chrf(reference, candidate)

    analyzer = initialize_apertium_analyzer()
    if not analyzer:
        print("ПРЕДУПРЕЖДЕНИЕ: Apertium не инициализирован. Возвращаем стандартный chrF.")
        return calculate_chrf(reference, candidate)

    # Очистка текста от лишних пробелов
    ref_cleaned = re.sub(r'\s+', ' ', reference).strip()
    cand_cleaned = re.sub(r'\s+', ' ', candidate).strip()

    ref_stems_info = get_stems_with_spans(ref_cleaned, analyzer)
    cand_stems_info = get_stems_with_spans(cand_cleaned, analyzer)

    # Вычисляем взвешенный chrF с агрессивной логикой взвешивания совпадений
    score = calculate_weighted_chrf(ref_cleaned, cand_cleaned, ref_stems_info, cand_stems_info,
                                    root_weight=root_weight, n=n, beta=beta)

    return score