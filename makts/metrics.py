# metrics.py

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# Закомментировал импорт, так как word_tokenize используется только в старой версии BLEU
# from nltk.tokenize import word_tokenize
import numpy as np
import re
from collections import Counter
import sacrebleu
try:
    import apertium
    from streamparser import LexicalUnit, SReading # Добавим SReading на всякий случай
    APERTIUM_AVAILABLE = True
except ImportError:
    print("="*80)
    # Исправленный текст предупреждения
    print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать библиотеку 'apertium'.")
    print("Вероятно, пакет 'apertium' для Python не установлен (pip install apertium)")
    print("или отсутствуют необходимые системные компоненты Apertium.")
    print("Новая метрика MAKTS (взвешенный chrF) будет недоступна и вернет стандартный chrF.")
    print("ВАЖНО: Требуется системная установка Apertium core, apertium-kaz и, возможно, python3-apertium-core.")
    print("См. инструкции по установке Apertium: https://wiki.apertium.org/wiki/Installation")
    print("="*80)
    APERTIUM_AVAILABLE = False
# --- КОНЕЦ НОВЫХ ИМПОРТОВ ---


# --- Примечание об установке ---
# Этот код требует:
# - nltk: pip install nltk
# - numpy: pip install numpy
# - sacrebleu: pip install sacrebleu (для chrF и как fallback для MAKTS)
# - apertium: pip install apertium (Python-обертка)
# - Системная установка: Apertium core, apertium-kaz, python3-apertium-core (или аналог)

# Download required NLTK data (оставляем для BLEU)
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download issue: {e}")

# ==============================================================================
# Вспомогательные функции
# ==============================================================================

def transliterate_cyrillic(text):
    """
    Transliterate Cyrillic text to Latin alphabet for BEER metric
    """
    cyrillic_to_latin = {
        'а': 'a', 'ә': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'ғ': 'g',
        'д': 'd', 'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
        'й': 'y', 'к': 'k', 'қ': 'q', 'л': 'l', 'м': 'm', 'н': 'n',
        'ң': 'n', 'о': 'o', 'ө': 'o', 'п': 'p', 'р': 'r', 'с': 's',
        'т': 't', 'у': 'u', 'ұ': 'u', 'ү': 'u', 'ф': 'f', 'х': 'h',
        'һ': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '',
        'ы': 'y', 'і': 'i', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
    }
    return ''.join(cyrillic_to_latin.get(c.lower(), c) for c in text)

def get_char_ngrams_simple(text, n=4):
    """Get character n-grams from text (simple version for BEER)"""
    if not text or len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text)-n+1)]

def get_char_ngrams_with_pos(text, n):
    """Генерирует n-граммы с их начальными и конечными позициями в тексте."""
    if not isinstance(text, str) or len(text) < n:
        return []
    text_lower = text.lower() # Сравнение без учета регистра
    return [(text_lower[i:i+n], i, i+n) for i in range(len(text_lower) - n + 1)]


# ==============================================================================
# Функции расчета стандартных метрик (кроме TER)
# ==============================================================================

def calculate_beer(reference, candidate):
    """
    Calculate BEER score with Cyrillic support through transliteration
    Implementation focuses on character n-grams and word overlap
    """
    if not reference or not candidate: return 0.0
    if any(ord(char) > 127 for char in reference[:50] + candidate[:50]):
        reference = transliterate_cyrillic(reference)
        candidate = transliterate_cyrillic(candidate)
    reference = reference.lower()
    candidate = candidate.lower()
    ref_words = set(reference.split())
    cand_words = set(candidate.split())
    word_denom = max(len(ref_words), len(cand_words), 1)
    word_overlap = len(ref_words & cand_words) / word_denom if word_denom > 0 else 0.0
    ref_ngrams = set(get_char_ngrams_simple(reference))
    cand_ngrams = set(get_char_ngrams_simple(candidate))
    char_denom = max(len(ref_ngrams), len(cand_ngrams), 1)
    char_overlap = len(ref_ngrams & cand_ngrams) / char_denom if char_denom > 0 else 0.0
    score = 0.6 * char_overlap + 0.4 * word_overlap
    return max(0.0, min(score, 1.0))

def calculate_bleu(reference, candidate, language='english'):
    """
    Calculate BLEU score between reference and candidate translations.
    Uses simple whitespace tokenization for robustness.
    """
    if not reference or not candidate: return 0.0
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    if not candidate_tokens: return 0.0
    if not reference_tokens: return 0.0
    reference_list = [reference_tokens]
    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(reference_list, candidate_tokens, smoothing_function=smoothing)
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0.0

def calculate_meteor(reference, candidate):
    """
    Calculate METEOR score (simplified version focusing on exact word matches, F1).
    """
    if not reference or not candidate: return 0.0
    ref_words = reference.lower().split()
    can_words = candidate.lower().split()
    len_ref = len(ref_words)
    len_can = len(can_words)
    if len_ref == 0 or len_can == 0: return 0.0
    ref_words_set = set(ref_words)
    can_words_set = set(can_words)
    matches = len(ref_words_set.intersection(can_words_set))
    if matches == 0: return 0.0
    precision = matches / len_can
    recall = matches / len_ref
    if precision + recall == 0: return 0.0
    f_mean = (2 * precision * recall) / (precision + recall)
    return f_mean

def calculate_chrf(reference, candidate):
    """
    Calculate chrF score between reference and candidate translations using sacrebleu.
    """
    if not reference or not candidate: return 0.0
    try:
        chrf_score_obj = sacrebleu.sentence_chrf(candidate, [reference])
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
    Использует эвристику выбора лучшей леммы И ДЕТАЛЬНУЮ ОТЛАДКУ.
    """
    if not analyzer or not text: return []
    stems_info = []
    print(f"\n--- Analyzing text: '{text[:50]}...'")

    for match in re.finditer(r'(\S+)|(\s+)', text):
        token = match.group(0)
        start_index = match.start()

        if match.group(1): # Process only non-whitespace tokens
            word_token_from_regex = token
            print(f"\n+++ Processing Word: '{word_token_from_regex}' at index {start_index}") # DEBUG WORD
            try:
                analyses = analyzer.analyze(word_token_from_regex)
                print(f"    +++ Analyze Result: {analyses}") # DEBUG ANALYSIS

                extracted_pairs = []

                if analyses and isinstance(analyses, list):
                    lu = analyses[0]
                    surface = lu.wordform if hasattr(lu, 'wordform') else word_token_from_regex
                    print(f"    +++ Got Surface: '{surface}'") # DEBUG SURFACE

                    if hasattr(lu, 'readings') and lu.readings:
                        print(f"    +++ Found {len(lu.readings)} reading path(s)") # DEBUG READINGS COUNT
                        for i, analysis_path in enumerate(lu.readings):
                            print(f"        +++ Processing Path {i}: {analysis_path}") # DEBUG PATH
                            if isinstance(analysis_path, list) and analysis_path and isinstance(analysis_path[0], SReading):
                                first_segment = analysis_path[0]
                                if hasattr(first_segment, 'baseform'):
                                    lemma = first_segment.baseform
                                    print(f"            +++ Extracted Lemma: '{lemma}'") # DEBUG LEMMA
                                    if lemma and surface and isinstance(lemma, str) and isinstance(surface, str):
                                        extracted_pairs.append((lemma, surface))
                                else:
                                     print(f"            !!! Segment has no baseform attribute") # DEBUG NO BASEFORM
                            else:
                                print(f"        !!! Invalid analysis path or segment type: {type(analysis_path)} or {type(analysis_path[0]) if analysis_path else 'Empty Path'}") # DEBUG INVALID PATH
                    else:
                        print(f"    !!! No 'readings' attribute found or readings is empty.") # DEBUG NO READINGS

                else:
                     print(f"    !!! Analyze returned None or not a list.") # DEBUG NO ANALYSIS

                # --- Apply heuristic ---
                best_lemma = None
                best_surface = None
                unique_extracted_pairs = list(set(extracted_pairs))
                print(f"    +++ Unique Extracted Pairs: {unique_extracted_pairs}") # DEBUG UNIQUE PAIRS

                if not unique_extracted_pairs:
                    print(f"    !!! No unique pairs found, skipping heuristic.") # DEBUG NO UNIQUE
                    continue

                shorter_lemmas = [(lem, surf) for lem, surf in unique_extracted_pairs if len(lem) < len(surf)]
                print(f"    +++ Shorter Lemmas: {shorter_lemmas}") # DEBUG SHORTER

                if shorter_lemmas:
                    best_lemma, best_surface = max(shorter_lemmas, key=lambda item: len(item[0]))
                    print(f"    +++ Heuristic chose from shorter: '{best_lemma}'") # DEBUG CHOICE
                else:
                    best_lemma, best_surface = unique_extracted_pairs[0]
                    print(f"    +++ Heuristic chose first (no shorter): '{best_lemma}'") # DEBUG CHOICE

                # --- Apply prefix check ---
                stem_len = len(best_lemma)
                # Убрали проверку best_surface.lower().startswith(best_lemma.lower())
                # Оставили только проверку, что лемма не длиннее исходного слова
                if stem_len <= len(best_surface):
                    stem_start_in_text = start_index
                    stem_end_in_text = start_index + stem_len # Граница основы определяется длиной леммы
                    stems_info.append((best_lemma, stem_start_in_text, stem_end_in_text))
                    print(f"    >>> SUCCESS (Prefix check ignored): Stem added: ('{best_lemma}', {stem_start_in_text}, {stem_end_in_text})") # DEBUG SUCCESS
                else:
                    # Эта ветка теперь маловероятна, но оставим на всякий случай
                    print(f"    !!! Length check failed for lemma '{best_lemma}' and surface '{best_surface}'") # DEBUG LEN FAIL

            except Exception as e:
                # Эта ошибка теперь будет ловить только непредвиденные исключения
                print(f"  XXX UNEXPECTED ERROR processing word '{word_token_from_regex}': {e}") # DEBUG UNEXPECTED ERROR
                pass

    # Final output
    print(f"\n--- Finished analyzing text. Stems extracted: {len(stems_info)}")
    if stems_info:
        # print(f"--- Extracted stems for: '{text[:50]}...'") # Already printed above
        for lem, s, e in stems_info:
             print(f"  Lemma: '{lem}', Span: ({s}, {e}), Text: '{text[s:e]}'")
    # else:
        # print(f"--- *** No stems extracted for text: '{text[:50]}...' ***") # Already printed above
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
    # Группируем позиции для каждой n-граммы для точного сопоставления
    ref_ngram_positions = {}
    for ngram, start, end in ref_ngrams_pos:
        if ngram not in ref_ngram_positions: ref_ngram_positions[ngram] = []
        ref_ngram_positions[ngram].append((start, end))

    cand_ngram_positions = {}
    for ngram, start, end in cand_ngrams_pos:
        if ngram not in cand_ngram_positions: cand_ngram_positions[ngram] = []
        cand_ngram_positions[ngram].append((start, end))

    weighted_matches = 0.0
    processed_cand_indices = {} # {ngram: set(indices_used)} - чтобы не использовать одну и ту же поз. кандидата дважды

    # Итерируем по УНИКАЛЬНЫМ позициям в эталоне, чтобы сопоставить с кандидатом
    for ngram, r_start, r_end in ref_ngrams_pos:
        if ngram in cand_ngram_positions:
            is_ref_ngram_root = ref_is_root_ngram_pos.get((r_start, r_end), False)

            # Ищем неиспользованную позицию этой же n-граммы в кандидате
            found_match = False
            if ngram not in processed_cand_indices:
                processed_cand_indices[ngram] = set()

            for c_idx, (c_start, c_end) in enumerate(cand_ngram_positions[ngram]):
                if c_idx not in processed_cand_indices[ngram]:
                    # Нашли совпадение! Определяем его вес
                    is_cand_ngram_root = cand_is_root_ngram_pos.get((c_start, c_end), False)
                    # Применяем повышенный вес, если ОБА вхождения корневые
                    match_weight = root_weight if (is_ref_ngram_root and is_cand_ngram_root) else 1.0
                    weighted_matches += match_weight
                    processed_cand_indices[ngram].add(c_idx) # Помечаем позицию кандидата как использованную
                    found_match = True
                    break # Переходим к следующей n-грамме эталона
            # (Если не нашли свободного совпадения в кандидате, found_match останется False)

    # --- Расчет P, R, F ---
    P_w = weighted_matches / weighted_total_cand
    R_w = weighted_matches / weighted_total_ref
    if P_w == 0 and R_w == 0: return 0.0
    denominator = (beta**2 * P_w) + R_w
    score = ((1 + beta**2) * P_w * R_w) / denominator if denominator > 0 else 0.0
    return score


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

    # Очистка текста для анализа Apertium (может помочь с некоторыми ошибками)
    ref_cleaned = re.sub(r'\s+', ' ', reference).strip()
    cand_cleaned = re.sub(r'\s+', ' ', candidate).strip()

    ref_stems_info = get_stems_with_spans(ref_cleaned, analyzer)
    cand_stems_info = get_stems_with_spans(cand_cleaned, analyzer)

    # Вычисляем взвешенный chrF с агрессивной логикой взвешивания совпадений
    score = calculate_weighted_chrf(ref_cleaned, cand_cleaned, ref_stems_info, cand_stems_info,
                                    root_weight=root_weight, n=n, beta=beta)
    return score

# ==============================================================================
# calculate_ter и СТАРАЯ РЕАЛИЗАЦИЯ MAKTS удалены / закомментированы
# ==============================================================================