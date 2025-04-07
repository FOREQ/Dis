import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
import re

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Warning: NLTK data download issue: {e}")

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

def get_char_ngrams(text, n=4):
    """Get character n-grams from text"""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def calculate_beer(reference, candidate):
    """
    Calculate BEER score with Cyrillic support through transliteration
    Implementation focuses on character n-grams and word overlap
    """
    if not reference or not candidate:
        return 0.0

    # Transliterate if text contains Cyrillic characters
    if any(ord(char) > 127 for char in reference + candidate):
        reference = transliterate_cyrillic(reference)
        candidate = transliterate_cyrillic(candidate)

    # Convert to lowercase
    reference = reference.lower()
    candidate = candidate.lower()

    # Get word-level similarity
    ref_words = set(reference.split())
    cand_words = set(candidate.split())
    word_overlap = len(ref_words & cand_words) / max(len(ref_words), len(cand_words), 1)

    # Get character n-gram similarity
    ref_ngrams = set(get_char_ngrams(reference))
    cand_ngrams = set(get_char_ngrams(candidate))
    char_overlap = len(ref_ngrams & cand_ngrams) / max(len(ref_ngrams), len(cand_ngrams), 1)

    # Combine scores (0.6 weight to character n-grams, 0.4 to word overlap)
    score = 0.6 * char_overlap + 0.4 * word_overlap

    return max(min(score, 1.0), 0.0)

def calculate_bleu(reference, candidate, language):
    """Calculate BLEU score between reference and candidate translations"""
    if not reference or not candidate:
        return 0.0

    # Simple whitespace tokenization for non-English languages
    if language.lower() not in ['english']:
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
    else:
        try:
            reference_tokens = word_tokenize(reference.lower())
            candidate_tokens = word_tokenize(candidate.lower())
        except:
            # Fallback to simple splitting if NLTK tokenization fails
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()

    reference_tokens = [reference_tokens]
    smoothing = SmoothingFunction().method1

    try:
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
    except:
        return 0.0

def calculate_ter(reference, candidate):
    """Calculate Translation Error Rate (TER)"""
    if not reference or not candidate:
        return 1.0

    ref_words = reference.lower().split()
    can_words = candidate.lower().split()

    # Dynamic programming matrix
    dp = np.zeros((len(ref_words) + 1, len(can_words) + 1))

    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(can_words) + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(can_words) + 1):
            if ref_words[i-1] == can_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                    dp[i][j-1],       # insertion
                                    dp[i-1][j-1])     # substitution

    # Calculate TER
    edits = dp[len(ref_words)][len(can_words)]
    ter = edits / max(len(ref_words), 1)

    return min(ter, 1.0)

def calculate_meteor(reference, candidate):
    """
    Calculate METEOR score (simplified version focusing on exact matches)
    """
    if not reference or not candidate:
        return 0.0

    ref_words = set(reference.lower().split())
    can_words = set(candidate.lower().split())

    matches = len(ref_words.intersection(can_words))

    if not matches:
        return 0.0

    precision = matches / max(len(can_words), 1)
    recall = matches / max(len(ref_words), 1)

    if precision + recall == 0:
        return 0.0

    f_mean = 2 * (precision * recall) / (precision + recall)
    return f_mean

def segment_kazakh_word_all(word, suffixes=None):
    """
    Возвращает все возможные варианты сегментации казахского слова на основу и суффиксы.
    Каждый вариант – это список, где первый элемент — основа, а далее идут суффиксы.
    """
    if suffixes is None:
        suffixes = [
            'лар', 'лер', 'дар', 'дер', 'тар', 'тер',  # множественное число
            'мын', 'мін', 'сың', 'сің', 'сыз', 'сіз',       # личные формы
            'да', 'де', 'та', 'те',                         # локатив
            'дың', 'дің', 'тың', 'тің',                     # притяжательные
            'ға', 'ге', 'қа', 'ке',                         # дательный
            'ды', 'ді', 'ты', 'ті'                          # винительный
        ]
    word = word.lower()
    variants = []

    def recursive_variants(remaining, current_variant):
        found = False
        # Сортируем окончания по убыванию длины, чтобы сперва брать длинные совпадения
        for suf in sorted(suffixes, key=len, reverse=True):
            if remaining.endswith(suf) and len(remaining) - len(suf) >= 2:
                new_remaining = remaining[:-len(suf)]
                recursive_variants(new_remaining, [suf] + current_variant)
                found = True
        if not found:
            variants.append([remaining] + current_variant)

    recursive_variants(word, [])
    return variants

def select_best_variant(variants):
    """
    Выбирает лучший вариант сегментации по простому критерию:
    максимальная суммарная длина суффиксов.
    """
    best_variant = None
    best_score = -1
    for var in variants:
        suffixes = var[1:]
        score = sum(len(suf) for suf in suffixes)
        if score > best_score:
            best_score = score
            best_variant = var
    return best_variant

def calculate_metric(stem, suffixes, weights):
    """
    Вычисляет морфологический балл для слова.
    Базовый балл = длина основы, к которому прибавляется сумма весов суффиксов.
    """
    base_score = len(stem)
    suffix_score = 0.0
    for suf in suffixes:
        suffix_score += weights.get(suf, 0.5)  # вес по умолчанию 0.5, если не указан
    total_score = base_score + suffix_score
    return total_score

def get_morph_score(word, weights):
    """
    Получает морфологический балл для слова, выбирая лучший вариант сегментации.
    """
    variants = segment_kazakh_word_all(word)
    best_variant = select_best_variant(variants)
    if best_variant is None or len(best_variant) == 0:
        return 0.0
    stem = best_variant[0]
    suffixes = best_variant[1:]
    return calculate_metric(stem, suffixes, weights)

def calculate_makts(reference, candidate, weights=None):
    """
    Функция calculate_makts вычисляет Morpheme-Aware Kazakh Translation Score (MAKTS)
    для сравнения морфологической структуры эталонного и кандидатского переводов.
    
    Алгоритм:
      1. Токенизация эталонного (reference) и кандидатского (candidate) предложений.
      2. Для каждого слова вычисляется морфологический балл (на основе сегментации).
      3. Для каждой пары слов (по порядку) вычисляется коэффициент:
         similarity = min(score_ref, score_cand) / max(score_ref, score_cand)
      4. Итоговый MAKTS — среднее значение этих коэффициентов (от 0 до 1).
    
    Аргументы:
      reference (str): эталонное предложение на казахском языке.
      candidate (str): кандидатский перевод.
      weights (dict): словарь весов для суффиксов. Если не передан, используется базовый.
    
    Возвращает:
      float: значение метрики MAKTS от 0 до 1.
    """
    if weights is None:
        weights = {
            "лар": 1.0,
            "лер": 1.0,
            "дар": 1.0,
            "дер": 1.0,
            "тар": 1.0,
            "тер": 1.0,
            "ымызда": 1.5,
            "імізда": 1.5,
            "ымыз": 1.2,
            "іміз": 1.2,
            "да": 0.8,
            "де": 0.8,
            "та": 0.8,
            "те": 0.8,
            "дың": 1.0,
            "дің": 1.0,
            "тың": 1.0,
            "тің": 1.0,
            "ға": 0.8,
            "ге": 0.8,
            "қа": 0.8,
            "ке": 0.8,
            "ды": 0.8,
            "ді": 0.8,
            "ты": 0.8,
            "ті": 0.8
        }
    
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    n = min(len(ref_words), len(cand_words))
    if n == 0:
        return 0.0

    similarities = []
    for i in range(n):
        score_ref = get_morph_score(ref_words[i], weights)
        score_cand = get_morph_score(cand_words[i], weights)
        # Если оба балла равны нулю, считаем совпадение идеальным
        if score_ref == 0 and score_cand == 0:
            sim = 1.0
        elif score_ref == 0 or score_cand == 0:
            sim = 0.0
        else:
            sim = min(score_ref, score_cand) / max(score_ref, score_cand)
        similarities.append(sim)
    return sum(similarities) / n