import sacrebleu

# Ваши предложения (гипотеза - машинный перевод, эталон(ы))
hypothesis = "Қазақстандықтарды адами тұрғыдан бағалау қажет"
# sacrebleu ожидает список эталонов, даже если он один
references = ["Қазақстандықтарды адамша бағалау керек"]

# Расчет chrF+ для одного предложения
# word_order=2 включает компонент F-меры по словам (делает метрику chrF+)
# char_order=6 это стандартный порядок символьных n-грамм (можно не указывать, это значение по умолчанию)
chrf_plus_score = sacrebleu.sentence_chrf(hypothesis, references, word_order=2).score

# Вывод результата (score обычно в диапазоне 0-100)
print(f"Предложение MT: {hypothesis}")
print(f"Предложение Ref: {references[0]}")
print(f"chrF++ Score: {chrf_plus_score:.2f}")

# --- Пример для нескольких предложений (оценка корпуса) ---

# Если у вас есть списки предложений
list_of_hypotheses = [hypothesis, "басқа мысал"]
list_of_references = [references, ["другой эталон"]] # Список списков эталонов

# Расчет chrF+ для корпуса
# corpus_chrf возвращает объект с разными данными, score содержит основную метрику
corpus_chrf_plus_score = sacrebleu.corpus_chrf(list_of_hypotheses, list_of_references, word_order=2).score
print(f"\nКорпусный chrF++ Score: {corpus_chrf_plus_score:.2f}")