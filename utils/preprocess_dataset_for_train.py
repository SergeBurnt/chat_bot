import torch
from copy import deepcopy

def preprocess_dataset(samples, tokenizer, max_length, promts, answers):
    """
    Предварительная обработка набора данных для модели.

    Аргументы:
    samples -- словарь с примерами данных.
    tokenizer -- токенизатор для преобразования текста в токены.
    max_length -- максимальная длина последовательности токенов.
    promts -- ключ для доступа к промптам в словаре samples.
    answers -- ключ для доступа к ответам в словаре samples.

    Возвращает:
    Словарь с токенизированными входными данными, маской внимания и метками.
    """

    # Объединяем промпты и ответы в один список текстов.
    texts = [promt + ans for promt, ans in zip(samples[promts], samples[answers])]

    # Токенизируем объединенные тексты с указанием максимальной длины и обрезкой.
    texts_tokenized = tokenizer(
        texts,                         # Список текстов для токенизации.
        max_length=max_length,         # Максимальная длина последовательности токенов.
        truncation=True,               # Обрезка текстов, которые превышают max_length.
        return_token_type_ids=False    # Не возвращать идентификаторы типов токенов.
    )

    # Токенизируем только промпты без указания максимальной длины.
    prompts_tokenized = tokenizer(
        samples[promts],               # Список промптов для токенизации.
        return_token_type_ids=False    # Не возвращать идентификаторы типов токенов.
    )

    # Создаем глубокую копию токенизированных текстов для использования в качестве меток.
    labels = deepcopy(texts_tokenized['input_ids'])

    # Заменяем токены промптов в метках на -100, чтобы они не учитывались при обучении.
    for i in range(len(labels)):
        prompt_len = len(prompts_tokenized['input_ids'][i])  # Длина промпта в токенах.
        labels[i][:prompt_len] = [-100] * prompt_len         # Заменяем токены промпта на -100.

    # Возвращаем словарь с токенизированными входными данными, маской внимания и метками.
    return {
        'input_ids': texts_tokenized['input_ids'],    # Токенизированные входные данные.
        'attention_mask': texts_tokenized['attention_mask'],  # Маска внимания.
        'labels': labels                             # Метки для обучения модели.
    }
