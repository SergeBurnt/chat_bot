import torch
import re
import numpy as np
from qdrant_client import QdrantClient
from utils.database import CreateDatabase, SearchDatabase

class ModelRag:
    """
    Класс ModelRag представляет собой модель, которая использует методы генерации текста и поиска
    для создания ответов на основе заданного запроса и контекста.
    """

    def __init__(self,
                 generation_pipeline,
                 embedding_model,
                 cross_encoder,
                 path_client: str,
                 device: torch.device,
                 collection_name: str,
                 max_new_tokens: int = 1024,
                 do_sample: bool = True,
                 temperature: float = 0.9,
                 top_p: float = 0.7,
                 limit_semantic_search: int = 50,
                 count_rephrase: int = 3,
                 max_history_size: int = 6,
                ):
        """
        Инициализация модели с заданными параметрами.
        """
        self.generation_pipeline = generation_pipeline  # Конвейер для генерации текста
        self.embedding_model = embedding_model  # Модель для создания эмбеддингов
        self.cross_encoder = cross_encoder  # Модель для оценки релевантности текста
        self.device = device  # Устройство для вычислений (CPU/GPU)
        self.max_new_tokens = max_new_tokens  # Максимальное количество новых токенов
        self.do_sample = do_sample  # Флаг для использования сэмплинга
        self.temperature = temperature  # Температура для контроля случайности
        self.top_p = top_p  # Параметр для выбора токенов с наибольшей вероятностью
        self.path_client = path_client  # Путь к клиенту Qdrant
        self.client = QdrantClient(path=self.path_client)  # Инициализация клиента Qdrant
        self.limit_semantic_search = limit_semantic_search  # Лимит на количество результатов в поиске
        self.collection_name = collection_name  # Имя коллекции в базе данных
        self.count_rephrase = count_rephrase  # Количество перефразировок запроса
        self.dialog_history = []  # Список для хранения истории диалога
        self.max_history_size = max_history_size  # Максимальное количество сообщений в истории

    def _llm_answer(self, query, context):
        """
        Генерация ответа на основе запроса и контекста.
        """
        # Формирование промпта для модели с учётом истории диалога
        # history_text = "\n".join([f"{entry[0]}: {entry[1]}" for entry in self.dialog_history])
        history_text = ''
        # Формирование промпта для модели
        prompt = f"""
        Ты русскоязычный эксперт в области атомной энергетики.
        У тебя есть доступ к набору нормативных документов из атомной отрасли и правилам проектирования
        систем, зданий, оборудования и трубопроводов, используй их, чтобы полно и точно ответить на следующий вопрос.
        Убедись, что ответ подробный, конкретный и непосредственно касается вопроса.
        Не добавляй информацию, которая не подтверждается предоставленными отзывами.

        История диалога:
        {history_text}
        
        Вопрос:
        {query}

        Норма:
        {context}
        """

        # Формирование сообщения для конвейера генерации
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Генерация ответа с использованием конвейера
        output = self.generation_pipeline(messages, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature=self.temperature, top_p=self.top_p)

        # Возврат сгенерированного текста
        return output[0]['generated_text'][1]['content']

    def _rephrase_query(self, query):
        """
        Перефразирование запроса для получения различных вариантов.
        """
        # Формирование промпта для перефразирования запроса
        prompt = f"""
            Твоя задача написать {self.count_rephrase} разных вариаций вопроса пользователя для того,
            чтобы по ним получить релевантные документы из векторной базы данных.
            Ты должен переформулировать вопрос с разных точек зрения.
            Это поможет избавить пользователя от недостатков поиска похожих документов на основе расстояния.
            Вопрос пользователя сфокусирован на теме атомной энергетики.
            Напиши ТОЛЬКО вариации вопроса и больше ничего, разделяя их символом новой строки \\\\n.
            НЕ пиши ответ на сам вопрос.
            -----------------
            {query}

            """

        # Формирование сообщения для конвейера генерации
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Генерация перефразированных запросов
        output = self.generation_pipeline(messages, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature=self.temperature, top_p=self.top_p)
        queries = output[0]['generated_text'][1]['content']

        # Разделение сгенерированных запросов по символу новой строки
        return re.split(r'\\n+', queries)

    def predict(self, query):
        """
        Основной метод для получения ответа на запрос.
        """
        # Получение перефразированных запросов
        # queries = self._rephrase_query(query)
        queries = query
        # print(f'Перефразированные вопросы: {queries}')

        all_chunks = []  # Список для хранения всех найденных фрагментов
        for rephrased_query in queries:
            # Инициализация объекта для поиска в базе данных
            search_database = SearchDatabase(client=self.client,
                                             embedding_model=self.embedding_model,
                                             collection_name=self.collection_name,
                                             device=self.device
            )

            # Поиск релевантных фрагментов в базе данных
            selected_chunks = search_database.semantic_search(query, limit=self.limit_semantic_search)
            all_chunks.extend(selected_chunks)  # Добавление найденных фрагментов в общий список

        # Формирование текстов из найденных фрагментов
        texts = [f"Название: {chunk['document_name']}. Пункт: {chunk['text']}" for chunk in all_chunks]
        texts = np.unique(texts)

        # Оценка релевантности текстов с использованием cross-encoder
        scores = self.cross_encoder.score([(query, text) for text in texts])

        # Получение индексов самых релевантных текстов
        idxs = np.argsort(list(scores))[-7:]

        # Формирование контекста из самых релевантных текстов
        context = ' ; '.join([texts[i] for i in idxs])
        
        # Получение ответа модели
        answer = self._llm_answer(query, context)

        # Обновление истории диалога
        self.dialog_history.append(("User", query))
        self.dialog_history.append(("Assistant", answer))

        # Ограничение размера истории
        if len(self.dialog_history) > self.max_history_size:
            self.dialog_history = self.dialog_history[-self.max_history_size:]

        # Возврат сгенерированного ответа и контекста
        return answer, context, self.dialog_history

    def clear_history(self):
        """
        Очистка истории диалога.
        """
        self.dialog_history = []