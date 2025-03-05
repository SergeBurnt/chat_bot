import torch
import uuid
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


class CreateDatabase:
    """
    Класс для создания и управления базой данных векторов текстовых фрагментов с использованием Qdrant.

    Инициализация параметров класса.

    Args:
        embedding_model (str): Модель для генерации эмбеддингов текста.
        dataset (list): Набор данных, содержащий текстовые документы для обработки.
        path (str): Путь к хранилищу Qdrant.
        size (int): Размер векторов эмбеддингов. По умолчанию 1024.
        chunk_size (int): Размер текстовых фрагментов. По умолчанию 1024.
        chunk_overlap (int): Перекрытие текстовых фрагментов. По умолчанию 100.
    """

    def __init__(self,
                 embedding_model,
                 dataset,
                 device: torch.device,
                 path: str,
                 collection_name: str,
                 size: int = 1024,
                 chunk_size: int = 1024,
                 chunk_overlap: int = 100,
                ):

        self.collection_name = collection_name  # Имя коллекции в Qdrant
        self.embedding_model = embedding_model  # Модель для генерации эмбеддингов
        self.dataset = dataset  # Набор данных для обработки
        self.path = path  # Путь к хранилищу Qdrant
        self.size = size  # Размер векторов эмбеддингов
        self.chunk_size = chunk_size  # Размер текстовых фрагментов
        self.chunk_overlap = chunk_overlap  # Перекрытие текстовых фрагментов

        # Инициализация модели эмбеддингов
        self.embedding_model = embedding_model
        # Инициализация разбиения текста на фрагменты
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        # Инициализация клиента Qdrant
        self.client = QdrantClient(path=self.path)
        # Определение устройства для вычислений (GPU, если доступно)
        self.device = device

    def _create_base(self):
        """
        Создание новой коллекции в Qdrant.

        Этот метод создает новую коллекцию в Qdrant с заданными параметрами векторов и метрикой расстояния.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,  # Имя создаваемой коллекции
            vectors_config=models.VectorParams(
                size=self.size,  # Размер векторов в коллекции (1024 измерения)
                distance=models.Distance.COSINE,  # Используемая метрика для измерения расстояния между векторами (косинусное расстояние)
                on_disk=True  # Указывает, что векторы будут храниться на диске
            ),
        )

    def create(self):
        """
        Создание базы данных и добавление данных в коллекцию.

        Этот метод создает новую коллекцию в Qdrant и добавляет в нее данные из набора данных.
        Текстовые документы разбиваются на фрагменты, для которых генерируются векторы эмбеддингов,
        и затем эти векторы добавляются в коллекцию вместе с метаданными.
        """
        self._create_base()  # Создание новой коллекции
        # Итерация по каждому элементу в наборе данных
        for i in tqdm(range(len(self.dataset))):
            # Разделение текста на части (chunks)
            self.text_chunks = self.text_splitter.split_text(self.dataset[i]['decription'])

            # Генерация векторов для каждого текстового фрагмента
            vectors = self.embedding_model.encode(
                self.text_chunks,
                normalize_embeddings=True,  # Нормализация векторов
                device=self.device  # Устройство, на котором выполняется вычисление (например, CPU или GPU)
            ).tolist()  # Преобразование массива векторов в список

            # Добавление векторов и метаданных в коллекцию Qdrant
            self.client.upsert(
                collection_name=self.collection_name,  # Имя коллекции, в которую добавляются данные
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),  # Генерация уникального идентификатора для каждого вектора
                        vector=vectors[j],  # Вектор, соответствующий текстовому фрагменту
                        payload={
                            'text': self.text_chunks[j],  # Текстовый фрагмент
                            'document_name': self.dataset[i]['document'],  # Название фильма (без последних 7 символов)
                            'point': self.dataset[i]['point'],  # Год выпуска фильма (извлекается из названия)
                        }
                    ) for j in range(len(self.text_chunks))  # Итерация по каждому текстовому фрагменту
                ]
            )


class SearchDatabase:
    def __init__(self, client, embedding_model: str, collection_name: str, device: str = 'cpu'):
        self.embedding_model = embedding_model
        self.device = device
        self.client = client
        self.collection_name = collection_name

    def semantic_search(self, query: str, limit: int = 50):
        """
        Поиск по семантическому запросу.

        Этот метод выполняет поиск в коллекции Qdrant на основе семантического запроса.
        Запрос преобразуется в вектор, который затем используется для поиска наиболее релевантных текстовых фрагментов.

        Args:
            query (str): Текстовый запрос для поиска.
            limit (int): Ограничение на количество возвращаемых результатов. По умолчанию 10.

        Returns:
            list: Список релевантных текстовых фрагментов.
        """
        query_vector = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            device=self.device
        ).tolist()  # Генерация вектора запроса

        # Поиск в коллекции Qdrant
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit  # Ограничение на количество возвращаемых результатов
        )

        # Извлечение релевантных текстовых фрагментов
        relevant_chunks = [hit.payload for hit in hits]

        return relevant_chunks  # Возврат релевантных текстовых фрагментов