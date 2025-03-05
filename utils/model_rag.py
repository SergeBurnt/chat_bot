import torch
import re
import numpy as np
from qdrant_client import QdrantClient
from utils.database import CreateDatabase, SearchDatabase

class ModelRag:
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
                 count_rephrase: int = 3
                ):
        
        self.generation_pipeline = generation_pipeline
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.path_client = path_client
        self.client = QdrantClient(path=self.path_client)
        self.limit_semantic_search = limit_semantic_search
        self.collection_name = collection_name
        self.count_rephrase = count_rephrase 

    def _llm_answer(self, query, context):
        prompt = f"""
        Ты русскоязычный эксперт в области атомной энергетики.
        У тебя есть доступ к набору нормативных документов из атомной отрасли и правилам проектирования
        систем, зданий, оборудования и трубопроводов, используй их, чтобы полно и точно ответить на следующий вопрос.
        Убедись, что ответ подробный, конкретный и непосредственно касается вопроса.
        Не добавляй информацию, которая не подтверждается предоставленными отзывами.
    
        Вопрос:
        {query}
        
        Отзывы:
        {context}
        """
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        output = self.generation_pipeline(messages, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature=self.temperature, top_p=self.top_p)
    
        return output[0]['generated_text'][1]['content']

    def _rephrase_query(self, query):
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
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        output = self.generation_pipeline(messages, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature=self.temperature, top_p=self.top_p)
        queries = output[0]['generated_text'][1]['content']
    
        return re.split(r'\\n+', queries)


    def predict(self, query):
        queries = self._rephrase_query(query)

        all_chunks = []
        for rephrased_query in queries:
            search_database = SearchDatabase(client=self.client,
                                             embedding_model=self.embedding_model,
                                             collection_name=self.collection_name,
                                             device=self.device
            )
            
            selected_chunks = search_database.semantic_search(query, limit=self.limit_semantic_search)
            all_chunks.extend(selected_chunks)
    
        texts = [f"Название: {chunk['document_name']}. Отзыв: {chunk['text']}" for chunk in all_chunks]
        texts = np.unique(texts)
        scores = self.cross_encoder.score([(query, text) for text in texts])
        
        idxs = np.argsort(list(scores))[-10:]
    
        context = ' ; '.join([texts[i] for i in idxs])
        
        return self._llm_answer(query, context), context