import nest_asyncio
import asyncio
import logging
import torch
import warnings
import numpy as np
import pandas as pd
import threading
from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from datasets import load_dataset, Dataset
from utils.database import CreateDatabase, SearchDatabase
from utils.model_rag import ModelRag
from transformers import pipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

path_client = 'aep_chat_bot'
collection_name = 'regulatory_documentation'
total = pd.read_excel('data/total.xlsx', dtype={'point': str})
dataset = Dataset.from_pandas(total)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print(f'Loading generation model...')
generation_pipeline = pipeline(
    'text-generation',
    # model='Qwen/Qwen2-1.5B-Instruct',
    model='train_model/Qwen_Qwen2_1.5B_Instruct',
    device=device,
    torch_dtype=torch.float16
)
print(f'Generation model loaded')

print(f'Loading embedding model...')
embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    model_kwargs={'torch_dtype': torch.float16}
)
print(f'Embedding model loaded')

print(f'Loading reranker model...')
cross_encoder = HuggingFaceCrossEncoder(
    model_name='amberoad/bert-multilingual-passage-reranking-msmarco',
    model_kwargs={'device': device}
)
print(f'Reranker model loaded')

model = ModelRag(
    generation_pipeline=generation_pipeline,
    embedding_model=embedding_model,
    cross_encoder=cross_encoder,
    path_client=path_client,
    device=device,
    collection_name=collection_name,
    temperature=0.9,
    top_p=0.6,
    max_new_tokens=1024
)


# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Вставьте ваш токен API, полученный от @BotFather
TOKEN = ''

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Привет! Я бот - эксперт в атомной энергетике, который отвечает на вопросы. Задайте мне вопрос!')

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print()
    answer, context_info, history = model.predict(user_message)
    await update.message.reply_text(answer)

# Обработчик команды /clear
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model.clear_history()
    await update.message.reply_text('История диалога очищена. Вы можете начать новый диалог.')

# Разрешение вложенных событийных циклов
nest_asyncio.apply()

# Создание и настройка приложения
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('clear', clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    await app.run_polling()

# Функция для запуска бота в отдельном потоке
def run_bot():
    asyncio.run(main())

# Запуск бота в отдельном потоке
bot_thread = threading.Thread(target=run_bot)
bot_thread.start()

print('Chatbot launched!')