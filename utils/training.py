import evaluate
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, PromptTuningConfig
from typing import Union

# Определяем класс TrainingModel для обучения модели.
class TrainingModel:
    def __init__(self,
                 model_name: str,                      # Имя модели для загрузки.
                 output_dir: str,                      # Директория для сохранения результатов.
                 type_tuning: Union['promt_tuning', 'lora'],  # Тип тонкой настройки (prompt tuning или LoRA).
                 run_name: str = None,                # Имя запуска для логирования.
                 train_batch_size: int = 1,           # Размер пакета для обучения.
                 eval_batch_size: int = 1,            # Размер пакета для оценки.
                 gradient_accumulation_steps: int = 4, # Количество шагов накопления градиентов.
                 gradient_checkpointing: bool = True,  # Использовать контрольные точки градиентов.
                 lr: float = 1e-3,                    # Скорость обучения.
                 weight_decay: float = 0.01,         # Коэффициент регуляризации весов.
                 fp16: bool = True,                   # Использовать смешанную точность (float16).
                 epochs: int = 1,                     # Количество эпох обучения.
                 logging_steps: int = 100,            # Шаги для логирования.
                 prompt_size: int = 20,                # Размер промпта для prompt tuning.
                 lora_r: int = 8,                      # Ранг для LoRA.
                 lora_dropout: float = 0.1,            # Dropout для LoRA.
                 count_save_checkpoint: int = 3,      # Ограничиваем количество хранимых чекпоинтов до трёх последних
                 report_flag: bool = False,           # Флаг для логирования в wandb.
                 metric: str = 'accuracy',            # Метрика для оценки модели.
                 device: str = 'cpu',                 # Устройство для выполнения вычислений (CPU или GPU).
                 continue_from_checkpoint: bool = False,  # запуск обучения с последнего checkpoint
                 save_steps=500,        # Каждые 500 шагов сохранять чекпоинт
                 checkpoint_path: str = None,  # Путь к чекпоинту для перезапуска обучения
                ):               

        # Инициализируем параметры класса.
        self.model_name = model_name
        self.run_name = run_name
        self.type_tuning = type_tuning
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.lr = lr
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.epochs = epochs
        self.logging_steps = logging_steps
        self.prompt_size = prompt_size
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout
        self.count_save_checkpoint = count_save_checkpoint
        self.output_dir = output_dir
        self.report_to = 'wandb' if report_flag else None
        self.metric = evaluate.load(metric)  # Загружаем метрику для оценки.
        self.device = torch.device(device)    # Устанавливаем устройство для вычислений.
        self.continue_from_checkpoint = continue_from_checkpoint
        self.save_steps = save_steps
        self.checkpoint_path = checkpoint_path

        # Загружаем токенизатор для модели.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Устанавливаем токен заполнения.

        # Выводим информацию о текущем устройстве.
        print(str(device))

        # Загружаем предобученную модель или модель с checkpoint
        if self.checkpoint_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                device_map=self.device,  # Указываем устройство для модели.
                torch_dtype=torch.float16  # Указываем тип данных для модели.
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,  # Указываем устройство для модели.
                torch_dtype=torch.float16  # Указываем тип данных для модели.
            )

        # Инициализируем аргументы для обучения модели.
        self.training_args = TrainingArguments(
            per_device_train_batch_size=self.train_batch_size,  # Размер батча для train.
            gradient_accumulation_steps=self.gradient_accumulation_steps,  # Шаги накопления градиентов.
            gradient_checkpointing=self.gradient_checkpointing,  # Контрольные точки градиентов.
            learning_rate=self.lr,  # Скорость обучения.
            weight_decay=self.weight_decay,  # Регуляризация весов.
            fp16=self.fp16,  # Смешанная точность.
            num_train_epochs=self.epochs,  # Количество эпох обучения.
            logging_steps=self.logging_steps,  # Шаги для логирования.
            output_dir=self.output_dir,  # Директория для сохранения результатов.
            report_to=self.report_to,  # Логирование в wandb.
            run_name=self.run_name,  # Имя запуска для логирования.
            save_total_limit=self.count_save_checkpoint,  # Ограничиваем количество хранимых чекпоинтов
            save_steps=self.save_steps # Частота сохранения checkpoint
        )

    def _get_peft(self):
        """
        Настраивает конфигурацию PEFT (Parameter-Efficient Fine-Tuning) в зависимости от типа настройки.
    
        Этот метод определяет конфигурацию PEFT на основе указанного типа настройки (prompt tuning или LoRA)
        и инициализирует модель с этой конфигурацией.
    
        :return: None
        """
        if self.type_tuning == 'promt_tuning':
            # Конфигурация для prompt tuning.
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,  # Тип задачи (например, CAUSAL_LM для языкового моделирования).
                num_virtual_tokens=self.prompt_size  # Размер промпта (количество виртуальных токенов).
            )
        elif self.type_tuning == 'lora':
            # Конфигурация для LoRA (Low-Rank Adaptation).
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # Тип задачи (например, CAUSAL_LM для языкового моделирования).
                r=self.lora_r,  # Ранг для LoRA (определяет количество параметров для адаптации).
                lora_dropout=self.lora_dropout  # Dropout для LoRA (используется для регуляризации).
            )
    
        # Получаем модель с конфигурацией PEFT и перемещаем её на устройство (например, GPU или CPU).
        self.peft_model = get_peft_model(self.model, self.peft_config).to(self.device)
    
        # Выводим информацию о тренируемых параметрах модели.
        self.peft_model.print_trainable_parameters()

    # Метод для вычисления метрик модели.
    def _compute_metrics(self, eval_pred):
        """
        Вычисляет метрики для оценки качества предсказаний модели.
    
        :param eval_pred: Кортеж, содержащий предсказания модели и истинные метки.
        :return: Словарь с вычисленными метриками.
        """
        # Получаем предсказания и метки из кортежа eval_pred.
        predictions, labels = eval_pred
    
        # Получаем индексы максимальных значений по последней оси для предсказаний.
        # Это преобразует вероятностные распределения в индексы предсказанных классов.
        predictions = predictions.argmax(axis=-1)
    
        # Удаляем предсказания для обучаемого промпта в случае prompt tuning.
        # Оставляем только те предсказания, которые соответствуют меткам.
        predictions = predictions[:, predictions.shape[1] - labels.shape[1]:]
    
        # Сдвигаем предсказания и метки, чтобы для каждого токена
        # считать правильность предсказания следующего токена.
        shifted_labels = labels[:, 1:]
        shifted_predictions = predictions[:, :-1]
    
        # Создаем маску для игнорирования специальных токенов (например, паддинговых токенов).
        # Специальные токены имеют значение -100 и должны быть исключены из вычисления метрик.
        mask = shifted_labels != -100
    
        # Вычисляем метрику для предсказаний и меток, используя только те токены,
        # которые не являются специальными (не попадают под маску).
        return self.metric.compute(
            predictions=shifted_predictions[mask],
            references=shifted_labels[mask]
        )

    # Метод для объединения пакетов данных.
    def _collate_fn(self, batch):
        """
        Объединяет пакеты данных для использования в процессе обучения или оценки модели.
    
        :param batch: Список словарей, где каждый словарь представляет собой один пример данных.
                      Каждый словарь должен содержать ключи 'input_ids', 'attention_mask' и 'labels'.
        :return: Словарь с объединенными данными пакета, включая 'input_ids', 'attention_mask' и 'labels'.
        """
        # Получаем идентификаторы входных токенов из каждого примера в пакете.
        input_ids = [sample['input_ids'] for sample in batch]
        # Дополняем последовательности input_ids до одинаковой длины, используя значение паддинга,
        # указанное в токенизаторе. Параметр batch_first=True указывает, что первая размерность - это размер пакета.
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    
        # Получаем маски внимания из каждого примера в пакете.
        attention_mask = [sample['attention_mask'] for sample in batch]
        # Дополняем последовательности attention_mask до одинаковой длины, используя значение паддинга 0.
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
        # Получаем метки из каждого примера в пакете.
        labels = [sample['labels'] for sample in batch]
        # Дополняем последовательности labels до одинаковой длины, используя значение паддинга -100.
        # Значение -100 используется для игнорирования паддинговых токенов при вычислении потерь.
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
        # Возвращаем словарь с объединенными данными пакета.
        return {
            'input_ids': input_ids,  # Идентификаторы входных токенов.
            'attention_mask': attention_mask,  # Маски внимания.
            'labels': labels  # Метки.
        }

    # Метод для обучения модели.
    def train_model(self, train_dataset, path_save_model, save_model: bool = True):
        """
        Обучает модель на основе предоставленного тренировочного набора данных.
    
        :param train_dataset: Тренировочный набор данных.
        :param path_save_model: Путь для сохранения обученной модели.
        :param save_model: Флаг, указывающий, нужно ли сохранять модель после обучения (по умолчанию True).
        :return: Возвращает объект Trainer после завершения обучения.
        """
        while True:
            try:
                # Получаем конфигурацию PEFT (Parameter-Efficient Fine-Tuning).
                self._get_peft()
    
                # Создаем объект Trainer для управления процессом обучения.
                trainer = Trainer(
                    model=self.peft_model,  # Модель с конфигурацией PEFT.
                    train_dataset=train_dataset,  # Тренировочный набор данных.
                    args=self.training_args,  # Аргументы для обучения.
                    tokenizer=self.tokenizer,  # Токенизатор для преобразования текста в токены.
                    data_collator=self._collate_fn,  # Функция для объединения пакетов данных.
                    compute_metrics=self._compute_metrics,  # Функция для вычисления метрик.
                )
    
                # Запускаем процесс обучения. Если указан чекпоинт, обучение продолжится с него.
                trainer.train(resume_from_checkpoint=self.continue_from_checkpoint)
                break  # Выходим из цикла, если обучение завершено без ошибок.
    
            except RuntimeError as e:
                # В случае возникновения ошибки выводим сообщение об ошибке и перезапускаем обучение.
                print(f"Произошла ошибка: {e}")
                print("Перезапускаем обучение...")
    
        if save_model:
            # Сохраняем обученную модель по указанному пути.
            trainer.save_model(path_save_model)
            print(f'Модель сохранена в {path_save_model}')
    
        return trainer  # Возвращаем объект Trainer.

    # Метод для оценки модели.
    def eval_model(self, eval_dataset, path_save_model=None):
        """
        Оценивает модель на основе предоставленного оценочного набора данных.
    
        :param eval_dataset: Оценочный набор данных.
        :param path_save_model: Путь к сохраненной модели (если не указан, используется текущая модель).
        :return: Возвращает результаты оценки модели.
        """
        # Получаем конфигурацию PEFT.
        # self._get_peft()

        eval_args = TrainingArguments(
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_checkpointing=False,  # Отключаем контрольные точки градиентов
            fp16=self.fp16,
            # output_dir=self.output_dir,
            # report_to=self.report_to,
            # run_name=self.run_name
        )
        
        if path_save_model:
            # Если указан путь к сохраненной модели, загружаем её.
            model = AutoModelForCausalLM.from_pretrained(
                path_save_model,
                device_map=self.device,  # Указываем устройство для модели (например, GPU или CPU).
                torch_dtype=torch.float16  # Указываем тип данных для модели (например, float16 для ускорения вычислений).
            )
        else:
            # Если путь не указан, используем текущую модель.
            model = self.model
            
    
        # Создаем объект Trainer для управления процессом оценки.
        eval_trainer = Trainer(
            model=model,  # Модель с конфигурацией PEFT.
            eval_dataset=eval_dataset,  # Оценочный набор данных.
            args=eval_args,  # Аргументы для оценки.
            tokenizer=self.tokenizer,  # Токенизатор для преобразования текста в токены.
            data_collator=self._collate_fn,  # Функция для объединения пакетов данных.
            compute_metrics=self._compute_metrics  # Функция для вычисления метрик.
        )
    
        # Запускаем процесс оценки и возвращаем результаты.
        return eval_trainer.evaluate()