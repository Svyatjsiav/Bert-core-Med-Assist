import torch
import os
import json
import glob
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import tkinter as tk
from tkinter import filedialog
import shutil
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class MedicalAssistant:
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", device="cpu"):
        """
        Инициализация медицинского ассистента на BERT
        
        Args:
            model_name (str): Название BERT модели или путь к локальной модели
            device (str): Устройство для вычислений ('cpu' или 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_name = model_name
        
        # Инициализация моделей
        print(f"🔄 Загрузка BERT модели: {model_name} на устройство {self.device}")
        
        # Модель для эмбеддингов (Sentence Transformers)
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            self.embedding_model.to(self.device)
        except:
            # Fallback на меньшую модель
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.embedding_model.to(self.device)
        
        # Модель для генерации текста
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.generation_model.to(self.device)
        except:
            # Если модель не Seq2Seq, используем другую
            print(f"⚠️  Модель {model_name} не поддерживает генерацию, используем альтернативу...")
            self.tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/fred_t5_ru_turbo_alpaca")
            self.generation_model = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/fred_t5_ru_turbo_alpaca")
            self.generation_model.to(self.device)
        
        # Пайплайн для вопрос-ответ (опционально)
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="DeepPavlov/rubert-base-cased-squad",
                tokenizer="DeepPavlov/rubert-base-cased-squad",
                device=0 if str(self.device) == "cuda" else -1
            )
        except:
            self.qa_pipeline = None
        
        self.conversation_history = []
        self.patient_data = {}
        self.vault_content = []
        self.vault_embeddings = None
        
        # Цвета для вывода в консоль
        self.PINK = '\033[95m'
        self.CYAN = '\033[96m'
        self.YELLOW = '\033[93m'
        self.NEON_GREEN = '\033[92m'
        self.RESET_COLOR = '\033[0m'

    def open_file(self, filepath: str) -> str:
        """Чтение файла"""
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def clear_ib_folder(self, data_path: str):
        """Очищает папку ИБ от всех файлов"""
        if os.path.exists(data_path):
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Удален старый файл: {filename}")
                except Exception as e:
                    print(f"Ошибка при удалении файла {filename}: {e}")

    def open_file_dialog(self) -> str:
        """Открывает диалоговое окно для выбора JSON файла"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Выберите JSON файл с данными пациента",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        root.destroy()
        return file_path

    def load_patient_data_simple(self, data_path='ИБ') -> Dict:
        """Загрузка данных пациента из JSON файла"""
        try:
            # Создаем папку если её нет
            if not os.path.exists(data_path):
                os.makedirs(data_path)
                print(f"Создана папка {data_path}")
            
            # Очищаем папку ИБ от старых файлов
            self.clear_ib_folder(data_path)
            
            # Открываем диалоговое окно для выбора файла
            json_filepath = self.open_file_dialog()
            
            if not json_filepath:
                print("Файл не выбран")
                return {}
            
            # Копируем выбранный файл в папку ИБ
            filename = os.path.basename(json_filepath)
            destination_path = os.path.join(data_path, filename)
            shutil.copy2(json_filepath, destination_path)
            print(f"Файл скопирован в: {destination_path}")
            
            # Загружаем данные из скопированного файла
            print(f"Загружаем данные из файла: {filename}")
            
            with open(destination_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Извлекаем данные
            patient_record = list(data["История болезни или наблюдений v.4"].values())[0]
            patient_info = patient_record["Данные"]["Сведения при обращении"]
            
            print("\nДанные пациента:")
            print("=" * 50)
            
            # Создаем словарь для возвращаемых данных
            patient_data = {}
            
            for field_name, field_data in patient_info.items():
                if isinstance(field_data, dict) and "Значение" in field_data:
                    value = field_data["Значение"]
                    
                    if value in [None, "", [], False]:
                        continue
                    
                    # Сохраняем данные в возвращаемый словарь
                    patient_data[field_name] = {
                        "Тип": field_data.get("Тип", ""),
                        "Значение": value
                    }
                    
                    # Простые значения
                    if not isinstance(value, list):
                        print(f"• {field_name}: {value}")
                    
                    # Списки простых значений
                    elif isinstance(value, list) and value and not isinstance(value[0], dict):
                        print(f"• {field_name}: {', '.join(map(str, value))}")
                    
                    # Сложные структуры
                    else:
                        print(f"• {field_name}:")
                        for item in value:
                            if isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, dict) and "Значение" in sub_value:
                                        nested_items = sub_value["Значение"]
                                        if nested_items:
                                            print(f"  └── {sub_key}:")
                                            for nested_item in nested_items:
                                                if isinstance(nested_item, dict):
                                                    for detail_key, detail_value in nested_item.items():
                                                        if isinstance(detail_value, dict) and "Значение" in detail_value:
                                                            detail_content = detail_value["Значение"]
                                                            if detail_content not in [None, "", []]:
                                                                print(f"      ├── {detail_key}: {detail_content}")
            
            print("=" * 50)
            return patient_data
            
        except Exception as e:
            print(f"Ошибка при загрузке данных пациента: {e}")
            return {}

    def get_paragraphs_file_by_diagnosis(self, patient_data: Dict) -> str:
        """Определяет какой файл параграфов использовать на основе клинического диагноза"""
        base_dir = os.path.dirname(__file__)
        
        # Извлекаем клинический диагноз из данных пациента
        clinical_diagnosis = ""
        if "Клинический диагноз" in patient_data:
            clinical_diagnosis = str(patient_data["Клинический диагноз"]["Значение"]).lower()
        
        print(f"Анализируем диагноз: '{clinical_diagnosis}'")
        
        # Регулярные выражения для поиска
        hepatitis_patterns = [
            r'хвгс',  # ХВГС
            r'гепатит',  # гепатит, гепатита, гепатитом и т.д.
            r'хрон\w* гепатит',  # хронический гепатит
            r'вирусн\w* гепатит',  # вирусный гепатит
            r'гепатит\s*с',  # гепатит с, гепатитс
        ]
        
        # Проверяем паттерны для гепатита
        for pattern in hepatitis_patterns:
            if re.search(pattern, clinical_diagnosis):
                paragraphs_file = os.path.join(base_dir, "data", "Хронический вирусный гепатит С (ХВГС) параграфы.txt")
                print(f"✅ Выбран файл для гепатита (найден паттерн: '{pattern}')")
                return paragraphs_file
        
        
        # По умолчанию используем гепатит, если диагноз не распознан
        paragraphs_file = os.path.join(base_dir, "data", "Хронический вирусный гепатит С (ХВГС) параграфы.txt")
        print("⚠️  Выбран файл по умолчанию (гепатит) - диагноз не распознан")
        return paragraphs_file

    def load_relevant_paragraphs(self, patient_data: Dict) -> List[str]:
        """Загружает параграфы соответствующие диагнозу пациента"""
        paragraphs_file = self.get_paragraphs_file_by_diagnosis(patient_data)
        
        print(self.NEON_GREEN + f"Загрузка {paragraphs_file}..." + self.RESET_COLOR)
        vault_content = []
        
        if os.path.exists(paragraphs_file):
            with open(paragraphs_file, "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
            print(f"Загружено {len(vault_content)} строк из {paragraphs_file}")
        else:
            print(f"Файл {paragraphs_file} не найден!")
        
        return vault_content

    def get_system_message_by_diagnosis(self, patient_data: Dict) -> str:
        """Возвращает соответствующее системное сообщение на основе диагноза"""
        clinical_diagnosis = ""
        if "Клинический диагноз" in patient_data:
            clinical_diagnosis = str(patient_data["Клинический диагноз"]["Значение"])
        
        # Для переломов
        if any(keyword in clinical_diagnosis.lower() for keyword in 
               ["перелом ключицы", "перелом лопатки", "ключицы и лопатки"]):
            return f"""Ты - медицинский ассистент, специализирующийся на травматологии и лечении переломов. 
СТРОГИЕ ПРАВИЛА:
1. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
2. ВСЕГДА ИСПОЛЬЗУЙ ДАННЫЕ ПАЦИЕНТА ДЛЯ ФОРМИРОВАНИЯ ОТВЕТА
3. ИСПОЛЬЗУЙ ТОЛЬКО ТЕКСТ ИЗ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА - НИЧЕГО НЕ ПРИДУМЫВАЙ
4. НЕ ИЗМЕНЯЙ ТЕРМИНОЛОГИЮ ИЗ КОНТЕКСТА
5. ЕСЛИ В КОНТЕКСТЕ НЕТ ИНФОРМАЦИИ - СКАЖИ "В предоставленном контексте нет информации"
6. НЕ ДОБАВЛЯЙ СВОИ ЗНАНИЯ ИЛИ ИНТЕРПРЕТАЦИИ
7. УБЕРИ РЕКОМЕНДАЦИИ ДЛЯ ДЕТЕЙ, ЕСЛИ ВОЗРАСТ ПАЦИЕНТА >=18

ДАННЫЕ ПАЦИЕНТА (ОБЯЗАТЕЛЬНО ИСПОЛЬЗОВАТЬ): {patient_data}
ПРАВИЛА ИСПОЛЬЗОВАНИЯ ДАННЫХ ПАЦИЕНТА:
- Учитывай возраст пациента при выборе лечения
- Учитывай противопоказания из анамнеза
- Учитывай уже проведенные лечения
- Адаптируй дозировки под параметры пациента
- Исключи рекомендации, не подходящие данному пациенту

ФОРМАТ ОТВЕТА:
Используй ТОЧНО такие же формулировки как в контексте. Не меняй слова, не перефразируй, не сокращай.

ОТВЕЧАЙ ТОЛЬКО НА ОСНОВЕ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА БЕЗ ИЗМЕНЕНИЙ!"""
        
        # Для гепатита
        elif any(keyword in clinical_diagnosis.lower() for keyword in 
                 ["хвгс", "гепатит", "гепатит с", "хронический вирусный гепатит"]):
            return f"""Ты - медицинский ассистент, специализирующийся на лечении хронического вирусного гепатита C. 
СТРОГИЕ ПРАВИЛА:
1. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
2. ВСЕГДА ИСПОЛЬЗУЙ ДАННЫЕ ПАЦИЕНТА ДЛЯ ФОРМИРОВАНИЯ ОТВЕТА
3. ИСПОЛЬЗУЙ ТОЛЬКО ТЕКСТ ИЗ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА - НИЧЕГО НЕ ПРИДУМЫВАЙ
4. НЕ ИЗМЕНЯЙ ТЕРМИНОЛОГИЮ ИЗ КОНТЕКСТА
5. ЕСЛИ В КОНТЕКСТЕ НЕТ ИНФОРМАЦИИ - СКАЖИ "В предоставленном контексте нет информации"
6. НЕ ДОБАВЛЯЙ СВОИ ЗНАНИЯ ИЛИ ИНТЕРПРЕТАЦИИ
7. УБЕРИ РЕКОМЕНДАЦИИ ДЛЯ ДЕТЕЙ, ЕСЛИ ВОЗРАСТ ПАЦИЕНТА >=18

ДАННЫЕ ПАЦИЕНТА (ОБЯЗАТЕЛЬНО ИСПОЛЬЗОВАТЬ): {patient_data}
ПРАВИЛА ИСПОЛЬЗОВАНИЯ ДАННЫХ ПАЦИЕНТА:
- Учитывай возраст пациента при выборе лечения
- Учитывай противопоказания из анамнеза
- Учитывай уже проведенные лечения
- Адаптируй дозировки под параметры пациента
- Исключи рекомендации, не подходящие данному пациенту

ФОРМАТ ОТВЕТА:
Используй ТОЧНО такие же формулировки как в контексте. Не меняй слова, не перефразируй, не сокращай.

ОТВЕЧАЙ ТОЛЬКО НА ОСНОВЕ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА БЕЗ ИЗМЕНЕНИЙ!"""
        
        # По умолчанию
        else:
            return f"""Ты - медицинский ассистент. 

СТРОГИЕ ПРАВИЛА:
1. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
2. ВСЕГДА ИСПОЛЬЗУЙ ДАННЫЕ ПАЦИЕНТА ДЛЯ ФОРМИРОВАНИЯ ОТВЕТА 
3. ИСПОЛЬЗУЙ ТОЛЬКО ТЕКСТ ИЗ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА - НИЧЕГО НЕ ПРИДУМЫВАЙ
4. НЕ ИЗМЕНЯЙ ТЕРМИНОЛОГИЮ ИЗ КОНТЕКСТА
5. ЕСЛИ В КОНТЕКСТЕ НЕТ ИНФОРМАЦИИ - СКАЖИ "В предоставленном контексте нет информации"
6. НЕ ДОБАВЛЯЙ СВОИ ЗНАНИЯ ИЛИ ИНТЕРПРЕТАЦИИ
7. УБЕРИ РЕКОМЕНДАЦИИ ДЛЯ ДЕТЕЙ, ЕСЛИ ВОЗРАСТ ПАЦИЕНТА >=18

ДАННЫЕ ПАЦИЕНТА (ОБЯЗАТЕЛЬНО ИСПОЛЬЗОВАТЬ): {patient_data}
ПРАВИЛА ИСПОЛЬЗОВАНИЯ ДАННЫХ ПАЦИЕНТА:
- Учитывай возраст пациента при выборе лечения
- Учитывай противопоказания из анамнеза
- Учитывай уже проведенные лечения
- Адаптируй дозировки под параметры пациента
- Исключи рекомендации, не подходящие данному пациенту

ОТВЕЧАЙ ТОЛЬКО НА ОСНОВЕ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА БЕЗ ИЗМЕНЕНИЙ!"""

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Генерация эмбеддингов с помощью BERT"""
        print(self.NEON_GREEN + "Генерация эмбеддингов BERT..." + self.RESET_COLOR)
        
        # Фильтруем пустые строки
        texts = [text for text in texts if text.strip()]
        
        # Генерируем эмбеддинги
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Сгенерировано {len(embeddings)} эмбеддингов размерностью {embeddings.shape[1]}")
        return embeddings.cpu().numpy()

    def get_relevant_context_bert(self, query: str, top_k: int = 3) -> List[str]:
        """Поиск релевантного контекста с использованием BERT эмбеддингов"""
        if self.vault_embeddings is None or len(self.vault_embeddings) == 0:
            return []
        
        # Фильтруем только параграфы с 3.
        filtered_indices = []
        filtered_content = []
        
        for i, content in enumerate(self.vault_content):
            if content.strip().startswith('3.'):
                filtered_indices.append(i)
                filtered_content.append(content)
        
        print(f"🔍 Из {len(self.vault_content)} параграфов отфильтровано {len(filtered_content)} с '3.'")
        
        if len(filtered_content) == 0:
            print("⚠️  Не найдено параграфов, начинающихся с '3.'")
            return []
        
        # Берем соответствующие эмбеддинги
        filtered_embeddings = self.vault_embeddings[filtered_indices]
        
        # Генерируем эмбеддинг для запроса
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy()
        
        # Вычисляем косинусное сходство
        similarities = np.dot(filtered_embeddings, query_embedding.T).flatten()
        
        # Порог релевантности
        similarity_threshold = 0.7
        above_threshold = similarities >= similarity_threshold
        
        if above_threshold.sum() > 0:
            # Берем только релевантные выше порога
            top_indices = np.where(above_threshold)[0]
            # Сортируем по убыванию схожести
            sorted_indices = np.argsort(similarities[top_indices])[::-1]
            top_indices = top_indices[sorted_indices][:top_k]
        else:
            # Если ничего выше порога - берем лучшие N
            top_k = min(top_k, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Получаем релевантный контекст
        relevant_context = [filtered_content[idx].strip() for idx in top_indices]
        
        print(f"✅ Найдено релевантных контекстов с 3.: {len(relevant_context)}")
        
        # Показываем найденные контексты для отладки
        if relevant_context:
            print("\n🔍 НАЙДЕННЫЕ КОНТЕКСТЫ С 3.:")
            for i, context in enumerate(relevant_context[:3]):
                preview = context.replace('\n', ' ').strip()[:150]
                print(f"   {i + 1}. {preview}...")
        
        return relevant_context

    def rewrite_query_bert(self, user_input: str, conversation_history: List[Dict], patient_data: Dict) -> str:
        """Переписывание запроса с помощью BERT модели"""
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
        
        patient_info_str = f"""
Данные пациента:
{patient_data}
"""
        
        prompt = f"""Ты - медицинский ассистент. Переформулируй следующий запрос для поиска релевантной медицинской информации. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.

{patient_info_str}

История разговора:
{context}

Исходный запрос: {user_input}

Переписанный запрос: """
        
        # Генерация переписанного запроса
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.generation_model.generate(
                **inputs,
                max_length=200,
                num_beams=4,
                temperature=0.3,
                do_sample=True,
                early_stopping=True
            )
        
        rewritten_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewritten_query

    def generate_response_bert(self, prompt: str, max_length: int = 300) -> str:
        """Генерация ответа с помощью BERT модели"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.generation_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.3,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def bert_chat(self, user_input: str, system_message: str, patient_data: Dict) -> str:
        """Основной чат с использованием BERT"""
        # Добавление пользовательского ввода в историю
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Переписывание запроса
        if len(self.conversation_history) > 1:
            rewritten_query = self.rewrite_query_bert(user_input, self.conversation_history, patient_data)
            print(self.PINK + "Исходный запрос: " + user_input + self.RESET_COLOR)
            print(self.PINK + "Переписанный запрос: " + rewritten_query + self.RESET_COLOR)
        else:
            rewritten_query = user_input
        
        # Извлечение релевантного контекста
        relevant_context = self.get_relevant_context_bert(rewritten_query)
        
        if relevant_context:
            context_str = "\n".join(relevant_context)
            print("Контекст найден: \n\n" + self.CYAN + context_str + self.RESET_COLOR)
            
            strict_context_instruction = """
ВАЖНО: Используй ТОЛЬКО информацию из предоставленного контекста. 
НЕ придумывай, НЕ интерпретируй, НЕ изменяй терминологию.
Копируй точные формулировки из контекста.
"""
            
            prompt = f"""{system_message}

{strict_context_instruction}

Релевантный контекст:
{context_str}

Вопрос: {user_input}

Ответ (используй ТОЛЬКО информацию из контекста выше):"""
        else:
            print(self.CYAN + "Контекст не найден" + self.RESET_COLOR)
            prompt = f"""{system_message}

Вопрос: {user_input}

Ответ (так как релевантный контекст не найден): 'В предоставленном контексте нет информации по данному вопросу.'"""
        
        # Обновление истории разговора
        self.conversation_history[-1]["content"] = user_input
        
        # Генерация ответа
        response = self.generate_response_bert(prompt)
        
        # Добавление ответа в историю
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    def initialize_system(self, data_path=None):
        """Полная инициализация системы"""
        if data_path is None:
            data_path = os.path.join(os.path.expanduser("~"), "MedicalAssistant", "ИБ")
        
        print(f"Используем путь: {data_path}")
        self.patient_data = self.load_patient_data_simple(data_path)
        
        if self.patient_data:
            self.vault_content = self.load_relevant_paragraphs(self.patient_data)
            if self.vault_content:
                self.vault_embeddings = self.generate_embeddings(self.vault_content)
            else:
                print("⚠️  Нет загруженного контекста для эмбеддингов")
                self.vault_embeddings = None
        else:
            print("❌ Не удалось загрузить данные пациента")