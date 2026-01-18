#!/usr/bin/env python3
import argparse
import sys
import os
import shutil
import glob

# Добавьте путь к текущей директории в Python path
sys.path.insert(0, os.path.dirname(__file__))

from core import MedicalAssistant


def copy_json_to_data_dir(json_filepath, data_dir='ИБ'):
    """
    Копирует JSON файл пациента в папку данных, предварительно очищая её
    """
    # Создаем папку если её нет
    os.makedirs(data_dir, exist_ok=True)

    # Очищаем папку от старых JSON файлов
    old_json_files = glob.glob(f'{data_dir}/*.json')
    for old_file in old_json_files:
        os.remove(old_file)
        print(f"🗑️  Удален старый файл: {os.path.basename(old_file)}")

    # Копируем новый файл
    filename = os.path.basename(json_filepath)
    destination = os.path.join(data_dir, filename)
    shutil.copy2(json_filepath, destination)
    print(f"✅ Файл скопирован: {filename} -> {data_dir}/")

    return destination


def main():
    parser = argparse.ArgumentParser(description='Медицинский ассистент для рекомендаций по лечению (BERT версия)')
    parser.add_argument('--model', default='DeepPavlov/rubert-base-cased', 
                       help='BERT модель для использования (путь к модели или название из HuggingFace)')
    parser.add_argument('--data-dir', default='ИБ', help='Путь к папке с данными пациента')
    parser.add_argument('--json-file', help='Путь к JSON файлу пациента (будет скопирован в папку данных)')
    parser.add_argument('--interactive', action='store_true', help='Интерактивный режим')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Устройство для вычислений (cpu или cuda)')

    args = parser.parse_args()

    try:
        # Если указан JSON файл, копируем его в папку данных
        if args.json_file:
            if not os.path.exists(args.json_file):
                print(f"❌ Файл не найден: {args.json_file}")
                sys.exit(1)

            print(f"📁 Загружаем файл пациента: {os.path.basename(args.json_file)}")
            copy_json_to_data_dir(args.json_file, args.data_dir)

        print("🚀 Инициализация медицинского ассистента (BERT)...")
        assistant = MedicalAssistant(
            model_name=args.model,
            device=args.device
        )
        assistant.initialize_system(data_path=args.data_dir)

        print("📋 Получение рекомендации по лечению...")

        # Получаем системное сообщение и рекомендацию
        system_message = assistant.get_system_message_by_diagnosis(assistant.patient_data)
        clinical_diagnosis = assistant.patient_data.get("Клинический диагноз", {}).get("Значение", "не указан")
        user_input = f"Назначьте лечение для пациента с диагнозом: {clinical_diagnosis}"

        recommendation = assistant.bert_chat(
            user_input=user_input,
            system_message=system_message,
            patient_data=assistant.patient_data
        )

        print("\n" + "=" * 60)
        print("РЕКОМЕНДАЦИЯ ПО ЛЕЧЕНИЮ (BERT)")
        print("=" * 60)
        print(recommendation)
        print("=" * 60)

        if args.interactive:
            print("\n💬 Вход в интерактивный режим...")
            system_message = assistant.get_system_message_by_diagnosis(assistant.patient_data)
            print("\n💬 ИНТЕРАКТИВНЫЙ РЕЖИМ BERT (введите 'exit' для выхода)")

            while True:
                try:
                    user_input = input("\n👤 Ваш вопрос: ").strip()

                    if user_input.lower() in ['exit', 'quit', 'выход']:
                        break

                    if not user_input:
                        continue

                    response = assistant.bert_chat(
                        user_input=user_input,
                        system_message=system_message,
                        patient_data=assistant.patient_data
                    )

                    print(f"\n🩺 {response}")

                except KeyboardInterrupt:
                    print("\n\nВыход из интерактивного режима.")
                    break
                except Exception as e:
                    print(f"\n❌ Ошибка: {e}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()