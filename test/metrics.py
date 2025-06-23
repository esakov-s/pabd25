import pandas as pd
import requests
import json
import dotenv

config = dotenv.dotenv_values(".env")
token = config["APP_TOKEN"]

HOST = 'http://127.0.0.1:5000'

def test_api_numbers():
    try:
        # Читаем данные из CSV файла
        df = pd.read_csv('data/raw/1_2025-06-23_12-59.csv')
        print(df.head())
        # Проверяем обязательные поля
        required_fields = ["total_meters", "rooms_count", "floors_count", "floor", "price"]
        for field in required_fields:
            if field not in df.columns:
                raise ValueError(f"CSV файл должен содержать поле: {field}")

        # Инициализируем счетчик успешных и неудачных запросов
        success_count = 0
        failure_count = 0
        
        # Проходим по каждой строке данных
        for index, row in df.iterrows():
            # Формируем JSON данные для запроса
            data = {
                "area": row["total_meters"], 
                "rooms": row["rooms_count"], 
                "total_floors": row["floors_count"], 
                "floor": row["floor"]
                }

            
            # Отправляем POST запрос
            response = requests.post(f'{HOST}/api/numbers', 
                                   json=data, 
                                   headers={
                                       'Content-Type': 'application/json',
                                       "Authorization": f"Bearer {token}",
                                            })
            
            # Проверяем ответ
            if response.status_code == 200:
                print(f"Успешно: {data}")
                success_count += 1
            else:
                print(f"Ошибка: {data} - {response.status_code}: {response.text}")
                failure_count += 1

        # Выводим статистику
        print(f"\nTEST COMPLETE:")
        print(f"SUCCESS: {success_count}")
        print(f"FAILS: {failure_count}")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    test_api_numbers()
