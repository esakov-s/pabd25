# Использовать официальный образ Python для базового изображения.  
FROM python:3.11-slim  
# Установить рабочий каталог в контейнере на /app.  
WORKDIR /app  
# Скопировать файлы из текущего каталога в /app контейнера.  
ADD . /app  
# Установить необходимые пакеты, указанные в файле requirements.txt.  
RUN pip install --no-cache-dir -r requirements.txt
RUN python src/download_from_s3.py
# Сделать порт 8000 доступным снаружи контейнера.  
EXPOSE 8000
# Запустить Gunicorn при запуске контейнера.  
CMD ["gunicorn", "-b", ":8000", "service:app"]