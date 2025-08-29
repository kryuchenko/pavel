# Quick Start для PAVEL

## Быстрый запуск с Docker Compose

### 1. Запуск MongoDB
```bash
# Запустить MongoDB 8 в контейнере
docker-compose up -d mongodb

# Проверить статус
docker-compose ps

# Посмотреть логи
docker-compose logs mongodb
```

### 2. Установка Python зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск сбора отзывов для Pokemon GO
```bash
python temp_ingest_script.py
```

## Остановка сервисов
```bash
# Остановить MongoDB
docker-compose down

# Остановить и удалить данные
docker-compose down -v
```

## Подключение к MongoDB
```bash
# Через mongosh в контейнере
docker exec -it pavel_mongodb mongosh

# Или напрямую с хоста (если установлен mongosh)
mongosh mongodb://localhost:27017/pavel_reviews
```

## Проверка данных
```javascript
// В mongosh
use pavel_reviews
db.reviews.countDocuments()
db.reviews.find().limit(5).pretty()
```