# MongoDB 8 Setup for PAVEL

## Установка MongoDB 8

### macOS (через Homebrew)
```bash
# Добавить MongoDB tap
brew tap mongodb/brew

# Установить MongoDB 8
brew install mongodb-community@8.0

# Запустить MongoDB как сервис
brew services start mongodb-community@8.0

# Или запустить вручную
mongod --config /opt/homebrew/etc/mongod.conf
```

### Linux (Ubuntu/Debian)
```bash
# Импорт публичного ключа
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

# Добавление репозитория
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

# Обновление и установка
sudo apt-get update
sudo apt-get install -y mongodb-org

# Запуск службы
sudo systemctl start mongod
sudo systemctl enable mongod
```

## Настройка базы данных

1. Подключитесь к MongoDB:
```bash
mongosh
```

2. Создайте базу данных для PAVEL:
```javascript
use pavel_reviews
```

3. Создайте коллекцию для отзывов:
```javascript
db.createCollection("reviews")
```

## Конфигурация проекта

1. Создайте файл `.env` в корне проекта:
```bash
touch /Users/andrey/Projects/pavel/.env
```

2. Добавьте настройки MongoDB:
```
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=pavel_reviews
```

## Установка Python зависимостей

```bash
cd /Users/andrey/Projects/pavel
pip install -r requirements.txt
```

## Проверка подключения

Запустите тестовый скрипт:
```bash
python temp_ingest_script.py
```

## Troubleshooting

### Если MongoDB не запускается:
- Проверьте логи: `tail -f /opt/homebrew/var/log/mongodb/mongo.log`
- Убедитесь, что порт 27017 свободен: `lsof -i :27017`

### Если не удается подключиться:
- Проверьте статус MongoDB: `brew services list | grep mongodb`
- Проверьте подключение: `mongosh --eval "db.adminCommand('ping')"`