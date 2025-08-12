#!/usr/bin/env python3
"""
Тест настройки семантической близости в PAVEL
"""

import asyncio
import numpy as np
from pavel.embeddings.embedding_pipeline import EmbeddingPipeline, PipelineConfig
from pavel.embeddings.semantic_search import SemanticSearchEngine
from pavel.embeddings.vector_store import SimilarityMetric
from pavel.embeddings.embedding_generator import SupportedModels

async def test_similarity_tuning():
    """Тестируем различные настройки семантической близости"""
    print("🔧 НАСТРОЙКА СЕМАНТИЧЕСКОЙ БЛИЗОСТИ")
    print("=" * 60)
    
    # Инициализация
    pipeline = EmbeddingPipeline(PipelineConfig(
        embedding_model=SupportedModels.E5_SMALL_MULTILINGUAL.value
    ))
    
    search_engine = pipeline.semantic_search
    
    print("🎯 Поисковый запрос: 'app crashes and freezes'")
    print("-" * 60)
    
    # Тестируем разные пороги близости
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    for threshold in thresholds:
        print(f"\n📊 Порог близости: {threshold:.1f}")
        
        results = search_engine.search_similar_reviews(
            query_text="app crashes and freezes",
            limit=5,
            min_similarity=threshold
        )
        
        print(f"   Найдено результатов: {len(results)}")
        
        for i, result in enumerate(results[:3], 1):  # Показываем топ-3
            print(f"   {i}. [{result.similarity_score:.3f}] {result.text[:70]}...")
    
    print("\n" + "=" * 60)
    print("🌍 КРОСС-ЯЗЫКОВАЯ БЛИЗОСТЬ")
    print("-" * 60)
    
    # Тестируем кросс-языковой поиск
    cross_language_queries = [
        ("crashes", "на английском"),
        ("глючит", "на русском"), 
        ("problemas", "на испанском"),
        ("проблемы", "на русском"),
        ("excellent", "на английском"),
        ("отличное", "на русском")
    ]
    
    for query, lang in cross_language_queries:
        print(f"\n🔍 Поиск '{query}' ({lang}):")
        
        results = search_engine.search_similar_reviews(
            query_text=query,
            limit=3,
            min_similarity=0.5
        )
        
        for i, result in enumerate(results, 1):
            # Определяем язык результата
            lang_detect = "🇷🇺" if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in result.text.lower()) else \
                         "🇪🇸" if any(word in result.text.lower() for word in ["aplicación", "excelente", "problemas"]) else \
                         "🇧🇷" if any(word in result.text.lower() for word in ["aplicativo", "muito", "bom"]) else \
                         "🇸🇦" if any(c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي" for c in result.text) else "🇺🇸"
            
            print(f"   {i}. {lang_detect} [{result.similarity_score:.3f}] {result.text[:60]}...")

async def test_advanced_similarity_config():
    """Тестируем продвинутые настройки близости"""
    print("\n" + "=" * 60) 
    print("⚙️  ПРОДВИНУТЫЕ НАСТРОЙКИ БЛИЗОСТИ")
    print("-" * 60)
    
    # Разные метрики близости
    metrics = [
        (SimilarityMetric.COSINE, "Косинусная близость"),
        (SimilarityMetric.DOT_PRODUCT, "Скалярное произведение"),
        (SimilarityMetric.EUCLIDEAN, "Евклидово расстояние")
    ]
    
    from pavel.embeddings.vector_store import VectorStore, VectorStoreConfig
    
    query_text = "app is great and works perfectly"
    
    for metric, name in metrics:
        print(f"\n📐 {name}:")
        
        # Создаем vector store с разными метриками
        config = VectorStoreConfig(
            similarity_metric=metric,
            collection_name="review_embeddings"
        )
        
        vector_store = VectorStore(config)
        
        # Получаем embedding для запроса
        from pavel.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
        generator = EmbeddingGenerator(EmbeddingConfig(
            model_name=SupportedModels.E5_SMALL_MULTILINGUAL.value
        ))
        
        query_embedding = generator.generate_single(query_text)
        
        # Поиск с разными метриками
        from pavel.embeddings.vector_store import SearchQuery
        search_query = SearchQuery(
            vector=query_embedding.embedding,
            limit=3,
            min_similarity=0.3
        )
        
        results = vector_store.search_similar(search_query)
        
        print(f"   Результатов: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"   {i}. [{result.score:.3f}] {result.text[:50]}...")

async def demo_similarity_use_cases():
    """Демонстрация практических случаев настройки близости"""
    print("\n" + "=" * 60)
    print("🎯 ПРАКТИЧЕСКИЕ СЛУЧАИ НАСТРОЙКИ")
    print("-" * 60)
    
    pipeline = EmbeddingPipeline()
    search_engine = pipeline.semantic_search
    
    use_cases = [
        {
            "name": "🔍 Поиск дубликатов (высокая точность)",
            "threshold": 0.85,
            "query": "app crashes during startup",
            "description": "Для поиска почти идентичных отзывов"
        },
        {
            "name": "🐛 Поиск похожих проблем (средняя точность)", 
            "threshold": 0.65,
            "query": "payment problems",
            "description": "Для группировки схожих проблем"
        },
        {
            "name": "📊 Анализ тематик (низкая точность)",
            "threshold": 0.45,
            "query": "user interface",
            "description": "Для широкого анализа тем"
        },
        {
            "name": "🌍 Кросс-языковой поиск",
            "threshold": 0.55,
            "query": "good application",
            "description": "Поиск по всем языкам"
        }
    ]
    
    for case in use_cases:
        print(f"\n{case['name']}")
        print(f"   Порог: {case['threshold']}")
        print(f"   Запрос: '{case['query']}'")
        print(f"   Цель: {case['description']}")
        
        results = search_engine.search_similar_reviews(
            query_text=case['query'],
            limit=3,
            min_similarity=case['threshold']
        )
        
        print(f"   Найдено: {len(results)} результатов")
        for i, result in enumerate(results, 1):
            print(f"   {i}. [{result.similarity_score:.3f}] {result.text[:55]}...")

async def show_configuration_options():
    """Показываем все доступные настройки"""
    print("\n" + "=" * 60)
    print("⚙️  ДОСТУПНЫЕ НАСТРОЙКИ БЛИЗОСТИ")
    print("-" * 60)
    
    print("""
🎛️  ПАРАМЕТРЫ ПОИСКА:

1. min_similarity (0.0-1.0):
   • 0.9+ : Практически идентичные отзывы
   • 0.8-0.9 : Очень похожие отзывы  
   • 0.7-0.8 : Похожие отзывы
   • 0.6-0.7 : Схожая тематика
   • 0.5-0.6 : Общие темы
   • 0.3-0.5 : Широкий тематический поиск
   • 0.1-0.3 : Очень широкий поиск

2. limit (количество результатов):
   • 1-10 : Топ результаты
   • 10-50 : Расширенный анализ
   • 50-100 : Массовый анализ

3. Метрики близости:
   • COSINE : Угол между векторами (по умолчанию)
   • DOT_PRODUCT : Скалярное произведение  
   • EUCLIDEAN : Евклидово расстояние
   • MANHATTAN : Манхэттенское расстояние

4. Модели эмбеддингов:
   • E5-small : Быстро, 384D (текущая)
   • E5-base : Баланс, 768D  
   • E5-large : Точно, 1024D
   • OpenAI ada-002 : 1536D (API)

5. Фильтры:
   • По языку : language='ru'
   • По дате : created_after='2024-01-01'
   • По приложению : app_id='specific_app'
   • По рейтингу : rating>=4

🎯 РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ:

• Поиск дубликатов: threshold=0.85+
• Группировка проблем: threshold=0.65-0.75  
• Анализ настроений: threshold=0.55-0.65
• Поиск трендов: threshold=0.45-0.55
• Широкий анализ: threshold=0.35-0.45
""")

async def main():
    """Основная функция демонстрации"""
    await test_similarity_tuning()
    await test_advanced_similarity_config() 
    await demo_similarity_use_cases()
    await show_configuration_options()
    
    print("\n" + "=" * 60)
    print("🎉 ВЫВОД: Близость настраивается очень гибко!")
    print("🔧 Можно точно контролировать результаты поиска")
    print("🌍 Работает across языков и культур")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())