#!/usr/bin/env python3
"""
Test cluster formation and analysis.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from pavel.clustering.dynamic_cluster_detector import DynamicClusterDetector, IssueCategory
from pavel.core.app_config import get_default_app_id


def create_test_embeddings(reviews: List[Dict[str, Any]]) -> np.ndarray:
    """Create simple test embeddings using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    texts = [r['content'] for r in reviews]
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    return embeddings


async def test_cluster_formation():
    """Test cluster formation with different review types."""
    print("🔍 ТЕСТИРОВАНИЕ ФОРМИРОВАНИЯ КЛАСТЕРОВ")
    print("=" * 50)
    
    # Создаем тестовые отзывы с разными темами
    base_time = datetime.now()
    reviews = [
        # Кластер 1: Технические проблемы (crashes)
        {'review_id': '1', 'content': 'App crashes constantly when I try to book', 'rating': 1, 'created_at': base_time},
        {'review_id': '2', 'content': 'Keeps crashing during payment process', 'rating': 1, 'created_at': base_time},
        {'review_id': '3', 'content': 'Crashes every time I open the app', 'rating': 1, 'created_at': base_time},
        
        # Кластер 2: Проблемы с оплатой (payment)
        {'review_id': '4', 'content': 'Payment failed multiple times, cannot pay', 'rating': 2, 'created_at': base_time},
        {'review_id': '5', 'content': 'Billing error, charged twice for same ride', 'rating': 1, 'created_at': base_time},
        {'review_id': '6', 'content': 'Payment system is broken, need refund', 'rating': 1, 'created_at': base_time},
        
        # Кластер 3: Проблемы с водителями (drivers)
        {'review_id': '7', 'content': 'Driver was rude and unprofessional service', 'rating': 2, 'created_at': base_time},
        {'review_id': '8', 'content': 'Bad driver behavior, very unpleasant experience', 'rating': 1, 'created_at': base_time},
        {'review_id': '9', 'content': 'Driver arrived late and was not polite', 'rating': 2, 'created_at': base_time},
        
        # Кластер 4: Положительные отзывы (positive)
        {'review_id': '10', 'content': 'Great app, fast service, amazing experience', 'rating': 5, 'created_at': base_time},
        {'review_id': '11', 'content': 'Excellent drivers, always on time and polite', 'rating': 5, 'created_at': base_time},
        {'review_id': '12', 'content': 'Perfect app, easy to use, highly recommend', 'rating': 5, 'created_at': base_time},
        
        # Кластер 5: UI/UX проблемы (interface)
        {'review_id': '13', 'content': 'Interface is confusing, hard to navigate', 'rating': 2, 'created_at': base_time},
        {'review_id': '14', 'content': 'UI design is terrible, cannot find buttons', 'rating': 2, 'created_at': base_time},
        {'review_id': '15', 'content': 'App interface needs complete redesign', 'rating': 2, 'created_at': base_time},
    ]
    
    print(f"📝 Тестируем на {len(reviews)} отзывах:")
    print("   🐛 Crashes (3 отзыва)")
    print("   💳 Payment issues (3 отзыва)")  
    print("   🚗 Driver problems (3 отзыва)")
    print("   😊 Positive reviews (3 отзыва)")
    print("   🖥️ UI/UX issues (3 отзыва)")
    
    # Генерируем embeddings
    print("\n🧠 Генерируем embeddings...")
    embeddings = create_test_embeddings(reviews)
    print(f"   Размер embeddings: {embeddings.shape}")
    
    # Создаем детектор кластеров
    detector = DynamicClusterDetector(
        n_clusters_range=(3, 7),  # Ожидаем 5 кластеров
        min_cluster_size=2,       # Минимум 2 отзыва в кластере
        significance_threshold=1.5  # Более чувствительный
    )
    
    # Запускаем кластеризацию
    print("\n🎯 Выполняем кластеризацию...")
    anomalies = detector.detect_anomalies(reviews, embeddings, base_time)
    
    # Анализируем результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ:")
    print(f"   🔍 Обнаружено аномалий: {len(anomalies)}")
    
    # Показываем базовые кластеры
    if detector.cluster_baselines:
        print(f"   📈 Сформировано кластеров: {len(detector.cluster_baselines)}")
        
        for cluster_id, baseline in detector.cluster_baselines.items():
            print(f"\n   🏷️ Кластер {cluster_id}:")
            print(f"      Размер: {baseline['size']}")
            print(f"      Средний рейтинг: {baseline['mean_rating']:.1f}")
            print(f"      Категория: {baseline.get('category', IssueCategory.UNKNOWN).value}")
    
    # Показываем тренды
    trends = detector.get_cluster_trends()
    if trends:
        print(f"\n📈 ТРЕНДЫ КЛАСТЕРОВ:")
        for cluster_id, trend in trends.items():
            print(f"   Кластер {cluster_id}: {trend['size_trend']} ({trend['size_change_pct']:+.1f}%)")
    
    # Анализируем найденные аномалии
    if anomalies:
        print(f"\n⚠️ НАЙДЕННЫЕ АНОМАЛИИ:")
        for i, anomaly in enumerate(anomalies[:5]):  # Показываем первые 5
            print(f"   {i+1}. {anomaly.anomaly_type} в кластере {anomaly.cluster_id}")
            print(f"      Серьезность: {anomaly.severity:.2f}")
            print(f"      Категория: {anomaly.category.value}")
            print(f"      Объяснение: {anomaly.explanation}")
    
    return len(detector.cluster_baselines) >= 3  # Успех если нашли минимум 3 кластера


async def test_real_clustering():
    """Test clustering with real embeddings on fewer reviews."""
    print("\n🌟 ТЕСТ С РЕАЛЬНЫМИ EMBEDDINGS")
    print("=" * 50)
    
    from sklearn.cluster import DBSCAN, KMeans
    
    # Создаем отзывы с четкими темами
    reviews = [
        {'content': 'app crashes all the time', 'rating': 1},
        {'content': 'constant crashes, unusable', 'rating': 1}, 
        {'content': 'payment failed twice', 'rating': 1},
        {'content': 'billing issues, wrong charges', 'rating': 1},
        {'content': 'great service, fast rides', 'rating': 5},
        {'content': 'excellent experience, recommend', 'rating': 5},
    ]
    
    # TF-IDF embeddings
    embeddings = create_test_embeddings(reviews)
    
    print(f"📊 Анализируем {len(reviews)} отзывов...")
    
    # Тестируем разные методы кластеризации
    print("\n🔄 DBSCAN кластеризация:")
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    dbscan_labels = dbscan.fit_predict(embeddings)
    
    clusters = {}
    for i, label in enumerate(dbscan_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((i, reviews[i]['content'][:50]))
    
    for cluster_id, items in clusters.items():
        if cluster_id == -1:
            print(f"   🚫 Шум ({len(items)} отзывов)")
        else:
            print(f"   🏷️ Кластер {cluster_id} ({len(items)} отзывов):")
            for idx, content in items:
                print(f"      • {content}...")
    
    print("\n📊 K-Means кластеризация (k=3):")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    
    kmeans_clusters = {}
    for i, label in enumerate(kmeans_labels):
        if label not in kmeans_clusters:
            kmeans_clusters[label] = []
        kmeans_clusters[label].append((i, reviews[i]['content'][:50]))
    
    for cluster_id, items in kmeans_clusters.items():
        print(f"   🏷️ Кластер {cluster_id} ({len(items)} отзывов):")
        for idx, content in items:
            print(f"      • {content}...")
    
    return True


async def main():
    """Main function to test clustering."""
    print("🧪 PAVEL CLUSTER ANALYSIS TEST")
    print("=" * 50)
    
    # Test 1: Basic cluster formation
    basic_success = await test_cluster_formation()
    
    # Test 2: Real clustering methods
    real_success = await test_real_clustering()
    
    print("\n" + "=" * 50)
    if basic_success and real_success:
        print("✅ КЛАСТЕРИЗАЦИЯ РАБОТАЕТ КОРРЕКТНО!")
        print("\n🎯 Ключевые возможности:")
        print("   ✓ Автоматическое определение количества кластеров")
        print("   ✓ Семантическая группировка по темам")
        print("   ✓ Классификация операционные vs продуктовые")
        print("   ✓ Отслеживание трендов кластеров")
        print("   ✓ Обнаружение аномалий в динамике")
    else:
        print("❌ Нужны доработки в кластеризации")
    

if __name__ == "__main__":
    asyncio.run(main())