import asyncio
import sys
import os

# Add the project src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from pavel.ingestion.google_play import GooglePlayIngester

async def main():
    """
    Script to ingest Google Play reviews for Pokemon GO.
    """
    app_id = "com.nianticlabs.pokemongo"
    # Собираем отзывы на всех основных языках
    locales_to_fetch = [
        'en', 'ru', 'es', 'pt', 'fr', 'de', 'it', 'ja', 
        'ko', 'zh', 'ar', 'hi', 'id', 'tr', 'pl', 'nl',
        'sv', 'no', 'da', 'fi', 'cs', 'hu', 'ro', 'uk'
    ]
    days_to_look_back = 90

    print(f"Starting review ingestion for app: {app_id}")
    print(f"Locales: {locales_to_fetch}, Days back: {days_to_look_back}")

    ingester = None
    try:
        ingester = GooglePlayIngester()
        stats = await ingester.ingest_batch_history(
            app_id=app_id,
            locales=locales_to_fetch,
            days_back=days_to_look_back
        )

        print("\n--- Ingestion Report ---")
        for s in stats:
            print(f"Locale: '{s.locale}'")
            print(f"  - Fetched: {s.total_fetched}")
            print(f"  - New:     {s.new_reviews}")
            print(f"  - Dups:    {s.duplicates}")
            print(f"  - Errors:  {s.errors}")
            print(f"  - Speed:   {s.reviews_per_second():.2f} reviews/sec")
        print("------------------------\n")

        # Show summary for the app
        summary = ingester.get_ingestion_summary(app_id)
        print(f"Total reviews in database: {summary.get('total_reviews', 0)}")
        if summary.get('avg_score'):
            print(f"Average score: {summary['avg_score']:.2f}")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ingester:
            ingester.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # This allows running the async main function
    asyncio.run(main())