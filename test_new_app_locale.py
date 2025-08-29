import asyncio
import sys
import os

# Add the project src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from pavel.ingestion.google_play import GooglePlayIngester

async def main():
    """
    Test script with a different app to verify locale/country fields.
    """
    # Используем другое популярное приложение
    app_id = "com.instagram.android"
    locales_to_fetch = ['es', 'fr']  # испанский и французский
    days_to_look_back = 7

    print(f"Testing locale/country saving for NEW app: {app_id}")
    print(f"Locales: {locales_to_fetch}, Days back: {days_to_look_back}")

    ingester = None
    try:
        ingester = GooglePlayIngester()
        stats = await ingester.ingest_batch_history(
            app_id=app_id,
            locales=locales_to_fetch,
            days_back=days_to_look_back
        )

        print("\n--- Test Results ---")
        for s in stats:
            print(f"Locale: '{s.locale}'")
            print(f"  - Fetched: {s.total_fetched}")
            print(f"  - New:     {s.new_reviews}")
        print("-------------------\n")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ingester:
            ingester.close()

if __name__ == "__main__":
    asyncio.run(main())