import asyncio
import sys
import os
import json

# Add the project src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from pavel.ingestion.google_play import GooglePlayIngester

async def main():
    """
    Test script to verify ALL fields are saved from Google Play scraper.
    """
    # Тестируем с новым приложением для чистых данных
    app_id = "com.zhiliaoapp.musically"  # TikTok
    locales_to_fetch = ['pt']  # Portuguese - активная аудитория
    days_to_look_back = 2

    print(f"Testing ALL fields saving for: {app_id}")
    print(f"Locales: {locales_to_fetch}, Days back: {days_to_look_back}")

    ingester = None
    try:
        ingester = GooglePlayIngester()
        stats = await ingester.ingest_batch_history(
            app_id=app_id,
            locales=locales_to_fetch,
            days_back=days_to_look_back
        )

        print("\n--- Results ---")
        for s in stats:
            print(f"Locale: '{s.locale}' - New: {s.new_reviews}")
        print("---------------\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ingester:
            ingester.close()

if __name__ == "__main__":
    asyncio.run(main())