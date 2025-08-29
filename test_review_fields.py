import json
from google_play_scraper import reviews, Sort

def test_review_fields():
    """Test what fields are available in Google Play reviews"""
    
    app_id = "com.instagram.android"
    
    print("Testing available fields from Google Play scraper...")
    
    # Get a few reviews
    result, token = reviews(
        app_id=app_id,
        lang='en',
        country='us',
        count=5
    )
    
    if result:
        print(f"\nFound {len(result)} reviews")
        print("Available fields in review:")
        
        # Print first review structure
        first_review = result[0]
        for key, value in first_review.items():
            print(f"  {key}: {type(value).__name__} = {value}")
            
        print("\nJSON structure:")
        print(json.dumps(first_review, indent=2, default=str))
    else:
        print("No reviews found")

if __name__ == "__main__":
    test_review_fields()