"""
Application configuration helpers.

Reads default app settings from environment variables.
"""

import os
from typing import Optional

def get_default_app_id() -> str:
    """
    Get default app ID from environment or fallback to inDriver.
    
    Environment variable: PAVEL_DEFAULT_APP_ID
    Default: sinet.startup.inDriver (inDriver)
    """
    return os.getenv('PAVEL_DEFAULT_APP_ID', 'sinet.startup.inDriver')

def get_default_app_name() -> str:
    """
    Get human-readable app name for the default app.
    """
    app_id = get_default_app_id()
    
    # Map common app IDs to readable names
    app_names = {
        'sinet.startup.inDriver': 'inDriver',
        'com.whatsapp': 'WhatsApp',
        'com.instagram.android': 'Instagram',
        'com.ubercab': 'Uber',
        'com.zhiliaoapp.musically': 'TikTok'
    }
    
    return app_names.get(app_id, app_id)

def get_default_app_url() -> str:
    """
    Get Google Play Store URL for the default app.
    """
    app_id = get_default_app_id()
    return f"https://play.google.com/store/apps/details?id={app_id}&hl=en"

def get_default_app_info() -> dict:
    """
    Get complete information about the default app.
    """
    app_id = get_default_app_id()
    return {
        'app_id': app_id,
        'name': get_default_app_name(),
        'url': get_default_app_url(),
        'description': _get_app_description(app_id)
    }

def _get_app_description(app_id: str) -> str:
    """Get description for known apps."""
    descriptions = {
        'sinet.startup.inDriver': 'Ride-hailing app with diverse operational and product issues',
        'com.whatsapp': 'Messaging app with high review volume',
        'com.instagram.android': 'Social media app with frequent updates',
        'com.ubercab': 'Ride-hailing with operational complexity',
        'com.zhiliaoapp.musically': 'Short video platform with diverse user base'
    }
    
    return descriptions.get(app_id, 'Mobile application')