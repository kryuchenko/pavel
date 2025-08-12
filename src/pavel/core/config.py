"""Configuration management for PAVEL"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()


class Config:
    """Central configuration for PAVEL pipeline"""
    
    def __init__(self):
        # Default app configuration
        self.DEFAULT_APP_ID = os.getenv("PAVEL_DEFAULT_APP_ID", "sinet.startup.inDriver")
        
        # MongoDB configuration
        self.DB_URI = os.getenv("PAVEL_DB_URI", "mongodb://localhost:27017")
        self.DB_NAME = os.getenv("PAVEL_DB_NAME", "gp")
        self.COLLECTION_REVIEWS = os.getenv("PAVEL_COLLECTION_REVIEWS", "reviews")
        self.COLLECTION_CLUSTERS = os.getenv("PAVEL_COLLECTION_CLUSTERS", "clusters")
        
        # Properties for ingester compatibility
        self.MONGODB_URI = self.DB_URI
        self.MONGODB_DATABASE = self.DB_NAME
        
        # Google Play API configuration
        self.GOOGLE_PLAY_DEVELOPER_EMAIL = os.getenv("GOOGLE_PLAY_DEVELOPER_EMAIL")
        self.GOOGLE_PLAY_KEY_FILE_PATH = os.getenv("GOOGLE_PLAY_KEY_FILE_PATH")
        self.GOOGLE_PLAY_PACKAGE_NAME = os.getenv("GOOGLE_PLAY_PACKAGE_NAME", self.DEFAULT_APP_ID)
        
        # Pipeline configuration
        self.BATCH_SIZE = int(os.getenv("PAVEL_BATCH_SIZE", "100"))
        self.RATE_LIMIT_REQUESTS = int(os.getenv("PAVEL_RATE_LIMIT_REQUESTS", "60"))
        self.RATE_LIMIT_PERIOD = int(os.getenv("PAVEL_RATE_LIMIT_PERIOD", "60"))
        self.RETRY_ATTEMPTS = int(os.getenv("PAVEL_RETRY_ATTEMPTS", "3"))
        self.RETRY_BACKOFF = float(os.getenv("PAVEL_RETRY_BACKOFF", "2"))
        
        # History window
        self.HISTORY_WINDOW_DAYS = int(os.getenv("PAVEL_HISTORY_WINDOW_DAYS", "90"))
        
        # Default locales
        locales_str = os.getenv("PAVEL_DEFAULT_LOCALES", "en_US,ru_RU,es_ES,pt_BR,id_ID,kk_KZ")
        self.DEFAULT_LOCALES = [loc.strip() for loc in locales_str.split(",")]
        
        # Logging configuration
        self.LOG_LEVEL = os.getenv("PAVEL_LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("PAVEL_LOG_FORMAT", "json")
        self.LOG_FILE = os.getenv("PAVEL_LOG_FILE", "./logs/pavel.log")
        
        # Feature flags
        self.ENABLE_DEDUPLICATION = os.getenv("PAVEL_ENABLE_DEDUPLICATION", "true").lower() == "true"
        self.ENABLE_COMPLAINT_FILTER = os.getenv("PAVEL_ENABLE_COMPLAINT_FILTER", "true").lower() == "true"
        self.ENABLE_AUTO_CLUSTERING = os.getenv("PAVEL_ENABLE_AUTO_CLUSTERING", "true").lower() == "true"
        self.ENABLE_TREND_ANALYSIS = os.getenv("PAVEL_ENABLE_TREND_ANALYSIS", "true").lower() == "true"
        
        # API configuration
        self.API_PORT = int(os.getenv("PAVEL_API_PORT", "8080"))
        self.API_HOST = os.getenv("PAVEL_API_HOST", "0.0.0.0")
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            Path(self.LOG_FILE).parent,
            Path("./models"),
            Path("./data"),
            Path("./secrets"),
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_app_id(self, app_id: Optional[str] = None) -> str:
        """
        Get app ID with fallback to default
        
        Args:
            app_id: Optional app ID override
            
        Returns:
            App ID to use (provided or default)
        """
        return app_id or self.DEFAULT_APP_ID
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required settings for Google Play API
        if not self.GOOGLE_PLAY_DEVELOPER_EMAIL:
            errors.append("GOOGLE_PLAY_DEVELOPER_EMAIL is not set")
        
        if not self.GOOGLE_PLAY_KEY_FILE_PATH:
            errors.append("GOOGLE_PLAY_KEY_FILE_PATH is not set")
        elif not Path(self.GOOGLE_PLAY_KEY_FILE_PATH).exists():
            errors.append(f"Google Play key file not found: {self.GOOGLE_PLAY_KEY_FILE_PATH}")
        
        # Check MongoDB connection
        if not self.DB_URI:
            errors.append("PAVEL_DB_URI is not set")
        
        return errors
    
    def __repr__(self):
        return f"<PAVEL Config: app={self.DEFAULT_APP_ID}, db={self.DB_NAME}>"


# Global config instance
config = Config()

def get_config() -> Config:
    """Get global config instance"""
    return config