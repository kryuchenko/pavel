"""
Incremental ingestion scheduler for automated review updates.

Runs periodic jobs to keep review data current.
"""

import asyncio
import schedule
from typing import List, Dict, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import json
import threading
import time

from pavel.core.config import get_config
from pavel.core.logger import get_logger
from .batch_processor import BatchProcessor, BatchJob

logger = get_logger(__name__)

@dataclass
class ScheduleConfig:
    """Configuration for scheduled ingestion"""
    app_id: str
    locales: List[str] = None
    schedule_type: str = "hourly"  # "hourly", "daily", "weekly"
    batch_size: int = 100
    enabled: bool = True
    last_run: Optional[datetime] = None
    
    def __post_init__(self):
        if self.locales is None:
            self.locales = ['en', 'ru']

class IncrementalScheduler:
    """
    Manages scheduled incremental ingestion for multiple apps.
    
    Features:
    - Flexible scheduling (hourly, daily, weekly)
    - Per-app configuration
    - Automatic error recovery
    - Status tracking and reporting
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_config()
        self.config_file = config_file
        self.app_configs: Dict[str, ScheduleConfig] = {}
        self.batch_processor = BatchProcessor(max_concurrent=2)
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Load configuration
        if config_file:
            self.load_config(config_file)
            
        # Set up progress tracking
        self.batch_processor.set_progress_callback(self._on_job_complete)
        
    def add_app(self, app_config: ScheduleConfig):
        """Add app to scheduled ingestion"""
        self.app_configs[app_config.app_id] = app_config
        logger.info(f"Added scheduled ingestion for {app_config.app_id} ({app_config.schedule_type})")
        
    def remove_app(self, app_id: str):
        """Remove app from scheduled ingestion"""
        if app_id in self.app_configs:
            del self.app_configs[app_id]
            logger.info(f"Removed scheduled ingestion for {app_id}")
            
    def enable_app(self, app_id: str, enabled: bool = True):
        """Enable/disable scheduled ingestion for app"""
        if app_id in self.app_configs:
            self.app_configs[app_id].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Scheduled ingestion {status} for {app_id}")
            
    def _schedule_app_jobs(self):
        """Set up schedule for all configured apps"""
        schedule.clear()  # Clear existing schedules
        
        for app_id, config in self.app_configs.items():
            if not config.enabled:
                continue
                
            job_func = lambda app_id=app_id: asyncio.create_task(self._run_incremental_job(app_id))
            
            if config.schedule_type == "hourly":
                schedule.every().hour.do(job_func)
            elif config.schedule_type == "daily":
                schedule.every().day.at("02:00").do(job_func)  # 2 AM
            elif config.schedule_type == "weekly":
                schedule.every().monday.at("01:00").do(job_func)  # Monday 1 AM
            else:
                logger.warning(f"Unknown schedule type: {config.schedule_type} for {app_id}")
                
        logger.info(f"Scheduled {len([c for c in self.app_configs.values() if c.enabled])} apps")
        
    async def _run_incremental_job(self, app_id: str):
        """Run incremental ingestion job for single app"""
        if app_id not in self.app_configs:
            logger.warning(f"No config found for {app_id}")
            return
            
        config = self.app_configs[app_id]
        
        if not config.enabled:
            logger.debug(f"Skipping disabled app: {app_id}")
            return
            
        logger.info(f"Running scheduled incremental ingestion for {app_id}")
        
        job = BatchJob(
            app_id=app_id,
            locales=config.locales,
            batch_size=config.batch_size,
            mode="incremental"
        )
        
        try:
            results = await self.batch_processor.process_jobs([job])
            
            if results and results[0].success:
                config.last_run = datetime.now(timezone.utc)
                logger.info(f"Scheduled job completed for {app_id}: {results[0].total_new_reviews()} new reviews")
            else:
                error = results[0].error if results else "Unknown error"
                logger.error(f"Scheduled job failed for {app_id}: {error}")
                
        except Exception as e:
            logger.error(f"Scheduled job exception for {app_id}: {e}")
            
    def _on_job_complete(self, result):
        """Callback for job completion (for monitoring)"""
        app_id = result.job.app_id
        if result.success:
            logger.debug(f"Job completed: {app_id} - {result.total_new_reviews()} new reviews")
        else:
            logger.warning(f"Job failed: {app_id} - {result.error}")
            
    def _scheduler_loop(self):
        """Main scheduler loop (runs in separate thread)"""
        logger.info("Scheduler loop started")
        
        while not self.stop_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
                
        logger.info("Scheduler loop stopped")
        
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        self._schedule_app_jobs()
        
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.is_running = True
        logger.info("Incremental scheduler started")
        
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
            
        self.stop_event.set()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
            
        self.is_running = False
        logger.info("Incremental scheduler stopped")
        
    async def run_manual_batch(self, app_ids: List[str]) -> List:
        """Run manual incremental batch for specified apps"""
        jobs = []
        
        for app_id in app_ids:
            if app_id in self.app_configs:
                config = self.app_configs[app_id]
                job = BatchJob(
                    app_id=app_id,
                    locales=config.locales,
                    batch_size=config.batch_size,
                    mode="incremental"
                )
                jobs.append(job)
            else:
                # Use default config
                job = BatchJob(app_id=app_id, mode="incremental")
                jobs.append(job)
                
        logger.info(f"Running manual incremental batch for {len(jobs)} apps")
        results = await self.batch_processor.process_jobs(jobs)
        
        # Update last_run timestamps
        for result in results:
            app_id = result.job.app_id
            if app_id in self.app_configs and result.success:
                self.app_configs[app_id].last_run = datetime.now(timezone.utc)
                
        return results
        
    def get_status(self) -> Dict:
        """Get scheduler status and app information"""
        now = datetime.now(timezone.utc)
        
        app_status = {}
        for app_id, config in self.app_configs.items():
            next_run = None
            if config.enabled and config.last_run:
                if config.schedule_type == "hourly":
                    next_run = config.last_run + timedelta(hours=1)
                elif config.schedule_type == "daily":
                    next_run = config.last_run + timedelta(days=1)
                elif config.schedule_type == "weekly":
                    next_run = config.last_run + timedelta(weeks=1)
                    
            app_status[app_id] = {
                "enabled": config.enabled,
                "schedule_type": config.schedule_type,
                "locales": config.locales,
                "last_run": config.last_run.isoformat() if config.last_run else None,
                "next_run": next_run.isoformat() if next_run else None,
                "overdue": next_run < now if next_run else False
            }
            
        return {
            "scheduler_running": self.is_running,
            "total_apps": len(self.app_configs),
            "enabled_apps": len([c for c in self.app_configs.values() if c.enabled]),
            "apps": app_status,
            "timestamp": now.isoformat()
        }
        
    def load_config(self, filepath: str):
        """Load scheduler configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.app_configs.clear()
            
            for app_data in data.get('apps', []):
                config = ScheduleConfig(
                    app_id=app_data['app_id'],
                    locales=app_data.get('locales', ['en', 'ru']),
                    schedule_type=app_data.get('schedule_type', 'hourly'),
                    batch_size=app_data.get('batch_size', 100),
                    enabled=app_data.get('enabled', True)
                )
                
                # Parse last_run if present
                last_run_str = app_data.get('last_run')
                if last_run_str:
                    config.last_run = datetime.fromisoformat(last_run_str.replace('Z', '+00:00'))
                    
                self.app_configs[config.app_id] = config
                
            logger.info(f"Loaded configuration for {len(self.app_configs)} apps from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            
    def save_config(self, filepath: str):
        """Save scheduler configuration to JSON file"""
        try:
            apps_data = []
            
            for config in self.app_configs.values():
                app_data = {
                    "app_id": config.app_id,
                    "locales": config.locales,
                    "schedule_type": config.schedule_type,
                    "batch_size": config.batch_size,
                    "enabled": config.enabled
                }
                
                if config.last_run:
                    app_data["last_run"] = config.last_run.isoformat()
                    
                apps_data.append(app_data)
                
            data = {
                "apps": apps_data,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved configuration for {len(apps_data)} apps to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {e}")
            
    async def close(self):
        """Clean up resources"""
        self.stop()
        await self.batch_processor.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.close())