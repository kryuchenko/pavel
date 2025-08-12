"""
Batch processing coordinator for large-scale ingestion operations.

Handles multiple apps, scheduling, and progress tracking.
"""

import asyncio
from typing import List, Dict, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import json

from pavel.core.config import get_config  
from pavel.core.logger import get_logger
from .google_play import GooglePlayIngester, IngestionStats

logger = get_logger(__name__)

@dataclass
class BatchJob:
    """Configuration for a batch ingestion job"""
    app_id: str
    locales: List[str] = None
    days_back: int = 90
    batch_size: int = 200
    mode: str = "batch"  # "batch" or "incremental"
    priority: int = 0    # Higher = more important
    
    def __post_init__(self):
        if self.locales is None:
            self.locales = ['en', 'ru']

@dataclass 
class BatchResult:
    """Result of batch processing job"""
    job: BatchJob
    stats: List[IngestionStats]
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    def total_reviews_fetched(self) -> int:
        return sum(s.total_fetched for s in self.stats)
        
    def total_new_reviews(self) -> int:
        return sum(s.new_reviews for s in self.stats)
        
    def total_duplicates(self) -> int:
        return sum(s.duplicates for s in self.stats)
        
    def total_errors(self) -> int:
        return sum(s.errors for s in self.stats)

class BatchProcessor:
    """
    Coordinates batch ingestion across multiple apps and locales.
    
    Features:
    - Concurrent processing with limits
    - Progress tracking and reporting
    - Error recovery and retry logic
    - Job prioritization
    """
    
    def __init__(self, max_concurrent: int = 3):
        self.config = get_config()
        self.max_concurrent = max_concurrent
        self.ingester = None
        self.progress_callback: Optional[Callable] = None
        
    def set_progress_callback(self, callback: Callable[[BatchResult], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    async def _process_single_job(self, job: BatchJob) -> BatchResult:
        """Process a single batch job"""
        logger.info(f"Starting batch job: {job.app_id} ({job.mode})")
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self.ingester:
                self.ingester = GooglePlayIngester()
                
            if job.mode == "batch":
                stats = await self.ingester.ingest_batch_history(
                    app_id=job.app_id,
                    locales=job.locales,
                    days_back=job.days_back,
                    batch_size=job.batch_size
                )
            elif job.mode == "incremental":
                stats = await self.ingester.ingest_incremental(
                    app_id=job.app_id,
                    locales=job.locales,
                    batch_size=job.batch_size
                )
            else:
                raise ValueError(f"Unknown mode: {job.mode}")
                
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = BatchResult(
                job=job,
                stats=stats,
                success=True,
                duration_seconds=duration
            )
            
            logger.info(f"Completed batch job: {job.app_id} - "
                       f"{result.total_new_reviews()} new reviews in {duration:.1f}s")
            
            return result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Batch job failed: {job.app_id} - {e}")
            
            return BatchResult(
                job=job,
                stats=[],
                success=False,
                error=str(e),
                duration_seconds=duration
            )
            
    async def process_jobs(self, jobs: List[BatchJob]) -> List[BatchResult]:
        """
        Process multiple batch jobs with concurrency control.
        
        Jobs are sorted by priority (highest first).
        """
        if not jobs:
            return []
            
        # Sort by priority (highest first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        logger.info(f"Processing {len(sorted_jobs)} batch jobs with max_concurrent={self.max_concurrent}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(job: BatchJob) -> BatchResult:
            async with semaphore:
                result = await self._process_single_job(job)
                
                # Call progress callback if set
                if self.progress_callback:
                    try:
                        self.progress_callback(result)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                        
                return result
                
        # Execute all jobs concurrently
        tasks = [process_with_semaphore(job) for job in sorted_jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Unhandled job exception: {result}")
                final_results.append(BatchResult(
                    job=sorted_jobs[i],
                    stats=[],
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)
                
        return final_results
        
    def create_default_jobs(self, app_ids: List[str], mode: str = "batch") -> List[BatchJob]:
        """Create default batch jobs for list of app IDs"""
        jobs = []
        
        for i, app_id in enumerate(app_ids):
            job = BatchJob(
                app_id=app_id,
                mode=mode,
                priority=len(app_ids) - i  # First apps get higher priority
            )
            jobs.append(job)
            
        return jobs
        
    def generate_report(self, results: List[BatchResult]) -> Dict:
        """Generate summary report from batch results"""
        total_jobs = len(results)
        successful_jobs = sum(1 for r in results if r.success)
        failed_jobs = total_jobs - successful_jobs
        
        total_fetched = sum(r.total_reviews_fetched() for r in results)
        total_new = sum(r.total_new_reviews() for r in results)
        total_duplicates = sum(r.total_duplicates() for r in results)
        total_errors = sum(r.total_errors() for r in results)
        
        total_duration = sum(r.duration_seconds for r in results)
        avg_duration = total_duration / total_jobs if total_jobs > 0 else 0
        
        # App-level breakdown
        app_breakdown = {}
        for result in results:
            app_id = result.job.app_id
            app_breakdown[app_id] = {
                "success": result.success,
                "fetched": result.total_reviews_fetched(),
                "new": result.total_new_reviews(),
                "duplicates": result.total_duplicates(),
                "errors": result.total_errors(),
                "duration": result.duration_seconds,
                "locales": result.job.locales,
                "error_message": result.error
            }
            
        report = {
            "summary": {
                "total_jobs": total_jobs,
                "successful_jobs": successful_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
                "total_reviews_fetched": total_fetched,
                "total_new_reviews": total_new,
                "total_duplicates": total_duplicates,
                "total_errors": total_errors,
                "total_duration_seconds": total_duration,
                "average_duration_seconds": avg_duration,
                "reviews_per_second": total_fetched / total_duration if total_duration > 0 else 0
            },
            "apps": app_breakdown,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return report
        
    def save_report(self, report: Dict, filepath: str):
        """Save report to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch report saved to {filepath}")
        
    async def close(self):
        """Clean up resources"""
        if self.ingester:
            self.ingester.close()