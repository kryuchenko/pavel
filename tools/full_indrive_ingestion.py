#!/usr/bin/env python3
"""
Full inDriver review ingestion across all available locales

This script comprehensively ingests ALL inDriver reviews from Google Play
across all supported locales for maximum dataset coverage.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pavel.core.logger import get_logger
from pavel.ingestion.google_play import GooglePlayIngester
from pavel.core.config import get_config
import pymongo

logger = get_logger(__name__)

class FullInDriverIngestion:
    """Complete inDriver review ingestion"""
    
    def __init__(self):
        self.config = get_config()
        self.ingester = GooglePlayIngester()
        
        # Comprehensive locale list for inDriver (ride-hailing is global)
        self.locales = [
            # Major markets
            'en_US', 'en_GB', 'en_AU', 'en_CA', 'en_ZA', 'en_IN',
            'ru_RU', 'ru_BY', 'ru_KZ', 'ru_UA', 'ru_MD',
            'es_ES', 'es_MX', 'es_AR', 'es_CO', 'es_PE', 'es_CL', 'es_EC',
            'pt_BR', 'pt_PT', 
            'fr_FR', 'fr_CA', 'fr_BE',
            'de_DE', 'de_AT', 'de_CH',
            'it_IT', 'it_CH',
            'pl_PL', 'pl_UA',
            'tr_TR',
            'ar_SA', 'ar_EG', 'ar_AE', 'ar_LB', 'ar_JO',
            'hi_IN', 'bn_BD', 'ur_PK',
            'zh_CN', 'zh_TW', 'zh_HK',
            'ja_JP', 'ko_KR',
            'th_TH', 'vi_VN', 'id_ID', 'ms_MY',
            'nl_NL', 'nl_BE',
            'sv_SE', 'no_NO', 'da_DK', 'fi_FI',
            'cs_CZ', 'sk_SK', 'hu_HU', 'ro_RO', 'bg_BG',
            'hr_HR', 'sr_RS', 'sl_SI', 'mk_MK', 'sq_AL',
            'et_EE', 'lv_LV', 'lt_LT',
            'el_GR', 'mt_MT',
            'ka_GE', 'am_ET', 'hy_AM', 'az_AZ',
            'he_IL', 'fa_IR',
            'sw_KE', 'sw_TZ', 'zu_ZA', 'af_ZA',
        ]
        
    async def get_current_stats(self):
        """Get current database statistics"""
        client = pymongo.MongoClient(self.config.DB_URI)
        db = client[self.config.DB_NAME]
        reviews = db.reviews
        
        total = reviews.count_documents({})
        indrive_total = reviews.count_documents({'appId': 'sinet.startup.inDriver'})
        
        # Get locale distribution
        pipeline = [
            {'$match': {'appId': 'sinet.startup.inDriver'}},
            {'$group': {'_id': '$locale', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 20}
        ]
        
        try:
            locale_stats = list(reviews.aggregate(pipeline))
        except:
            # Fallback if aggregation fails
            locale_stats = []
            locales = reviews.distinct('locale', {'appId': 'sinet.startup.inDriver'})[:10]
            for locale in locales:
                count = reviews.count_documents({'appId': 'sinet.startup.inDriver', 'locale': locale})
                locale_stats.append({'_id': locale, 'count': count})
        
        return {
            'total_reviews': total,
            'indrive_reviews': indrive_total,
            'locale_distribution': locale_stats
        }
    
    async def ingest_comprehensive(self, days_back: int = 365):
        """Ingest reviews from all locales comprehensively"""
        
        print(f"🚀 Starting comprehensive inDriver ingestion")
        print(f"📅 History: {days_back} days")
        print(f"🌍 Locales to try: {len(self.locales)}")
        
        # Get baseline stats
        initial_stats = await self.get_current_stats()
        print(f"📊 Initial database state:")
        print(f"   Total reviews: {initial_stats['total_reviews']:,}")
        print(f"   inDriver reviews: {initial_stats['indrive_reviews']:,}")
        
        successful_locales = []
        failed_locales = []
        total_new_reviews = 0
        
        for i, locale in enumerate(self.locales, 1):
            print(f"\n🔄 [{i}/{len(self.locales)}] Processing {locale}...")
            
            try:
                stats_list = await self.ingester.ingest_batch_history(
                    app_id="sinet.startup.inDriver",
                    locales=[locale],
                    days_back=days_back
                )
                
                # ingest_batch_history returns a list of stats
                stats = stats_list[0] if stats_list else None
                if stats and stats.total_fetched > 0:
                    successful_locales.append((locale, stats.new_reviews))
                    total_new_reviews += stats.new_reviews
                    print(f"   ✅ {locale}: {stats.new_reviews} new reviews "
                          f"({stats.total_fetched} total fetched)")
                else:
                    print(f"   ⚪ {locale}: No reviews available")
                    
            except Exception as e:
                failed_locales.append((locale, str(e)))
                print(f"   ❌ {locale}: {str(e)[:50]}")
                continue
        
        # Final stats
        final_stats = await self.get_current_stats()
        
        print(f"\n🎯 Comprehensive Ingestion Results:")
        print(f"   📈 Reviews before: {initial_stats['indrive_reviews']:,}")
        print(f"   📈 Reviews after: {final_stats['indrive_reviews']:,}")
        print(f"   📈 New reviews added: {final_stats['indrive_reviews'] - initial_stats['indrive_reviews']:,}")
        print(f"   ✅ Successful locales: {len(successful_locales)}")
        print(f"   ❌ Failed locales: {len(failed_locales)}")
        
        if successful_locales:
            print(f"\n🏆 Top productive locales:")
            sorted_locales = sorted(successful_locales, key=lambda x: x[1], reverse=True)
            for locale, count in sorted_locales[:10]:
                print(f"      {locale}: {count:,} new reviews")
        
        print(f"\n🌍 Current locale distribution:")
        for locale_stat in final_stats['locale_distribution'][:10]:
            print(f"      {locale_stat['_id']}: {locale_stat['count']:,} reviews")
        
        return final_stats

async def main():
    """Main ingestion process"""
    print("🏁 Full inDriver Review Ingestion")
    print("=" * 50)
    
    ingestion = FullInDriverIngestion()
    
    # Show plan
    print(f"🎯 Plan:")
    print(f"   📱 Target app: sinet.startup.inDriver")
    print(f"   🌍 Locales: {len(ingestion.locales)} worldwide")  
    print(f"   📅 History: 365 days (1 year)")
    print(f"   🎲 Expected: 5,000-15,000 reviews")
    
    # Auto-confirm for automation
    print(f"\n🚀 Starting comprehensive ingestion...")
    print(f"⏱️ Estimated time: 1-2 hours")
    
    try:
        start_time = datetime.now()
        
        # Run comprehensive ingestion
        final_stats = await ingestion.ingest_comprehensive(days_back=365)
        
        duration = datetime.now() - start_time
        print(f"\n⏱️ Total time: {duration}")
        print(f"🎉 Comprehensive ingestion completed!")
        print(f"📊 Final dataset: {final_stats['indrive_reviews']:,} inDriver reviews")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Ingestion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))