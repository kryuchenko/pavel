#!/usr/bin/env python3
"""
CLI tool for searching complaints and visualizing trends over time.
Usage: python analyze_complaints.py com.example.app "payment issues" --months 6
"""

import asyncio
import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from scipy import ndimage

# Add project to path
sys.path.insert(0, 'src')

from pavel.search.vector_search import VectorSearchEngine, SearchQuery
from pavel.core.config import get_config
from pavel.core.logger import get_logger
from pymongo import MongoClient
import matplotlib
matplotlib.use('Agg')  # Set backend before pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

logger = get_logger(__name__)

class ComplaintAnalyzer:
    """Analyze complaints and visualize trends."""
    
    def __init__(self):
        self.config = get_config()
        self.mongo_client = MongoClient(self.config.MONGODB_URI)
        self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        
    async def search_and_analyze(self,
                                app_id: str,
                                query: str,
                                months: int = 3,
                                min_similarity: float = 0.8,
                                output_file: str = None,
                                normalized: bool = False) -> Dict:
        """
        Search for complaints and analyze trends over time.
        
        Args:
            app_id: Google Play app ID
            query: Search query for complaints
            months: Number of months to analyze
            min_similarity: Minimum similarity threshold
            output_file: Path to save graph (default: auto-generated)
            
        Returns:
            Analysis results
        """
        print(f"\n🔍 Analyzing: '{query}'")
        print(f"📱 App: {app_id}")
        print(f"📅 Period: last {months} months")
        print("-" * 50)
        
        # Initialize search engine
        search_engine = VectorSearchEngine(collection_name=app_id, mongo_client=self.mongo_client)
        
        # Calculate date range - use UTC to avoid timezone issues
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=months * 30)
        
        # Get total reviews count for normalization if needed
        total_reviews_count = 0
        daily_totals = {}
        weekly_totals = {}
        if normalized:
            print(f"📊 Getting daily/weekly review counts for normalization...")
            collection = self.db[app_id]
            
            # Get all reviews in the period and count by day/week (optimized query)
            all_reviews = collection.find({
                'at': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }, {'_id': 0, 'at': 1})
            
            for review in all_reviews:
                review_date = review.get('at')
                if not review_date:
                    continue
                
                # Convert to UTC consistently with main pipeline
                if isinstance(review_date, str):
                    review_date = datetime.fromisoformat(review_date.replace('Z', '+00:00'))
                if review_date.tzinfo is None:
                    review_date = review_date.replace(tzinfo=timezone.utc)
                else:
                    review_date = review_date.astimezone(timezone.utc)
                    
                # Count by day and week
                day_key = review_date.date()
                week_key = review_date.isocalendar()[:2]  # (year, week)
                
                daily_totals[day_key] = daily_totals.get(day_key, 0) + 1
                weekly_totals[week_key] = weekly_totals.get(week_key, 0) + 1
            
            total_reviews_count = sum(daily_totals.values())
            print(f"📈 Total reviews in period: {total_reviews_count:,}")
            print(f"📅 Daily totals calculated for {len(daily_totals)} days")
            print(f"📅 Weekly totals calculated for {len(weekly_totals)} weeks")
        
        # Search for similar complaints
        print(f"\n🔎 Searching for complaints similar to: '{query}'")
        
        search_query = SearchQuery(
            text=query,
            limit=100_000,  # Large limit instead of None (safer for some drivers)
            min_similarity=min_similarity,
            filter_params={
                'at': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            },  # Filter by date at MongoDB level for efficiency
            include_fields=['at', 'score', 'userName', 'locale', 'content']
        )
        
        raw_results = await search_engine.search(search_query)
        
        if not raw_results:
            print("❌ No matching complaints found")
            search_engine.close()
            return {}
        
        # Process results (date filtering now handled by MongoDB filter_params)
        results = []
        for result in raw_results:
            # Parse and normalize date
            review_date = result.document.get('at')
            if isinstance(review_date, str):
                try:
                    review_date = datetime.fromisoformat(review_date.replace('Z', '+00:00'))
                except:
                    continue
            
            if not review_date:
                continue
                
            # Convert to UTC if needed
            if review_date.tzinfo is None:
                review_date = review_date.replace(tzinfo=timezone.utc)
            else:
                review_date = review_date.astimezone(timezone.utc)
                
            results.append({
                'date': review_date,
                'rating': result.document.get('score', 0),  # Rating from review
                'similarity': result.similarity,  # Search similarity score
                'content': result.document.get('content', ''),
                'locale': result.document.get('locale', '')
            })
        
        if not results:
            print("❌ No matching complaints found in the specified period")
            search_engine.close()
            return {}
        
        print(f"✅ Found {len(results)} matching reviews (similarity >= {min_similarity}) in period")
        
        # Process results by time
        daily_counts = defaultdict(int)
        weekly_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        
        reviews_by_date = []
        scores_by_date = []
        
        for result in results:
            review_date = result['date']
            
            # Count by period
            day_key = review_date.date()
            week_key = review_date.isocalendar()[:2]  # (year, week)
            month_key = (review_date.year, review_date.month)
            
            daily_counts[day_key] += 1
            weekly_counts[week_key] += 1
            monthly_counts[month_key] += 1
            
            reviews_by_date.append({
                'date': review_date,
                'rating': result['rating'],
                'similarity': result['similarity'],
                'content': result['content'][:100],
                'locale': result['locale']
            })
            
            scores_by_date.append((review_date, result['rating']))
        
        # Create visualization
        print("\n📊 Creating visualization...")
        
        if not output_file:
            output_file = f"complaints_{app_id.replace('.', '_')}_{query.replace(' ', '_')[:20]}_{months}m.png"
        
        self._create_trend_graph(
            daily_counts=daily_counts,
            weekly_counts=weekly_counts,
            monthly_counts=monthly_counts,
            scores_by_date=scores_by_date,
            query=query,
            app_id=app_id,
            months=months,
            total_complaints=len(results),
            output_file=output_file,
            normalized=normalized,
            total_reviews_count=total_reviews_count,
            daily_totals=daily_totals,
            weekly_totals=weekly_totals,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate statistics - use the same method as normalization
        if normalized and total_reviews_count > 0:
            total_reviews = total_reviews_count
        else:
            total_reviews = self.db[app_id].count_documents({
                'at': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            })
        
        complaint_rate = (len(results) / total_reviews * 100) if total_reviews > 0 else 0
        
        # Find peak periods - for normalized mode, find peak by percentage not absolute count
        peak_day = max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else (None, 0)
        
        # For weekly peak, use same Bayesian smoothing as in visualization
        if normalized and weekly_totals:
            base_rate = (len(results) / total_reviews_count) if total_reviews_count > 0 else 0.0
            ALPHA_WEEK = 7 * 20
            def week_rate(w):
                tot = weekly_totals.get(w, 0)
                comp = weekly_counts.get(w, 0)
                return ((comp) + ALPHA_WEEK * base_rate) / (tot + ALPHA_WEEK) if (tot + ALPHA_WEEK) > 0 else 0.0
            if weekly_counts:
                w_best = max(weekly_counts.keys(), key=week_rate)
                peak_week = (w_best, weekly_counts[w_best])
            else:
                peak_week = (None, 0)
        else:
            peak_week = max(weekly_counts.items(), key=lambda x: x[1]) if weekly_counts else (None, 0)
        
        # Statistically robust trend analysis
        def calc_mann_kendall_trend(weekly_counts):
            """Calculate trend using Mann-Kendall test (non-parametric)."""
            if len(weekly_counts) < 8:  # Need at least ~2 months
                return "➡️ Too few points", 0
            
            # Get time-ordered values
            ordered_weeks = sorted(weekly_counts.keys())
            values = [weekly_counts[w] for w in ordered_weeks]
            n = len(values)
            
            # Mann-Kendall S statistic
            S = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    if values[j] > values[i]:
                        S += 1
                    elif values[j] < values[i]:
                        S -= 1
            
            # Variance calculation with tie correction
            from collections import Counter
            tie_counts = list(Counter(values).values())
            var_S = (n * (n - 1) * (2 * n + 5) - 
                     sum(t * (t - 1) * (2 * t + 5) for t in tie_counts)) / 18
            
            # Z-score
            if S > 0:
                Z = (S - 1) / np.sqrt(var_S)
            elif S < 0:
                Z = (S + 1) / np.sqrt(var_S)
            else:
                Z = 0
            
            # p-value (two-tailed)
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(Z)))
            
            # Determine significance (α = 0.05)
            if p_value > 0.05:
                return "➡️ Stable", 0
            else:
                trend_dir = "📈 Increasing" if S > 0 else "📉 Decreasing"
                # Sen's slope as effect size (converted to percentage)
                slopes = []
                for i in range(n-1):
                    for j in range(i+1, n):
                        if j != i:
                            slopes.append((values[j] - values[i]) / (j - i))
                sen_slope = np.median(slopes) if slopes else 0
                # Convert to percentage change per week relative to baseline
                base = max(np.mean(values), 1e-9)
                pct_change = sen_slope / base * 100
                return trend_dir, pct_change
        
        try:
            trend_direction, trend_change = calc_mann_kendall_trend(weekly_counts)
        except Exception:
            # Fallback to simple comparison if Mann-Kendall fails
            if len(weekly_counts) >= 2:
                weeks_sorted = sorted(weekly_counts.keys())
                first_half = weeks_sorted[:len(weeks_sorted)//2]
                second_half = weeks_sorted[len(weeks_sorted)//2:]
                
                avg_first = np.mean([weekly_counts[w] for w in first_half])
                avg_second = np.mean([weekly_counts[w] for w in second_half])
                
                delta = avg_second - avg_first
                if abs(delta) < max(1.0, 0.1 * max(avg_first, 1e-9)):
                    trend_direction, trend_change = "➡️ Stable", 0
                else:
                    trend_direction = "📈 Increasing" if delta > 0 else "📉 Decreasing"
                    trend_change = (delta / avg_first * 100) if avg_first > 0 else 0
            else:
                trend_direction = "➡️ Stable"
                trend_change = 0
        
        # Print analysis
        print(f"\n📈 Analysis Results:")
        print(f"  • Total matching complaints: {len(results)}")
        print(f"  • Total reviews in period: {total_reviews}")
        print(f"  • Complaint rate: {complaint_rate:.2f}%")
        print(f"  • Average similarity: {np.mean([r['similarity'] for r in results]):.3f}")
        print(f"  • Peak day: {peak_day[0]} ({peak_day[1]} complaints)")
        # Format peak week as human-readable
        pk_week_label = (f"{peak_week[0][0]}-W{peak_week[0][1]:02d}" if peak_week[0] else "—")
        print(f"  • Peak week: {pk_week_label} ({peak_week[1]} complaints)")
        print(f"  • Trend: {trend_direction} ({trend_change:+.1f}%)")
        
        # Language distribution - clean empty/invalid locales
        lang_dist = defaultdict(int)
        for r in reviews_by_date:
            lang = (r['locale'] or '').split('-')[0].lower()
            if lang: 
                lang_dist[lang] += 1
        
        print(f"\n🌍 Language Distribution:")
        for lang, count in sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / len(results) * 100
            print(f"  • {lang}: {count} ({pct:.1f}%)")
        
        # Sample complaints
        print(f"\n💬 Sample Complaints (highest similarity):")
        for i, result in enumerate(sorted(results, key=lambda x: x['similarity'], reverse=True)[:3], 1):
            print(f"  {i}. [{result['locale']}] (similarity: {result['similarity']:.3f})")
            print(f"     \"{result['content'][:100]}...\"")
        
        print(f"\n✅ Graph saved to: {output_file}")
        
        # Close the search engine (it has its own connection)
        search_engine.close()
        
        return {
            'total_complaints': len(results),
            'complaint_rate': complaint_rate,
            'trend': trend_direction,
            'trend_change': trend_change,
            'peak_day': peak_day,
            'peak_week': peak_week,
            'graph_file': output_file,
            'daily_counts': dict(daily_counts),  # Add actual daily trend data
            'avg_similarity': np.mean([r['similarity'] for r in results])
        }
    
    def _create_trend_graph(self, 
                           daily_counts: Dict,
                           weekly_counts: Dict,
                           monthly_counts: Dict,
                           scores_by_date: List[Tuple],
                           query: str,
                           app_id: str,
                           months: int,
                           total_complaints: int,
                           output_file: str,
                           normalized: bool = False,
                           total_reviews_count: int = 0,
                           daily_totals: Dict = None,
                           weekly_totals: Dict = None,
                           start_date: datetime = None,
                           end_date: datetime = None):
        """Create beautiful trend visualization."""
        
        # Set style with pastel colors
        plt.style.use('seaborn-v0_8-whitegrid')
        # Define beautiful pastel color palette
        pastel_colors = ['#8FB3D3', '#C7E9B4', '#FDB863', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']
        sns.set_palette(pastel_colors)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # базовая доля жалоб за весь период для сглаживания
        base_rate = (total_complaints / total_reviews_count) if (normalized and total_reviews_count > 0) else 0.0
        ALPHA_DAY = 20          # эквивалент 20 отзывов в день
        ALPHA_WEEK = 7 * ALPHA_DAY  # ~140 отзывов в неделю
        
        # Main title
        title_suffix = " (Bayesian-smoothed, α=20/day, 140/week)" if normalized else ""
        fig.suptitle(f'Complaint Analysis: "{query}"\n{app_id} - Last {months} months{title_suffix}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Daily trend (main plot)
        ax1 = plt.subplot(2, 2, (1, 2))
        
        # Create full date range and fill missing days with zeros
        all_days = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        df_daily = pd.DataFrame({'date': all_days})
        df_daily['count'] = df_daily['date'].dt.date.map(lambda d: daily_counts.get(d, 0)).astype(float)
        
        # Apply normalization if requested - байесовское сглаживание без NaN
        if normalized and daily_totals:
            def day_rate(d):
                tot = daily_totals.get(d, 0)
                comp = daily_counts.get(d, 0)
                return 100.0 * ((comp) + ALPHA_DAY * base_rate) / (tot + ALPHA_DAY) if (tot + ALPHA_DAY) > 0 else 0.0
            df_daily['count'] = df_daily['date'].dt.date.map(day_rate).astype(float)
            # ничего не дропаем; дальше rolling уже по календарному времени
            
        # Check if we have any data left after filtering
        if df_daily.empty:
            print("⚠️ No data points left after filtering noisy days")
            ax1.text(0.5, 0.5, 'Insufficient data\n(all days < 20 reviews)', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=14)
        
        if len(df_daily) > 0:
            # Plot raw daily counts (more transparent to reduce noise)
            label_daily = 'Daily rate (raw)' if normalized else 'Daily count (raw)'
            ax1.plot(df_daily['date'], df_daily['count'], 
                    color='#8FB3D3', linewidth=0.8, alpha=0.3, label=label_daily)
            
            # Add time-based rolling averages (not point-based)
            df_daily = df_daily.sort_values('date')
            df_daily['ma7'] = df_daily.set_index('date')['count'].rolling('7D', min_periods=3).mean().values
            ax1.plot(df_daily['date'], df_daily['ma7'], 
                    color='#C7E9B4', linewidth=3, alpha=0.9, label='7-day trend')
            
            # Add 14-day moving average for longer trend
            if len(df_daily) > 14:
                df_daily['ma14'] = df_daily.set_index('date')['count'].rolling('14D', min_periods=7).mean().values
                ax1.plot(df_daily['date'], df_daily['ma14'], 
                        color='#FDB863', linewidth=2.5, alpha=0.8, label='14-day trend')
            
            # Fill area under 7-day average instead of raw data
            ax1.fill_between(df_daily['date'], df_daily['ma7'], 
                           alpha=0.15, color='#C7E9B4')
            
            # Mark peak day on 7-day average (only if there's actual data)
            if df_daily['ma7'].max() > 0:
                peak_idx = df_daily['ma7'].idxmax()
                ax1.scatter(df_daily.loc[peak_idx, 'date'], 
                           df_daily.loc[peak_idx, 'ma7'],
                           color='red', s=120, zorder=5, edgecolors='darkred', linewidth=2)
                peak_value = df_daily.loc[peak_idx, 'ma7']
                ax1.annotate(f'Peak: {peak_value:.1f}{"%" if normalized else ""}',
                            xy=(df_daily.loc[peak_idx, 'date'], peak_value),
                            xytext=(15, 15), textcoords='offset points',
                            fontsize=10, color='darkred', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.8, edgecolor='red'))
            
            # Format x-axis - adapt to period length
            period_days = (end_date - start_date).days
            if period_days > 365:  # > 12 months
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            elif period_days > 120:  # > 4 months
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            else:
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Calculate trend line using 7-day average (smoother trend)
            show_trend = False
            if len(df_daily) > 7:
                try:
                    x_numeric = (df_daily['date'] - df_daily['date'].min()).dt.days.values
                    z = np.polyfit(x_numeric, df_daily['ma7'].values, 1)
                    p = np.poly1d(z)
                    show_trend = True
                except (np.linalg.LinAlgError, ValueError):
                    # Skip trend line if calculation fails
                    pass
            
            if show_trend:
                ax1.plot(df_daily['date'], p(x_numeric), 
                        "--", color='gray', alpha=0.7, linewidth=2, label='Overall trend')
            
        ax1.set_xlabel('Date')
        if normalized:
            ax1.set_ylabel('Daily Complaint Rate (%)')
            ax1.set_title(f'Daily Complaint Rate Trend (Matched: {total_complaints}, Total reviews: {total_reviews_count:,})')
        else:
            ax1.set_ylabel('Daily Complaint Count')
            ax1.set_title(f'Daily Complaint Count Trend (Total: {total_complaints})')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Weekly aggregation
        ax2 = plt.subplot(2, 2, 3)
        
        if weekly_counts:
            weeks = sorted(weekly_counts.keys())
            week_labels = [f"{w[0]}-W{w[1]:02d}" for w in weeks]  # Include year
            week_counts_list = [weekly_counts[w] for w in weeks]
            
            # Apply normalization if requested - байесовское сглаживание
            weighted_avg = None
            if normalized and weekly_totals:
                def week_rate(w):
                    tot = weekly_totals.get(w, 0)
                    comp = weekly_counts.get(w, 0)
                    return 100.0 * ((comp) + ALPHA_WEEK * base_rate) / (tot + ALPHA_WEEK) if (tot + ALPHA_WEEK) > 0 else 0.0

                week_counts_list = [week_rate(w) for w in weeks]

                # честная взвешенная средняя (без сглаживания): это глобальная доля
                tot_comp = sum(weekly_counts[w] for w in weeks)
                tot_rev  = sum(weekly_totals.get(w, 0) for w in weeks)
                weighted_avg = (100.0 * tot_comp / tot_rev) if tot_rev else 0.0
            
            # Color based on whether above/below mean
            colors = []
            mean_val = np.mean(week_counts_list) if week_counts_list else 0
            for c in week_counts_list:
                if c > mean_val:
                    colors.append('#FDB863')
                else:
                    colors.append('#C7E9B4')
            
            # подсвети «тонкие» недели меньшей прозрачностью, но не скрывай
            alphas = [0.4 if weekly_totals.get(w, 0) < ALPHA_WEEK else 1.0 for w in weeks]
            bars = ax2.bar(range(len(weeks)), week_counts_list, color=colors)
            for bar, a in zip(bars, alphas):
                bar.set_alpha(a)
            ax2.set_xticks(range(len(weeks)))
            ax2.set_xticklabels(week_labels, rotation=45)
            
            # Add value labels on bars with proper formatting
            for bar, count in zip(bars, week_counts_list):
                height = bar.get_height()
                label = f"{count:.1f}%" if normalized else f"{int(count)}"
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.2 if normalized else 0.5),
                        label, ha='center', va='bottom', fontsize=8)
            
            # Add average line
            if normalized:
                ax2.axhline(y=weighted_avg, color='red', linestyle='--', alpha=0.6,
                           label=f'Weighted avg: {weighted_avg:.1f}%')
            else:
                avg = np.mean(week_counts_list) if week_counts_list else 0
                ax2.axhline(y=avg, color='red', linestyle='--', alpha=0.5, 
                           label=f'Avg: {avg:.1f}')
            
        ax2.set_xlabel('Week')
        if normalized:
            ax2.set_ylabel('Complaint Rate (%)')
            # Calculate peak week for normalized display
            valid_counts = [c for c in week_counts_list if not np.isnan(c)]
            if valid_counts:
                max_week_idx = next(i for i, c in enumerate(week_counts_list) if c == max(valid_counts))
                peak_week_label = week_labels[max_week_idx]
                ax2.set_title(f'Weekly Complaint Rate (Peak: {peak_week_label} at {max(valid_counts):.1f}%)')
            else:
                ax2.set_title('Weekly Complaint Rate - Normalized')
        else:
            ax2.set_ylabel('Complaint Count')
            # Find peak week for absolute counts
            if week_counts_list:
                max_week_idx = week_counts_list.index(max(week_counts_list))
                peak_week_label = week_labels[max_week_idx]
                ax2.set_title(f'Weekly Complaint Count (Peak: {peak_week_label} with {max(week_counts_list)})')
            else:
                ax2.set_title('Weekly Complaint Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Score distribution over time
        ax3 = plt.subplot(2, 2, 4)
        
        if scores_by_date:
            dates_scores = [(d, s) for d, s in scores_by_date]
            dates_scores.sort(key=lambda x: x[0])
            
            # Group by day and calculate average score
            daily_scores = defaultdict(list)
            for date, score in dates_scores:
                daily_scores[date.date()].append(score)
            
            avg_dates = []
            avg_scores = []
            for date, scores in sorted(daily_scores.items()):
                avg_dates.append(date)
                avg_scores.append(np.mean(scores))
            
            # Create color map based on score (pastel colors)
            colors = ['#E78AC3' if s <= 2 else '#FDB863' if s <= 3 else '#C7E9B4' 
                     for s in avg_scores]
            
            ax3.scatter(pd.to_datetime(avg_dates), avg_scores, 
                       c=colors, alpha=0.6, s=50)
            
            # Add trend line
            if len(avg_dates) > 1:
                try:
                    x_numeric = (pd.to_datetime(avg_dates) - pd.to_datetime(avg_dates[0])).days
                    z = np.polyfit(x_numeric, avg_scores, 1)
                    p = np.poly1d(z)
                    ax3.plot(pd.to_datetime(avg_dates), p(x_numeric), 
                            "--", color='gray', alpha=0.8)
                except (np.linalg.LinAlgError, ValueError):
                    # Skip trend line if calculation fails
                    pass
            
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Average Rating')
        ax3.set_ylim(0.5, 5.5)
        ax3.set_title('Average Rating of Matching Reviews')
        ax3.grid(True, alpha=0.3)
        
        # Add legend for colors (pastel)
        legend_elements = [
            plt.scatter([], [], c='#E78AC3', alpha=0.6, s=50, label='1-2 stars'),
            plt.scatter([], [], c='#FDB863', alpha=0.6, s=50, label='3 stars'),
            plt.scatter([], [], c='#C7E9B4', alpha=0.6, s=50, label='4-5 stars')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout with space for title
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_aggregated_visualization(self, all_results: List[Dict], app_id: str, months: int, output_file: str):
        """Create aggregated visualization for multiple queries."""
        
        # Set style with pastel colors
        plt.style.use('seaborn-v0_8-whitegrid')
        # Define beautiful pastel color palette
        pastel_colors = ['#8FB3D3', '#C7E9B4', '#FDB863', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']
        sns.set_palette(pastel_colors)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Main title
        queries_str = ", ".join([r['query'] for r in all_results])
        fig.suptitle(f'Multi-Query Complaint Analysis: {queries_str}\n{app_id} - Last {months} months', 
                    fontsize=16, fontweight='bold')
        
        # 1. Comparison bar chart (top left)
        ax1 = plt.subplot(2, 3, 1)
        
        queries = [r['query'] for r in all_results if r['results']]
        complaints = [r['results']['total_complaints'] for r in all_results if r['results']]
        
        # Use our pastel color palette
        pastel_colors = ['#8FB3D3', '#C7E9B4', '#FDB863', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']
        colors = pastel_colors[:len(queries)] if len(queries) <= len(pastel_colors) else plt.cm.Set3(np.linspace(0, 1, len(queries)))
        bars = ax1.bar(range(len(queries)), complaints, color=colors)
        
        ax1.set_xticks(range(len(queries)))
        ax1.set_xticklabels([q[:15] + '...' if len(q) > 15 else q for q in queries], rotation=45, ha='right')
        ax1.set_ylabel('Number of Complaints')
        ax1.set_title('Complaints by Query Category')
        
        # Add value labels on bars
        for bar, count in zip(bars, complaints):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Trends comparison (top middle and right)
        ax2 = plt.subplot(2, 3, (2, 3))
        
        # Collect real trend data for all queries
        query_data = {}
        all_dates = set()
        for result in all_results:
            if result['results'] and 'daily_counts' in result['results']:
                query = result['query']
                daily_counts = result['results']['daily_counts']
                
                # Convert date strings back to date objects and collect all dates
                date_counts = {}
                for date_str, count in daily_counts.items():
                    if isinstance(date_str, str):
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except ValueError:
                            continue
                    else:
                        date_obj = date_str
                    date_counts[date_obj] = count
                    all_dates.add(date_obj)
                
                query_data[query] = date_counts
        
        if query_data and all_dates:
            # Create full date range and fill missing dates with 0
            min_date = min(all_dates)
            max_date = max(all_dates)
            date_range = []
            current_date = min_date
            while current_date <= max_date:
                date_range.append(current_date)
                current_date += timedelta(days=1)
            
            # Convert to days ago for plotting
            days = [(max_date - date).days for date in reversed(date_range)]
        
            # Plot trends for each query
            for i, (query, date_counts) in enumerate(query_data.items()):
                # Create counts array for full date range
                counts = []
                for date in reversed(date_range):  # Reverse to match days calculation
                    counts.append(date_counts.get(date, 0))
                
                # Original data (faint)
                ax2.plot(days, counts, color=colors[i], linewidth=1, alpha=0.3, 
                        marker='o', markersize=2)
                
                # Gaussian smoothed (main line)
                if len(counts) > 5:
                    gaussian_smoothed = ndimage.gaussian_filter1d(counts, sigma=3)
                    ax2.plot(days, gaussian_smoothed, color=colors[i], linewidth=3, 
                            label=query[:20] + '...' if len(query) > 20 else query)
                else:
                    ax2.plot(days, counts, color=colors[i], linewidth=2, 
                            label=query[:20] + '...' if len(query) > 20 else query)
        
        total_days = len(date_range) if query_data and all_dates else months * 30
        
        ax2.set_xlabel('Days Ago')
        ax2.set_ylabel('Daily Complaints')
        ax2.set_title(f'Complaint Trends Comparison (Last {total_days} Days)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Reverse x-axis so most recent is on the right
        ax2.invert_xaxis()
        
        # 3. Similarity scores comparison (bottom left)
        ax3 = plt.subplot(2, 3, 4)
        
        avg_similarities = []
        query_labels = []
        for result in all_results:
            if result['results']:
                # Use real similarity data
                avg_sim = result['results'].get('avg_similarity', 0.8)
                avg_similarities.append(avg_sim)
                query_labels.append(result['query'][:10] + '...' if len(result['query']) > 10 else result['query'])
        
        bars3 = ax3.bar(range(len(query_labels)), avg_similarities, color=colors[:len(query_labels)])
        ax3.set_xticks(range(len(query_labels)))
        ax3.set_xticklabels(query_labels, rotation=45, ha='right')
        ax3.set_ylabel('Average Similarity')
        ax3.set_ylim(0.5, 1.0)
        ax3.set_title('Query Match Quality')
        
        # Add value labels
        for bar, sim in zip(bars3, avg_similarities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Total distribution pie chart (bottom middle)
        ax4 = plt.subplot(2, 3, 5)
        
        sizes = [r['results']['total_complaints'] for r in all_results if r['results']]
        labels = [f"{r['query']}\n({r['results']['total_complaints']})" for r in all_results if r['results']]
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
        ax4.set_title('Complaint Distribution')
        
        # 5. Summary statistics (bottom right)
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('off')  # Turn off axis
        
        # Create summary text
        total_complaints = sum(complaints)
        avg_complaints = np.mean(complaints)
        max_query = queries[complaints.index(max(complaints))]
        min_query = queries[complaints.index(min(complaints))]
        
        summary_text = f"""
SUMMARY STATISTICS

Total Complaints: {total_complaints:,}
Average per Query: {avg_complaints:.1f}
Number of Queries: {len(queries)}

Highest: {max_query}
   ({max(complaints)} complaints)

Lowest: {min_query}
   ({min(complaints)} complaints)

Analysis Period: {months} month(s)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout with space for title
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        plt.savefig(output_file, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def close(self):
        """Close connections."""
        self.mongo_client.close()

async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Analyze complaints and visualize trends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single query
    python analyze_complaints.py com.example.app "payment issues" --months 6
    
    # Analyze multiple queries (space-separated)
    python analyze_complaints.py com.example.app "payment issues" "crashes" "login problems" --months 3
    
    # Analyze multiple queries (comma-separated)
    python analyze_complaints.py com.example.app "payment issues,crashes,login problems" --months 3
    
    # Mixed format
    python analyze_complaints.py com.example.app "payment issues,crashes" "login problems" --months 2
        """
    )
    
    parser.add_argument('app_id', help='Google Play application ID')
    parser.add_argument('queries', nargs='+', help='Search queries for complaints (can specify multiple separated by space or comma)')
    parser.add_argument('--months', type=int, default=3,
                       help='Number of months to analyze (default: 3)')
    parser.add_argument('--similarity', type=float, default=0.8,
                       help='Minimum similarity threshold (default: 0.8)')
    parser.add_argument('--output', help='Output file for graph')
    parser.add_argument('--db-uri', help='MongoDB URI (overrides config)')
    parser.add_argument('--normalized', action='store_true', 
                       help='Show normalized percentages from total reviews instead of absolute numbers')
    
    args = parser.parse_args()
    
    # Parse queries - handle comma-separated or space-separated
    queries = []
    for query_part in args.queries:
        # Split by comma if present
        if ',' in query_part:
            queries.extend([q.strip() for q in query_part.split(',') if q.strip()])
        else:
            queries.append(query_part)
    
    # Override DB URI if provided
    if args.db_uri:
        import os
        os.environ['MONGODB_URI'] = args.db_uri  # Use correct env var
    
    # Run analysis for each query
    analyzer = ComplaintAnalyzer()
    
    try:
        all_results = []
        
        for i, query in enumerate(queries):
            print(f"\n{'='*60}")
            print(f"📊 QUERY {i+1}/{len(queries)}: '{query}'")
            print(f"{'='*60}")
            
            # Generate output filename for each query
            if args.output:
                if len(queries) == 1:
                    output_file = args.output
                else:
                    base, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'png')
                    output_file = f"{base}_q{i+1}_{query.replace(' ', '_')[:15]}.{ext}"
            else:
                query_clean = query.replace(' ', '_')[:20]
                output_file = f"complaints_{args.app_id.replace('.', '_')}_{query_clean}_{args.months}m.png"
            
            results = await analyzer.search_and_analyze(
                app_id=args.app_id,
                query=query,
                months=args.months,
                min_similarity=args.similarity,
                output_file=output_file,
                normalized=args.normalized
            )
            
            if results:
                all_results.append({
                    'query': query,
                    'results': results,
                    'graph_file': output_file
                })
        
        # Summary for multiple queries
        if len(queries) > 1:
            print(f"\n{'='*60}")
            print(f"📊 MULTI-QUERY ANALYSIS SUMMARY")
            print(f"{'='*60}")
            
            total_complaints = sum(r['results']['total_complaints'] for r in all_results if r['results'])
            
            print(f"\n📈 Results by Query:")
            for result in all_results:
                if result['results']:
                    complaints = result['results']['total_complaints']
                    trend = result['results']['trend']
                    print(f"  • '{result['query']}': {complaints} complaints ({trend})")
                    print(f"    Graph: {result['graph_file']}")
            
            # Create aggregated visualization
            print(f"\n📊 Creating aggregated visualization...")
            aggregated_file = f"complaints_aggregated_{args.app_id.replace('.', '_')}_{args.months}m.png"
            analyzer._create_aggregated_visualization(all_results, args.app_id, args.months, aggregated_file)
            print(f"✅ Aggregated graph saved to: {aggregated_file}")
            
            print(f"\n✅ Total complaints found: {total_complaints}")
            print(f"📁 Generated {len(all_results)} individual + 1 aggregated visualization")
        
        results = all_results
        
        if results:
            print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())