"""
Enhanced Data Loader
增强数据加载器 - 添加更多数据模态

新增数据源:
1. 市场情绪 (RapidAPI Stock Sentiment)
2. 资金流向 (FMP ETF/Fund Flows)
3. 经济日历 (FMP Economic Calendar)
4. 期权数据 (FMP Options)
5. 财报数据 (FMP Earnings)
"""

import os
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache

from finsage.config import AssetConfig
from finsage.data.fmp_client import FMPClient

logger = logging.getLogger(__name__)


class EnhancedDataLoader:
    """
    增强数据加载器

    整合多种数据模态:
    - 市场情绪 (RapidAPI)
    - 资金流向 (FMP)
    - 经济日历 (FMP)
    - 期权数据 (FMP)
    - 财报数据 (FMP)

    注意: 所有 FMP 端点统一使用 FMPClient.BASE_URL (stable)
    """

    def __init__(self):
        # API Keys from environment
        self.rapidapi_key = os.environ.get("RAPIDAPI_KEY")
        self.fmp_api_key = os.environ.get("FMP_API_KEY")

        # API endpoints - 统一使用 FMPClient 中的 URL 常量
        self.sentiment_host = "stock-sentiment-api.p.rapidapi.com"
        self.fmp_base_url = FMPClient.BASE_URL  # stable API
        self.fmp_v3_url = FMPClient.V3_URL      # v3 fallback (经济日历等)
        self.fmp_v4_url = FMPClient.V4_URL      # v4 API (期权等)

        # Cache
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info(f"EnhancedDataLoader initialized (RAPIDAPI_KEY: {'set' if self.rapidapi_key else 'NOT SET'}, FMP_KEY: {'set' if self.fmp_api_key else 'NOT SET'})")

    # ==================== 1. 市场情绪数据 ====================

    def get_stock_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取个股情绪数据 (RapidAPI Stock Sentiment)

        Args:
            symbol: 股票代码 (如 AAPL)

        Returns:
            情绪数据字典
        """
        if not self.rapidapi_key:
            logger.warning("RAPIDAPI_KEY not set, returning default sentiment")
            return self._get_default_sentiment(symbol)

        cache_key = f"sentiment_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            # 正确的端点是 /avg_sentiment/ 而不是 /api/sentiment
            url = f"https://{self.sentiment_host}/avg_sentiment/"
            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": self.sentiment_host
            }
            params = {"ticker": symbol}

            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # 解析响应 - API 返回 avg_sentiment (0-1), number_of_articles
                # avg_sentiment: 1 = 非常看涨, 0 = 非常看跌, 0.5 = 中性
                avg_sent = data.get("avg_sentiment", 0.5)
                news_count = data.get("number_of_articles", 0)

                # 确保 avg_sent 在 [0, 1] 范围内（API 有时返回异常值）
                if avg_sent < 0 or avg_sent > 1:
                    logger.warning(f"{symbol} sentiment out of range: {avg_sent}, clamping to [0,1]")
                    avg_sent = max(0.0, min(1.0, avg_sent))

                # 将 0-1 分数转换为 -1 到 1 的标准情感分数
                sentiment_score = (avg_sent - 0.5) * 2  # 0->-1, 0.5->0, 1->1

                # 确定情感标签
                if sentiment_score > 0.2:
                    sentiment_label = "bullish"
                elif sentiment_score < -0.2:
                    sentiment_label = "bearish"
                else:
                    sentiment_label = "neutral"

                # 计算看涨/看跌百分比
                bullish_percent = avg_sent * 100
                bearish_percent = (1 - avg_sent) * 100

                sentiment = {
                    "symbol": symbol,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "bullish_percent": bullish_percent,
                    "bearish_percent": bearish_percent,
                    "news_count": news_count,
                    "social_volume": 0,  # API 不提供此数据
                    "timestamp": datetime.now().isoformat(),
                    "source": "rapidapi",
                    "raw_avg_sentiment": avg_sent,  # 保留原始值供参考
                }

                self._set_cache(cache_key, sentiment)
                logger.info(f"[SENTIMENT] {symbol}: {sentiment_label} (score={sentiment_score:.2f}, articles={news_count})")
                return sentiment
            elif response.status_code == 403:
                logger.warning(f"[SENTIMENT] RapidAPI returned 403 for {symbol} - check API key")
            elif response.status_code == 429:
                logger.warning(f"[SENTIMENT] RapidAPI rate limited for {symbol}")
            else:
                logger.warning(f"[SENTIMENT] RapidAPI returned {response.status_code} for {symbol}")

            return self._get_default_sentiment(symbol)

        except Exception as e:
            logger.warning(f"[SENTIMENT] Failed to get sentiment for {symbol}: {e}")
            return self._get_default_sentiment(symbol)

    def get_market_sentiment(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        获取市场整体情绪

        Args:
            symbols: 可选，特定股票列表

        Returns:
            市场情绪汇总
        """
        if symbols is None:
            # 使用 config 中的股票列表前5个
            symbols = AssetConfig().default_universe.get("stocks", ["SPY", "QQQ"])[:5]

        sentiments = []
        for symbol in symbols[:5]:  # 限制 API 调用
            sent = self.get_stock_sentiment(symbol)
            if sent:
                sentiments.append(sent)

        if not sentiments:
            return self._get_default_market_sentiment()

        # 计算平均情绪
        avg_score = sum(s.get("sentiment_score", 0) for s in sentiments) / len(sentiments)
        avg_bullish = sum(s.get("bullish_percent", 50) for s in sentiments) / len(sentiments)

        # 确定市场情绪
        if avg_score > 0.3:
            market_label = "bullish"
        elif avg_score < -0.3:
            market_label = "bearish"
        else:
            market_label = "neutral"

        return {
            "market_sentiment_score": avg_score,
            "market_sentiment_label": market_label,
            "bullish_percent": avg_bullish,
            "bearish_percent": 100 - avg_bullish,
            "sample_size": len(sentiments),
            "individual_sentiments": sentiments,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_sentiment_from_fmp_news(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        使用 FMP 新闻数据估算情绪 (RapidAPI 失败时的 fallback)

        基于新闻标题中的关键词进行简单情绪分析
        """
        if not self.fmp_api_key:
            return None

        try:
            from finsage.data.fmp_client import get_news_client
            news_client = get_news_client()
            news_items = news_client.get_stock_news([symbol], limit=10)

            if not news_items:
                return None

            # 简单情绪分析：基于标题关键词
            bullish_words = ['surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'beat',
                          'strong', 'growth', 'profit', 'upgrade', 'buy', 'bullish',
                          'record', 'high', 'positive', 'optimistic', 'outperform']
            bearish_words = ['drop', 'fall', 'decline', 'loss', 'down', 'crash', 'miss',
                          'weak', 'cut', 'sell', 'bearish', 'concern', 'warning',
                          'low', 'negative', 'downgrade', 'underperform', 'risk']

            bullish_count = 0
            bearish_count = 0

            for item in news_items:
                title = item.get('title', '').lower()
                text = item.get('text', '').lower()
                content = title + ' ' + text[:200]  # 标题 + 前200字

                for word in bullish_words:
                    if word in content:
                        bullish_count += 1
                for word in bearish_words:
                    if word in content:
                        bearish_count += 1

            total = bullish_count + bearish_count
            if total == 0:
                return None

            # 计算情绪分数 (-1 to 1)
            sentiment_score = (bullish_count - bearish_count) / total
            bullish_percent = (bullish_count / total) * 100 if total > 0 else 50
            bearish_percent = (bearish_count / total) * 100 if total > 0 else 50

            if sentiment_score > 0.2:
                label = "bullish"
            elif sentiment_score < -0.2:
                label = "bearish"
            else:
                label = "neutral"

            sentiment = {
                "symbol": symbol,
                "sentiment_score": sentiment_score,
                "sentiment_label": label,
                "bullish_percent": bullish_percent,
                "bearish_percent": bearish_percent,
                "news_count": len(news_items),
                "social_volume": 0,
                "timestamp": datetime.now().isoformat(),
                "source": "fmp_news",
            }

            logger.info(f"[SENTIMENT] {symbol} (FMP): {label} (score={sentiment_score:.2f}, news={len(news_items)})")
            return sentiment

        except Exception as e:
            logger.debug(f"[SENTIMENT] FMP news fallback failed for {symbol}: {e}")
            return None

    def _get_default_sentiment(self, symbol: str) -> Dict[str, Any]:
        """默认情绪数据 - 先尝试 FMP 新闻, 失败则返回中性默认值"""
        # 尝试 FMP 新闻作为 fallback
        fmp_sentiment = self._get_sentiment_from_fmp_news(symbol)
        if fmp_sentiment:
            return fmp_sentiment

        # 返回默认中性值
        return {
            "symbol": symbol,
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "bullish_percent": 50,
            "bearish_percent": 50,
            "news_count": 0,
            "social_volume": 0,
            "timestamp": datetime.now().isoformat(),
            "is_default": True,
        }

    def _get_default_market_sentiment(self) -> Dict[str, Any]:
        """默认市场情绪"""
        return {
            "market_sentiment_score": 0,
            "market_sentiment_label": "neutral",
            "bullish_percent": 50,
            "bearish_percent": 50,
            "sample_size": 0,
            "individual_sentiments": [],
            "timestamp": datetime.now().isoformat(),
            "is_default": True,
        }

    # ==================== 2. 资金流向数据 ====================

    def get_etf_fund_flows(self, symbol: str = "SPY") -> Optional[Dict[str, Any]]:
        """
        获取 ETF 资金流向

        Args:
            symbol: ETF 代码

        Returns:
            资金流向数据
        """
        if not self.fmp_api_key:
            logger.warning("[ETF] FMP_API_KEY not set")
            return None

        cache_key = f"etf_flow_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            # 使用正确的 stable 端点: /stable/etf/holdings?symbol=XXX
            url = f"{self.fmp_base_url}/etf/holdings?symbol={symbol}&apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    result = {
                        "symbol": symbol,
                        "holdings_count": len(data),
                        "top_holdings": data[:10] if len(data) > 10 else data,
                        "timestamp": datetime.now().isoformat(),
                        "source": "fmp_stable",
                    }
                    self._set_cache(cache_key, result)
                    logger.info(f"[ETF] {symbol}: {len(data)} holdings loaded")
                    return result
            else:
                logger.warning(f"[ETF] FMP returned {response.status_code} for {symbol}")

        except Exception as e:
            logger.warning(f"[ETF] Failed to get ETF flows for {symbol}: {e}")

        return None

    def get_institutional_ownership(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取机构持股数据

        Args:
            symbol: 股票代码

        Returns:
            机构持股数据
        """
        if not self.fmp_api_key:
            logger.warning("[INST] FMP_API_KEY not set")
            return None

        cache_key = f"inst_ownership_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            # 使用正确的 stable 端点: /stable/institutional-ownership/symbol-positions-summary
            # 需要指定 year 和 quarter
            current_year = datetime.now().year
            current_quarter = (datetime.now().month - 1) // 3 + 1
            # 使用上一季度的数据 (当前季度可能还没出)
            if current_quarter == 1:
                year, quarter = current_year - 1, 4
            else:
                year, quarter = current_year, current_quarter - 1

            url = f"{self.fmp_base_url}/institutional-ownership/symbol-positions-summary?symbol={symbol}&year={year}&quarter={quarter}&apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    # 计算统计
                    total_shares = sum(h.get("shares", h.get("totalShares", 0)) for h in data)
                    total_value = sum(h.get("value", h.get("totalValue", 0)) for h in data)

                    result = {
                        "symbol": symbol,
                        "institutional_holders": len(data),
                        "total_institutional_shares": total_shares,
                        "total_institutional_value": total_value,
                        "top_holders": data[:10] if len(data) > 10 else data,
                        "year": year,
                        "quarter": quarter,
                        "timestamp": datetime.now().isoformat(),
                        "source": "fmp_stable",
                    }
                    self._set_cache(cache_key, result)
                    logger.info(f"[INST] {symbol}: {len(data)} institutional holders (Q{quarter} {year})")
                    return result
            else:
                logger.warning(f"[INST] FMP returned {response.status_code} for {symbol}")

        except Exception as e:
            logger.warning(f"[INST] Failed to get institutional ownership for {symbol}: {e}")

        return None

    # ==================== 3. 经济日历数据 ====================

    def get_economic_calendar(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict[str, Any]]:
        """
        获取经济日历

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            经济事件列表
        """
        if not self.fmp_api_key:
            logger.warning("[CALENDAR] FMP_API_KEY not set")
            return []

        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        cache_key = f"econ_calendar_{start_date}_{end_date}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            # 使用 stable 端点
            url = f"{self.fmp_base_url}/economic-calendar?from={start_date}&to={end_date}&apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    # 过滤重要事件
                    important_events = [
                        e for e in data
                        if e.get("impact", "").lower() in ["high", "medium"]
                    ]
                    self._set_cache(cache_key, important_events)
                    logger.info(f"[CALENDAR] Loaded {len(important_events)} important economic events")
                    return important_events
            elif response.status_code == 403:
                logger.warning(f"[CALENDAR] FMP economic calendar returned 403 - trying v3 endpoint")
                # 尝试 v3 端点作为备用
                url = f"{self.fmp_v3_url}/economic_calendar?from={start_date}&to={end_date}&apikey={self.fmp_api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        important_events = [e for e in data if e.get("impact", "").lower() in ["high", "medium"]]
                        self._set_cache(cache_key, important_events)
                        logger.info(f"[CALENDAR] Loaded {len(important_events)} economic events via v3")
                        return important_events
            else:
                logger.warning(f"[CALENDAR] FMP returned {response.status_code}")

            return []

        except Exception as e:
            logger.warning(f"[CALENDAR] Failed to get economic calendar: {e}")
            return []

    def get_upcoming_fomc(self) -> Optional[Dict[str, Any]]:
        """获取即将到来的 FOMC 会议"""
        calendar = self.get_economic_calendar()

        for event in calendar:
            if "FOMC" in event.get("event", "") or "Federal Reserve" in event.get("event", ""):
                return {
                    "date": event.get("date"),
                    "event": event.get("event"),
                    "impact": event.get("impact"),
                    "forecast": event.get("estimate"),
                    "previous": event.get("previous"),
                }

        return None

    # ==================== 4. 期权数据 ====================

    def get_options_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取期权链数据

        Args:
            symbol: 股票代码

        Returns:
            期权链数据
        """
        if not self.fmp_api_key:
            return None

        cache_key = f"options_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            # 获取期权链 (注: options-expiration 仍需使用 v4 端点，stable 不支持)
            url = f"{self.fmp_v4_url}/options-expiration/{symbol}?apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            expirations = response.json()

            if not expirations:
                return None

            # 获取最近到期日的期权数据
            nearest_exp = expirations[0] if expirations else None

            result = {
                "symbol": symbol,
                "expiration_dates": expirations[:5] if len(expirations) > 5 else expirations,
                "nearest_expiration": nearest_exp,
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get options chain for {symbol}: {e}")
            return None

    def get_put_call_ratio(self, symbol: str = "SPY") -> Optional[Dict[str, Any]]:
        """
        获取 Put/Call Ratio (通过计算估算)

        Args:
            symbol: 股票代码

        Returns:
            Put/Call 比率
        """
        # FMP 免费版可能没有这个端点，返回估算值
        return {
            "symbol": symbol,
            "put_call_ratio": 0.85,  # 市场平均水平
            "interpretation": "neutral",
            "note": "Estimated value - upgrade API for real data",
            "timestamp": datetime.now().isoformat(),
        }

    # ==================== 5. 财报数据 ====================

    def get_earnings_calendar(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict[str, Any]]:
        """
        获取财报日历

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            财报事件列表
        """
        if not self.fmp_api_key:
            logger.warning("[EARNINGS] FMP_API_KEY not set")
            return []

        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        cache_key = f"earnings_cal_{start_date}_{end_date}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            # 使用 stable 端点 - 注意是 earnings-calendar (带s)
            url = f"{self.fmp_base_url}/earnings-calendar?from={start_date}&to={end_date}&apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self._set_cache(cache_key, data)
                    logger.info(f"[EARNINGS] Loaded {len(data)} earnings events")
                    return data
            elif response.status_code == 404:
                logger.warning("[EARNINGS] FMP stable returned 404 - trying v3")
                # 尝试 v3 端点
                url = f"{self.fmp_v3_url}/earning_calendar?from={start_date}&to={end_date}&apikey={self.fmp_api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        self._set_cache(cache_key, data)
                        logger.info(f"[EARNINGS] Loaded {len(data)} earnings events via v3")
                        return data
            else:
                logger.warning(f"[EARNINGS] FMP returned {response.status_code}")

            return []

        except Exception as e:
            logger.warning(f"[EARNINGS] Failed to get earnings calendar: {e}")
            return []

    def get_earnings_surprise(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取历史财报惊喜

        Args:
            symbol: 股票代码

        Returns:
            财报惊喜数据
        """
        if not self.fmp_api_key:
            return None

        cache_key = f"earnings_surprise_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            url = f"{self.fmp_base_url}/earnings-surpises/{symbol}?apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data:
                # 计算平均惊喜
                recent = data[:4]  # 最近 4 个季度
                avg_surprise = sum(
                    (e.get("actualEarningResult", 0) - e.get("estimatedEarning", 0))
                    for e in recent
                ) / len(recent) if recent else 0

                result = {
                    "symbol": symbol,
                    "avg_surprise": avg_surprise,
                    "beat_rate": sum(1 for e in recent if e.get("actualEarningResult", 0) > e.get("estimatedEarning", 0)) / len(recent) if recent else 0,
                    "recent_surprises": recent,
                    "timestamp": datetime.now().isoformat(),
                }

                self._set_cache(cache_key, result)
                return result

        except Exception as e:
            logger.warning(f"Failed to get earnings surprise for {symbol}: {e}")

        return None

    # ==================== 综合数据获取 ====================

    def get_enhanced_market_data(
        self,
        symbols: List[str],
        include_sentiment: bool = True,
        include_flows: bool = True,
        include_calendar: bool = True,
        include_options: bool = False,
        include_earnings: bool = True,
    ) -> Dict[str, Any]:
        """
        获取增强市场数据

        Args:
            symbols: 股票代码列表
            include_sentiment: 是否包含情绪数据
            include_flows: 是否包含资金流向
            include_calendar: 是否包含经济日历
            include_options: 是否包含期权数据
            include_earnings: 是否包含财报数据

        Returns:
            综合市场数据
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
        }

        # 1. 市场情绪
        if include_sentiment:
            result["sentiment"] = self.get_market_sentiment(symbols)

        # 2. 资金流向
        if include_flows:
            result["fund_flows"] = {
                "spy_flows": self.get_etf_fund_flows("SPY"),
                "qqq_flows": self.get_etf_fund_flows("QQQ"),
            }

        # 3. 经济日历
        if include_calendar:
            result["economic_calendar"] = self.get_economic_calendar()
            result["upcoming_fomc"] = self.get_upcoming_fomc()

        # 4. 期权数据
        if include_options:
            result["options"] = {
                symbol: self.get_options_chain(symbol)
                for symbol in symbols[:3]  # 限制调用次数
            }
            result["put_call_ratio"] = self.get_put_call_ratio()

        # 5. 财报数据
        if include_earnings:
            result["earnings_calendar"] = self.get_earnings_calendar()
            result["earnings_surprises"] = {
                symbol: self.get_earnings_surprise(symbol)
                for symbol in symbols[:5]
            }

        return result

    # ==================== 缓存管理 ====================

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any):
        """设置缓存"""
        self._cache[key] = (data, datetime.now())

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("Enhanced data loader cache cleared")
