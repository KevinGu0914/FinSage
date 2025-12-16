"""
Intraday Data Loader
日内数据加载器 - 获取小时级和分钟级数据

使用 FMP (Financial Modeling Prep) API 获取真实日内数据。

FMP API 端点 (Premium 用户使用 stable 端点):
- 日内图表: https://financialmodelingprep.com/stable/historical-chart/{timeframe}/{symbol}
- 实时报价: https://financialmodelingprep.com/stable/quote/{symbol}
- VIX: https://financialmodelingprep.com/stable/quote/^VIX

支持的时间间隔: 1min, 5min, 15min, 30min, 1hour, 4hour
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from finsage.data.fmp_client import FMPClient

logger = logging.getLogger(__name__)


class IntradayDataLoader:
    """
    日内数据加载器 - 使用 FMP API

    专门用于获取小时级和分钟级数据，
    配合 IntradayRiskMonitor 进行实时风险监控。

    注意: 所有端点统一使用 FMPClient.BASE_URL (stable)
    """

    # 使用统一的 FMP 基础 URL (从 fmp_client.py 获取)
    FMP_BASE_URL = FMPClient.BASE_URL  # "https://financialmodelingprep.com/stable"

    # 支持的时间间隔映射 (用户格式 -> FMP格式)
    INTERVAL_MAPPING = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "60m": "1hour",
        "4h": "4hour",
    }

    # 各间隔对应的分钟数
    INTERVAL_MINUTES = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "60m": 60, "4h": 240,
    }

    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 5,
    ):
        """
        初始化日内数据加载器

        Args:
            fmp_api_key: FMP API密钥 (默认从环境变量 OA_FMP_KEY 获取)
            cache_enabled: 是否启用缓存
            cache_ttl_minutes: 缓存过期时间 (分钟)
        """
        # 获取 API Key
        self.api_key = fmp_api_key or os.getenv("OA_FMP_KEY")
        if not self.api_key:
            logger.warning("FMP API key not found. Set OA_FMP_KEY environment variable.")

        self.cache_enabled = cache_enabled
        self.cache_ttl_minutes = cache_ttl_minutes

        # 缓存
        self._cache: Dict[str, Dict] = {}

        # Session for connection pooling
        self.session = requests.Session()

        logger.info(f"IntradayDataLoader initialized (FMP API)")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_fmp_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Any:
        """
        发送 FMP API 请求

        Args:
            endpoint: API 端点 (e.g., "/historical-chart/1hour/AAPL")
            params: 额外的查询参数

        Returns:
            API 响应数据
        """
        if not self.api_key:
            raise ValueError("FMP API key not configured")

        url = f"{self.FMP_BASE_URL}{endpoint}"

        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        try:
            response = self.session.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FMP API request failed: {e}")
            raise

    def load_hourly_data(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        end_datetime: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载小时级数据

        Args:
            symbols: 股票代码列表
            lookback_hours: 回看小时数
            end_datetime: 结束时间 (默认当前)

        Returns:
            {symbol: DataFrame} 包含 OHLCV 数据
        """
        return self._load_intraday_data(
            symbols=symbols,
            interval="1h",
            lookback_periods=lookback_hours,
            end_datetime=end_datetime,
        )

    def load_minute_data(
        self,
        symbols: List[str],
        interval: str = "5m",
        lookback_periods: int = 100,
        end_datetime: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载分钟级数据

        Args:
            symbols: 股票代码列表
            interval: 时间间隔 ("1m", "5m", "15m", "30m")
            lookback_periods: 回看周期数
            end_datetime: 结束时间 (默认当前)

        Returns:
            {symbol: DataFrame} 包含 OHLCV 数据
        """
        if interval not in self.INTERVAL_MAPPING:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(self.INTERVAL_MAPPING.keys())}")

        return self._load_intraday_data(
            symbols=symbols,
            interval=interval,
            lookback_periods=lookback_periods,
            end_datetime=end_datetime,
        )

    def _load_intraday_data(
        self,
        symbols: List[str],
        interval: str,
        lookback_periods: int,
        end_datetime: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """加载日内数据的核心方法"""
        end_dt = end_datetime or datetime.now()
        result = {}

        for symbol in symbols:
            # 检查缓存
            cache_key = f"{symbol}_{interval}_{lookback_periods}"
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                result[symbol] = cached
                continue

            # 从 FMP API 加载
            try:
                df = self._load_from_fmp(symbol, interval, lookback_periods, end_dt)

                if df is not None and not df.empty:
                    result[symbol] = df
                    self._save_to_cache(cache_key, df)
                    logger.info(f"Loaded {len(df)} rows of {interval} data for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to load intraday data for {symbol}: {e}")

        return result

    def _load_from_fmp(
        self,
        symbol: str,
        interval: str,
        lookback_periods: int,
        end_dt: datetime,
    ) -> Optional[pd.DataFrame]:
        """从 FMP API 加载日内数据"""
        # 转换间隔格式
        fmp_interval = self.INTERVAL_MAPPING.get(interval, "1hour")

        try:
            # FMP stable API 日内数据端点 - 使用查询参数而非路径参数
            endpoint = f"/historical-chart/{fmp_interval}"
            data = self._make_fmp_request(endpoint, params={"symbol": symbol})

            if not data or isinstance(data, dict):
                # 可能是错误响应
                if isinstance(data, dict) and "Error Message" in str(data):
                    logger.warning(f"FMP API error for {symbol}: {data}")
                return None

            # 转换为 DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                logger.warning(f"No FMP data returned for {symbol}")
                return None

            # FMP 返回的数据是降序 (最新在前)，需要反转
            df = df.iloc[::-1].reset_index(drop=True)

            # 处理时间列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            # 标准化列名 (首字母大写)
            df.columns = [c.title() for c in df.columns]

            # 确保有必要的列
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} for {symbol}")

            # 只保留最近的数据
            if len(df) > lookback_periods:
                df = df.tail(lookback_periods)

            return df

        except Exception as e:
            logger.error(f"FMP data load error for {symbol}: {e}")
            return None

    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时报价

        Args:
            symbol: 股票代码

        Returns:
            报价信息字典
        """
        try:
            # FMP stable API 使用查询参数
            endpoint = "/quote"
            data = self._make_fmp_request(endpoint, params={"symbol": symbol})

            if data and isinstance(data, list) and len(data) > 0:
                quote = data[0]
                return {
                    "symbol": quote.get("symbol"),
                    "price": quote.get("price"),
                    "change": quote.get("change"),
                    "change_percent": quote.get("changesPercentage"),
                    "open": quote.get("open"),
                    "high": quote.get("dayHigh"),
                    "low": quote.get("dayLow"),
                    "previous_close": quote.get("previousClose"),
                    "volume": quote.get("volume"),
                    "avg_volume": quote.get("avgVolume"),
                    "market_cap": quote.get("marketCap"),
                    "pe_ratio": quote.get("pe"),
                    "52_week_high": quote.get("yearHigh"),
                    "52_week_low": quote.get("yearLow"),
                    "timestamp": quote.get("timestamp"),
                }
            return None

        except Exception as e:
            logger.warning(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_realtime_snapshot(
        self,
        symbols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取多个股票的实时快照

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: quote_dict}
        """
        result = {}

        # FMP stable API 批量查询使用逗号分隔的 symbol 参数
        try:
            symbols_str = ",".join(symbols)
            endpoint = "/quote"
            data = self._make_fmp_request(endpoint, params={"symbol": symbols_str})

            if data and isinstance(data, list):
                for quote in data:
                    symbol = quote.get("symbol")
                    if symbol:
                        result[symbol] = {
                            "price": quote.get("price"),
                            "change": quote.get("change"),
                            "change_percent": quote.get("changesPercentage"),
                            "open": quote.get("open"),
                            "high": quote.get("dayHigh"),
                            "low": quote.get("dayLow"),
                            "previous_close": quote.get("previousClose"),
                            "volume": quote.get("volume"),
                        }

        except Exception as e:
            logger.warning(f"Failed to get batch quotes: {e}")
            # 降级为逐个查询
            for symbol in symbols:
                quote = self.get_realtime_quote(symbol)
                if quote:
                    result[symbol] = quote

        return result

    def get_vix_level(self) -> Optional[float]:
        """
        获取当前 VIX 水平

        使用 FMP 的市场指数接口
        """
        try:
            # FMP stable API 获取 VIX
            endpoint = "/quote"
            data = self._make_fmp_request(endpoint, params={"symbol": "^VIX"})

            if data and isinstance(data, list) and len(data) > 0:
                return float(data[0].get("price", 0))

            # 备用方案：编码后的 VIX 符号
            data = self._make_fmp_request(endpoint, params={"symbol": "%5EVIX"})

            if data and isinstance(data, list) and len(data) > 0:
                return float(data[0].get("price", 0))

            return None

        except Exception as e:
            logger.warning(f"Failed to get VIX: {e}")
            return None

    def get_market_hours_status(self) -> Dict[str, Any]:
        """
        获取市场交易时段状态

        Returns:
            市场状态信息
        """
        try:
            endpoint = "/is-the-market-open"
            data = self._make_fmp_request(endpoint)
            return data if data else {}
        except Exception as e:
            logger.warning(f"Failed to get market hours: {e}")
            return {}

    def get_stock_news(
        self,
        symbol: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        获取股票相关新闻

        Args:
            symbol: 股票代码
            limit: 新闻数量限制

        Returns:
            新闻列表
        """
        try:
            # 使用正确的 stable 端点: /news/stock?symbols=XXX
            endpoint = f"/news/stock"
            params = {"symbols": symbol, "limit": limit}
            data = self._make_fmp_request(endpoint, params)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to get news for {symbol}: {e}")
            return []

    def get_price_change(
        self,
        symbol: str,
    ) -> Optional[Dict[str, float]]:
        """
        获取价格变化信息

        Args:
            symbol: 股票代码

        Returns:
            价格变化信息
        """
        try:
            endpoint = f"/stock-price-change/{symbol}"
            data = self._make_fmp_request(endpoint)

            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            return None

        except Exception as e:
            logger.warning(f"Failed to get price change for {symbol}: {e}")
            return None

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        if not self.cache_enabled:
            return None

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (datetime.now() - cached["timestamp"]).total_seconds() / 60

            if age < self.cache_ttl_minutes:
                logger.debug(f"Cache hit for {cache_key}")
                return cached["data"]
            else:
                # 缓存过期
                del self._cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if not self.cache_enabled:
            return

        self._cache[cache_key] = {
            "data": data.copy(),
            "timestamp": datetime.now(),
        }

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("IntradayDataLoader cache cleared")

    def check_api_connection(self) -> bool:
        """
        检查 API 连接是否正常

        Returns:
            True if API is reachable
        """
        try:
            # 尝试获取一个简单的报价 (stable API 使用查询参数)
            data = self._make_fmp_request("/quote", params={"symbol": "AAPL"})
            return data is not None and len(data) > 0
        except Exception as e:
            logger.warning(f"API connection check failed: {e}")
            return False


# 便捷函数
def create_fmp_loader(api_key: Optional[str] = None) -> IntradayDataLoader:
    """创建 FMP 数据加载器的便捷函数"""
    return IntradayDataLoader(fmp_api_key=api_key)
