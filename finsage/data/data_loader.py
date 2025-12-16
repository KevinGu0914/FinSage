"""
Data Loader
数据加载器 - 支持多种数据源
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote
import pandas as pd
import numpy as np
import logging
import os

from finsage.data.fmp_client import FMPClient

logger = logging.getLogger(__name__)


class DataLoader:
    """
    多数据源数据加载器

    支持:
    - Yahoo Finance (yfinance)
    - FMP (Financial Modeling Prep)
    - 本地CSV文件
    - 自定义数据源
    """

    def __init__(
        self,
        data_source: str = "yfinance",
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化数据加载器

        Args:
            data_source: 数据源 ("yfinance", "fmp", "local")
            cache_dir: 缓存目录
            api_key: API密钥 (用于FMP等)
        """
        self.data_source = data_source
        self.cache_dir = cache_dir
        self.api_key = api_key or os.environ.get("FMP_API_KEY")

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        logger.info(f"DataLoader initialized with source={data_source}")

    def load_price_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        加载价格数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            interval: 时间间隔 ("1d", "1h", etc.)

        Returns:
            价格数据DataFrame
        """
        if self.data_source == "yfinance":
            return self._load_yfinance(symbols, start_date, end_date, interval)
        elif self.data_source == "fmp":
            return self._load_fmp(symbols, start_date, end_date)
        elif self.data_source == "local":
            return self._load_local(symbols, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

    def _load_yfinance(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """通过yfinance加载数据 (带速率限制和重试)"""
        try:
            import yfinance as yf
            import time

            all_data = {}
            batch_size = 3  # 每批处理3个，减少速率限制

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                # 分批下载，避免速率限制
                for symbol in batch:
                    try:
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(start=start_date, end=end_date, interval=interval)

                        if not df.empty:
                            # 重命名列以匹配标准格式
                            df = df.rename(columns={
                                "Open": "Open",
                                "High": "High",
                                "Low": "Low",
                                "Close": "Close",
                                "Volume": "Volume"
                            })
                            all_data[symbol] = df
                            logger.debug(f"Loaded {len(df)} days of data for {symbol}")
                        else:
                            logger.warning(f"No data for {symbol}")

                        time.sleep(0.5)  # 每个符号之间休息

                    except Exception as e:
                        logger.warning(f"Failed to load {symbol}: {e}")
                        time.sleep(1)  # 错误后多休息一会

                # 批次之间休息更长时间
                if i + batch_size < len(symbols):
                    time.sleep(1)

            if not all_data:
                logger.warning("No data loaded from yfinance for any symbol")
                return pd.DataFrame()

            # 合并数据
            combined = pd.concat(all_data, axis=1)
            logger.info(f"yfinance loaded data for {len(all_data)} symbols")
            return combined

        except Exception as e:
            logger.error(f"Failed to load from yfinance: {e}")
            return pd.DataFrame()

    def _load_fmp(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """通过FMP API加载数据 (使用新的 stable 端点)"""
        if not self.api_key:
            logger.warning("FMP API key not set")
            return pd.DataFrame()

        try:
            import requests
            import time

            all_data = {}
            batch_size = 5  # 每批处理5个股票，避免API限制

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                for symbol in batch:
                    # 处理加密货币符号（BTC-USD -> BTCUSD）
                    fmp_symbol = symbol.replace("-", "")

                    # 使用 stable 端点 (旧的 /api/v3/ 已废弃)
                    url = (
                        f"{FMPClient.BASE_URL}/historical-price-eod/full"
                        f"?symbol={quote(fmp_symbol, safe='')}&from={start_date}&to={end_date}&apikey={self.api_key}"
                    )

                    try:
                        response = requests.get(url, timeout=30)
                        if response.status_code == 200:
                            data = response.json()
                            # 新端点直接返回数组，不是 {"historical": [...]}
                            if isinstance(data, list) and len(data) > 0:
                                df = pd.DataFrame(data)
                                df["date"] = pd.to_datetime(df["date"])
                                df.set_index("date", inplace=True)
                                df = df.sort_index()  # 确保按日期排序

                                # 标准化列名 (新端点使用小写)
                                df = df.rename(columns={
                                    "open": "Open",
                                    "high": "High",
                                    "low": "Low",
                                    "close": "Close",
                                    "volume": "Volume",
                                    "vwap": "VWAP"
                                })

                                all_data[symbol] = df
                                logger.debug(f"Loaded {len(df)} days of data for {symbol}")
                            else:
                                logger.warning(f"No historical data for {symbol}")
                        elif response.status_code == 429:
                            logger.warning(f"Rate limited, waiting 2 seconds...")
                            time.sleep(2)
                        else:
                            logger.warning(f"Failed to load {symbol}: HTTP {response.status_code}")
                    except requests.exceptions.Timeout:
                        logger.warning(f"Timeout loading {symbol}")
                    except Exception as e:
                        logger.warning(f"Error loading {symbol}: {e}")

                # 批次间休息，避免触发限制
                if i + batch_size < len(symbols):
                    time.sleep(0.5)

            if not all_data:
                logger.warning("No data loaded from FMP for any symbol")
                return pd.DataFrame()

            # 合并数据，使用MultiIndex
            combined = pd.concat(all_data, axis=1)
            logger.info(f"FMP loaded data for {len(all_data)} symbols")
            return combined

        except Exception as e:
            logger.error(f"Failed to load from FMP: {e}")
            return pd.DataFrame()

    def _load_local(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """从本地文件加载数据"""
        if not self.cache_dir:
            logger.warning("Cache directory not set for local loading")
            return pd.DataFrame()

        all_data = {}
        for symbol in symbols:
            file_path = os.path.join(self.cache_dir, f"{symbol}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                all_data[symbol] = df

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, axis=1)
        return combined

    def load_news(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        加载新闻数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            limit: 每个股票的新闻数量限制

        Returns:
            新闻列表
        """
        if self.data_source == "fmp" and self.api_key:
            return self._load_fmp_news(symbols, start_date, end_date, limit)
        return []

    def _load_fmp_news(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit: int
    ) -> List[Dict]:
        """从FMP加载新闻 (使用 stable API 端点)"""
        try:
            import requests
            import time

            all_news = []
            batch_size = 5  # 每批处理5个股票

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                for symbol in batch:
                    # 使用 stable 端点: /stable/news/stock?symbols=XXX
                    url = (
                        f"{FMPClient.BASE_URL}/news/stock"
                        f"?symbols={quote(symbol, safe='')}&limit={limit}&apikey={self.api_key}"
                    )
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            # stable 端点直接返回数组
                            if isinstance(data, list):
                                for item in data:
                                    item["symbol"] = symbol
                                    # 标准化字段名
                                    if "publishedDate" in item:
                                        item["date"] = item["publishedDate"]
                                    if "url" in item and "link" not in item:
                                        item["link"] = item["url"]
                                    all_news.append(item)
                                logger.debug(f"Loaded {len(data)} news for {symbol}")
                        elif response.status_code == 403:
                            logger.warning(f"FMP news API returned 403 for {symbol}")
                        else:
                            logger.warning(f"FMP news failed for {symbol}: HTTP {response.status_code}")
                    except requests.exceptions.Timeout:
                        logger.warning(f"Timeout loading news for {symbol}")
                    except Exception as e:
                        logger.warning(f"Error loading news for {symbol}: {e}")

                    time.sleep(0.2)  # 避免速率限制

                # 批次间休息
                if i + batch_size < len(symbols):
                    time.sleep(0.3)

            logger.info(f"FMP loaded {len(all_news)} news items for {len(symbols)} symbols")
            return all_news

        except Exception as e:
            logger.error(f"Failed to load news from FMP: {e}")
            return []

    def calculate_returns(
        self,
        price_data: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        计算收益率

        Args:
            price_data: 价格数据
            method: 计算方法 ("log" or "simple")

        Returns:
            收益率DataFrame
        """
        if "Close" in price_data.columns or "Adj Close" in price_data.columns:
            prices = price_data.get("Adj Close", price_data.get("Close"))
        else:
            prices = price_data

        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()

    def get_technical_indicators(
        self,
        price_data: pd.DataFrame,
        indicators: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        计算技术指标

        Args:
            price_data: 价格数据
            indicators: 指标列表

        Returns:
            技术指标字典
        """
        if indicators is None:
            indicators = ["sma_20", "sma_50", "rsi_14", "macd", "bb"]

        result = {}

        close = price_data.get("Close", price_data)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0] if close.shape[1] == 1 else close

        for indicator in indicators:
            if indicator.startswith("sma_"):
                period = int(indicator.split("_")[1])
                result[indicator] = close.rolling(window=period).mean()

            elif indicator.startswith("ema_"):
                period = int(indicator.split("_")[1])
                result[indicator] = close.ewm(span=period, adjust=False).mean()

            elif indicator.startswith("rsi_"):
                period = int(indicator.split("_")[1])
                result[indicator] = self._calculate_rsi(close, period)

            elif indicator == "macd":
                result["macd"], result["macd_signal"], result["macd_hist"] = \
                    self._calculate_macd(close)

            elif indicator == "bb":
                result["bb_upper"], result["bb_middle"], result["bb_lower"] = \
                    self._calculate_bollinger_bands(close)

        return result

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        # 安全除法：避免 loss=0 时除零错误
        rs = np.where(loss > 1e-10, gain / loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # 将 numpy array 转回 pandas Series 以保持一致性
        return pd.Series(rsi, index=prices.index)

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """计算MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
