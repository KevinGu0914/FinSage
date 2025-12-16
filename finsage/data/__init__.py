"""
FinSage Data Module
数据加载和处理模块

FMP API 统一使用:
- FMPClient.BASE_URL = "https://financialmodelingprep.com/stable"
- 所有新代码应使用 fmp_client.py 中的统一接口
"""

from finsage.data.data_loader import DataLoader
from finsage.data.market_data import MarketDataProvider
from finsage.data.enhanced_data_loader import EnhancedDataLoader
from finsage.data.fmp_client import (
    FMPClient,
    FactorScreener,
    FMPNewsClient,
    FMPETFClient,
    get_fmp_client,
    get_news_client,
    get_etf_client,
    get_factor_screener,
    screen_stocks,
    get_quote,
    get_historical_price,
    get_stock_news,
    get_profile,
)

__all__ = [
    # Core data loaders
    "DataLoader",
    "MarketDataProvider",
    "EnhancedDataLoader",
    # Unified FMP client
    "FMPClient",
    "FactorScreener",
    "FMPNewsClient",
    "FMPETFClient",
    # Singleton getters
    "get_fmp_client",
    "get_news_client",
    "get_etf_client",
    "get_factor_screener",
    # Convenience functions
    "screen_stocks",
    "get_quote",
    "get_historical_price",
    "get_stock_news",
    "get_profile",
]
