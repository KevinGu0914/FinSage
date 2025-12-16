"""
FMP Ultra API Client
FMP Ultra 统一 API 客户端

Documentation: https://site.financialmodelingprep.com/developer/docs#chart
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import requests
import logging
import time
import os
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class FMPClient:
    """
    FMP Ultra API 统一客户端

    支持:
    - 全市场股票筛选 (company-screener)
    - 批量报价 (batch-quote)
    - 财务指标 (key-metrics, ratios)
    - 历史价格 (historical-price-eod)
    - 财务报表 (income-statement, balance-sheet, cash-flow)

    URL 常量 (所有 FMP URL 应从这里引用):
    - BASE_URL: stable API (推荐，大部分端点)
    - V3_URL: v3 API (fallback，经济日历等)
    - V4_URL: v4 API (期权、COT报告等)
    """

    # ==========================================================================
    # FMP API URL 常量 - 所有文件应从这里引用
    # ==========================================================================
    BASE_URL = "https://financialmodelingprep.com/stable"  # 推荐使用
    V3_URL = "https://financialmodelingprep.com/api/v3"    # 经济日历 fallback
    V4_URL = "https://financialmodelingprep.com/api/v4"    # 期权/COT 报告

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rate_limit: int = 750,  # Ultra tier: 750 requests/min
    ):
        """
        初始化 FMP 客户端

        Args:
            api_key: FMP API Key (或从环境变量读取)
            cache_dir: 缓存目录
            rate_limit: 每分钟请求限制
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY") or os.environ.get("OA_FMP_KEY")
        if not self.api_key:
            raise ValueError("FMP API key not found. Set FMP_API_KEY environment variable.")

        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/cache/fmp")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit = rate_limit
        self._request_times: List[float] = []

        logger.info(f"FMPClient initialized (rate_limit={rate_limit}/min)")

    def _rate_limit_wait(self):
        """速率限制等待"""
        now = time.time()
        # 清理超过1分钟的请求记录
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self._request_times.append(time.time())

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = 30,
        _retry_count: int = 0,
    ) -> Any:
        """
        发送 API 请求

        Args:
            endpoint: API 端点 (不含 base URL)
            params: 请求参数
            timeout: 超时时间
            _retry_count: 内部重试计数器

        Returns:
            JSON 响应数据
        """
        MAX_RETRIES = 3
        self._rate_limit_wait()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apikey"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                if _retry_count >= MAX_RETRIES:
                    logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries: {endpoint}")
                    return None
                wait_time = 5 * (2 ** _retry_count)  # Exponential backoff: 5, 10, 20 seconds
                logger.warning(f"Rate limited by FMP, waiting {wait_time} seconds (retry {_retry_count + 1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                return self._request(endpoint, params, timeout, _retry_count + 1)
            else:
                logger.error(f"FMP API error: {response.status_code} - {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    # =========================================================================
    # Stock Screener - 全市场筛选
    # =========================================================================

    def screen_stocks(
        self,
        market_cap_min: Optional[float] = None,
        market_cap_max: Optional[float] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        volume_min: Optional[float] = None,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        dividend_min: Optional[float] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        country: str = "US",
        exchange: str = "NYSE,NASDAQ,AMEX",
        is_actively_trading: bool = True,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        全市场股票筛选

        Args:
            market_cap_min: 最小市值
            market_cap_max: 最大市值
            price_min: 最低价格
            price_max: 最高价格
            volume_min: 最小成交量
            beta_min: 最小 Beta
            beta_max: 最大 Beta
            dividend_min: 最小股息率
            sector: 行业 (Technology, Healthcare, etc.)
            industry: 细分行业
            country: 国家 (US, CN, etc.)
            exchange: 交易所
            is_actively_trading: 是否活跃交易
            limit: 返回数量限制

        Returns:
            筛选结果 DataFrame
        """
        params = {
            "country": country,
            "exchange": exchange,
            "isActivelyTrading": is_actively_trading,
            "limit": limit,
        }

        if market_cap_min:
            params["marketCapMoreThan"] = market_cap_min
        if market_cap_max:
            params["marketCapLowerThan"] = market_cap_max
        if price_min:
            params["priceMoreThan"] = price_min
        if price_max:
            params["priceLowerThan"] = price_max
        if volume_min:
            params["volumeMoreThan"] = volume_min
        if beta_min:
            params["betaMoreThan"] = beta_min
        if beta_max:
            params["betaLowerThan"] = beta_max
        if dividend_min:
            params["dividendMoreThan"] = dividend_min
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry

        data = self._request("/company-screener", params)

        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            logger.info(f"Screened {len(df)} stocks with given criteria")
            return df

        return pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        """获取所有股票列表"""
        data = self._request("/stock-list")
        if data and isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_etf_list(self) -> pd.DataFrame:
        """获取所有 ETF 列表"""
        data = self._request("/etf-list")
        if data and isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    # =========================================================================
    # Multi-Asset Class - 多资产类别
    # =========================================================================

    def get_cryptocurrency_list(self) -> pd.DataFrame:
        """
        获取加密货币列表

        Returns:
            加密货币 DataFrame (4,786+ 种)
        """
        data = self._request("/cryptocurrency-list")
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            df["asset_class"] = "crypto"
            logger.info(f"Loaded {len(df)} cryptocurrencies")
            return df
        return pd.DataFrame()

    def get_commodities_list(self) -> pd.DataFrame:
        """
        获取大宗商品列表

        Returns:
            商品期货 DataFrame (40+ 种)
        """
        data = self._request("/commodities-list")
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            df["asset_class"] = "commodities"
            logger.info(f"Loaded {len(df)} commodities")
            return df
        return pd.DataFrame()

    def get_reits(
        self,
        country: str = "US",
        market_cap_min: Optional[float] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        获取 REITs (房地产信托) 列表

        Args:
            country: 国家
            market_cap_min: 最小市值
            limit: 返回数量限制

        Returns:
            REITs DataFrame
        """
        df = self.screen_stocks(
            sector="Real Estate",
            country=country,
            market_cap_min=market_cap_min,
            limit=limit,
        )
        if not df.empty:
            df["asset_class"] = "reits"
            logger.info(f"Loaded {len(df)} REITs")
        return df

    def get_bond_etfs(self) -> pd.DataFrame:
        """
        获取债券 ETF 列表

        Returns:
            债券类 ETF DataFrame (~1,920 只)
        """
        # 从缓存加载 ETF 列表
        cached = self._load_cache("etf_list", max_age_hours=24)
        if cached is not None:
            etfs = pd.DataFrame(cached)
        else:
            etfs = self.get_etf_list()
            if not etfs.empty:
                self._save_cache("etf_list", etfs.to_dict("records"))

        if etfs.empty:
            return pd.DataFrame()

        # 按名称关键词筛选债券 ETF
        bond_keywords = [
            'bond', 'treasury', 'fixed income', 'corporate', 'municipal',
            'aggregate', 'government', 'investment grade', 'high yield',
            'tips', 'inflation', 'short-term', 'intermediate', 'long-term'
        ]

        name_col = 'name' if 'name' in etfs.columns else 'companyName'
        if name_col not in etfs.columns:
            return pd.DataFrame()

        mask = etfs[name_col].str.lower().apply(
            lambda x: any(kw in str(x) for kw in bond_keywords) if pd.notna(x) else False
        )
        bond_etfs = etfs[mask].copy()
        bond_etfs["asset_class"] = "bonds"

        logger.info(f"Filtered {len(bond_etfs)} bond ETFs")
        return bond_etfs

    def get_commodity_etfs(self) -> pd.DataFrame:
        """
        获取商品 ETF 列表

        Returns:
            商品类 ETF DataFrame (~599 只)
        """
        cached = self._load_cache("etf_list", max_age_hours=24)
        if cached is not None:
            etfs = pd.DataFrame(cached)
        else:
            etfs = self.get_etf_list()
            if not etfs.empty:
                self._save_cache("etf_list", etfs.to_dict("records"))

        if etfs.empty:
            return pd.DataFrame()

        commodity_keywords = [
            'gold', 'silver', 'oil', 'commodity', 'metal', 'energy',
            'natural gas', 'agriculture', 'copper', 'platinum', 'palladium',
            'grain', 'wheat', 'corn', 'soybean', 'crude', 'mining'
        ]

        name_col = 'name' if 'name' in etfs.columns else 'companyName'
        if name_col not in etfs.columns:
            return pd.DataFrame()

        mask = etfs[name_col].str.lower().apply(
            lambda x: any(kw in str(x) for kw in commodity_keywords) if pd.notna(x) else False
        )
        commodity_etfs = etfs[mask].copy()
        commodity_etfs["asset_class"] = "commodities"

        logger.info(f"Filtered {len(commodity_etfs)} commodity ETFs")
        return commodity_etfs

    def get_reit_etfs(self) -> pd.DataFrame:
        """
        获取 REIT ETF 列表

        Returns:
            REIT 类 ETF DataFrame (~200 只)
        """
        cached = self._load_cache("etf_list", max_age_hours=24)
        if cached is not None:
            etfs = pd.DataFrame(cached)
        else:
            etfs = self.get_etf_list()
            if not etfs.empty:
                self._save_cache("etf_list", etfs.to_dict("records"))

        if etfs.empty:
            return pd.DataFrame()

        reit_keywords = [
            'real estate', 'reit', 'property', 'mortgage', 'housing',
            'residential', 'commercial', 'industrial real'
        ]

        name_col = 'name' if 'name' in etfs.columns else 'companyName'
        if name_col not in etfs.columns:
            return pd.DataFrame()

        mask = etfs[name_col].str.lower().apply(
            lambda x: any(kw in str(x) for kw in reit_keywords) if pd.notna(x) else False
        )
        reit_etfs = etfs[mask].copy()
        reit_etfs["asset_class"] = "reits"

        logger.info(f"Filtered {len(reit_etfs)} REIT ETFs")
        return reit_etfs

    def get_full_asset_universe(
        self,
        include_stocks: bool = True,
        include_crypto: bool = True,
        include_commodities: bool = True,
        include_reits: bool = True,
        include_bonds: bool = True,
        stock_market_cap_min: float = 1e9,
        stock_limit: int = 1000,
        cache_hours: int = 24,
    ) -> Dict[str, pd.DataFrame]:
        """
        获取全资产类别投资范围

        Args:
            include_stocks: 包含股票
            include_crypto: 包含加密货币
            include_commodities: 包含大宗商品
            include_reits: 包含 REITs
            include_bonds: 包含债券 ETF
            stock_market_cap_min: 股票最小市值
            stock_limit: 股票数量限制
            cache_hours: 缓存有效期

        Returns:
            按资产类别分组的 DataFrame 字典
        """
        universe = {}

        # 股票
        if include_stocks:
            cache_key = f"stocks_{stock_market_cap_min}_{stock_limit}"
            cached = self._load_cache(cache_key, cache_hours)
            if cached is not None:
                universe["stocks"] = pd.DataFrame(cached)
            else:
                stocks = self.screen_stocks(
                    market_cap_min=stock_market_cap_min,
                    limit=stock_limit,
                )
                if not stocks.empty:
                    stocks["asset_class"] = "stocks"
                    self._save_cache(cache_key, stocks.to_dict("records"))
                    universe["stocks"] = stocks

        # 加密货币
        if include_crypto:
            cached = self._load_cache("crypto_list", cache_hours)
            if cached is not None:
                universe["crypto"] = pd.DataFrame(cached)
            else:
                crypto = self.get_cryptocurrency_list()
                if not crypto.empty:
                    self._save_cache("crypto_list", crypto.to_dict("records"))
                    universe["crypto"] = crypto

        # 大宗商品 (期货 + ETF)
        if include_commodities:
            cached = self._load_cache("commodities_combined", cache_hours)
            if cached is not None:
                universe["commodities"] = pd.DataFrame(cached)
            else:
                commodities = self.get_commodities_list()
                commodity_etfs = self.get_commodity_etfs()
                combined = pd.concat([commodities, commodity_etfs], ignore_index=True)
                if not combined.empty:
                    self._save_cache("commodities_combined", combined.to_dict("records"))
                    universe["commodities"] = combined

        # REITs (个股 + ETF)
        if include_reits:
            cached = self._load_cache("reits_combined", cache_hours)
            if cached is not None:
                universe["reits"] = pd.DataFrame(cached)
            else:
                reits = self.get_reits()
                reit_etfs = self.get_reit_etfs()
                combined = pd.concat([reits, reit_etfs], ignore_index=True)
                if not combined.empty:
                    self._save_cache("reits_combined", combined.to_dict("records"))
                    universe["reits"] = combined

        # 债券 (ETF)
        if include_bonds:
            cached = self._load_cache("bond_etfs", cache_hours)
            if cached is not None:
                universe["bonds"] = pd.DataFrame(cached)
            else:
                bonds = self.get_bond_etfs()
                if not bonds.empty:
                    self._save_cache("bond_etfs", bonds.to_dict("records"))
                    universe["bonds"] = bonds

        # 统计
        total = sum(len(df) for df in universe.values())
        logger.info(f"Full asset universe: {total} assets across {len(universe)} classes")
        for cls, df in universe.items():
            logger.info(f"  {cls}: {len(df)}")

        return universe

    # =========================================================================
    # Market Data - 市场数据
    # =========================================================================

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """获取单个股票报价"""
        data = self._request("/quote", {"symbol": symbol})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_batch_quote(self, symbols: List[str]) -> pd.DataFrame:
        """批量获取股票报价"""
        if not symbols:
            return pd.DataFrame()

        # FMP 批量查询限制
        batch_size = 100
        all_data = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            symbols_str = ",".join(batch)
            data = self._request("/batch-quote", {"symbols": symbols_str})

            if data and isinstance(data, list):
                all_data.extend(data)

            if i + batch_size < len(symbols):
                time.sleep(0.1)  # 批次间短暂休息

        if all_data:
            return pd.DataFrame(all_data)
        return pd.DataFrame()

    def get_profile(self, symbol: str) -> Optional[Dict]:
        """获取公司概况"""
        data = self._request("/profile", {"symbol": symbol})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    # =========================================================================
    # Financial Metrics - 财务指标
    # =========================================================================

    def get_key_metrics_ttm(self, symbol: str) -> Optional[Dict]:
        """获取 TTM 关键指标"""
        data = self._request("/key-metrics-ttm", {"symbol": symbol})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_key_metrics_batch(self, symbols: List[str]) -> pd.DataFrame:
        """批量获取关键指标"""
        all_data = []

        for symbol in symbols:
            metrics = self.get_key_metrics_ttm(symbol)
            if metrics:
                metrics["symbol"] = symbol
                all_data.append(metrics)

        if all_data:
            return pd.DataFrame(all_data)
        return pd.DataFrame()

    def get_ratios_ttm(self, symbol: str) -> Optional[Dict]:
        """获取 TTM 财务比率"""
        data = self._request("/ratios-ttm", {"symbol": symbol})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_financial_scores(self, symbol: str) -> Optional[Dict]:
        """获取财务评分 (Piotroski, Altman Z-Score 等)"""
        data = self._request("/financial-scores", {"symbol": symbol})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    # =========================================================================
    # Financial Statements - 财务报表
    # =========================================================================

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> pd.DataFrame:
        """获取利润表"""
        data = self._request(
            "/income-statement",
            {"symbol": symbol, "period": period, "limit": limit}
        )
        if data and isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> pd.DataFrame:
        """获取资产负债表"""
        data = self._request(
            "/balance-sheet-statement",
            {"symbol": symbol, "period": period, "limit": limit}
        )
        if data and isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> pd.DataFrame:
        """获取现金流量表"""
        data = self._request(
            "/cash-flow-statement",
            {"symbol": symbol, "period": period, "limit": limit}
        )
        if data and isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    # =========================================================================
    # Historical Data - 历史数据
    # =========================================================================

    def get_historical_price(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取历史日线数据

        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            历史价格 DataFrame
        """
        params = {"symbol": symbol}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        data = self._request("/historical-price-eod/full", params)

        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df = df.sort_index()
                # 标准化列名
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "adjClose": "Adj Close",
                })
            return df

        return pd.DataFrame()

    def get_historical_prices_batch(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量获取历史价格"""
        result = {}

        for symbol in symbols:
            df = self.get_historical_price(symbol, start_date, end_date)
            if not df.empty:
                result[symbol] = df
            time.sleep(0.1)  # 避免触发限流

        return result

    # =========================================================================
    # Caching - 缓存管理
    # =========================================================================

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[Any]:
        """加载缓存"""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=max_age_hours):
                return None

            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def _save_cache(self, cache_key: str, data: Any):
        """保存缓存"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def screen_stocks_cached(
        self,
        cache_key: str = "screener_default",
        cache_hours: int = 24,
        **kwargs
    ) -> pd.DataFrame:
        """
        带缓存的股票筛选

        Args:
            cache_key: 缓存键名
            cache_hours: 缓存有效期 (小时)
            **kwargs: 传递给 screen_stocks 的参数

        Returns:
            筛选结果 DataFrame
        """
        cached = self._load_cache(cache_key, cache_hours)
        if cached is not None:
            logger.info(f"Loaded {len(cached)} stocks from cache ({cache_key})")
            return pd.DataFrame(cached)

        df = self.screen_stocks(**kwargs)
        if not df.empty:
            self._save_cache(cache_key, df.to_dict("records"))

        return df


# =============================================================================
# Factor Screening - 因子筛选
# =============================================================================

class FactorScreener:
    """
    因子筛选器

    基于 FMP 数据进行量化因子筛选，支持:
    - 价值因子 (P/E, P/B, EV/EBITDA)
    - 质量因子 (ROE, ROA, Gross Margin)
    - 动量因子 (Price Change)
    - 成长因子 (Revenue Growth, EPS Growth)
    """

    def __init__(self, fmp_client: Optional[FMPClient] = None):
        self.client = fmp_client or FMPClient()

    def screen_by_factors(
        self,
        base_universe: Optional[pd.DataFrame] = None,
        value_weight: float = 0.25,
        quality_weight: float = 0.25,
        momentum_weight: float = 0.25,
        growth_weight: float = 0.25,
        top_n: int = 100,
        sector_filter: Optional[str] = None,
        market_cap_min: float = 1e9,  # 10亿美元
    ) -> pd.DataFrame:
        """
        多因子筛选

        Args:
            base_universe: 基础股票池 (如果为空则从 screener 获取)
            value_weight: 价值因子权重
            quality_weight: 质量因子权重
            momentum_weight: 动量因子权重
            growth_weight: 成长因子权重
            top_n: 返回排名前 N 只
            sector_filter: 行业过滤
            market_cap_min: 最小市值

        Returns:
            排序后的股票 DataFrame (含因子得分)
        """
        # 1. 获取基础股票池
        if base_universe is None or base_universe.empty:
            base_universe = self.client.screen_stocks_cached(
                cache_key="factor_base_universe",
                market_cap_min=market_cap_min,
                sector=sector_filter,
                limit=2000,
            )

        if base_universe.empty:
            logger.warning("No stocks found in base universe")
            return pd.DataFrame()

        logger.info(f"Screening {len(base_universe)} stocks with factors")

        # 2. 提取因子字段并计算得分
        df = base_universe.copy()

        # 确保必要字段存在
        factor_cols = ["pe", "priceToBook", "roe", "roa", "grossProfitMargin",
                       "priceChange1Y", "revenueGrowth", "epsGrowth"]

        for col in factor_cols:
            if col not in df.columns:
                df[col] = np.nan

        # 价值得分 (低 P/E, 低 P/B 更好)
        df["value_score"] = self._rank_score(df, ["pe", "priceToBook"], ascending=True)

        # 质量得分 (高 ROE, ROA, Margin 更好)
        df["quality_score"] = self._rank_score(
            df, ["roe", "roa", "grossProfitMargin"], ascending=False
        )

        # 动量得分 (高涨幅更好)
        df["momentum_score"] = self._rank_score(df, ["priceChange1Y"], ascending=False)

        # 成长得分 (高增长更好)
        df["growth_score"] = self._rank_score(
            df, ["revenueGrowth", "epsGrowth"], ascending=False
        )

        # 3. 计算综合得分
        df["composite_score"] = (
            value_weight * df["value_score"] +
            quality_weight * df["quality_score"] +
            momentum_weight * df["momentum_score"] +
            growth_weight * df["growth_score"]
        )

        # 4. 排序并返回 Top N
        df = df.sort_values("composite_score", ascending=False).head(top_n)

        logger.info(f"Selected top {len(df)} stocks by composite factor score")

        return df

    def _rank_score(
        self,
        df: pd.DataFrame,
        columns: List[str],
        ascending: bool = True
    ) -> pd.Series:
        """
        计算排名得分 (0-1 标准化)

        Args:
            df: 数据 DataFrame
            columns: 用于计算的列
            ascending: 是否升序 (True = 低值高分)

        Returns:
            归一化得分 Series
        """
        scores = []

        for col in columns:
            if col in df.columns:
                # 处理无穷值和极端值
                series = df[col].replace([np.inf, -np.inf], np.nan)
                # 百分位排名
                rank = series.rank(pct=True, ascending=ascending)
                scores.append(rank.fillna(0.5))

        if scores:
            return pd.concat(scores, axis=1).mean(axis=1)
        return pd.Series(0.5, index=df.index)


# =============================================================================
# News & Sentiment - 新闻情绪
# =============================================================================

class FMPNewsClient:
    """
    FMP 新闻客户端

    使用正确的 stable 端点:
    - /news/stock?symbols=XXX (股票新闻)
    - /news/general (通用新闻)
    - /news/crypto (加密货币新闻)
    - /news/forex (外汇新闻)
    """

    def __init__(self, fmp_client: Optional[FMPClient] = None):
        self.client = fmp_client or FMPClient()

    def get_stock_news(
        self,
        symbols: List[str],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        获取股票新闻

        Args:
            symbols: 股票代码列表
            limit: 每个股票的新闻数量

        Returns:
            新闻列表
        """
        all_news = []

        for symbol in symbols:
            # 正确的端点: /news/stock?symbols=XXX
            data = self.client._request(
                "/news/stock",
                {"symbols": symbol, "limit": limit}
            )

            if data and isinstance(data, list):
                for item in data:
                    item["symbol"] = symbol
                    # 标准化字段名
                    if "publishedDate" in item:
                        item["date"] = item["publishedDate"]
                    if "url" in item and "link" not in item:
                        item["link"] = item["url"]
                    all_news.append(item)

            time.sleep(0.2)  # 避免速率限制

        logger.info(f"Loaded {len(all_news)} news for {len(symbols)} symbols")
        return all_news

    def get_general_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取通用市场新闻"""
        data = self.client._request("/news/general", {"limit": limit})
        if data and isinstance(data, list):
            return data
        return []

    def get_crypto_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取加密货币新闻"""
        data = self.client._request("/news/crypto", {"limit": limit})
        if data and isinstance(data, list):
            return data
        return []


# =============================================================================
# ETF Holdings - ETF 持仓
# =============================================================================

class FMPETFClient:
    """
    FMP ETF 客户端

    使用正确的 stable 端点:
    - /etf/holdings?symbol=XXX (ETF 持仓)
    - /etf-list (ETF 列表)
    """

    def __init__(self, fmp_client: Optional[FMPClient] = None):
        self.client = fmp_client or FMPClient()

    def get_etf_holdings(self, symbol: str) -> pd.DataFrame:
        """
        获取 ETF 持仓

        Args:
            symbol: ETF 代码

        Returns:
            持仓 DataFrame
        """
        # 正确的端点: /etf/holdings?symbol=XXX
        data = self.client._request("/etf/holdings", {"symbol": symbol})

        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} holdings for ETF {symbol}")
            return df

        return pd.DataFrame()

    def get_etf_sector_weightings(self, symbol: str) -> Dict[str, float]:
        """获取 ETF 行业权重"""
        data = self.client._request("/etf/sector-weighting", {"symbol": symbol})

        if data and isinstance(data, list):
            return {item.get("sector", "Unknown"): item.get("weightPercentage", 0)
                    for item in data}
        return {}


# =============================================================================
# Singleton Pattern - 单例模式 (线程安全)
# =============================================================================

_fmp_client_instance: Optional[FMPClient] = None
_fmp_news_client_instance: Optional[FMPNewsClient] = None
_fmp_etf_client_instance: Optional[FMPETFClient] = None
_factor_screener_instance: Optional[FactorScreener] = None
_singleton_lock = threading.Lock()


def get_fmp_client() -> FMPClient:
    """
    获取 FMP 客户端单例 (线程安全)

    Returns:
        FMPClient 实例
    """
    global _fmp_client_instance
    if _fmp_client_instance is None:
        with _singleton_lock:
            if _fmp_client_instance is None:
                _fmp_client_instance = FMPClient()
    return _fmp_client_instance


def get_news_client() -> FMPNewsClient:
    """获取新闻客户端单例 (线程安全)"""
    global _fmp_news_client_instance
    if _fmp_news_client_instance is None:
        with _singleton_lock:
            if _fmp_news_client_instance is None:
                _fmp_news_client_instance = FMPNewsClient(get_fmp_client())
    return _fmp_news_client_instance


def get_etf_client() -> FMPETFClient:
    """获取 ETF 客户端单例 (线程安全)"""
    global _fmp_etf_client_instance
    if _fmp_etf_client_instance is None:
        with _singleton_lock:
            if _fmp_etf_client_instance is None:
                _fmp_etf_client_instance = FMPETFClient(get_fmp_client())
    return _fmp_etf_client_instance


def get_factor_screener() -> FactorScreener:
    """获取因子筛选器单例 (线程安全)"""
    global _factor_screener_instance
    if _factor_screener_instance is None:
        with _singleton_lock:
            if _factor_screener_instance is None:
                _factor_screener_instance = FactorScreener(get_fmp_client())
    return _factor_screener_instance


# =============================================================================
# Convenience Functions - 便捷函数
# =============================================================================

def screen_stocks(**kwargs) -> pd.DataFrame:
    """便捷函数: 股票筛选"""
    return get_fmp_client().screen_stocks(**kwargs)


def get_quote(symbol: str) -> Optional[Dict]:
    """便捷函数: 获取报价"""
    return get_fmp_client().get_quote(symbol)


def get_historical_price(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """便捷函数: 获取历史价格"""
    return get_fmp_client().get_historical_price(symbol, start_date, end_date)


def get_stock_news(symbols: List[str], limit: int = 50) -> List[Dict]:
    """便捷函数: 获取股票新闻"""
    return get_news_client().get_stock_news(symbols, limit)


def get_profile(symbol: str) -> Optional[Dict]:
    """便捷函数: 获取公司概况"""
    return get_fmp_client().get_profile(symbol)
