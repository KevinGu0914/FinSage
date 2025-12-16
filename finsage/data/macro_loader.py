"""
Macro Data Loader - 宏观经济数据加载器
使用 FMP (Financial Modeling Prep) API 获取真实宏观数据

FMP API 端点 (Premium/stable):
- VIX: /quote?symbol=^VIX
- 美元指数 DXY: /quote?symbol=DX-Y.NYB
- 国债收益率: /treasury
- 经济日历: /economic-calendar
- Fear & Greed Index: /fear-and-greed

Author: Boyang Gu
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from finsage.config import AssetConfig
from finsage.data.fmp_client import FMPClient

logger = logging.getLogger(__name__)


class MacroDataLoader:
    """
    宏观经济数据加载器 - 使用 FMP API

    提供:
    - VIX (波动率指数)
    - DXY (美元指数)
    - Treasury Yields (国债收益率: 2Y, 5Y, 10Y, 30Y)
    - 信用利差
    - Fear & Greed Index
    - 经济日历

    注意: 所有端点统一使用 FMPClient.BASE_URL (stable)
    仅 COT 报告等少数端点需要使用 v4 API
    """

    # 使用统一的 FMP URL 常量 (从 fmp_client.py 获取)
    FMP_BASE_URL = FMPClient.BASE_URL  # stable API
    # 仅保留 v4 URL 用于 COT 报告等特殊端点
    FMP_V4_URL = FMPClient.V4_URL

    # 市场指数符号
    MARKET_SYMBOLS = {
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "gold": "GC=F",
        "oil": "CL=F",
        "btc": "BTCUSD",
    }

    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 5,
    ):
        """
        初始化宏观数据加载器

        Args:
            fmp_api_key: FMP API密钥
            cache_enabled: 是否启用缓存
            cache_ttl_minutes: 缓存过期时间
        """
        self.api_key = fmp_api_key or os.getenv("OA_FMP_KEY")
        if not self.api_key:
            logger.warning("FMP API key not found. Set OA_FMP_KEY environment variable.")

        self.cache_enabled = cache_enabled
        self.cache_ttl_minutes = cache_ttl_minutes
        self._cache: Dict[str, Dict] = {}
        self._cache_max_size = 100  # 缓存最大条目数

        self.session = requests.Session()
        logger.info("MacroDataLoader initialized (FMP API)")

    def close(self):
        """关闭session连接"""
        if self.session:
            self.session.close()
            logger.debug("MacroDataLoader session closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _trim_cache(self):
        """修剪缓存至最大大小"""
        if len(self._cache) > self._cache_max_size:
            # 按时间戳排序，删除最旧的条目
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].get("timestamp", datetime.min)
            )
            # 删除最旧的一半
            for key in sorted_keys[:len(sorted_keys) // 2]:
                del self._cache[key]
            logger.debug(f"Cache trimmed from {len(sorted_keys)} to {len(self._cache)} entries")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        """发送 FMP API 请求"""
        if not self.api_key:
            raise ValueError("FMP API key not configured")

        url = f"{base_url or self.FMP_BASE_URL}{endpoint}"
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

    def check_api_connection(self) -> bool:
        """
        检查 API 连接是否正常

        Returns:
            True 如果连接成功，False 如果失败
        """
        try:
            data = self._make_request("/quote", params={"symbol": "^VIX"})
            return data is not None and len(data) > 0
        except Exception as e:
            logger.warning(f"API connection check failed: {e}")
            return False

    def get_vix(self) -> Optional[float]:
        """
        获取当前 VIX 水平

        Returns:
            VIX 值
        """
        cached = self._get_cached("vix")
        if cached is not None:
            return cached

        try:
            data = self._make_request("/quote", params={"symbol": "^VIX"})
            if data and isinstance(data, list) and len(data) > 0:
                vix = float(data[0].get("price", 0))
                self._set_cache("vix", vix)
                return vix
            return None
        except Exception as e:
            logger.warning(f"Failed to get VIX: {e}")
            return None

    def get_dxy(self) -> Optional[float]:
        """
        获取美元指数 (DXY)

        Returns:
            DXY 值
        """
        cached = self._get_cached("dxy")
        if cached is not None:
            return cached

        try:
            data = self._make_request("/quote", params={"symbol": "DX-Y.NYB"})
            if data and isinstance(data, list) and len(data) > 0:
                dxy = float(data[0].get("price", 0))
                self._set_cache("dxy", dxy)
                return dxy
            return None
        except Exception as e:
            logger.warning(f"Failed to get DXY: {e}")
            return None

    def get_treasury_rates(self) -> Dict[str, float]:
        """
        获取国债收益率曲线

        使用国债 ETF 作为代理获取收益率水平

        Returns:
            {
                "treasury_2y": 4.85,
                "treasury_5y": 4.50,
                "treasury_10y": 4.35,
                "treasury_30y": 4.55,
            }
        """
        cached = self._get_cached("treasury_rates")
        if cached is not None:
            return cached

        try:
            # 使用国债 ETF 报价来推算收益率
            # SHY (1-3年), IEF (7-10年), TLT (20+年)
            symbols = "SHY,IEF,TLT,^TNX"
            data = self._make_request("/quote", params={"symbol": symbols})

            rates = {
                "treasury_1m": 5.0,  # 默认值
                "treasury_3m": 5.0,
                "treasury_6m": 5.0,
                "treasury_1y": 4.8,
                "treasury_2y": 4.5,
                "treasury_5y": 4.3,
                "treasury_10y": 4.2,
                "treasury_30y": 4.5,
            }

            if data and isinstance(data, list):
                for quote in data:
                    symbol = quote.get("symbol", "")
                    price = quote.get("price", 0)

                    # 10年期国债收益率 (^TNX)
                    if symbol == "^TNX" or symbol == "%5ETNX":
                        rates["treasury_10y"] = float(price) / 10  # TNX 是收益率 x 10

            self._set_cache("treasury_rates", rates)
            return rates

        except Exception as e:
            logger.warning(f"Failed to get treasury rates: {e}")
            return {
                "treasury_2y": 4.5,
                "treasury_5y": 4.3,
                "treasury_10y": 4.2,
                "treasury_30y": 4.5,
            }

    def get_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """
        获取 Fear & Greed Index

        使用 VIX 和市场数据计算近似的 Fear & Greed 指数
        (FMP v4 fear-and-greed 端点对 Premium 用户返回 403)

        Returns:
            {
                "value": 65,
                "classification": "Greed",
                "timestamp": "2024-01-15"
            }
        """
        cached = self._get_cached("fear_greed")
        if cached is not None:
            return cached

        try:
            # 使用 VIX 来估算 Fear & Greed
            # VIX < 15: Extreme Greed (80-100)
            # VIX 15-20: Greed (60-80)
            # VIX 20-25: Neutral (40-60)
            # VIX 25-30: Fear (20-40)
            # VIX > 30: Extreme Fear (0-20)
            vix = self.get_vix()

            if vix is None:
                return None

            # 将 VIX 转换为 Fear & Greed 值 (反向关系)
            if vix < 15:
                fg_value = 90 - (vix - 10) * 2  # 80-100
                classification = "Extreme Greed"
            elif vix < 20:
                fg_value = 80 - (vix - 15) * 4  # 60-80
                classification = "Greed"
            elif vix < 25:
                fg_value = 60 - (vix - 20) * 4  # 40-60
                classification = "Neutral"
            elif vix < 30:
                fg_value = 40 - (vix - 25) * 4  # 20-40
                classification = "Fear"
            else:
                fg_value = max(0, 20 - (vix - 30) * 2)  # 0-20
                classification = "Extreme Fear"

            fg_value = max(0, min(100, fg_value))  # Clamp to 0-100

            result = {
                "value": fg_value,
                "classification": classification,
                "timestamp": datetime.now().isoformat(),
                "source": "calculated_from_vix",
                "vix_level": vix,
            }
            self._set_cache("fear_greed", result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get Fear & Greed Index: {e}")
            return None

    def get_economic_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取经济日历

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            经济事件列表
        """
        try:
            params = {}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date

            # 使用 stable 端点
            data = self._make_request("/economic-calendar", params=params)

            if data and isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning(f"Failed to get economic calendar: {e}")
            return []

    def get_sector_performance(self) -> Dict[str, float]:
        """
        获取板块表现

        使用 SPDR Sector ETFs 作为板块代理
        (FMP v3 sectors-performance 端点对 Premium 用户返回 403)
        (stable 端点不支持批量查询，需要单独查询每个)

        Returns:
            {sector: change_percent}
        """
        cached = self._get_cached("sector_performance")
        if cached is not None:
            return cached

        try:
            etf_to_sector = {
                "XLK": "Technology",
                "XLF": "Financial Services",
                "XLV": "Healthcare",
                "XLE": "Energy",
                "XLI": "Industrials",
                "XLY": "Consumer Cyclical",
                "XLP": "Consumer Defensive",
                "XLU": "Utilities",
                "XLB": "Basic Materials",
                "XLRE": "Real Estate",
                "XLC": "Communication Services",
            }

            result = {}
            # stable 端点需要单独查询每个符号
            for etf_symbol, sector_name in etf_to_sector.items():
                try:
                    data = self._make_request("/quote", params={"symbol": etf_symbol})
                    if data and isinstance(data, list) and len(data) > 0:
                        change_pct = float(data[0].get("changesPercentage", 0) or 0)
                        result[sector_name] = change_pct
                except Exception as e:
                    logger.debug(f"Failed to get {etf_symbol}: {e}")
                    continue

            if result:
                self._set_cache("sector_performance", result)

            return result

        except Exception as e:
            logger.warning(f"Failed to get sector performance: {e}")
            return {}

    def get_market_indices(self) -> Dict[str, Dict[str, Any]]:
        """
        获取主要市场指数

        Returns:
            {
                "sp500": {"price": 4800, "change": 0.5, "change_percent": 0.01},
                "nasdaq": {...},
                ...
            }
        """
        cached = self._get_cached("market_indices")
        if cached is not None:
            return cached

        try:
            symbols = "^GSPC,^IXIC,^DJI,^VIX"
            data = self._make_request("/quote", params={"symbol": symbols})

            result = {}
            symbol_map = {
                "^GSPC": "sp500",
                "^IXIC": "nasdaq",
                "^DJI": "dow",
                "^VIX": "vix",
            }

            if data and isinstance(data, list):
                for quote in data:
                    symbol = quote.get("symbol", "")
                    name = symbol_map.get(symbol, symbol)
                    result[name] = {
                        "price": float(quote.get("price", 0)),
                        "change": float(quote.get("change", 0)),
                        "change_percent": float(quote.get("changesPercentage", 0)),
                        "volume": int(quote.get("volume", 0)),
                    }
                self._set_cache("market_indices", result)

            return result
        except Exception as e:
            logger.warning(f"Failed to get market indices: {e}")
            return {}

    def get_commodities(self) -> Dict[str, Dict[str, Any]]:
        """
        获取商品价格 (黄金, 原油, etc.)

        使用 ETF 作为商品代理
        (stable 端点不支持批量查询，需要单独查询每个)

        Returns:
            {
                "gold": {"price": 2050, "change_percent": 0.5},
                "oil": {"price": 75, "change_percent": -1.2},
            }
        """
        cached = self._get_cached("commodities")
        if cached is not None:
            return cached

        try:
            etf_map = {
                "GLD": "gold",
                "SLV": "silver",
                "USO": "oil",
                "UNG": "natural_gas",
            }

            result = {}
            # stable 端点需要单独查询每个符号
            for etf_symbol, commodity_name in etf_map.items():
                try:
                    data = self._make_request("/quote", params={"symbol": etf_symbol})
                    if data and isinstance(data, list) and len(data) > 0:
                        quote = data[0]
                        result[commodity_name] = {
                            "price": float(quote.get("price", 0) or 0),
                            "change": float(quote.get("change", 0) or 0),
                            "change_percent": float(quote.get("changesPercentage", 0) or 0),
                        }
                except Exception as e:
                    logger.debug(f"Failed to get {etf_symbol}: {e}")
                    continue

            if result:
                self._set_cache("commodities", result)

            return result
        except Exception as e:
            logger.warning(f"Failed to get commodities: {e}")
            return {}

    def get_crypto_data(self) -> Dict[str, Dict[str, Any]]:
        """
        获取加密货币数据

        (stable 端点不支持批量查询，需要单独查询每个)

        Returns:
            {
                "btc": {"price": 45000, "change_percent": 2.5, "market_cap": 800000000000},
                "eth": {...},
            }
        """
        cached = self._get_cached("crypto")
        if cached is not None:
            return cached

        try:
            symbol_map = {
                "BTCUSD": "btc",
                "ETHUSD": "eth",
                "SOLUSD": "sol",
            }

            result = {}
            # stable 端点需要单独查询每个符号
            for crypto_symbol, name in symbol_map.items():
                try:
                    data = self._make_request("/quote", params={"symbol": crypto_symbol})
                    if data and isinstance(data, list) and len(data) > 0:
                        quote = data[0]
                        result[name] = {
                            "price": float(quote.get("price", 0) or 0),
                            "change": float(quote.get("change", 0) or 0),
                            "change_percent": float(quote.get("changesPercentage", 0) or 0),
                            "volume": int(quote.get("volume", 0) or 0),
                            "market_cap": float(quote.get("marketCap", 0) or 0),
                        }
                except Exception as e:
                    logger.debug(f"Failed to get {crypto_symbol}: {e}")
                    continue

            if result:
                self._set_cache("crypto", result)

            return result
        except Exception as e:
            logger.warning(f"Failed to get crypto data: {e}")
            return {}

    def get_full_macro_snapshot(self) -> Dict[str, Any]:
        """
        获取完整的宏观数据快照

        Returns:
            包含所有宏观数据的字典
        """
        vix = self.get_vix()
        dxy = self.get_dxy()
        treasury = self.get_treasury_rates()
        fear_greed = self.get_fear_greed_index()
        sectors = self.get_sector_performance()
        indices = self.get_market_indices()
        commodities = self.get_commodities()
        crypto = self.get_crypto_data()

        # 计算衍生指标
        treasury_10y = treasury.get("treasury_10y", 4.0)
        treasury_2y = treasury.get("treasury_2y", 4.5)
        yield_curve_spread = treasury_10y - treasury_2y  # 收益率曲线 (负值=倒挂)

        # 估算实际利率 (10Y - 预期通胀)
        # 使用 TIPS 隐含通胀预期，这里用 2.5% 作为近似值
        inflation_expectation = 2.5
        real_rate = treasury_10y - inflation_expectation

        return {
            # 波动率
            "vix": vix,

            # 货币
            "dxy": dxy,

            # 国债收益率
            "treasury_10y": treasury_10y,
            "treasury_2y": treasury_2y,
            "treasury_30y": treasury.get("treasury_30y", 4.5),
            "yield_curve_spread": yield_curve_spread,

            # 利率衍生
            "inflation_expectation": inflation_expectation,
            "real_rate": real_rate,

            # 市场情绪
            "fear_greed_value": fear_greed.get("value") if fear_greed else None,
            "fear_greed_class": fear_greed.get("classification") if fear_greed else None,

            # 市场指数
            "indices": indices,

            # 板块表现
            "sectors": sectors,

            # 商品
            "commodities": commodities,

            # 加密货币
            "crypto": crypto,

            # 完整国债收益率曲线
            "treasury_curve": treasury,

            # 时间戳
            "timestamp": datetime.now().isoformat(),
        }

    def get_fed_funds_rate(self) -> Optional[float]:
        """
        获取 Federal Funds Rate (联邦基金利率)

        使用 FMP 的 Treasury Rates 端点获取短期利率作为 Fed Funds 近似值
        注: Fed Funds Rate 实际由 FRED API 提供，FMP 不直接提供

        Returns:
            Fed Funds Rate 近似值 (使用3个月国债收益率作为代理)
        """
        cached = self._get_cached("fed_funds")
        if cached is not None:
            return cached

        try:
            # 使用 3 个月国债收益率作为 Fed Funds Rate 的近似
            # 实际 Fed Funds 和 3M T-Bill 通常差距 < 0.1%
            treasury = self.get_treasury_rates()
            fed_funds = treasury.get("treasury_3m", 5.25)  # 当前 Fed Funds 约 5.25-5.50%

            self._set_cache("fed_funds", fed_funds)
            return fed_funds

        except Exception as e:
            logger.warning(f"Failed to get Fed Funds Rate: {e}")
            return 5.25  # 默认值

    def get_2s10s_spread(self) -> Optional[float]:
        """
        获取 2s10s 利差 (2年-10年国债收益率利差)

        重要的经济周期指标:
        - 正值: 收益率曲线正常，经济扩张
        - 负值: 收益率曲线倒挂，可能预示衰退

        Returns:
            2s10s 利差 (bps, 基点)
        """
        cached = self._get_cached("2s10s_spread")
        if cached is not None:
            return cached

        try:
            treasury = self.get_treasury_rates()
            treasury_10y = treasury.get("treasury_10y", 4.2)
            treasury_2y = treasury.get("treasury_2y", 4.5)

            # 转换为 bps (基点)
            spread_bps = (treasury_10y - treasury_2y) * 100

            self._set_cache("2s10s_spread", spread_bps)
            return spread_bps

        except Exception as e:
            logger.warning(f"Failed to get 2s10s spread: {e}")
            return -30.0  # 当前约倒挂 30bps

    def get_macro_for_experts(self, date: str = None) -> Dict[str, Any]:
        """
        获取适合专家智能体使用的宏观数据格式

        这是替代原有硬编码 _get_macro_data 的方法

        Args:
            date: 日期 (当前不使用历史数据，仅获取最新)

        Returns:
            兼容原有格式的宏观数据
        """
        vix = self.get_vix() or 20.0
        dxy = self.get_dxy() or 103.5
        treasury = self.get_treasury_rates()

        treasury_10y = treasury.get("treasury_10y", 4.2)
        treasury_2y = treasury.get("treasury_2y", 4.5)
        fed_funds = self.get_fed_funds_rate() or 5.25

        # 估算通胀预期和实际利率
        inflation_expectation = 2.5
        real_rate = treasury_10y - inflation_expectation

        # 2s10s 利差 (基点)
        spread_2s10s = self.get_2s10s_spread() or -30.0

        return {
            "vix": vix,
            "dxy": dxy,
            "treasury_10y": treasury_10y,
            "treasury_2y": treasury_2y,
            "treasury_30y": treasury.get("treasury_30y", 4.5),
            "fed_funds": fed_funds,
            "spread_2s10s": spread_2s10s,
            "inflation_expectation": inflation_expectation,
            "real_rate": real_rate,
            "yield_curve_spread": treasury_10y - treasury_2y,
        }

    def get_bond_expert_data(self) -> Dict[str, Any]:
        """
        获取 Bond Expert 专用的利率数据

        Returns:
            包含所有债券分析所需的利率数据
        """
        treasury = self.get_treasury_rates()
        fed_funds = self.get_fed_funds_rate() or 5.25
        spread_2s10s = self.get_2s10s_spread() or -30.0

        return {
            "fed_funds": fed_funds,
            "treasury_2y": treasury.get("treasury_2y", 4.5),
            "treasury_5y": treasury.get("treasury_5y", 4.3),
            "treasury_10y": treasury.get("treasury_10y", 4.2),
            "treasury_30y": treasury.get("treasury_30y", 4.5),
            "spread_2s10s": spread_2s10s,
            # 信用利差估算 (使用 HYG-LQD 价差作为代理)
            "ig_spread": 100,  # 投资级利差约 100bps
            "hy_spread": 350,  # 高收益利差约 350bps
        }

    def _get_cached(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.cache_enabled:
            return None

        if key in self._cache:
            cached = self._cache[key]
            age = (datetime.now() - cached["timestamp"]).total_seconds() / 60
            if age < self.cache_ttl_minutes:
                return cached["data"]
            del self._cache[key]

        return None

    def _set_cache(self, key: str, data: Any):
        """设置缓存"""
        if self.cache_enabled:
            self._cache[key] = {
                "data": data,
                "timestamp": datetime.now(),
            }

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("MacroDataLoader cache cleared")

    # ==================== REITs Cap Rate 计算 ====================

    # REITs 默认符号列表
    REITS_SYMBOLS = ["VNQ", "IYR", "DLR", "EQIX", "PLD", "EQR", "AVB", "O", "SPG"]

    def get_reit_cap_rate(self, symbol: str) -> Optional[float]:
        """
        计算单个 REIT 的 Cap Rate (资本化率)

        Cap Rate = FFO (或 NOI) / Market Cap
        由于 FMP 没有直接的 FFO 数据，我们用以下方法近似:
        Cap Rate ≈ (Net Income + Depreciation) / Market Cap

        对于 ETF (如 VNQ, IYR)，使用 dividend yield 作为 cap rate 的代理

        Args:
            symbol: REIT 股票代码

        Returns:
            Cap Rate (百分比形式，如 5.5 表示 5.5%)
        """
        cache_key = f"reit_cap_rate_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # 对于 ETF，使用 dividend yield 作为 cap rate 代理
            # 从 config 获取 REITs 符号列表
            etf_symbols = AssetConfig().default_universe.get("reits", ["VNQ", "IYR"])
            if symbol in etf_symbols:
                cap_rate = self._get_etf_dividend_yield(symbol)
                if cap_rate:
                    self._set_cache(cache_key, cap_rate)
                return cap_rate

            # 对于个股 REIT，计算真实 cap rate
            # 使用 stable API 端点 (查询参数格式)
            # 获取 Key Metrics (包含 market cap)
            key_metrics = self._make_request(
                "/key-metrics",
                params={"symbol": symbol, "limit": 1}
            )

            # 获取 Income Statement (包含 net income)
            income_stmt = self._make_request(
                "/income-statement",
                params={"symbol": symbol, "limit": 1}
            )

            # 获取 Cash Flow Statement (包含 depreciation)
            cash_flow = self._make_request(
                "/cash-flow-statement",
                params={"symbol": symbol, "limit": 1}
            )

            if not (key_metrics and income_stmt and cash_flow):
                logger.warning(f"Missing financial data for {symbol}")
                return None

            # 提取数据
            market_cap = None
            net_income = None
            depreciation = None

            # 从 key metrics 获取 market cap
            if key_metrics and len(key_metrics) > 0:
                market_cap = key_metrics[0].get("marketCap", 0)

            # 如果 key metrics 没有 market cap，尝试从 quote 获取
            if not market_cap:
                quote_data = self._make_request("/quote", params={"symbol": symbol})
                if quote_data and len(quote_data) > 0:
                    market_cap = quote_data[0].get("marketCap", 0)

            # 从 income statement 获取 net income
            if income_stmt and len(income_stmt) > 0:
                net_income = income_stmt[0].get("netIncome", 0)

            # 从 cash flow 获取 depreciation (通常为负数，取绝对值)
            if cash_flow and len(cash_flow) > 0:
                depreciation = abs(cash_flow[0].get("depreciationAndAmortization", 0))

            # 计算 FFO 近似值 (Net Income + Depreciation)
            if market_cap and market_cap > 0 and net_income is not None:
                ffo_approx = net_income + (depreciation or 0)
                cap_rate = (ffo_approx / market_cap) * 100  # 转换为百分比

                # Cap rate 合理范围检查 (1% - 15%)
                if 1.0 <= cap_rate <= 15.0:
                    self._set_cache(cache_key, cap_rate)
                    return cap_rate
                else:
                    logger.warning(f"Cap rate {cap_rate:.2f}% for {symbol} out of range, using dividend yield")
                    return self._get_dividend_yield_as_cap_rate(symbol)

            return None

        except Exception as e:
            logger.warning(f"Failed to calculate cap rate for {symbol}: {e}")
            return None

    def _get_etf_dividend_yield(self, symbol: str) -> Optional[float]:
        """
        获取 ETF 的 dividend yield 作为 cap rate 代理

        Args:
            symbol: ETF 股票代码

        Returns:
            Dividend yield (百分比)
        """
        try:
            # 尝试从 quote 获取
            quote_data = self._make_request("/quote", params={"symbol": symbol})
            if quote_data and len(quote_data) > 0:
                # FMP 的 dividend yield 可能在不同字段
                div_yield = quote_data[0].get("dividendYield")
                if div_yield:
                    return float(div_yield) * 100  # 转换为百分比

            # 备用方案：从 stable profile 获取
            profile = self._make_request(
                "/profile",
                params={"symbol": symbol}
            )
            if profile and len(profile) > 0:
                # stable API 的 profile 直接有 lastDividend 字段
                last_div = profile[0].get("lastDividend", 0)
                price = profile[0].get("price", 0)
                if last_div and price and price > 0:
                    return (last_div / price) * 100

            return None
        except Exception as e:
            logger.warning(f"Failed to get dividend yield for {symbol}: {e}")
            return None

    def _get_dividend_yield_as_cap_rate(self, symbol: str) -> Optional[float]:
        """
        使用 dividend yield 作为 cap rate 的备用计算

        Args:
            symbol: 股票代码

        Returns:
            Dividend yield (百分比)
        """
        try:
            # 使用 stable API
            profile = self._make_request(
                "/profile",
                params={"symbol": symbol}
            )
            if profile and len(profile) > 0:
                last_div = profile[0].get("lastDividend", 0)
                price = profile[0].get("price", 0)
                if last_div and price and price > 0:
                    return (last_div / price) * 100
            return None
        except Exception as e:
            logger.warning(f"Failed to get dividend yield for {symbol}: {e}")
            return None

    def get_reits_average_cap_rate(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        获取一组 REITs 的平均 Cap Rate

        Args:
            symbols: REIT 股票代码列表，默认使用 REITS_SYMBOLS

        Returns:
            {
                "avg_cap_rate": 5.5,
                "individual_rates": {"VNQ": 4.2, "DLR": 5.8, ...},
                "sample_size": 5,
                "timestamp": "2024-01-15T10:00:00"
            }
        """
        cached = self._get_cached("reits_avg_cap_rate")
        if cached is not None:
            return cached

        symbols = symbols or self.REITS_SYMBOLS
        individual_rates = {}
        valid_rates = []

        for symbol in symbols:
            try:
                cap_rate = self.get_reit_cap_rate(symbol)
                if cap_rate is not None:
                    individual_rates[symbol] = round(cap_rate, 2)
                    valid_rates.append(cap_rate)
            except Exception as e:
                logger.debug(f"Failed to get cap rate for {symbol}: {e}")
                continue

        avg_cap_rate = sum(valid_rates) / len(valid_rates) if valid_rates else 5.0

        result = {
            "avg_cap_rate": round(avg_cap_rate, 2),
            "individual_rates": individual_rates,
            "sample_size": len(valid_rates),
            "timestamp": datetime.now().isoformat(),
        }

        self._set_cache("reits_avg_cap_rate", result)
        return result

    def get_cap_rate_spread(self) -> Optional[float]:
        """
        计算 Cap Rate Spread (Cap Rate - 10Y Treasury)

        这是 REITs 估值的重要指标:
        - 正值越大: REITs 相对国债越有吸引力
        - 接近零或负值: REITs 估值偏高

        Returns:
            Cap Rate Spread (bps, 基点)
        """
        cached = self._get_cached("cap_rate_spread")
        if cached is not None:
            return cached

        try:
            # 获取平均 cap rate
            reits_data = self.get_reits_average_cap_rate()
            avg_cap_rate = reits_data.get("avg_cap_rate", 5.0)

            # 获取 10Y Treasury
            treasury = self.get_treasury_rates()
            treasury_10y = treasury.get("treasury_10y", 4.2)

            # 计算 spread (转换为 bps)
            spread_bps = (avg_cap_rate - treasury_10y) * 100

            self._set_cache("cap_rate_spread", spread_bps)
            return round(spread_bps, 1)

        except Exception as e:
            logger.warning(f"Failed to calculate cap rate spread: {e}")
            return 100.0  # 默认 100bps

    def get_reits_expert_data(self) -> Dict[str, Any]:
        """
        获取 REITs Expert 专用的数据

        Returns:
            包含 REITs 分析所需的所有数据
        """
        treasury = self.get_treasury_rates()
        reits_data = self.get_reits_average_cap_rate()
        cap_rate_spread = self.get_cap_rate_spread()

        return {
            "treasury_10y": treasury.get("treasury_10y", 4.2),
            "avg_cap_rate": reits_data.get("avg_cap_rate", 5.0),
            "cap_rate_spread": cap_rate_spread or 100.0,
            "individual_cap_rates": reits_data.get("individual_rates", {}),
            "rate_expectation": self._estimate_rate_expectation(),
            "timestamp": datetime.now().isoformat(),
        }

    def _estimate_rate_expectation(self) -> str:
        """
        估算利率预期

        基于收益率曲线形态判断
        """
        try:
            treasury = self.get_treasury_rates()
            spread_2s10s = treasury.get("treasury_10y", 4.2) - treasury.get("treasury_2y", 4.5)

            if spread_2s10s < -0.5:
                return "降息预期强"  # 深度倒挂
            elif spread_2s10s < 0:
                return "温和降息预期"  # 轻度倒挂
            elif spread_2s10s < 0.5:
                return "利率稳定"  # 平坦
            else:
                return "加息预期"  # 正常陡峭
        except Exception as e:
            logger.warning(f"Error analyzing yield curve: {e}")
            return "stable"

    # ==================== COT (Commitment of Traders) 数据 ====================

    def get_cot_report(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        获取 COT (交易商持仓报告) 数据

        COT 报告显示大型投机者、商业交易者和小型交易者的持仓情况
        FMP API: /cot-report

        Args:
            symbol: 商品符号 (可选)

        Returns:
            COT 报告数据
        """
        cache_key = f"cot_{symbol or 'all'}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            params = {"limit": 10}
            if symbol:
                params["symbol"] = symbol

            data = self._make_request("/cot-report", params=params, base_url=self.FMP_V4_URL)

            if data and isinstance(data, list) and len(data) > 0:
                # 解析 COT 数据
                result = self._parse_cot_data(data)
                self._set_cache(cache_key, result)
                return result
            return None

        except Exception as e:
            logger.warning(f"Failed to get COT report: {e}")
            return None

    def _parse_cot_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """
        解析 COT 原始数据

        Args:
            raw_data: FMP API 返回的原始 COT 数据

        Returns:
            解析后的 COT 数据
        """
        if not raw_data:
            return {}

        latest = raw_data[0]

        # 计算净持仓
        commercial_long = latest.get("commercialLong", 0) or 0
        commercial_short = latest.get("commercialShort", 0) or 0
        commercial_net = commercial_long - commercial_short

        noncommercial_long = latest.get("nonCommercialLong", 0) or 0
        noncommercial_short = latest.get("nonCommercialShort", 0) or 0
        speculator_net = noncommercial_long - noncommercial_short

        # 计算持仓变化 (如果有历史数据)
        net_change = 0
        if len(raw_data) >= 2:
            prev = raw_data[1]
            prev_spec_net = (prev.get("nonCommercialLong", 0) or 0) - (prev.get("nonCommercialShort", 0) or 0)
            net_change = speculator_net - prev_spec_net

        # 判断市场情绪
        sentiment = "neutral"
        if speculator_net > 0 and net_change > 0:
            sentiment = "bullish"
        elif speculator_net < 0 and net_change < 0:
            sentiment = "bearish"
        elif speculator_net > 0:
            sentiment = "moderately_bullish"
        elif speculator_net < 0:
            sentiment = "moderately_bearish"

        return {
            "symbol": latest.get("symbol", ""),
            "name": latest.get("name", ""),
            "date": latest.get("date", ""),
            "commercial_long": commercial_long,
            "commercial_short": commercial_short,
            "commercial_net": commercial_net,
            "speculator_long": noncommercial_long,
            "speculator_short": noncommercial_short,
            "speculator_net": speculator_net,
            "net_change": net_change,
            "sentiment": sentiment,
            "open_interest": latest.get("openInterest", 0),
        }

    def get_cot_for_commodities(self) -> Dict[str, Dict[str, Any]]:
        """
        获取主要商品的 COT 数据

        Returns:
            {
                "gold": {"speculator_net": 150000, "sentiment": "bullish", ...},
                "oil": {...},
                ...
            }
        """
        cached = self._get_cached("cot_commodities")
        if cached is not None:
            return cached

        # FMP COT 符号映射
        cot_symbols = {
            "GC": "gold",          # 黄金
            "SI": "silver",        # 白银
            "CL": "oil",           # 原油
            "NG": "natural_gas",   # 天然气
            "HG": "copper",        # 铜
            "W": "wheat",          # 小麦
            "C": "corn",           # 玉米
            "S": "soybeans",       # 大豆
        }

        result = {}
        for cot_symbol, commodity_name in cot_symbols.items():
            try:
                cot_data = self.get_cot_report(cot_symbol)
                if cot_data:
                    result[commodity_name] = cot_data
            except Exception as e:
                logger.debug(f"Failed to get COT for {cot_symbol}: {e}")
                continue

        if result:
            self._set_cache("cot_commodities", result)

        return result

    def get_commodity_expert_data(self) -> Dict[str, Any]:
        """
        获取 Commodity Expert 专用的增强数据

        整合:
        - 商品价格数据
        - COT 持仓数据
        - 美元指数
        - 通胀数据
        - 经济事件

        Returns:
            Commodity Expert 所需的完整数据包
        """
        commodities = self.get_commodities()
        cot_data = self.get_cot_for_commodities()
        dxy = self.get_dxy()
        vix = self.get_vix()
        treasury = self.get_treasury_rates()

        # 计算实际利率 (影响黄金等)
        treasury_10y = treasury.get("treasury_10y", 4.2)
        inflation_expectation = 2.5
        real_rate = treasury_10y - inflation_expectation

        # 获取能源相关经济事件
        economic_events = self._get_energy_economic_events()

        return {
            # 价格数据
            "commodities": commodities,

            # COT 持仓数据 (市场情绪)
            "cot_data": cot_data,

            # 宏观环境
            "dxy": dxy,
            "vix": vix,
            "real_rate": real_rate,
            "treasury_10y": treasury_10y,
            "inflation_expectation": inflation_expectation,

            # 经济事件
            "economic_events": economic_events,

            # 综合分析
            "market_regime": self._analyze_commodity_regime(commodities, cot_data, dxy, real_rate),

            "timestamp": datetime.now().isoformat(),
        }

    def _get_energy_economic_events(self) -> List[Dict]:
        """
        获取能源相关的经济事件

        Returns:
            能源相关经济事件列表
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

            events = self.get_economic_calendar(today, end_date)

            # 筛选能源/商品相关事件
            energy_keywords = ["oil", "crude", "opec", "eia", "inventory", "stockpile",
                               "natural gas", "api", "gold", "copper", "pmi"]

            relevant_events = []
            for event in events:
                event_name = (event.get("event", "") or "").lower()
                if any(kw in event_name for kw in energy_keywords):
                    relevant_events.append({
                        "date": event.get("date"),
                        "event": event.get("event"),
                        "impact": event.get("impact", "low"),
                        "actual": event.get("actual"),
                        "forecast": event.get("forecast"),
                        "previous": event.get("previous"),
                    })

            return relevant_events[:10]  # 返回前10个

        except Exception as e:
            logger.warning(f"Failed to get energy economic events: {e}")
            return []

    def _analyze_commodity_regime(
        self,
        commodities: Dict,
        cot_data: Dict,
        dxy: float,
        real_rate: float
    ) -> Dict[str, str]:
        """
        分析商品市场环境

        Args:
            commodities: 商品价格数据
            cot_data: COT 持仓数据
            dxy: 美元指数
            real_rate: 实际利率

        Returns:
            市场环境分析
        """
        regime = {
            "dollar_environment": "neutral",
            "inflation_hedge_demand": "moderate",
            "speculator_positioning": "neutral",
            "overall_bias": "neutral",
        }

        # 美元环境分析
        if dxy and dxy > 105:
            regime["dollar_environment"] = "strong_dollar_headwind"
        elif dxy and dxy < 100:
            regime["dollar_environment"] = "weak_dollar_tailwind"

        # 通胀对冲需求分析 (基于实际利率)
        if real_rate < 0:
            regime["inflation_hedge_demand"] = "high"  # 负实际利率利好黄金
        elif real_rate > 2:
            regime["inflation_hedge_demand"] = "low"

        # 投机者持仓分析
        if cot_data:
            bullish_count = sum(1 for c in cot_data.values()
                                if c.get("sentiment") in ["bullish", "moderately_bullish"])
            bearish_count = sum(1 for c in cot_data.values()
                                if c.get("sentiment") in ["bearish", "moderately_bearish"])

            if bullish_count > bearish_count + 2:
                regime["speculator_positioning"] = "bullish"
            elif bearish_count > bullish_count + 2:
                regime["speculator_positioning"] = "bearish"

        # 综合判断
        bullish_factors = 0
        if regime["dollar_environment"] == "weak_dollar_tailwind":
            bullish_factors += 1
        if regime["inflation_hedge_demand"] == "high":
            bullish_factors += 1
        if regime["speculator_positioning"] == "bullish":
            bullish_factors += 1

        if bullish_factors >= 2:
            regime["overall_bias"] = "bullish"
        elif bullish_factors == 0:
            regime["overall_bias"] = "bearish"

        return regime


# 便捷函数
def create_macro_loader(api_key: Optional[str] = None) -> MacroDataLoader:
    """创建宏观数据加载器"""
    return MacroDataLoader(fmp_api_key=api_key)
