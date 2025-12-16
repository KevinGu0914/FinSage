"""
Dynamic Asset Screener
动态资产筛选器 - 基于学术因子模型从全市场筛选候选标的

支持资产类别:
- Stocks: 基于 Fama-French 五因子 + 动量
- Bonds: 基于 Carry, Duration, Value, Low-Risk
- Commodities: 基于 Term Structure, Momentum, Carry
- REITs: 基于 NAV, Sector, Idiosyncratic Risk
- Crypto: 基于 Network, Adoption, Momentum, Crash Risk
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

from finsage.data.fmp_client import FMPClient

logger = logging.getLogger(__name__)


@dataclass
class ScreenedAsset:
    """筛选出的资产"""
    symbol: str
    asset_class: str
    factor_score: float  # 综合因子评分 [0, 1]
    signal: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    top_factors: List[Tuple[str, float]]  # 最强因子
    expected_alpha: float  # 预期Alpha


class DynamicAssetScreener:
    """
    动态资产筛选器

    基于学术因子模型对各类资产进行评分和筛选。

    学术基础:
    - Stocks: Fama-French (2015) Five-Factor Model
    - Bonds: Ilmanen (2011) Expected Returns
    - Commodities: Miffre & Rallis (2007), Szymanowska (2014)
    - REITs: Sagi (2021), Giacoletti (2021)
    - Crypto: Biais et al. (2023), Liu & Tsyvinski (2021)
    """

    # 各资产类别的候选池 (用于因子筛选)
    CANDIDATE_POOLS = {
        "stocks": {
            # 科技
            "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
                     "ADBE", "ORCL", "CSCO", "AVGO", "TXN"],
            # 金融
            "financial": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V", "MA"],
            # 医疗
            "healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY"],
            # 消费
            "consumer": ["WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX"],
            # 工业
            "industrial": ["CAT", "BA", "HON", "UPS", "UNP", "RTX", "DE", "GE", "LMT"],
            # 能源
            "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
            # 防御
            "defensive": ["PG", "KO", "PEP", "CL", "KMB", "GIS", "K", "HSY"],
            # ETF
            "etf": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLK", "XLF", "XLV", "XLE"],
        },
        "bonds": {
            # 国债 ETF
            "treasuries": ["TLT", "IEF", "SHY", "GOVT", "VGSH", "VGIT", "VGLT", "SCHO", "SCHR"],
            # 投资级公司债
            "investment_grade": ["LQD", "VCIT", "VCSH", "IGIB", "IGSB", "USIG"],
            # 高收益债
            "high_yield": ["HYG", "JNK", "SHYG", "USHY", "HYLB"],
            # 综合债券
            "aggregate": ["AGG", "BND", "SCHZ", "FBND"],
            # 通胀保护
            "tips": ["TIP", "VTIP", "SCHP", "STIP"],
            # 市政债
            "muni": ["MUB", "VTEB", "TFI"],
        },
        "commodities": {
            # 贵金属
            "precious_metals": ["GLD", "SLV", "IAU", "SGOL", "GLDM", "PALL", "PPLT"],
            # 能源
            "energy": ["USO", "UNG", "BNO", "UGA", "BOIL"],
            # 工业金属
            "industrial_metals": ["COPX", "JJC", "DBB"],
            # 农产品
            "agriculture": ["DBA", "CORN", "WEAT", "SOYB", "COW"],
            # 综合商品
            "broad": ["DBC", "PDBC", "GSG", "COMT", "DJP"],
        },
        "reits": {
            # 综合 REITs ETF
            "diversified": ["VNQ", "IYR", "SCHH", "XLRE", "RWR", "USRT"],
            # 数据中心
            "data_center": ["DLR", "EQIX", "COR"],
            # 物流仓储
            "logistics": ["PLD", "STAG", "REXR", "EGP"],
            # 住宅
            "residential": ["EQR", "AVB", "ESS", "MAA", "UDR", "CPT"],
            # 医疗
            "healthcare": ["WELL", "VTR", "PEAK", "OHI", "HR"],
            # 零售
            "retail": ["SPG", "O", "NNN", "STOR", "REG", "KIM"],
            # 办公
            "office": ["BXP", "ARE", "KRC", "HIW"],
            # 通信基础设施
            "tower": ["AMT", "CCI", "SBAC"],
        },
        "crypto": {
            # 主流币 (使用 Yahoo Finance 符号)
            "major": ["BTC-USD", "ETH-USD"],
            # Layer 1
            "layer1": ["SOL-USD", "AVAX-USD", "ADA-USD", "DOT-USD", "ATOM-USD"],
            # Crypto ETF
            "etf": ["BITO", "GBTC", "ETHE", "ARKB", "IBIT", "FBTC"],
        },
    }

    # 各资产类别的筛选配置
    SCREENING_CONFIGS = {
        "stocks": {
            "max_assets": 25,
            "min_score": 0.45,  # 最低因子评分
            "core_etfs": ["SPY", "QQQ"],  # 始终包含
        },
        "bonds": {
            "max_assets": 12,
            "min_score": 0.40,
            "core_etfs": ["AGG", "TLT"],
        },
        "commodities": {
            "max_assets": 10,
            "min_score": 0.30,  # 降低阈值，商品因子分数通常较低
            "core_etfs": ["GLD"],
        },
        "reits": {
            "max_assets": 10,
            "min_score": 0.40,
            "core_etfs": ["VNQ"],
        },
        "crypto": {
            "max_assets": 6,
            "min_score": 0.35,
            "core_etfs": ["BTC-USD", "ETH-USD"],
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_hours: int = 168,
        data_provider: Optional[Any] = None,
    ):
        """
        初始化筛选器

        Args:
            api_key: FMP API密钥 (用于获取基本面数据)
            cache_hours: 缓存有效期（小时），默认168小时=1周
            data_provider: 数据提供者 (用于获取市场数据)
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self.base_url = FMPClient.BASE_URL
        self.cache_hours = cache_hours
        self.data_provider = data_provider

        # 缓存
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}

        # 因子评分器 (延迟初始化)
        self._scorers: Dict[str, Any] = {}

        if not self.api_key:
            logger.warning("FMP API key not found, using simplified factor scoring")

    def _get_scorer(self, asset_class: str):
        """获取因子评分器 (延迟初始化)"""
        if asset_class not in self._scorers:
            try:
                if asset_class == "stocks":
                    from finsage.factors.stock_factors import StockFactorScorer
                    self._scorers[asset_class] = StockFactorScorer()
                elif asset_class == "bonds":
                    from finsage.factors.bond_factors import BondFactorScorer
                    self._scorers[asset_class] = BondFactorScorer()
                elif asset_class == "commodities":
                    from finsage.factors.commodity_factors import CommodityFactorScorer
                    self._scorers[asset_class] = CommodityFactorScorer()
                elif asset_class == "reits":
                    from finsage.factors.reits_factors import REITsFactorScorer
                    self._scorers[asset_class] = REITsFactorScorer()
                elif asset_class == "crypto":
                    from finsage.factors.crypto_factors import CryptoFactorScorer
                    self._scorers[asset_class] = CryptoFactorScorer()
                else:
                    logger.warning(f"No scorer for {asset_class}")
                    return None
            except ImportError as e:
                logger.warning(f"Failed to import scorer for {asset_class}: {e}")
                return None

        return self._scorers.get(asset_class)

    def get_dynamic_universe(
        self,
        asset_class: str,
        date: Optional[str] = None,
        force_refresh: bool = False,
        market_data: Optional[Dict[str, Any]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        market_regime: Optional[str] = None,
    ) -> List[str]:
        """
        获取动态资产池 (基于因子评分)

        Args:
            asset_class: 资产类别 (stocks, bonds, commodities, reits, crypto)
            date: 日期（用于回测）
            force_refresh: 强制刷新缓存
            market_data: 市场数据字典 {symbol: {price, volume, ...}}
            returns_df: 收益率 DataFrame
            market_regime: 市场体制 ("bull", "bear", "neutral")

        Returns:
            符号列表 (按因子评分排序)
        """
        # 包含所有影响结果的参数，避免缓存键冲突
        regime_str = market_regime or 'default'
        cache_key = f"{asset_class}_{date or 'current'}_{regime_str}"

        # 检查缓存
        if not force_refresh and self._is_cache_valid(cache_key):
            logger.debug(f"Using cached universe for {asset_class} (regime={regime_str})")
            return self._cache[cache_key]

        # 基于因子评分筛选
        try:
            symbols = self._screen_by_factors(
                asset_class=asset_class,
                market_data=market_data,
                returns_df=returns_df,
                market_regime=market_regime,
            )
            self._update_cache(cache_key, symbols)
            return symbols
        except Exception as e:
            logger.warning(f"Factor screening failed for {asset_class}: {e}, using fallback")
            return self._get_fallback_symbols(asset_class)

    def _screen_by_factors(
        self,
        asset_class: str,
        market_data: Optional[Dict[str, Any]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        market_regime: Optional[str] = None,
    ) -> List[str]:
        """
        基于因子评分筛选资产

        Args:
            asset_class: 资产类别
            market_data: 市场数据
            returns_df: 收益率数据
            market_regime: 市场体制

        Returns:
            按因子评分排序的符号列表
        """
        config = self.SCREENING_CONFIGS.get(asset_class, {})
        max_assets = config.get("max_assets", 15)
        min_score = config.get("min_score", 0.4)
        core_etfs = config.get("core_etfs", [])

        # 获取候选池
        candidates = self._get_candidate_symbols(asset_class)

        # 获取因子评分器
        scorer = self._get_scorer(asset_class)

        if scorer is None:
            logger.warning(f"No scorer available for {asset_class}, using fallback")
            return self._get_fallback_symbols(asset_class)

        # 评分
        scored_assets: List[Tuple[str, float, str]] = []

        for symbol in candidates:
            try:
                # 准备数据
                symbol_data = self._prepare_factor_data(
                    symbol=symbol,
                    asset_class=asset_class,
                    market_data=market_data,
                )

                # 获取收益率序列
                symbol_returns = None
                if returns_df is not None and symbol in returns_df.columns:
                    symbol_returns = returns_df[symbol].dropna()

                # 计算因子评分
                score = scorer.score(
                    symbol=symbol,
                    data=symbol_data,
                    returns=symbol_returns,
                    market_regime=market_regime,
                )

                if score.composite_score >= min_score:
                    scored_assets.append((
                        symbol,
                        score.composite_score,
                        score.signal,
                    ))

            except Exception as e:
                logger.debug(f"Failed to score {symbol}: {e}")
                continue

        # 按评分排序
        scored_assets.sort(key=lambda x: -x[1])

        # 取前 N 个
        result = [s[0] for s in scored_assets[:max_assets]]

        # 确保核心 ETF 在列表中
        for etf in core_etfs:
            if etf not in result:
                result.append(etf)

        # 详细记录因子筛选结果
        logger.info(
            f"[Factor Screening] {asset_class.upper()}: "
            f"{len(candidates)} candidates -> {len(scored_assets)} passed min_score({min_score}) -> {len(result)} selected"
        )

        # 记录选中资产的详细信息
        if scored_assets:
            top_details = ", ".join([
                f"{s[0]}({s[1]:.3f}/{s[2]})"
                for s in scored_assets[:min(5, len(scored_assets))]
            ])
            logger.info(f"[Factor Screening] {asset_class.upper()} Top 5: {top_details}")

        return result if result else self._get_fallback_symbols(asset_class)

    def _get_candidate_symbols(self, asset_class: str) -> List[str]:
        """获取候选符号池"""
        pools = self.CANDIDATE_POOLS.get(asset_class, {})
        candidates = []
        for category, symbols in pools.items():
            candidates.extend(symbols)
        return list(set(candidates))  # 去重

    def _prepare_factor_data(
        self,
        symbol: str,
        asset_class: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        准备因子计算所需的数据

        如果提供了 market_data 则使用，否则使用默认值
        """
        # 基础数据
        data = {}

        # 从市场数据获取
        if market_data and symbol in market_data:
            md = market_data[symbol]
            data["price"] = md.get("close", md.get("price", 100))
            data["volume"] = md.get("volume", 0)
            data["price_change_1m"] = md.get("return_1m", 0)
            data["price_change_12m"] = md.get("return_12m", 0)

        # 根据资产类别补充默认数据
        if asset_class == "stocks":
            data.setdefault("price", 100)
            data.setdefault("market_cap", 50e9)  # 默认 500 亿
            data.setdefault("book_to_market", 0.4)
            data.setdefault("pe_ratio", 20)
            data.setdefault("pb_ratio", 3)
            data.setdefault("roe", 0.15)
            data.setdefault("operating_margin", 0.15)
            data.setdefault("gross_margin", 0.40)
            data.setdefault("asset_growth", 0.10)
            data.setdefault("beta", 1.0)
            data.setdefault("price_change_12m", 0.10)
            data.setdefault("price_change_1m", 0.02)

            # 尝试从 FMP 获取基本面数据
            if self.api_key:
                fundamental_data = self._fetch_fundamental_data(symbol)
                data.update(fundamental_data)

        elif asset_class == "bonds":
            data.setdefault("price", 100)
            data.setdefault("yield_to_maturity", 0.05)
            data.setdefault("duration", 7)
            data.setdefault("credit_spread", 0.01)
            data.setdefault("yield_curve_slope", 0.005)
            data.setdefault("yield_change_3m", -0.001)
            data.setdefault("return_3m", 0.02)

        elif asset_class == "commodities":
            data.setdefault("price", 100)
            data.setdefault("front_price", 100)
            data.setdefault("back_price", 102)  # 默认轻微 contango
            data.setdefault("spot_price", 99)
            data.setdefault("days_to_roll", 30)
            data.setdefault("price_change_12m", 0.05)
            data.setdefault("storage_cost", 0.02)
            data.setdefault("risk_free_rate", 0.04)

        elif asset_class == "reits":
            data.setdefault("price", 100)
            data.setdefault("nav", 95)  # 默认小幅折价
            data.setdefault("nav_premium", -0.05)
            data.setdefault("dividend_yield", 0.045)
            data.setdefault("p_ffo", 15)
            data.setdefault("cap_rate", 0.055)
            data.setdefault("risk_free_rate", 0.04)
            data.setdefault("sector", self._infer_reit_sector(symbol))
            data.setdefault("price_change_12m", 0.05)

        elif asset_class == "crypto":
            data.setdefault("price", 50000)
            data.setdefault("market_cap", 1e12)
            data.setdefault("active_addresses", 500000)
            data.setdefault("active_addresses_percentile", 60)
            data.setdefault("transaction_volume", 1e10)
            data.setdefault("developer_activity_percentile", 50)
            data.setdefault("new_addresses_growth_30d", 0.05)
            data.setdefault("whale_concentration", 0.4)
            data.setdefault("exchange_netflow_7d", 0)
            data.setdefault("institutional_holdings_pct", 0.05)
            data.setdefault("return_7d", 0.02)
            data.setdefault("return_30d", 0.05)
            data.setdefault("return_90d", 0.10)
            data.setdefault("volatility_30d", 0.60)
            data.setdefault("max_drawdown_90d", -0.25)
            data.setdefault("funding_rate", 0.0001)
            data.setdefault("open_interest_change_7d", 0.05)
            data.setdefault("liquidation_volume_24h", 1e8)

        return data

    def _fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """从 FMP API 获取基本面数据"""
        try:
            # 使用正确的 stable 端点: /stable/profile?symbol=XXX
            url = f"{self.base_url}/profile"
            params = {"symbol": symbol, "apikey": self.api_key}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data_list = response.json()
            if not data_list:
                return {}

            profile = data_list[0] if isinstance(data_list, list) else data_list
            return {
                "market_cap": profile.get("mktCap", 50e9),
                "beta": profile.get("beta", 1.0),
                "pe_ratio": profile.get("peRatio") or 20,
            }
        except Exception as e:
            logger.debug(f"Failed to fetch FMP data for {symbol}: {e}")
            return {}

    def _infer_reit_sector(self, symbol: str) -> str:
        """推断 REITs 的行业分类"""
        symbol_upper = symbol.upper()

        sector_mapping = {
            "data_center": ["DLR", "EQIX", "COR"],
            "logistics": ["PLD", "STAG", "REXR", "EGP"],
            "residential": ["EQR", "AVB", "ESS", "MAA", "UDR", "CPT"],
            "healthcare": ["WELL", "VTR", "PEAK", "OHI", "HR"],
            "retail": ["SPG", "O", "NNN", "STOR", "REG", "KIM"],
            "office": ["BXP", "ARE", "KRC", "HIW"],
            "tower": ["AMT", "CCI", "SBAC"],
            "diversified": ["VNQ", "IYR", "SCHH", "XLRE", "RWR", "USRT"],
        }

        for sector, symbols in sector_mapping.items():
            if symbol_upper in symbols:
                return sector

        return "diversified"

    def _get_fallback_symbols(self, asset_class: str) -> List[str]:
        """回退到静态符号列表"""
        fallbacks = {
            "stocks": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                "JPM", "BAC", "GS", "V", "MA",
                "UNH", "JNJ", "PFE", "ABBV",
                "WMT", "COST", "HD", "MCD",
                "CAT", "BA", "HON",
                "XOM", "CVX",
                "SPY", "QQQ", "IWM", "DIA", "VTI",
            ],
            "bonds": [
                "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND",
                "VCIT", "VCSH", "TIP", "GOVT",
            ],
            "commodities": [
                "GLD", "SLV", "USO", "UNG", "DBA", "COPX", "PDBC", "DBC",
            ],
            "reits": [
                "VNQ", "IYR", "SCHH", "DLR", "EQIX", "AMT", "PLD", "SPG",
            ],
            "crypto": [
                "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "BITO",
            ],
        }
        return fallbacks.get(asset_class, [])

    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache:
            return False
        if key not in self._cache_time:
            return False

        age = datetime.now() - self._cache_time[key]
        return age < timedelta(hours=self.cache_hours)

    def _update_cache(self, key: str, value: Any):
        """更新缓存"""
        self._cache[key] = value
        self._cache_time[key] = datetime.now()

    def get_sector_rotation_picks(
        self,
        market_regime: str = "neutral",
        market_data: Optional[Dict[str, Any]] = None,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[str]]:
        """
        根据市场状态获取行业轮动推荐 (基于因子)

        Args:
            market_regime: "risk_on", "risk_off", "neutral"
            market_data: 市场数据
            returns_df: 收益率数据

        Returns:
            各资产类别推荐标的
        """
        result = {}

        # 根据市场状态选择资产类别
        if market_regime == "risk_on":
            asset_classes = ["stocks", "commodities", "crypto"]
        elif market_regime == "risk_off":
            asset_classes = ["bonds", "commodities"]  # 黄金作为避险
        else:
            asset_classes = ["stocks", "bonds", "commodities"]

        for asset_class in asset_classes:
            symbols = self.get_dynamic_universe(
                asset_class=asset_class,
                market_data=market_data,
                returns_df=returns_df,
                market_regime=market_regime,
            )
            result[asset_class] = symbols[:10]  # 每类最多 10 个

        return result

    def refresh_all(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[str]]:
        """刷新所有资产类别的候选池"""
        result = {}
        for asset_class in self.SCREENING_CONFIGS.keys():
            result[asset_class] = self.get_dynamic_universe(
                asset_class=asset_class,
                force_refresh=True,
                market_data=market_data,
                returns_df=returns_df,
            )
        return result

    def get_factor_rankings(
        self,
        asset_class: str,
        market_data: Optional[Dict[str, Any]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        market_regime: Optional[str] = None,
        top_n: int = 20,
    ) -> List[ScreenedAsset]:
        """
        获取详细的因子排名

        Args:
            asset_class: 资产类别
            market_data: 市场数据
            returns_df: 收益率数据
            market_regime: 市场体制
            top_n: 返回前 N 个资产

        Returns:
            ScreenedAsset 列表
        """
        candidates = self._get_candidate_symbols(asset_class)
        scorer = self._get_scorer(asset_class)

        if scorer is None:
            return []

        results = []

        for symbol in candidates:
            try:
                symbol_data = self._prepare_factor_data(
                    symbol=symbol,
                    asset_class=asset_class,
                    market_data=market_data,
                )

                symbol_returns = None
                if returns_df is not None and symbol in returns_df.columns:
                    symbol_returns = returns_df[symbol].dropna()

                score = scorer.score(
                    symbol=symbol,
                    data=symbol_data,
                    returns=symbol_returns,
                    market_regime=market_regime,
                )

                # 获取最强因子
                top_factors = sorted(
                    [(k, v.exposure) for k, v in score.factor_exposures.items()],
                    key=lambda x: -x[1],
                )[:3]

                results.append(ScreenedAsset(
                    symbol=symbol,
                    asset_class=asset_class,
                    factor_score=score.composite_score,
                    signal=score.signal,
                    top_factors=top_factors,
                    expected_alpha=score.expected_alpha,
                ))

            except Exception as e:
                logger.debug(f"Failed to get ranking for {symbol}: {e}")
                continue

        # 排序
        results.sort(key=lambda x: -x.factor_score)

        return results[:top_n]


# 向后兼容的别名
DynamicStockScreener = DynamicAssetScreener
