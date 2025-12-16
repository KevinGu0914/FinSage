"""
Hedge Asset Universe
对冲资产全集 - 定义所有可用的对冲工具

扩展了原有的固定7个工具到70+个动态可选工具
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HedgeCategory(Enum):
    """对冲资产类别"""
    INVERSE_EQUITY = "inverse_equity"       # 反向股票ETF
    INVERSE_SECTOR = "inverse_sector"       # 反向行业ETF
    SECTOR_ETF = "sector_etf"               # 行业ETF (用于配对交易)
    VOLATILITY = "volatility"               # 波动率工具
    SAFE_HAVEN = "safe_haven"               # 避险资产
    FIXED_INCOME = "fixed_income"           # 固定收益
    COMMODITY = "commodity"                 # 商品
    INTERNATIONAL = "international"         # 国际市场
    CURRENCY = "currency"                   # 货币对冲


@dataclass
class HedgeAsset:
    """对冲资产定义"""
    symbol: str
    name: str
    category: HedgeCategory
    leverage: float = 1.0                   # 杠杆倍数 (负数表示反向)
    expense_ratio: float = 0.001            # 费率
    avg_daily_volume: float = 1e6           # 日均成交量 (估计值)
    typical_spread: float = 0.001           # 典型价差
    underlying: Optional[str] = None        # 底层资产
    sector: Optional[str] = None            # 行业
    tags: List[str] = field(default_factory=list)

    @property
    def is_inverse(self) -> bool:
        """是否为反向ETF"""
        return self.leverage < 0

    @property
    def is_leveraged(self) -> bool:
        """是否为杠杆ETF"""
        return abs(self.leverage) > 1

    @property
    def total_cost_estimate(self) -> float:
        """估算总持有成本 (年化)"""
        return self.expense_ratio + self.typical_spread * 252

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "category": self.category.value,
            "leverage": self.leverage,
            "expense_ratio": self.expense_ratio,
            "avg_daily_volume": self.avg_daily_volume,
            "underlying": self.underlying,
            "sector": self.sector,
            "tags": self.tags,
            "is_inverse": self.is_inverse,
            "total_cost": self.total_cost_estimate,
        }


class HedgeAssetUniverse:
    """
    对冲资产全集

    管理所有可用的对冲资产，按类别组织，支持多种查询方式。
    """

    def __init__(self):
        """初始化资产全集"""
        self._assets: Dict[str, HedgeAsset] = {}
        self._register_all_assets()
        logger.info(f"HedgeAssetUniverse initialized with {len(self._assets)} assets")

    def _register_all_assets(self):
        """注册所有对冲资产"""
        self._register_inverse_equity()
        self._register_inverse_sector()
        self._register_sector_etf()
        self._register_volatility()
        self._register_safe_haven()
        self._register_fixed_income()
        self._register_commodity()
        self._register_international()
        self._register_currency()

    def _register(self, asset: HedgeAsset):
        """注册单个资产"""
        self._assets[asset.symbol] = asset

    # ============ 反向股票ETF ============
    def _register_inverse_equity(self):
        """注册反向股票ETF"""
        assets = [
            # S&P 500 反向
            HedgeAsset("SH", "ProShares Short S&P500", HedgeCategory.INVERSE_EQUITY,
                       leverage=-1.0, expense_ratio=0.0089, avg_daily_volume=3e6,
                       underlying="SPY", tags=["sp500", "short"]),
            HedgeAsset("SDS", "ProShares UltraShort S&P500", HedgeCategory.INVERSE_EQUITY,
                       leverage=-2.0, expense_ratio=0.0089, avg_daily_volume=2e6,
                       underlying="SPY", tags=["sp500", "short", "leveraged"]),
            HedgeAsset("SPXU", "ProShares UltraPro Short S&P500", HedgeCategory.INVERSE_EQUITY,
                       leverage=-3.0, expense_ratio=0.0091, avg_daily_volume=5e6,
                       underlying="SPY", tags=["sp500", "short", "leveraged"]),
            HedgeAsset("SPXS", "Direxion Daily S&P 500 Bear 3X", HedgeCategory.INVERSE_EQUITY,
                       leverage=-3.0, expense_ratio=0.0099, avg_daily_volume=4e6,
                       underlying="SPY", tags=["sp500", "short", "leveraged"]),

            # Nasdaq 反向
            HedgeAsset("PSQ", "ProShares Short QQQ", HedgeCategory.INVERSE_EQUITY,
                       leverage=-1.0, expense_ratio=0.0095, avg_daily_volume=2e6,
                       underlying="QQQ", tags=["nasdaq", "short", "tech"]),
            HedgeAsset("QID", "ProShares UltraShort QQQ", HedgeCategory.INVERSE_EQUITY,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=3e6,
                       underlying="QQQ", tags=["nasdaq", "short", "tech", "leveraged"]),
            HedgeAsset("SQQQ", "ProShares UltraPro Short QQQ", HedgeCategory.INVERSE_EQUITY,
                       leverage=-3.0, expense_ratio=0.0095, avg_daily_volume=8e7,
                       underlying="QQQ", tags=["nasdaq", "short", "tech", "leveraged"]),

            # Russell 2000 反向
            HedgeAsset("RWM", "ProShares Short Russell2000", HedgeCategory.INVERSE_EQUITY,
                       leverage=-1.0, expense_ratio=0.0095, avg_daily_volume=1e6,
                       underlying="IWM", tags=["russell", "small_cap", "short"]),
            HedgeAsset("TWM", "ProShares UltraShort Russell2000", HedgeCategory.INVERSE_EQUITY,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=5e5,
                       underlying="IWM", tags=["russell", "small_cap", "short", "leveraged"]),
            HedgeAsset("SRTY", "ProShares UltraPro Short Russell2000", HedgeCategory.INVERSE_EQUITY,
                       leverage=-3.0, expense_ratio=0.0095, avg_daily_volume=2e6,
                       underlying="IWM", tags=["russell", "small_cap", "short", "leveraged"]),

            # Dow Jones 反向
            HedgeAsset("DOG", "ProShares Short Dow30", HedgeCategory.INVERSE_EQUITY,
                       leverage=-1.0, expense_ratio=0.0095, avg_daily_volume=5e5,
                       underlying="DIA", tags=["dow", "short"]),
            HedgeAsset("DXD", "ProShares UltraShort Dow30", HedgeCategory.INVERSE_EQUITY,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=3e5,
                       underlying="DIA", tags=["dow", "short", "leveraged"]),
            HedgeAsset("SDOW", "ProShares UltraPro Short Dow30", HedgeCategory.INVERSE_EQUITY,
                       leverage=-3.0, expense_ratio=0.0095, avg_daily_volume=1e6,
                       underlying="DIA", tags=["dow", "short", "leveraged"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 反向行业ETF ============
    def _register_inverse_sector(self):
        """注册反向行业ETF"""
        assets = [
            # 科技
            HedgeAsset("REW", "ProShares UltraShort Technology", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=5e4,
                       sector="technology", tags=["tech", "short"]),

            # 金融
            HedgeAsset("SKF", "ProShares UltraShort Financials", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=1e5,
                       sector="financials", tags=["finance", "short"]),
            HedgeAsset("SEF", "ProShares Short Financials", HedgeCategory.INVERSE_SECTOR,
                       leverage=-1.0, expense_ratio=0.0095, avg_daily_volume=5e4,
                       sector="financials", tags=["finance", "short"]),
            HedgeAsset("FAZ", "Direxion Daily Financial Bear 3X", HedgeCategory.INVERSE_SECTOR,
                       leverage=-3.0, expense_ratio=0.0099, avg_daily_volume=2e6,
                       sector="financials", tags=["finance", "short", "leveraged"]),

            # 能源
            HedgeAsset("ERY", "Direxion Daily Energy Bear 2X", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0099, avg_daily_volume=5e5,
                       sector="energy", tags=["energy", "short"]),
            HedgeAsset("DUG", "ProShares UltraShort Oil & Gas", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=1e5,
                       sector="energy", tags=["energy", "oil", "short"]),

            # 房地产
            HedgeAsset("SRS", "ProShares UltraShort Real Estate", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=1e5,
                       sector="real_estate", tags=["reits", "short"]),
            HedgeAsset("DRV", "Direxion Daily Real Estate Bear 3X", HedgeCategory.INVERSE_SECTOR,
                       leverage=-3.0, expense_ratio=0.0099, avg_daily_volume=3e5,
                       sector="real_estate", tags=["reits", "short", "leveraged"]),

            # 医疗
            HedgeAsset("RXD", "ProShares UltraShort Health Care", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=3e4,
                       sector="healthcare", tags=["healthcare", "short"]),

            # 半导体
            HedgeAsset("SOXS", "Direxion Daily Semiconductor Bear 3X", HedgeCategory.INVERSE_SECTOR,
                       leverage=-3.0, expense_ratio=0.0099, avg_daily_volume=3e7,
                       sector="semiconductors", tags=["semis", "tech", "short", "leveraged"]),

            # 工业
            HedgeAsset("SIJ", "ProShares UltraShort Industrials", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=2e4,
                       sector="industrials", tags=["industrial", "short"]),

            # 材料
            HedgeAsset("SMN", "ProShares UltraShort Basic Materials", HedgeCategory.INVERSE_SECTOR,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=2e4,
                       sector="materials", tags=["materials", "short"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 行业ETF (用于配对交易) ============
    def _register_sector_etf(self):
        """注册行业ETF"""
        assets = [
            HedgeAsset("XLK", "Technology Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=1e7, sector="technology"),
            HedgeAsset("XLF", "Financial Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=5e7, sector="financials"),
            HedgeAsset("XLE", "Energy Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=2e7, sector="energy"),
            HedgeAsset("XLV", "Health Care Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=1e7, sector="healthcare"),
            HedgeAsset("XLI", "Industrial Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=1e7, sector="industrials"),
            HedgeAsset("XLY", "Consumer Discretionary Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=8e6, sector="consumer_discretionary"),
            HedgeAsset("XLP", "Consumer Staples Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=1e7, sector="consumer_staples"),
            HedgeAsset("XLU", "Utilities Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=1e7, sector="utilities"),
            HedgeAsset("XLB", "Materials Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=5e6, sector="materials"),
            HedgeAsset("XLRE", "Real Estate Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=5e6, sector="real_estate"),
            HedgeAsset("XLC", "Communication Services Select Sector SPDR", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.001, avg_daily_volume=5e6, sector="communication"),
            HedgeAsset("SMH", "VanEck Semiconductor ETF", HedgeCategory.SECTOR_ETF,
                       expense_ratio=0.0035, avg_daily_volume=1e7, sector="semiconductors"),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 波动率工具 ============
    def _register_volatility(self):
        """注册波动率工具"""
        assets = [
            HedgeAsset("VXX", "iPath Series B S&P 500 VIX Short-Term Futures", HedgeCategory.VOLATILITY,
                       leverage=1.0, expense_ratio=0.0089, avg_daily_volume=3e7,
                       tags=["vix", "short_term", "volatility"]),
            HedgeAsset("UVXY", "ProShares Ultra VIX Short-Term Futures", HedgeCategory.VOLATILITY,
                       leverage=1.5, expense_ratio=0.0095, avg_daily_volume=5e7,
                       tags=["vix", "leveraged", "volatility"]),
            HedgeAsset("VIXY", "ProShares VIX Short-Term Futures ETF", HedgeCategory.VOLATILITY,
                       leverage=1.0, expense_ratio=0.0087, avg_daily_volume=5e6,
                       tags=["vix", "volatility"]),
            HedgeAsset("SVXY", "ProShares Short VIX Short-Term Futures", HedgeCategory.VOLATILITY,
                       leverage=-0.5, expense_ratio=0.0095, avg_daily_volume=5e6,
                       tags=["vix", "inverse", "volatility"]),
            HedgeAsset("TAIL", "Cambria Tail Risk ETF", HedgeCategory.VOLATILITY,
                       leverage=1.0, expense_ratio=0.0059, avg_daily_volume=5e5,
                       tags=["tail_risk", "put_spread", "protection"]),
            HedgeAsset("VIXM", "ProShares VIX Mid-Term Futures ETF", HedgeCategory.VOLATILITY,
                       leverage=1.0, expense_ratio=0.0085, avg_daily_volume=1e5,
                       tags=["vix", "mid_term", "volatility"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 避险资产 ============
    def _register_safe_haven(self):
        """注册避险资产"""
        assets = [
            HedgeAsset("GLD", "SPDR Gold Shares", HedgeCategory.SAFE_HAVEN,
                       expense_ratio=0.004, avg_daily_volume=8e6,
                       tags=["gold", "precious_metal"]),
            HedgeAsset("IAU", "iShares Gold Trust", HedgeCategory.SAFE_HAVEN,
                       expense_ratio=0.0025, avg_daily_volume=1e7,
                       tags=["gold", "precious_metal"]),
            HedgeAsset("GLDM", "SPDR Gold MiniShares Trust", HedgeCategory.SAFE_HAVEN,
                       expense_ratio=0.001, avg_daily_volume=2e6,
                       tags=["gold", "precious_metal", "low_cost"]),
            HedgeAsset("SLV", "iShares Silver Trust", HedgeCategory.SAFE_HAVEN,
                       expense_ratio=0.005, avg_daily_volume=2e7,
                       tags=["silver", "precious_metal"]),
            HedgeAsset("SGOL", "Aberdeen Standard Physical Gold Shares", HedgeCategory.SAFE_HAVEN,
                       expense_ratio=0.0017, avg_daily_volume=1e6,
                       tags=["gold", "precious_metal"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 固定收益 ============
    def _register_fixed_income(self):
        """注册固定收益资产"""
        assets = [
            # 长期国债
            HedgeAsset("TLT", "iShares 20+ Year Treasury Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0015, avg_daily_volume=2e7,
                       tags=["treasury", "long_term", "duration_high"]),
            HedgeAsset("EDV", "Vanguard Extended Duration Treasury ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0006, avg_daily_volume=5e5,
                       tags=["treasury", "extended_duration"]),

            # 中期国债
            HedgeAsset("IEF", "iShares 7-10 Year Treasury Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0015, avg_daily_volume=5e6,
                       tags=["treasury", "mid_term"]),
            HedgeAsset("IEI", "iShares 3-7 Year Treasury Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0015, avg_daily_volume=1e6,
                       tags=["treasury", "mid_term"]),

            # 短期国债
            HedgeAsset("SHY", "iShares 1-3 Year Treasury Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0015, avg_daily_volume=3e6,
                       tags=["treasury", "short_term", "low_risk"]),
            HedgeAsset("SHV", "iShares Short Treasury Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0015, avg_daily_volume=2e6,
                       tags=["treasury", "ultra_short", "cash_equivalent"]),
            HedgeAsset("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0014, avg_daily_volume=3e6,
                       tags=["t_bill", "ultra_short", "cash_equivalent"]),

            # 综合债券
            HedgeAsset("AGG", "iShares Core U.S. Aggregate Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0003, avg_daily_volume=8e6,
                       tags=["aggregate", "investment_grade"]),
            HedgeAsset("BND", "Vanguard Total Bond Market ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0003, avg_daily_volume=7e6,
                       tags=["aggregate", "investment_grade"]),

            # 通胀保护
            HedgeAsset("TIP", "iShares TIPS Bond ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0019, avg_daily_volume=3e6,
                       tags=["tips", "inflation_protection"]),
            HedgeAsset("SCHP", "Schwab U.S. TIPS ETF", HedgeCategory.FIXED_INCOME,
                       expense_ratio=0.0004, avg_daily_volume=1e6,
                       tags=["tips", "inflation_protection", "low_cost"]),

            # 反向债券 (利率上升对冲)
            HedgeAsset("TBF", "ProShares Short 20+ Year Treasury", HedgeCategory.FIXED_INCOME,
                       leverage=-1.0, expense_ratio=0.0092, avg_daily_volume=2e5,
                       tags=["treasury", "inverse", "rate_hedge"]),
            HedgeAsset("TBT", "ProShares UltraShort 20+ Year Treasury", HedgeCategory.FIXED_INCOME,
                       leverage=-2.0, expense_ratio=0.0090, avg_daily_volume=2e6,
                       tags=["treasury", "inverse", "rate_hedge", "leveraged"]),
            HedgeAsset("TTT", "ProShares UltraPro Short 20+ Year Treasury", HedgeCategory.FIXED_INCOME,
                       leverage=-3.0, expense_ratio=0.0095, avg_daily_volume=5e5,
                       tags=["treasury", "inverse", "rate_hedge", "leveraged"]),
            HedgeAsset("TBX", "ProShares Short 7-10 Year Treasury", HedgeCategory.FIXED_INCOME,
                       leverage=-1.0, expense_ratio=0.0095, avg_daily_volume=5e4,
                       tags=["treasury", "inverse", "rate_hedge", "mid_term"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 商品 ============
    def _register_commodity(self):
        """注册商品资产"""
        assets = [
            # 原油
            HedgeAsset("USO", "United States Oil Fund", HedgeCategory.COMMODITY,
                       expense_ratio=0.0079, avg_daily_volume=5e6,
                       tags=["oil", "crude", "energy"]),
            HedgeAsset("SCO", "ProShares UltraShort Bloomberg Crude Oil", HedgeCategory.COMMODITY,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=1e6,
                       tags=["oil", "crude", "inverse", "energy"]),

            # 综合商品
            HedgeAsset("DBC", "Invesco DB Commodity Index Tracking Fund", HedgeCategory.COMMODITY,
                       expense_ratio=0.0087, avg_daily_volume=2e6,
                       tags=["diversified", "broad_commodity"]),
            HedgeAsset("GSG", "iShares S&P GSCI Commodity-Indexed Trust", HedgeCategory.COMMODITY,
                       expense_ratio=0.0075, avg_daily_volume=5e5,
                       tags=["diversified", "broad_commodity"]),

            # 农产品
            HedgeAsset("DBA", "Invesco DB Agriculture Fund", HedgeCategory.COMMODITY,
                       expense_ratio=0.0093, avg_daily_volume=1e6,
                       tags=["agriculture", "soft_commodity"]),

            # 金属
            HedgeAsset("COPX", "Global X Copper Miners ETF", HedgeCategory.COMMODITY,
                       expense_ratio=0.0065, avg_daily_volume=1e6,
                       tags=["copper", "base_metal", "miners"]),
            HedgeAsset("PPLT", "abrdn Physical Platinum Shares ETF", HedgeCategory.COMMODITY,
                       expense_ratio=0.006, avg_daily_volume=1e5,
                       tags=["platinum", "precious_metal"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 国际市场 ============
    def _register_international(self):
        """注册国际市场资产"""
        assets = [
            # 发达市场
            HedgeAsset("EFA", "iShares MSCI EAFE ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0032, avg_daily_volume=2e7,
                       tags=["developed", "ex_us", "diversified"]),
            HedgeAsset("VEA", "Vanguard FTSE Developed Markets ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0005, avg_daily_volume=1e7,
                       tags=["developed", "ex_us", "low_cost"]),

            # 新兴市场
            HedgeAsset("EEM", "iShares MSCI Emerging Markets ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0068, avg_daily_volume=4e7,
                       tags=["emerging", "diversified"]),
            HedgeAsset("VWO", "Vanguard FTSE Emerging Markets ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0008, avg_daily_volume=1e7,
                       tags=["emerging", "diversified", "low_cost"]),

            # 中国
            HedgeAsset("FXI", "iShares China Large-Cap ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0074, avg_daily_volume=3e7,
                       tags=["china", "large_cap"]),
            HedgeAsset("KWEB", "KraneShares CSI China Internet ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.007, avg_daily_volume=1e7,
                       tags=["china", "internet", "tech"]),
            HedgeAsset("MCHI", "iShares MSCI China ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0059, avg_daily_volume=5e6,
                       tags=["china", "broad"]),

            # 日本
            HedgeAsset("EWJ", "iShares MSCI Japan ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.005, avg_daily_volume=5e6,
                       tags=["japan", "developed"]),

            # 其他
            HedgeAsset("EWZ", "iShares MSCI Brazil ETF", HedgeCategory.INTERNATIONAL,
                       expense_ratio=0.0059, avg_daily_volume=2e7,
                       tags=["brazil", "emerging", "latam"]),

            # 反向国际
            HedgeAsset("EUM", "ProShares Short MSCI Emerging Markets", HedgeCategory.INTERNATIONAL,
                       leverage=-1.0, expense_ratio=0.0095, avg_daily_volume=2e5,
                       tags=["emerging", "inverse"]),
            HedgeAsset("EEV", "ProShares UltraShort MSCI Emerging Markets", HedgeCategory.INTERNATIONAL,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=1e5,
                       tags=["emerging", "inverse", "leveraged"]),
            HedgeAsset("FXP", "ProShares UltraShort FTSE China 50", HedgeCategory.INTERNATIONAL,
                       leverage=-2.0, expense_ratio=0.0095, avg_daily_volume=5e5,
                       tags=["china", "inverse", "leveraged"]),
            HedgeAsset("YANG", "Direxion Daily FTSE China Bear 3X", HedgeCategory.INTERNATIONAL,
                       leverage=-3.0, expense_ratio=0.0099, avg_daily_volume=5e6,
                       tags=["china", "inverse", "leveraged"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 货币对冲 ============
    def _register_currency(self):
        """注册货币对冲资产"""
        assets = [
            HedgeAsset("UUP", "Invesco DB US Dollar Index Bullish Fund", HedgeCategory.CURRENCY,
                       expense_ratio=0.0078, avg_daily_volume=2e6,
                       tags=["usd", "dollar_bull"]),
            HedgeAsset("UDN", "Invesco DB US Dollar Index Bearish Fund", HedgeCategory.CURRENCY,
                       leverage=-1.0, expense_ratio=0.0078, avg_daily_volume=2e5,
                       tags=["usd", "dollar_bear"]),
            HedgeAsset("FXE", "Invesco CurrencyShares Euro Trust", HedgeCategory.CURRENCY,
                       expense_ratio=0.004, avg_daily_volume=5e5,
                       tags=["euro", "eur"]),
            HedgeAsset("FXY", "Invesco CurrencyShares Japanese Yen Trust", HedgeCategory.CURRENCY,
                       expense_ratio=0.004, avg_daily_volume=3e5,
                       tags=["yen", "jpy", "safe_haven"]),
            HedgeAsset("FXB", "Invesco CurrencyShares British Pound Sterling Trust", HedgeCategory.CURRENCY,
                       expense_ratio=0.004, avg_daily_volume=1e5,
                       tags=["pound", "gbp"]),
            HedgeAsset("FXC", "Invesco CurrencyShares Canadian Dollar Trust", HedgeCategory.CURRENCY,
                       expense_ratio=0.004, avg_daily_volume=1e5,
                       tags=["cad", "commodity_currency"]),
            HedgeAsset("FXA", "Invesco CurrencyShares Australian Dollar Trust", HedgeCategory.CURRENCY,
                       expense_ratio=0.004, avg_daily_volume=1e5,
                       tags=["aud", "commodity_currency"]),
        ]
        for asset in assets:
            self._register(asset)

    # ============ 查询方法 ============

    def get(self, symbol: str) -> Optional[HedgeAsset]:
        """获取单个资产"""
        return self._assets.get(symbol)

    def get_all(self) -> Dict[str, HedgeAsset]:
        """获取所有资产"""
        return self._assets.copy()

    def get_all_symbols(self) -> List[str]:
        """获取所有资产代码"""
        return list(self._assets.keys())

    def get_by_category(self, category: HedgeCategory) -> List[HedgeAsset]:
        """按类别获取资产"""
        return [a for a in self._assets.values() if a.category == category]

    def get_by_sector(self, sector: str) -> List[HedgeAsset]:
        """按行业获取资产"""
        return [a for a in self._assets.values() if a.sector == sector]

    def get_by_tag(self, tag: str) -> List[HedgeAsset]:
        """按标签获取资产"""
        return [a for a in self._assets.values() if tag in a.tags]

    def get_inverse_assets(self) -> List[HedgeAsset]:
        """获取所有反向资产"""
        return [a for a in self._assets.values() if a.is_inverse]

    def get_leveraged_assets(self) -> List[HedgeAsset]:
        """获取所有杠杆资产"""
        return [a for a in self._assets.values() if a.is_leveraged]

    def get_inverse_for(self, underlying: str) -> List[HedgeAsset]:
        """获取特定标的的反向ETF"""
        return [a for a in self._assets.values()
                if a.underlying == underlying and a.is_inverse]

    def filter(
        self,
        category: Optional[HedgeCategory] = None,
        sector: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_volume: float = 0,
        max_expense: float = 1.0,
        inverse_only: bool = False,
        leveraged_only: bool = False,
    ) -> List[HedgeAsset]:
        """
        多条件过滤资产

        Args:
            category: 资产类别
            sector: 行业
            tags: 标签列表 (任一匹配)
            min_volume: 最小日均成交量
            max_expense: 最大费率
            inverse_only: 仅反向资产
            leveraged_only: 仅杠杆资产

        Returns:
            List[HedgeAsset]: 符合条件的资产列表
        """
        result = list(self._assets.values())

        if category:
            result = [a for a in result if a.category == category]

        if sector:
            result = [a for a in result if a.sector == sector]

        if tags:
            result = [a for a in result if any(t in a.tags for t in tags)]

        if min_volume > 0:
            result = [a for a in result if a.avg_daily_volume >= min_volume]

        if max_expense < 1.0:
            result = [a for a in result if a.expense_ratio <= max_expense]

        if inverse_only:
            result = [a for a in result if a.is_inverse]

        if leveraged_only:
            result = [a for a in result if a.is_leveraged]

        return result

    def summary(self) -> Dict[str, int]:
        """获取资产统计摘要"""
        summary = {
            "total": len(self._assets),
            "inverse": len(self.get_inverse_assets()),
            "leveraged": len(self.get_leveraged_assets()),
        }

        for category in HedgeCategory:
            summary[category.value] = len(self.get_by_category(category))

        return summary
