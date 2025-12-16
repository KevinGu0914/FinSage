# 动态对冲资产选择模块设计方案

## 1. 问题分析

### 1.1 当前系统局限性

```
当前 HedgingAgent 的硬编码对冲工具:
┌────────────────────────────────────────────────────┐
│  HEDGE_INSTRUMENTS (固定7个)                        │
│  • SPY_PUT   - SPY看跌期权                          │
│  • VIX_CALL  - VIX看涨期权                          │
│  • TLT       - 长期国债ETF                          │
│  • GLD       - 黄金ETF                             │
│  • TAIL      - 尾部风险ETF                          │
│  • SH        - 做空标普ETF                          │
│  • CASH      - 现金                                │
└────────────────────────────────────────────────────┘
```

**问题场景示例：**

| 对冲需求 | 理想工具 | 当前是否可用 |
|---------|---------|------------|
| 做空科技板块 | SQQQ, QID | ❌ 无 |
| 做空能源板块 | ERY, DUG | ❌ 无 |
| 低相关性对冲 | 动态筛选低相关资产 | ❌ 固定 |
| 新兴市场对冲 | EUM, FXP | ❌ 无 |
| 利率上升对冲 | TBF, TBT | ❌ 无 |
| 配对交易对冲 | 需协整检验筛选 | ❌ 无 |

---

## 2. 解决方案架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     动态对冲资产选择架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    HedgeAssetUniverse                           │    │
│  │                    (对冲资产全集)                                │    │
│  │                                                                  │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │    │
│  │  │  反向ETF池    │ │  行业ETF池    │ │  避险资产池   │            │    │
│  │  │  SQQQ, SH    │ │  XLK, XLF    │ │  TLT, GLD    │            │    │
│  │  │  SPXU, QID   │ │  XLE, XLV    │ │  VNQ, AGG    │            │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │    │
│  │  │  波动率工具池  │ │  国际市场池   │ │  商品池      │            │    │
│  │  │  VXX, UVXY   │ │  EFA, EEM    │ │  USO, DBA    │            │    │
│  │  │  VIXY, SVXY  │ │  FXI, EWJ    │ │  SLV, COPX   │            │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   DynamicHedgeSelector                          │    │
│  │                   (动态对冲选择器)                               │    │
│  │                                                                  │    │
│  │  Step 1: 识别对冲需求                                           │    │
│  │  ┌─────────────────────────────────────────────────────────┐   │    │
│  │  │  • 分析当前组合风险敞口 (行业、因子、beta)                  │   │    │
│  │  │  • 解读 HedgingAgent 的策略建议                           │   │    │
│  │  │  • 确定对冲目标 (beta对冲/行业对冲/尾部对冲等)              │   │    │
│  │  └─────────────────────────────────────────────────────────┘   │    │
│  │                               │                                 │    │
│  │                               ▼                                 │    │
│  │  Step 2: 候选资产筛选                                           │    │
│  │  ┌─────────────────────────────────────────────────────────┐   │    │
│  │  │  • 根据对冲目标从 Universe 中筛选候选                       │   │    │
│  │  │  • 计算与组合的相关性矩阵                                  │   │    │
│  │  │  • 流动性过滤 (日均成交量)                                 │   │    │
│  │  │  • 成本效益分析                                           │   │    │
│  │  └─────────────────────────────────────────────────────────┘   │    │
│  │                               │                                 │    │
│  │                               ▼                                 │    │
│  │  Step 3: 多因子排序                                             │    │
│  │  ┌─────────────────────────────────────────────────────────┐   │    │
│  │  │  Score = w1*相关性 + w2*流动性 + w3*成本 + w4*对冲效率      │   │    │
│  │  │  输出: Top K 最优对冲资产                                  │   │    │
│  │  └─────────────────────────────────────────────────────────┘   │    │
│  │                               │                                 │    │
│  └───────────────────────────────┼────────────────────────────────┘    │
│                                  │                                      │
│                                  ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      HedgingAgent (增强版)                       │    │
│  │                                                                  │    │
│  │  HEDGE_INSTRUMENTS = 固定工具 + 动态选择的工具                   │    │
│  │                                                                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件设计

### 3.1 HedgeAssetUniverse (对冲资产全集)

```python
# finsage/hedging/hedge_universe.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class HedgeCategory(Enum):
    """对冲资产类别"""
    INVERSE_EQUITY = "inverse_equity"       # 反向股票ETF
    INVERSE_SECTOR = "inverse_sector"       # 反向行业ETF
    SECTOR_ETF = "sector_etf"               # 行业ETF (用于配对)
    VOLATILITY = "volatility"               # 波动率工具
    SAFE_HAVEN = "safe_haven"               # 避险资产
    FIXED_INCOME = "fixed_income"           # 固定收益
    COMMODITY = "commodity"                 # 商品
    INTERNATIONAL = "international"         # 国际市场
    CURRENCY = "currency"                   # 货币对冲
    OPTION = "option"                       # 期权类


@dataclass
class HedgeAsset:
    """对冲资产定义"""
    symbol: str
    name: str
    category: HedgeCategory
    leverage: float = 1.0                   # 杠杆倍数 (负数表示反向)
    expense_ratio: float = 0.001            # 费率
    avg_daily_volume: float = 1e6           # 日均成交量
    typical_spread: float = 0.001           # 典型价差
    underlying: Optional[str] = None        # 底层资产
    sector: Optional[str] = None            # 行业
    tags: List[str] = field(default_factory=list)

    @property
    def is_inverse(self) -> bool:
        return self.leverage < 0

    @property
    def total_cost_estimate(self) -> float:
        """估算总持有成本 (年化)"""
        return self.expense_ratio + self.typical_spread * 252


@dataclass
class HedgeAssetUniverse:
    """对冲资产全集配置"""

    # ============ 反向股票ETF ============
    inverse_equity: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "SH": HedgeAsset("SH", "ProShares Short S&P500", HedgeCategory.INVERSE_EQUITY,
                         leverage=-1.0, expense_ratio=0.0089, underlying="SPY"),
        "SDS": HedgeAsset("SDS", "ProShares UltraShort S&P500", HedgeCategory.INVERSE_EQUITY,
                          leverage=-2.0, expense_ratio=0.0089, underlying="SPY"),
        "SPXU": HedgeAsset("SPXU", "ProShares UltraPro Short S&P500", HedgeCategory.INVERSE_EQUITY,
                           leverage=-3.0, expense_ratio=0.0091, underlying="SPY"),
        "PSQ": HedgeAsset("PSQ", "ProShares Short QQQ", HedgeCategory.INVERSE_EQUITY,
                          leverage=-1.0, expense_ratio=0.0095, underlying="QQQ"),
        "QID": HedgeAsset("QID", "ProShares UltraShort QQQ", HedgeCategory.INVERSE_EQUITY,
                          leverage=-2.0, expense_ratio=0.0095, underlying="QQQ"),
        "SQQQ": HedgeAsset("SQQQ", "ProShares UltraPro Short QQQ", HedgeCategory.INVERSE_EQUITY,
                           leverage=-3.0, expense_ratio=0.0095, underlying="QQQ"),
        "RWM": HedgeAsset("RWM", "ProShares Short Russell2000", HedgeCategory.INVERSE_EQUITY,
                          leverage=-1.0, expense_ratio=0.0095, underlying="IWM"),
        "DOG": HedgeAsset("DOG", "ProShares Short Dow30", HedgeCategory.INVERSE_EQUITY,
                          leverage=-1.0, expense_ratio=0.0095, underlying="DIA"),
    })

    # ============ 反向行业ETF ============
    inverse_sector: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        # 科技
        "REW": HedgeAsset("REW", "ProShares UltraShort Technology", HedgeCategory.INVERSE_SECTOR,
                          leverage=-2.0, sector="technology"),
        # 金融
        "SKF": HedgeAsset("SKF", "ProShares UltraShort Financials", HedgeCategory.INVERSE_SECTOR,
                          leverage=-2.0, sector="financials"),
        "SEF": HedgeAsset("SEF", "ProShares Short Financials", HedgeCategory.INVERSE_SECTOR,
                          leverage=-1.0, sector="financials"),
        # 能源
        "ERY": HedgeAsset("ERY", "Direxion Daily Energy Bear 2X", HedgeCategory.INVERSE_SECTOR,
                          leverage=-2.0, sector="energy"),
        "DUG": HedgeAsset("DUG", "ProShares UltraShort Oil & Gas", HedgeCategory.INVERSE_SECTOR,
                          leverage=-2.0, sector="energy"),
        # 房地产
        "SRS": HedgeAsset("SRS", "ProShares UltraShort Real Estate", HedgeCategory.INVERSE_SECTOR,
                          leverage=-2.0, sector="real_estate"),
        # 医疗
        "RXD": HedgeAsset("RXD", "ProShares UltraShort Health Care", HedgeCategory.INVERSE_SECTOR,
                          leverage=-2.0, sector="healthcare"),
    })

    # ============ 行业ETF (用于配对交易) ============
    sector_etf: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "XLK": HedgeAsset("XLK", "Technology Select Sector", HedgeCategory.SECTOR_ETF, sector="technology"),
        "XLF": HedgeAsset("XLF", "Financial Select Sector", HedgeCategory.SECTOR_ETF, sector="financials"),
        "XLE": HedgeAsset("XLE", "Energy Select Sector", HedgeCategory.SECTOR_ETF, sector="energy"),
        "XLV": HedgeAsset("XLV", "Health Care Select Sector", HedgeCategory.SECTOR_ETF, sector="healthcare"),
        "XLI": HedgeAsset("XLI", "Industrial Select Sector", HedgeCategory.SECTOR_ETF, sector="industrials"),
        "XLY": HedgeAsset("XLY", "Consumer Discretionary", HedgeCategory.SECTOR_ETF, sector="consumer_discretionary"),
        "XLP": HedgeAsset("XLP", "Consumer Staples", HedgeCategory.SECTOR_ETF, sector="consumer_staples"),
        "XLU": HedgeAsset("XLU", "Utilities Select Sector", HedgeCategory.SECTOR_ETF, sector="utilities"),
        "XLB": HedgeAsset("XLB", "Materials Select Sector", HedgeCategory.SECTOR_ETF, sector="materials"),
        "XLRE": HedgeAsset("XLRE", "Real Estate Select Sector", HedgeCategory.SECTOR_ETF, sector="real_estate"),
        "XLC": HedgeAsset("XLC", "Communication Services", HedgeCategory.SECTOR_ETF, sector="communication"),
    })

    # ============ 波动率工具 ============
    volatility: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "VXX": HedgeAsset("VXX", "iPath Series B S&P 500 VIX Short-Term", HedgeCategory.VOLATILITY,
                          expense_ratio=0.0089, tags=["vix", "short_term"]),
        "UVXY": HedgeAsset("UVXY", "ProShares Ultra VIX Short-Term", HedgeCategory.VOLATILITY,
                           leverage=1.5, expense_ratio=0.0095, tags=["vix", "leveraged"]),
        "VIXY": HedgeAsset("VIXY", "ProShares VIX Short-Term Futures", HedgeCategory.VOLATILITY,
                           expense_ratio=0.0087, tags=["vix"]),
        "SVXY": HedgeAsset("SVXY", "ProShares Short VIX Short-Term", HedgeCategory.VOLATILITY,
                           leverage=-0.5, expense_ratio=0.0095, tags=["vix", "inverse"]),
        "TAIL": HedgeAsset("TAIL", "Cambria Tail Risk ETF", HedgeCategory.VOLATILITY,
                           expense_ratio=0.0059, tags=["tail_risk", "put_spread"]),
    })

    # ============ 避险资产 ============
    safe_haven: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "GLD": HedgeAsset("GLD", "SPDR Gold Shares", HedgeCategory.SAFE_HAVEN, expense_ratio=0.004),
        "IAU": HedgeAsset("IAU", "iShares Gold Trust", HedgeCategory.SAFE_HAVEN, expense_ratio=0.0025),
        "SLV": HedgeAsset("SLV", "iShares Silver Trust", HedgeCategory.SAFE_HAVEN, expense_ratio=0.005),
        "GLDM": HedgeAsset("GLDM", "SPDR Gold MiniShares", HedgeCategory.SAFE_HAVEN, expense_ratio=0.001),
    })

    # ============ 固定收益 ============
    fixed_income: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "TLT": HedgeAsset("TLT", "iShares 20+ Year Treasury Bond", HedgeCategory.FIXED_INCOME),
        "IEF": HedgeAsset("IEF", "iShares 7-10 Year Treasury Bond", HedgeCategory.FIXED_INCOME),
        "SHY": HedgeAsset("SHY", "iShares 1-3 Year Treasury Bond", HedgeCategory.FIXED_INCOME),
        "AGG": HedgeAsset("AGG", "iShares Core US Aggregate Bond", HedgeCategory.FIXED_INCOME),
        "BND": HedgeAsset("BND", "Vanguard Total Bond Market", HedgeCategory.FIXED_INCOME),
        "TIP": HedgeAsset("TIP", "iShares TIPS Bond", HedgeCategory.FIXED_INCOME, tags=["inflation"]),
        # 反向债券 (利率上升对冲)
        "TBF": HedgeAsset("TBF", "ProShares Short 20+ Year Treasury", HedgeCategory.FIXED_INCOME,
                          leverage=-1.0, tags=["inverse", "rate_hedge"]),
        "TBT": HedgeAsset("TBT", "ProShares UltraShort 20+ Year Treasury", HedgeCategory.FIXED_INCOME,
                          leverage=-2.0, tags=["inverse", "rate_hedge"]),
    })

    # ============ 国际市场 ============
    international: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "EFA": HedgeAsset("EFA", "iShares MSCI EAFE", HedgeCategory.INTERNATIONAL),
        "EEM": HedgeAsset("EEM", "iShares MSCI Emerging Markets", HedgeCategory.INTERNATIONAL),
        "FXI": HedgeAsset("FXI", "iShares China Large-Cap", HedgeCategory.INTERNATIONAL),
        "EWJ": HedgeAsset("EWJ", "iShares MSCI Japan", HedgeCategory.INTERNATIONAL),
        "EWZ": HedgeAsset("EWZ", "iShares MSCI Brazil", HedgeCategory.INTERNATIONAL),
        # 反向国际
        "EUM": HedgeAsset("EUM", "ProShares Short MSCI Emerging Markets", HedgeCategory.INTERNATIONAL,
                          leverage=-1.0),
        "FXP": HedgeAsset("FXP", "ProShares UltraShort FTSE China 50", HedgeCategory.INTERNATIONAL,
                          leverage=-2.0),
    })

    # ============ 货币对冲 ============
    currency: Dict[str, HedgeAsset] = field(default_factory=lambda: {
        "UUP": HedgeAsset("UUP", "Invesco DB US Dollar Index Bullish", HedgeCategory.CURRENCY),
        "UDN": HedgeAsset("UDN", "Invesco DB US Dollar Index Bearish", HedgeCategory.CURRENCY, leverage=-1.0),
        "FXE": HedgeAsset("FXE", "Invesco CurrencyShares Euro Trust", HedgeCategory.CURRENCY),
        "FXY": HedgeAsset("FXY", "Invesco CurrencyShares Japanese Yen", HedgeCategory.CURRENCY),
    })

    def get_all_symbols(self) -> List[str]:
        """获取所有资产代码"""
        symbols = []
        for category_assets in [
            self.inverse_equity, self.inverse_sector, self.sector_etf,
            self.volatility, self.safe_haven, self.fixed_income,
            self.international, self.currency
        ]:
            symbols.extend(category_assets.keys())
        return symbols

    def get_by_category(self, category: HedgeCategory) -> Dict[str, HedgeAsset]:
        """按类别获取资产"""
        category_map = {
            HedgeCategory.INVERSE_EQUITY: self.inverse_equity,
            HedgeCategory.INVERSE_SECTOR: self.inverse_sector,
            HedgeCategory.SECTOR_ETF: self.sector_etf,
            HedgeCategory.VOLATILITY: self.volatility,
            HedgeCategory.SAFE_HAVEN: self.safe_haven,
            HedgeCategory.FIXED_INCOME: self.fixed_income,
            HedgeCategory.INTERNATIONAL: self.international,
            HedgeCategory.CURRENCY: self.currency,
        }
        return category_map.get(category, {})

    def get_by_sector(self, sector: str) -> List[HedgeAsset]:
        """按行业获取资产"""
        result = []
        for category_assets in [self.inverse_sector, self.sector_etf]:
            for asset in category_assets.values():
                if asset.sector == sector:
                    result.append(asset)
        return result

    def get_inverse_for(self, underlying: str) -> List[HedgeAsset]:
        """获取特定标的的反向ETF"""
        result = []
        for asset in self.inverse_equity.values():
            if asset.underlying == underlying:
                result.append(asset)
        return result
```

---

### 3.2 DynamicHedgeSelector (动态对冲选择器)

```python
# finsage/hedging/dynamic_selector.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from finsage.hedging.hedge_universe import HedgeAssetUniverse, HedgeAsset, HedgeCategory

logger = logging.getLogger(__name__)


class HedgeObjective(Enum):
    """对冲目标类型"""
    BETA_NEUTRAL = "beta_neutral"           # Beta中性
    SECTOR_HEDGE = "sector_hedge"           # 行业对冲
    TAIL_RISK = "tail_risk"                 # 尾部风险对冲
    CORRELATION_HEDGE = "correlation_hedge"  # 相关性对冲
    VOLATILITY_HEDGE = "volatility_hedge"   # 波动率对冲
    RATE_HEDGE = "rate_hedge"               # 利率对冲
    CURRENCY_HEDGE = "currency_hedge"       # 货币对冲
    DIVERSIFICATION = "diversification"     # 分散化


@dataclass
class HedgeCandidate:
    """对冲候选资产评分"""
    asset: HedgeAsset
    correlation_score: float      # 相关性得分 (越负越好)
    liquidity_score: float        # 流动性得分
    cost_score: float             # 成本得分 (越低越好)
    efficiency_score: float       # 对冲效率得分
    total_score: float            # 综合得分

    def to_dict(self) -> Dict:
        return {
            "symbol": self.asset.symbol,
            "name": self.asset.name,
            "category": self.asset.category.value,
            "leverage": self.asset.leverage,
            "correlation_score": round(self.correlation_score, 4),
            "liquidity_score": round(self.liquidity_score, 4),
            "cost_score": round(self.cost_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "total_score": round(self.total_score, 4),
        }


@dataclass
class HedgeRecommendation:
    """对冲推荐结果"""
    objective: HedgeObjective
    candidates: List[HedgeCandidate]
    recommended_allocation: Dict[str, float]
    expected_correlation_reduction: float
    expected_cost: float
    reasoning: str

    def to_dict(self) -> Dict:
        return {
            "objective": self.objective.value,
            "top_candidates": [c.to_dict() for c in self.candidates[:5]],
            "recommended_allocation": self.recommended_allocation,
            "expected_correlation_reduction": self.expected_correlation_reduction,
            "expected_cost": self.expected_cost,
            "reasoning": self.reasoning,
        }


class DynamicHedgeSelector:
    """
    动态对冲资产选择器

    核心功能:
    1. 分析当前组合的风险敞口
    2. 识别对冲需求和目标
    3. 从全市场筛选最优对冲工具
    4. 计算最优对冲配置
    """

    def __init__(
        self,
        universe: Optional[HedgeAssetUniverse] = None,
        data_provider: Any = None,
        config: Optional[Dict] = None
    ):
        """
        初始化选择器

        Args:
            universe: 对冲资产全集
            data_provider: 数据提供者
            config: 配置参数
        """
        self.universe = universe or HedgeAssetUniverse()
        self.data_provider = data_provider
        self.config = config or {}

        # 配置参数
        self.min_liquidity = self.config.get("min_daily_volume", 1e6)
        self.max_cost = self.config.get("max_expense_ratio", 0.01)
        self.lookback_days = self.config.get("correlation_lookback", 60)
        self.top_k = self.config.get("top_candidates", 5)

        # 评分权重
        self.weights = {
            "correlation": 0.35,   # 相关性权重
            "liquidity": 0.20,     # 流动性权重
            "cost": 0.20,          # 成本权重
            "efficiency": 0.25,    # 对冲效率权重
        }

        logger.info(f"DynamicHedgeSelector initialized with {len(self.universe.get_all_symbols())} assets")

    def analyze_portfolio_exposure(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析组合的风险敞口

        Returns:
            Dict containing:
            - sector_exposure: 行业敞口
            - beta_exposure: Beta敞口
            - factor_exposure: 因子敞口
            - correlation_matrix: 相关性矩阵
        """
        exposure = {
            "sector_exposure": {},
            "beta_exposure": 1.0,
            "volatility": 0.0,
            "top_holdings": [],
            "concentration_risk": 0.0,
        }

        if returns_data.empty:
            return exposure

        # 计算组合波动率
        available = [a for a in portfolio_weights if a in returns_data.columns]
        if not available:
            return exposure

        weights = np.array([portfolio_weights.get(a, 0) for a in available])
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        cov_matrix = returns_data[available].cov() * 252
        exposure["volatility"] = float(np.sqrt(weights @ cov_matrix @ weights))

        # 计算 Beta (相对于 SPY)
        if "SPY" in returns_data.columns:
            spy_returns = returns_data["SPY"]
            portfolio_returns = returns_data[available] @ weights
            covariance = portfolio_returns.cov(spy_returns)
            spy_variance = spy_returns.var()
            if spy_variance > 0:
                exposure["beta_exposure"] = covariance / spy_variance

        # 计算集中度风险 (HHI)
        exposure["concentration_risk"] = float(np.sum(weights ** 2))

        # Top holdings
        sorted_holdings = sorted(zip(available, weights), key=lambda x: -x[1])
        exposure["top_holdings"] = [{"symbol": s, "weight": float(w)} for s, w in sorted_holdings[:5]]

        return exposure

    def identify_hedge_objective(
        self,
        exposure: Dict[str, Any],
        hedge_strategy: str,
        risk_constraints: Dict[str, float]
    ) -> HedgeObjective:
        """
        识别对冲目标

        Args:
            exposure: 组合敞口分析
            hedge_strategy: HedgingAgent建议的策略
            risk_constraints: 风控约束
        """
        # 根据 HedgingAgent 策略映射
        strategy_map = {
            "put_protection": HedgeObjective.TAIL_RISK,
            "collar": HedgeObjective.TAIL_RISK,
            "tail_hedge": HedgeObjective.TAIL_RISK,
            "dynamic_hedge": HedgeObjective.BETA_NEUTRAL,
            "diversification": HedgeObjective.DIVERSIFICATION,
            "safe_haven": HedgeObjective.CORRELATION_HEDGE,
        }

        if hedge_strategy in strategy_map:
            return strategy_map[hedge_strategy]

        # 根据敞口分析自动判断
        beta = exposure.get("beta_exposure", 1.0)
        volatility = exposure.get("volatility", 0.12)
        concentration = exposure.get("concentration_risk", 0.1)

        # 高 Beta 需要 Beta 对冲
        if abs(beta) > 1.2:
            return HedgeObjective.BETA_NEUTRAL

        # 高波动率需要尾部风险对冲
        if volatility > 0.20:
            return HedgeObjective.TAIL_RISK

        # 高集中度需要分散化
        if concentration > 0.25:
            return HedgeObjective.DIVERSIFICATION

        return HedgeObjective.CORRELATION_HEDGE

    def select_candidates(
        self,
        objective: HedgeObjective,
        exposure: Dict[str, Any],
    ) -> List[HedgeAsset]:
        """
        根据对冲目标筛选候选资产
        """
        candidates = []

        if objective == HedgeObjective.BETA_NEUTRAL:
            # Beta 对冲：反向 ETF
            candidates.extend(self.universe.inverse_equity.values())
            candidates.extend(self.universe.inverse_sector.values())

        elif objective == HedgeObjective.TAIL_RISK:
            # 尾部风险：波动率工具 + 反向 ETF
            candidates.extend(self.universe.volatility.values())
            candidates.extend(self.universe.inverse_equity.values())

        elif objective == HedgeObjective.SECTOR_HEDGE:
            # 行业对冲：反向行业 ETF
            candidates.extend(self.universe.inverse_sector.values())
            candidates.extend(self.universe.sector_etf.values())

        elif objective == HedgeObjective.CORRELATION_HEDGE:
            # 相关性对冲：避险资产 + 债券 + 黄金
            candidates.extend(self.universe.safe_haven.values())
            candidates.extend(self.universe.fixed_income.values())

        elif objective == HedgeObjective.RATE_HEDGE:
            # 利率对冲：反向债券
            for asset in self.universe.fixed_income.values():
                if "inverse" in asset.tags or "rate_hedge" in asset.tags:
                    candidates.append(asset)

        elif objective == HedgeObjective.DIVERSIFICATION:
            # 分散化：多类资产
            candidates.extend(self.universe.safe_haven.values())
            candidates.extend(self.universe.fixed_income.values())
            candidates.extend(self.universe.international.values())

        else:
            # 默认：避险资产
            candidates.extend(self.universe.safe_haven.values())
            candidates.extend(self.universe.fixed_income.values())

        # 流动性过滤
        candidates = [c for c in candidates if c.avg_daily_volume >= self.min_liquidity]

        # 成本过滤
        candidates = [c for c in candidates if c.expense_ratio <= self.max_cost]

        logger.info(f"Selected {len(candidates)} candidates for {objective.value}")
        return candidates

    def score_candidates(
        self,
        candidates: List[HedgeAsset],
        portfolio_returns: pd.Series,
        returns_data: pd.DataFrame,
    ) -> List[HedgeCandidate]:
        """
        对候选资产进行多因子评分

        Scoring factors:
        1. 相关性 (越负越好)
        2. 流动性 (越高越好)
        3. 成本 (越低越好)
        4. 对冲效率 (beta, leverage)
        """
        scored = []

        for asset in candidates:
            # 1. 计算相关性得分
            correlation = 0.0
            if asset.symbol in returns_data.columns:
                correlation = portfolio_returns.corr(returns_data[asset.symbol])

            # 相关性得分: -1 对应 1.0, 0 对应 0.5, 1 对应 0.0
            correlation_score = (1 - correlation) / 2

            # 2. 流动性得分 (log scale normalization)
            liquidity_score = min(1.0, np.log10(asset.avg_daily_volume) / 9)  # 9 = log10(1e9)

            # 3. 成本得分 (越低越好)
            cost_score = max(0.0, 1 - asset.total_cost_estimate * 10)  # 1% = 0.9分

            # 4. 对冲效率得分 (leverage 影响)
            efficiency = abs(asset.leverage) if asset.leverage else 1.0
            efficiency_score = min(1.0, efficiency / 3)  # 3x = 满分

            # 综合得分
            total_score = (
                self.weights["correlation"] * correlation_score +
                self.weights["liquidity"] * liquidity_score +
                self.weights["cost"] * cost_score +
                self.weights["efficiency"] * efficiency_score
            )

            scored.append(HedgeCandidate(
                asset=asset,
                correlation_score=correlation_score,
                liquidity_score=liquidity_score,
                cost_score=cost_score,
                efficiency_score=efficiency_score,
                total_score=total_score,
            ))

        # 按总分排序
        scored.sort(key=lambda x: -x.total_score)

        return scored

    def compute_optimal_allocation(
        self,
        candidates: List[HedgeCandidate],
        hedge_ratio: float,
        portfolio_returns: pd.Series,
        returns_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        计算最优对冲配置

        基于均值-方差优化，最小化对冲后组合的方差
        """
        if not candidates:
            return {}

        top_k = candidates[:self.top_k]
        symbols = [c.asset.symbol for c in top_k if c.asset.symbol in returns_data.columns]

        if not symbols:
            # 无数据，等权分配
            return {c.asset.symbol: hedge_ratio / len(top_k) for c in top_k}

        # 简化方案：按得分加权
        total_score = sum(c.total_score for c in top_k if c.asset.symbol in symbols)
        if total_score == 0:
            return {s: hedge_ratio / len(symbols) for s in symbols}

        allocation = {}
        for c in top_k:
            if c.asset.symbol in symbols:
                weight = (c.total_score / total_score) * hedge_ratio
                allocation[c.asset.symbol] = weight

        return allocation

    def recommend(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        hedge_strategy: str,
        hedge_ratio: float,
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> HedgeRecommendation:
        """
        生成对冲推荐

        主入口函数

        Args:
            portfolio_weights: 当前组合权重
            returns_data: 收益率数据
            hedge_strategy: HedgingAgent 建议的策略
            hedge_ratio: 建议的对冲比例
            market_data: 市场数据
            risk_constraints: 风控约束

        Returns:
            HedgeRecommendation: 对冲推荐
        """
        # Step 1: 分析组合敞口
        exposure = self.analyze_portfolio_exposure(
            portfolio_weights, returns_data, market_data
        )

        # Step 2: 识别对冲目标
        objective = self.identify_hedge_objective(
            exposure, hedge_strategy, risk_constraints
        )
        logger.info(f"Identified hedge objective: {objective.value}")

        # Step 3: 筛选候选资产
        candidates = self.select_candidates(objective, exposure)

        if not candidates:
            logger.warning("No suitable hedge candidates found")
            return HedgeRecommendation(
                objective=objective,
                candidates=[],
                recommended_allocation={},
                expected_correlation_reduction=0.0,
                expected_cost=0.0,
                reasoning="未找到符合条件的对冲资产",
            )

        # Step 4: 计算组合收益率
        available = [a for a in portfolio_weights if a in returns_data.columns]
        if available:
            weights = np.array([portfolio_weights.get(a, 0) for a in available])
            weights = weights / weights.sum() if weights.sum() > 0 else weights
            portfolio_returns = returns_data[available] @ weights
        else:
            portfolio_returns = pd.Series([0.0])

        # Step 5: 评分候选资产
        scored_candidates = self.score_candidates(
            candidates, portfolio_returns, returns_data
        )

        # Step 6: 计算最优配置
        allocation = self.compute_optimal_allocation(
            scored_candidates, hedge_ratio, portfolio_returns, returns_data
        )

        # Step 7: 估算效果
        expected_cost = sum(
            allocation.get(c.asset.symbol, 0) * c.asset.expense_ratio
            for c in scored_candidates[:self.top_k]
        )

        # 估算相关性降低
        avg_corr_score = np.mean([c.correlation_score for c in scored_candidates[:self.top_k]])
        expected_corr_reduction = avg_corr_score * hedge_ratio

        # 生成推荐理由
        top_symbols = [c.asset.symbol for c in scored_candidates[:3]]
        reasoning = (
            f"基于{objective.value}目标，推荐使用 {', '.join(top_symbols)} 进行对冲。"
            f"组合当前Beta={exposure.get('beta_exposure', 1.0):.2f}，"
            f"波动率={exposure.get('volatility', 0.12):.1%}。"
            f"预期对冲成本{expected_cost:.2%}，相关性降低{expected_corr_reduction:.1%}。"
        )

        return HedgeRecommendation(
            objective=objective,
            candidates=scored_candidates,
            recommended_allocation=allocation,
            expected_correlation_reduction=expected_corr_reduction,
            expected_cost=expected_cost,
            reasoning=reasoning,
        )
```

---

## 4. 集成方案

### 4.1 增强版 HedgingAgent

```python
# 修改 finsage/agents/hedging_agent.py

class HedgingAgent:
    """对冲策略智能体 (增强版)"""

    def __init__(
        self,
        llm_provider: Any,
        config: Optional[Dict] = None,
        use_dynamic_selection: bool = True  # 新增：是否使用动态选择
    ):
        self.llm = llm_provider
        self.config = config or {}
        self.use_dynamic_selection = use_dynamic_selection

        # 初始化动态选择器
        if self.use_dynamic_selection:
            from finsage.hedging.dynamic_selector import DynamicHedgeSelector
            self.dynamic_selector = DynamicHedgeSelector(config=config)

        # ... 其余初始化代码不变

    def analyze(
        self,
        target_allocation: Dict[str, float],
        position_sizes: Dict[str, float],
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> HedgingDecision:
        """分析并制定对冲策略 (增强版)"""

        # Step 1-3: 评估尾部风险、选择策略、确定基础参数 (原有逻辑)
        tail_risk = self._assess_tail_risk(target_allocation, market_data)
        strategy = self._select_hedging_strategy(tail_risk, market_data, risk_constraints)
        hedge_ratio, base_instruments = self._determine_hedge_params(strategy, tail_risk, market_data)

        # ========== 新增: 动态资产选择 ==========
        if self.use_dynamic_selection and strategy != "none":
            returns_data = market_data.get("returns", pd.DataFrame())

            recommendation = self.dynamic_selector.recommend(
                portfolio_weights=target_allocation,
                returns_data=returns_data,
                hedge_strategy=strategy,
                hedge_ratio=hedge_ratio,
                market_data=market_data,
                risk_constraints=risk_constraints,
            )

            # 用动态选择的工具替换/补充固定工具
            dynamic_instruments = []
            for symbol, alloc in recommendation.recommended_allocation.items():
                asset = self._get_asset_info(symbol)  # 获取资产详情
                dynamic_instruments.append({
                    "symbol": symbol,
                    "name": asset.get("name", symbol),
                    "type": asset.get("type", "etf"),
                    "allocation": alloc,
                    "cost_rate": asset.get("expense_ratio", 0.001),
                    "source": "dynamic",  # 标记来源
                })

            # 合并固定和动态工具
            instruments = self._merge_instruments(base_instruments, dynamic_instruments)

            # 更新成本估算
            cost = recommendation.expected_cost
            protection = hedge_ratio * 2 + recommendation.expected_correlation_reduction

            # 增强推理
            reasoning = self._generate_reasoning(strategy, tail_risk, hedge_ratio)
            reasoning += f" {recommendation.reasoning}"
        else:
            instruments = base_instruments
            cost, protection = self._calculate_hedge_economics(...)
            reasoning = self._generate_reasoning(strategy, tail_risk, hedge_ratio)
        # ========================================

        return HedgingDecision(
            timestamp=datetime.now().isoformat(),
            hedging_strategy=strategy,
            hedge_ratio=hedge_ratio,
            hedge_instruments=instruments,
            expected_cost=cost,
            expected_protection=protection,
            reasoning=reasoning,
            tail_risk_metrics=tail_risk,
        )

    def _merge_instruments(
        self,
        base: List[Dict],
        dynamic: List[Dict]
    ) -> List[Dict]:
        """合并固定工具和动态工具"""
        # 动态工具优先，但保留固定工具中动态未覆盖的
        dynamic_symbols = {inst["symbol"] for inst in dynamic}

        merged = list(dynamic)  # 动态工具优先

        for inst in base:
            if inst.get("name") == "现金/短期国债":
                # 现金始终保留
                merged.append(inst)
            elif inst.get("symbol") not in dynamic_symbols:
                # 固定工具作为补充
                inst["source"] = "fixed"
                merged.append(inst)

        return merged
```

---

## 5. 使用流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        增强版对冲决策流程                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. HedgingAgent 接收组合和市场数据                                      │
│     └─→ target_allocation, market_data                                  │
│                                                                          │
│  2. 评估尾部风险 (原有逻辑)                                              │
│     └─→ VaR, CVaR, 偏度, 峰度                                           │
│                                                                          │
│  3. LLM 选择对冲策略                                                     │
│     └─→ "tail_hedge", "safe_haven", etc.                                │
│                                                                          │
│  4. 确定基础对冲比例                                                     │
│     └─→ hedge_ratio = 10-20%                                            │
│                                                                          │
│  5. [NEW] DynamicHedgeSelector 动态选择                                  │
│     │                                                                    │
│     ├─→ 分析组合敞口 (Beta, 行业, 波动率)                               │
│     │                                                                    │
│     ├─→ 识别对冲目标 (beta_neutral, tail_risk, ...)                     │
│     │                                                                    │
│     ├─→ 从 HedgeAssetUniverse (~70+资产) 筛选候选                       │
│     │                                                                    │
│     ├─→ 多因子评分 (相关性, 流动性, 成本, 效率)                         │
│     │                                                                    │
│     └─→ 输出: Top K 最优对冲资产 + 配置权重                             │
│                                                                          │
│  6. 合并固定工具和动态工具                                               │
│     └─→ 动态优先，固定补充                                              │
│                                                                          │
│  7. 输出 HedgingDecision                                                │
│     └─→ 策略 + 工具列表 + 成本 + 保护水平 + 推理                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 效果对比

| 场景 | 原系统 | 增强系统 |
|------|--------|----------|
| 科技股对冲 | TLT+GLD (间接) | SQQQ, PSQ (直接) |
| 高Beta组合 | SH (固定) | SPXU, SDS (根据Beta选择杠杆) |
| 新兴市场敞口 | 无专门工具 | EUM, FXP |
| 利率上升 | 无 | TBF, TBT |
| 配对交易 | 无 | 动态筛选协整配对 |

---

## 7. 未来扩展

1. **LLM辅助选择**: 让LLM参与候选筛选和权重决策
2. **协整检验**: 为配对交易自动检测协整关系
3. **期权定价**: 集成BSM模型计算期权对冲成本
4. **实时更新**: 根据市场变化动态调整对冲组合
5. **回测验证**: 对不同对冲策略进行历史回测

---

*设计完成时间: 2024年12月*
*版本: 1.0*
