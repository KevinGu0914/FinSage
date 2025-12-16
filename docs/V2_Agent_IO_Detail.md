# FinSage V2 智能体输入输出详解

## 1. 专家智能体 (Expert Agents)

所有专家继承自 `BaseExpert`，共享相同的输入输出结构。

### 1.1 StockExpert (股票专家)

**输入:**
```python
market_data: Dict[str, Any]
# 示例:
{
    "SPY": {"close": 450.25, "open": 448.0, "high": 451.0, "low": 447.5, "volume": 85000000, "change_pct": 0.5},
    "QQQ": {"close": 380.50, "open": 378.0, "high": 382.0, "low": 377.0, "volume": 45000000, "change_pct": 0.66},
    "AAPL": {...},
    "NVDA": {...},
}

news_data: List[Dict]
# 示例:
[
    {"title": "Fed signals rate cut", "published": "2024-10-15", "sentiment": "positive", "symbols": ["SPY"]},
    {"title": "Tech earnings beat expectations", "sentiment": "positive", "symbols": ["QQQ", "AAPL"]},
]

technical_indicators: Dict[str, Any]
# 示例:
{
    "SPY": {"rsi_14": 58.5, "macd": 2.3, "macd_signal": 1.8, "sma_20": 445.0, "sma_50": 440.0},
    "QQQ": {"rsi_14": 62.0, "macd": 3.1, "macd_signal": 2.5, "sma_20": 375.0, "sma_50": 368.0},
}

macro_data: Optional[Dict[str, Any]]  # 可选
# 示例:
{
    "vix": 18.5,
    "sp500_pe": 22.3,
    "market_breadth": 0.65,
}
```

**输出:**
```python
ExpertReport(
    expert_name="Stock Expert",
    asset_class="stocks",
    timestamp="2024-10-15T14:30:00",
    recommendations=[
        ExpertRecommendation(
            asset_class="stocks",
            symbol="SPY",
            action=Action.BUY_50,  # V2支持: SHORT_25/50/75/100
            confidence=0.75,
            target_weight=0.12,   # 负数表示做空
            reasoning="Strong momentum with Fed support",
            market_view={"trend": "bullish", "momentum": "positive"},
            risk_assessment={"volatility_risk": 0.15, "downside_risk": 0.08},
        ),
        ExpertRecommendation(
            asset_class="stocks",
            symbol="NVDA",
            action=Action.BUY_75,
            confidence=0.80,
            target_weight=0.10,
            reasoning="AI demand driving growth",
            market_view={"trend": "bullish", "sector": "technology"},
            risk_assessment={"volatility_risk": 0.25, "downside_risk": 0.12},
        ),
    ],
    overall_view="bullish",  # bullish / bearish / neutral
    sector_allocation={"SPY": 0.30, "QQQ": 0.25, "NVDA": 0.25, "AAPL": 0.20},
    key_factors=["Fed dovish stance", "Strong earnings", "AI momentum"],
)
```

---

### 1.2 BondExpert (债券专家)

**输入:**
```python
market_data: Dict[str, Any]
# 示例:
{
    "TLT": {"close": 92.45, "change_pct": -0.85, "yield": 4.85},
    "IEF": {"close": 95.20, "change_pct": -0.42, "yield": 4.25},
    "SHY": {"close": 82.10, "change_pct": -0.05, "yield": 5.10},
    "rates": {
        "fed_funds": 5.25,
        "treasury_2y": 4.95,
        "treasury_10y": 4.65,
        "treasury_30y": 4.85,
        "spread_2s10s": -30,  # 倒挂 bps
    }
}

news_data: List[Dict]
# 示例:
[
    {"title": "FOMC minutes reveal hawkish stance", "sentiment": "negative"},
    {"title": "Inflation remains sticky", "sentiment": "negative"},
]

technical_indicators: Dict[str, Any]
# 示例:
{
    "TLT": {"rsi_14": 35.2, "macd": -1.25, "trend": "downtrend"},
}
```

**输出:**
```python
ExpertReport(
    expert_name="Bond Expert",
    asset_class="bonds",
    timestamp="2024-10-15T14:30:00",
    recommendations=[
        ExpertRecommendation(
            asset_class="bonds",
            symbol="TLT",
            action=Action.SHORT_50,  # V2关键: 利率上升时做空长债
            confidence=0.70,
            target_weight=-0.08,    # 负权重 = 做空
            reasoning="Rising rates pressure long-duration bonds",
            market_view={
                "rate_view": "hawkish",
                "duration_preference": "short",
                "credit_view": "neutral",
            },
            risk_assessment={"duration_risk": 0.35, "credit_risk": 0.05},
        ),
        ExpertRecommendation(
            asset_class="bonds",
            symbol="SHY",
            action=Action.BUY_25,
            confidence=0.65,
            target_weight=0.05,
            reasoning="Short duration safer in rising rate environment",
            market_view={"rate_view": "hawkish", "duration_preference": "short"},
            risk_assessment={"duration_risk": 0.08, "credit_risk": 0.02},
        ),
    ],
    overall_view="bearish",
    sector_allocation={"TLT": -0.40, "SHY": 0.35, "IEF": 0.25},
    key_factors=["Fed hawkish stance", "Inverted yield curve", "Rising long-term rates"],
)
```

---

### 1.3 CommodityExpert (商品专家)

**输入:**
```python
market_data: Dict[str, Any]
# 示例:
{
    "GLD": {"close": 185.50, "change_pct": 0.65, "volume": 12000000},
    "SLV": {"close": 21.80, "change_pct": 0.92, "volume": 8000000},
    "USO": {"close": 72.30, "change_pct": -1.25, "volume": 8500000},
    "commodities": {
        "gold_spot": 1985.50,
        "silver_spot": 23.45,
        "crude_wti": 82.50,
        "natural_gas": 2.85,
        "copper": 3.75,
    }
}

news_data: List[Dict]
# 示例:
[
    {"title": "OPEC+ considers production cuts", "sentiment": "positive", "symbols": ["USO"]},
    {"title": "Gold rises on safe-haven demand", "sentiment": "positive", "symbols": ["GLD"]},
]

macro_data: Dict[str, Any]
# 示例:
{
    "dollar_index": 105.5,
    "inflation_expectation": 2.8,
    "real_rates": 1.5,
}
```

**输出:**
```python
ExpertReport(
    expert_name="Commodity Expert",
    asset_class="commodities",
    timestamp="2024-10-15T14:30:00",
    recommendations=[
        ExpertRecommendation(
            asset_class="commodities",
            symbol="GLD",
            action=Action.BUY_50,
            confidence=0.72,
            target_weight=0.08,
            reasoning="Safe haven demand amid market uncertainty",
            market_view={"gold_view": "bullish", "dollar_impact": "neutral"},
            risk_assessment={"volatility_risk": 0.12, "currency_risk": 0.08},
        ),
        ExpertRecommendation(
            asset_class="commodities",
            symbol="USO",
            action=Action.SHORT_25,  # V2: 做空原油
            confidence=0.55,
            target_weight=-0.03,    # 负权重
            reasoning="Demand concerns outweigh OPEC cuts",
            market_view={"oil_view": "bearish", "supply_demand": "oversupply"},
            risk_assessment={"volatility_risk": 0.25, "geopolitical_risk": 0.30},
        ),
    ],
    overall_view="neutral",
    sector_allocation={"GLD": 0.50, "SLV": 0.20, "USO": -0.15, "DBA": 0.15},
    key_factors=["Dollar strength", "Safe haven flows", "Oil demand uncertainty"],
)
```

---

### 1.4 REITsExpert (房地产信托专家)

**输入:**
```python
market_data: Dict[str, Any]
# 示例:
{
    "VNQ": {"close": 82.50, "change_pct": -0.95, "dividend_yield": 4.25},
    "DLR": {"close": 125.80, "change_pct": 0.45, "dividend_yield": 3.85, "sector": "data_center"},
    "EQIX": {"close": 780.50, "change_pct": 0.62, "dividend_yield": 2.10, "sector": "data_center"},
    "rates": {
        "mortgage_30y": 7.25,
        "treasury_10y": 4.65,
    }
}

news_data: List[Dict]
# 示例:
[
    {"title": "Office vacancy rates hit record high", "sentiment": "negative", "sector": "office"},
    {"title": "Data center demand surges on AI boom", "sentiment": "positive", "sector": "data_center"},
]
```

**输出:**
```python
ExpertReport(
    expert_name="REITs Expert",
    asset_class="reits",
    timestamp="2024-10-15T14:30:00",
    recommendations=[
        ExpertRecommendation(
            asset_class="reits",
            symbol="VNQ",
            action=Action.SHORT_25,  # V2: 做空广泛REITs
            confidence=0.60,
            target_weight=-0.03,
            reasoning="Rising rates and office vacancy concerns",
            market_view={"sector_view": "bearish", "rate_sensitivity": "high"},
            risk_assessment={"rate_risk": 0.35, "vacancy_risk": 0.25},
        ),
        ExpertRecommendation(
            asset_class="reits",
            symbol="DLR",
            action=Action.BUY_50,
            confidence=0.75,
            target_weight=0.05,
            reasoning="Data center demand driven by AI infrastructure",
            market_view={"sector_view": "bullish", "growth_driver": "AI/cloud"},
            risk_assessment={"rate_risk": 0.20, "tech_dependency": 0.15},
        ),
    ],
    overall_view="neutral",
    sector_allocation={"DLR": 0.45, "EQIX": 0.35, "VNQ": -0.10, "IYR": -0.10},
    key_factors=["High interest rates", "Office sector weakness", "Data center strength"],
)
```

---

### 1.5 CryptoExpert (加密货币专家)

**输入:**
```python
market_data: Dict[str, Any]
# 示例:
{
    "BTC-USD": {
        "close": 42500.00,
        "change_pct": 2.35,
        "volume_24h": 28000000000,
        "market_cap": 830000000000,
        "dominance": 52.5,
    },
    "ETH-USD": {
        "close": 2250.00,
        "change_pct": 3.15,
        "volume_24h": 15000000000,
        "gas_price": 25,
    },
    "market_metrics": {
        "total_market_cap": 1580000000000,
        "fear_greed_index": 65,
        "btc_dominance": 52.5,
    }
}

news_data: List[Dict]
# 示例:
[
    {"title": "Bitcoin ETF approval speculation intensifies", "sentiment": "positive"},
    {"title": "Ethereum Shanghai upgrade successful", "sentiment": "positive"},
]

on_chain_data: Optional[Dict]  # 可选
# 示例:
{
    "btc_exchange_netflow": -15000,  # 负数=流出交易所
    "eth_staking_ratio": 0.22,
    "whale_activity": "accumulating",
}
```

**输出:**
```python
ExpertReport(
    expert_name="Crypto Expert",
    asset_class="crypto",
    timestamp="2024-10-15T14:30:00",
    recommendations=[
        ExpertRecommendation(
            asset_class="crypto",
            symbol="BTC-USD",
            action=Action.BUY_75,
            confidence=0.70,
            target_weight=0.04,
            reasoning="ETF approval momentum, exchange outflows bullish",
            market_view={"trend": "bullish", "catalyst": "ETF_approval", "on_chain": "accumulation"},
            risk_assessment={"volatility_risk": 0.45, "regulatory_risk": 0.25},
        ),
        ExpertRecommendation(
            asset_class="crypto",
            symbol="ETH-USD",
            action=Action.BUY_50,
            confidence=0.65,
            target_weight=0.03,
            reasoning="Network upgrade success, growing staking",
            market_view={"trend": "bullish", "network_health": "strong"},
            risk_assessment={"volatility_risk": 0.50, "smart_contract_risk": 0.15},
        ),
    ],
    overall_view="bullish",
    sector_allocation={"BTC-USD": 0.60, "ETH-USD": 0.40},
    key_factors=["Bitcoin ETF catalyst", "Institutional accumulation", "Network upgrades"],
)
```

---

## 2. Portfolio Manager (组合管理器)

**输入:**
```python
expert_reports: Dict[str, ExpertReport]
# 示例:
{
    "stocks": ExpertReport(...),   # 来自 StockExpert
    "bonds": ExpertReport(...),    # 来自 BondExpert
    "commodities": ExpertReport(...),
    "reits": ExpertReport(...),
    "crypto": ExpertReport(...),
}

portfolio_state: PortfolioState
# 当前组合状态 (详见第4节)

risk_constraints: Dict
# 示例:
{
    "max_single_asset": 0.15,
    "max_asset_class": 0.50,
    "max_short_exposure": 0.30,  # V2新增: 最大做空敞口
    "target_volatility": 0.12,
    "max_drawdown": 0.15,
}

allocation_bounds: Dict[str, Dict]
# 示例:
{
    "stocks": {"min": 0.30, "max": 0.50},
    "bonds": {"min": -0.20, "max": 0.35},  # V2: min可为负数
    "commodities": {"min": -0.10, "max": 0.25},
    "reits": {"min": -0.05, "max": 0.15},
    "crypto": {"min": 0.00, "max": 0.10},
    "cash": {"min": 0.05, "max": 0.20},
}
```

**输出:**
```python
target_allocation: Dict[str, float]
# 示例 (V2: 负权重表示做空):
{
    "SPY": 0.12,
    "QQQ": 0.10,
    "NVDA": 0.08,
    "AAPL": 0.06,
    "TLT": -0.08,   # 做空
    "IEF": -0.04,   # 做空
    "SHY": 0.05,
    "GLD": 0.08,
    "USO": -0.03,   # 做空
    "DLR": 0.05,
    "BTC-USD": 0.04,
    "ETH-USD": 0.03,
    "cash": 0.10,
}

trade_orders: List[TradeOrder]
# 示例:
[
    TradeOrder(symbol="TLT", action="SHORT", shares=100, target_weight=-0.08),
    TradeOrder(symbol="SPY", action="BUY", shares=50, target_weight=0.12),
    TradeOrder(symbol="USO", action="SHORT", shares=200, target_weight=-0.03),
]

optimization_report: Dict
# 示例:
{
    "expected_return": 0.085,
    "expected_volatility": 0.118,
    "sharpe_ratio": 0.72,
    "max_drawdown_estimate": 0.12,
    "gross_exposure": 1.15,   # V2新增: 总敞口
    "net_exposure": 0.85,     # V2新增: 净敞口
    "short_exposure": 0.15,   # V2新增: 空头敞口
}
```

---

## 3. Risk Controller (风控控制器)

**输入:**
```python
proposed_allocation: Dict[str, float]
# 来自 Portfolio Manager 的建议配置 (含负权重)
# 示例:
{
    "SPY": 0.15,
    "TLT": -0.10,  # 做空请求
    "GLD": 0.08,
}

portfolio_state: PortfolioState
# 当前组合状态

historical_returns: pd.DataFrame
# 历史收益率数据 (252天)

risk_params: Dict
# 示例 (V2新增做空相关参数):
{
    "max_single_asset": 0.15,
    "max_asset_class": 0.50,
    "max_var_95": 0.03,
    "max_drawdown": 0.15,
    # V2 新增
    "max_short_single": 0.10,    # 单一做空上限
    "max_total_short": 0.30,     # 总做空上限
    "margin_requirement": 0.50,   # 保证金比例
}
```

**输出:**
```python
adjusted_allocation: Dict[str, float]
# 风险调整后的配置
# 示例:
{
    "SPY": 0.15,      # 保持不变
    "TLT": -0.08,     # 从-0.10调整到-0.08 (超出单一做空上限)
    "GLD": 0.08,      # 保持不变
}

risk_report: Dict
# 示例:
{
    "checks_passed": True,
    "violations": [],
    "warnings": [
        {
            "type": "SHORT_CONCENTRATION",
            "message": "TLT short reduced from 10% to 8%",
            "original": -0.10,
            "adjusted": -0.08,
        }
    ],
    "risk_metrics": {
        "portfolio_var_95": 0.025,
        "expected_shortfall": 0.032,
        "gross_exposure": 1.12,
        "net_exposure": 0.88,
        "short_exposure": 0.12,
        "margin_utilization": 0.24,  # V2新增
        "diversification_score": 0.85,
    },
    # V2新增: 做空风险检查
    "short_checks": {
        "single_short_ok": True,      # 单一做空 <= 10%
        "total_short_ok": True,       # 总做空 <= 30%
        "margin_ok": True,            # 保证金充足
        "borrow_available": True,     # 可借券
        "short_squeeze_risk": "low",  # 逼空风险
    },
    "stress_tests": {
        "market_crash_10pct": -0.065,
        "rate_spike_100bps": 0.015,   # V2: 做空债券获利
        "volatility_spike": -0.042,
    },
}
```

---

## 4. Trading Environment (交易环境)

### 4.1 MultiAssetEnv

**输入:**
```python
config: FinSageConfig
# 系统配置

market_data_stream: Dict
# 每日市场数据
# 示例:
{
    "date": "2024-10-15",
    "prices": {
        "SPY": 450.25,
        "TLT": 92.45,
        "GLD": 185.50,
    },
    "volumes": {...},
    "news": [...],
}

expert_weights: Dict
# 专家权重建议
# 示例:
{
    "SPY": {"action": "BUY_50%", "confidence": 0.75},
    "TLT": {"action": "SHORT_50%", "confidence": 0.70},  # V2
    "GLD": {"action": "BUY_25%", "confidence": 0.65},
}
```

**step() 输出:**
```python
observation: Dict
# 新的市场状态

reward: float
# 当日收益

done: bool
# 是否结束

info: Dict
# 示例:
{
    "date": "2024-10-15",
    "portfolio_value": 1005230.50,
    "daily_return": 0.0052,
    "trades_executed": [
        {"symbol": "TLT", "action": "SHORT", "shares": 100, "price": 92.45},
        {"symbol": "SPY", "action": "BUY", "shares": 25, "price": 450.25},
    ],
    "positions": {
        "SPY": {"shares": 225, "value": 101306.25, "weight": 0.101},
        "TLT": {"shares": -100, "value": -9245.00, "weight": -0.009},  # V2: 空头
        "GLD": {"shares": 150, "value": 27825.00, "weight": 0.028},
    },
    # V2新增
    "short_metrics": {
        "short_market_value": 9245.00,
        "margin_used": 4622.50,
        "borrowing_cost_daily": 0.51,
    },
}
```

---

### 4.2 PortfolioState

**Position 结构:**
```python
@dataclass
class Position:
    symbol: str
    shares: float           # V2关键: 正数=多头, 负数=空头
    avg_cost: float
    current_price: float
    asset_class: str

    # V2新增属性
    @property
    def is_short(self) -> bool:
        return self.shares < 0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price  # 空头为负

    @property
    def unrealized_pnl(self) -> float:
        if self.is_short:
            # 空头: 价格下跌 = 盈利
            return abs(self.shares) * (self.avg_cost - self.current_price)
        else:
            # 多头: 价格上涨 = 盈利
            return self.shares * (self.current_price - self.avg_cost)

    @property
    def margin_requirement(self) -> float:
        if self.is_short:
            return abs(self.market_value) * 0.5  # 50%保证金
        return 0.0
```

**PortfolioState 结构:**
```python
@dataclass
class PortfolioState:
    initial_capital: float = 1_000_000.0
    cash: float = 1_000_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    value_history: List[Dict] = field(default_factory=list)

    # V2新增
    short_borrow_rate: float = 0.02   # 年化借股费率 2%
    short_margin_ratio: float = 0.5   # 空头保证金比例 50%

    # V2新增属性
    @property
    def long_market_value(self) -> float:
        """多头总市值"""
        return sum(pos.market_value for pos in self.positions.values() if not pos.is_short)

    @property
    def short_market_value(self) -> float:
        """空头总市值 (负数)"""
        return sum(pos.market_value for pos in self.positions.values() if pos.is_short)

    @property
    def gross_exposure(self) -> float:
        """总敞口 = |多头| + |空头|"""
        return self.long_market_value + abs(self.short_market_value)

    @property
    def net_exposure(self) -> float:
        """净敞口 = 多头 - |空头|"""
        return self.long_market_value - abs(self.short_market_value)

    @property
    def short_margin_required(self) -> float:
        """空头所需保证金"""
        return abs(self.short_market_value) * self.short_margin_ratio
```

---

## 5. V2 做空机制

### 5.1 Action 枚举 (13-Action Space)

```python
class Action(Enum):
    # 做空动作 (V2新增)
    SHORT_100 = "SHORT_100%"   # multiplier = -1.0
    SHORT_75 = "SHORT_75%"     # multiplier = -0.75
    SHORT_50 = "SHORT_50%"     # multiplier = -0.5
    SHORT_25 = "SHORT_25%"     # multiplier = -0.25

    # 卖出/减持
    SELL_100 = "SELL_100%"     # multiplier = 0.0
    SELL_75 = "SELL_75%"       # multiplier = 0.25
    SELL_50 = "SELL_50%"       # multiplier = 0.5
    SELL_25 = "SELL_25%"       # multiplier = 0.75

    # 持有
    HOLD = "HOLD"              # multiplier = 1.0

    # 买入/加仓
    BUY_25 = "BUY_25%"         # multiplier = 1.25
    BUY_50 = "BUY_50%"         # multiplier = 1.5
    BUY_75 = "BUY_75%"         # multiplier = 1.75
    BUY_100 = "BUY_100%"       # multiplier = 2.0
```

### 5.2 做空交易类型

| 交易类型 | 触发条件 | 说明 |
|----------|----------|------|
| `SHORT` | shares < 0, 无现有仓位 | 开新空仓 |
| `ADD_SHORT` | shares < 0, 有空头仓位 | 加空仓 |
| `COVER_SHORT` | shares > 0, 有空头仓位 | 平空仓 |
| `SELL_AND_SHORT` | shares < 0, 有多头仓位, is_short=True | 先卖后空 |
| `COVER_AND_BUY` | shares > 平空所需, 有空头仓位 | 先平空后做多 |

### 5.3 空头盈亏计算

```
多头 PnL = shares × (current_price - avg_cost)
         价格上涨 → 盈利

空头 PnL = |shares| × (avg_cost - current_price)
         价格下跌 → 盈利
```

### 5.4 保证金和借股成本

| 项目 | 数值 | 公式 |
|------|------|------|
| 保证金比例 | 50% | `margin = |short_value| × 0.5` |
| 借股费率 | 2% 年化 | `daily_cost = |short_value| × 0.02 / 365` |

---

## 6. 完整系统架构图

### 6.1 系统总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FinSage V2 Multi-Agent System                              │
│                                  (支持做空的多资产配置系统)                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA LAYER (数据层)                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   FMP API   │  │  Yahoo Fin  │  │  News API   │  │ Technical   │  │   Macro     │   │
│  │  (Prices)   │  │  (Backup)   │  │ (Sentiment) │  │ Indicators  │  │   Data      │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │                │          │
│         └────────────────┴────────────────┴────────────────┴────────────────┘          │
│                                           │                                             │
│                              ┌────────────▼────────────┐                               │
│                              │   DataProvider (统一)    │                               │
│                              │   market_data, news,    │                               │
│                              │   technical, macro      │                               │
│                              └────────────┬────────────┘                               │
└──────────────────────────────────────────┼──────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────────────────────────┐
│                               EXPERT LAYER (专家层)                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │  StockExpert    │ │  BondExpert     │ │ CommodityExpert │ │  REITsExpert    │       │
│  │                 │ │                 │ │                 │ │                 │       │
│  │ Input:          │ │ Input:          │ │ Input:          │ │ Input:          │       │
│  │ - prices        │ │ - prices        │ │ - prices        │ │ - prices        │       │
│  │ - news          │ │ - rates         │ │ - commodities   │ │ - mortgage rate │       │
│  │ - technicals    │ │ - FOMC news     │ │ - dollar index  │ │ - vacancy news  │       │
│  │ - VIX           │ │ - inflation     │ │ - geopolitics   │ │ - sector news   │       │
│  │                 │ │                 │ │                 │ │                 │       │
│  │ Output:         │ │ Output:         │ │ Output:         │ │ Output:         │       │
│  │ ExpertReport    │ │ ExpertReport    │ │ ExpertReport    │ │ ExpertReport    │       │
│  │ - BUY/HOLD/SELL │ │ - BUY/HOLD/SELL │ │ - BUY/HOLD/SELL │ │ - BUY/HOLD/SELL │       │
│  │ - SHORT (V2)    │ │ - SHORT (V2)    │ │ - SHORT (V2)    │ │ - SHORT (V2)    │       │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘ └────────┬────────┘       │
│           │                   │                   │                   │                │
│           │  ┌────────────────┴────────────────┐  │                   │                │
│           │  │         CryptoExpert            │  │                   │                │
│           │  │                                 │  │                   │                │
│           │  │ Input: prices, on-chain, news   │  │                   │                │
│           │  │ Output: ExpertReport            │  │                   │                │
│           │  └────────────────┬────────────────┘  │                   │                │
│           │                   │                   │                   │                │
│           └───────────────────┴───────────────────┴───────────────────┘                │
│                                           │                                             │
│                              ┌────────────▼────────────┐                               │
│                              │  5个 ExpertReport 汇总   │                               │
│                              │  含 recommendations[]   │                               │
│                              │  含 SHORT 信号 (V2)     │                               │
│                              └────────────┬────────────┘                               │
└──────────────────────────────────────────┼──────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────────────────────────┐
│                            DECISION LAYER (决策层)                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          Portfolio Manager (组合管理器)                          │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │  Input:                              │  Output:                                 │   │
│  │  ├── expert_reports (5个专家报告)     │  ├── target_allocation                   │   │
│  │  ├── portfolio_state (当前状态)       │  │   ├── SPY: 0.12                       │   │
│  │  ├── risk_constraints                │  │   ├── TLT: -0.08  ← V2: 负权重=做空   │   │
│  │  └── allocation_bounds               │  │   ├── GLD: 0.08                       │   │
│  │                                      │  │   └── ...                             │   │
│  │                                      │  ├── trade_orders                        │   │
│  │  Process:                            │  └── optimization_report                 │   │
│  │  1. 聚合专家建议                      │      ├── gross_exposure: 1.15            │   │
│  │  2. 计算 action_multiplier           │      ├── net_exposure: 0.85              │   │
│  │     SHORT_50% → -0.5                 │      └── short_exposure: 0.15            │   │
│  │  3. 生成目标权重                      │                                          │   │
│  │                                      │                                          │   │
│  └──────────────────────────────────────┴──────────────────────────────────────────┘   │
│                                           │                                             │
│                              ┌────────────▼────────────┐                               │
│                              │   target_allocation     │                               │
│                              │   (含负权重做空配置)      │                               │
│                              └────────────┬────────────┘                               │
│                                           │                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          Risk Controller (风控控制器)                            │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │  Input:                              │  Output:                                 │   │
│  │  ├── proposed_allocation             │  ├── adjusted_allocation                 │   │
│  │  ├── portfolio_state                 │  │   (风险调整后的配置)                    │   │
│  │  ├── historical_returns              │  │                                       │   │
│  │  └── risk_params                     │  └── risk_report                         │   │
│  │      ├── max_single_asset: 0.15      │      ├── checks_passed                   │   │
│  │      ├── max_short_single: 0.10 (V2) │      ├── violations                      │   │
│  │      ├── max_total_short: 0.30 (V2)  │      ├── risk_metrics                    │   │
│  │      └── margin_requirement: 0.50    │      │   ├── VaR_95                       │   │
│  │                                      │      │   ├── margin_utilization          │   │
│  │  Checks:                             │      │   └── short_exposure               │   │
│  │  ├── 单一资产上限检查                 │      └── short_checks (V2)               │   │
│  │  ├── 单一做空上限检查 (V2)            │          ├── single_short_ok             │   │
│  │  ├── 总做空敞口检查 (V2)              │          ├── total_short_ok              │   │
│  │  ├── 保证金充足检查 (V2)              │          └── margin_ok                   │   │
│  │  └── VaR限制检查                     │                                          │   │
│  │                                      │                                          │   │
│  └──────────────────────────────────────┴──────────────────────────────────────────┘   │
│                                           │                                             │
│                              ┌────────────▼────────────┐                               │
│                              │  adjusted_allocation    │                               │
│                              │  (风险调整后的最终配置)   │                               │
│                              └────────────┬────────────┘                               │
└──────────────────────────────────────────┼──────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────────────────────────┐
│                            EXECUTION LAYER (执行层)                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           MultiAssetEnv (交易环境)                               │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │  Input:                              │  step() Output:                          │   │
│  │  ├── config                          │  ├── observation (新市场状态)             │   │
│  │  ├── market_data_stream              │  ├── reward (当日收益)                   │   │
│  │  └── adjusted_allocation             │  ├── done (是否结束)                     │   │
│  │                                      │  └── info                                │   │
│  │                                      │      ├── portfolio_value                 │   │
│  │  _rebalance():                       │      ├── trades_executed                 │   │
│  │  ├── 计算目标股数                     │      ├── positions                       │   │
│  │  ├── 检测做空信号 (target < 0)        │      └── short_metrics (V2)             │   │
│  │  ├── 执行 BUY/SELL/SHORT             │          ├── short_market_value          │   │
│  │  └── 记录交易                         │          ├── margin_used                 │   │
│  │                                      │          └── borrowing_cost_daily        │   │
│  └──────────────────────────────────────┴──────────────────────────────────────────┘   │
│                                           │                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           PortfolioState (组合状态)                              │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                         Position (持仓)                                  │   │   │
│  │  ├─────────────────────────────────────────────────────────────────────────┤   │   │
│  │  │  symbol: str        shares: float (正=多头, 负=空头)                     │   │   │
│  │  │  avg_cost: float    current_price: float    asset_class: str            │   │   │
│  │  │                                                                         │   │   │
│  │  │  V2 属性:                                                                │   │   │
│  │  │  ├── is_short → shares < 0                                              │   │   │
│  │  │  ├── market_value → shares × price (空头为负)                            │   │   │
│  │  │  ├── unrealized_pnl → 多头: (price-cost)×shares                         │   │   │
│  │  │  │                    空头: (cost-price)×|shares| ← 价格跌=盈利          │   │   │
│  │  │  └── margin_requirement → |market_value| × 0.5 (仅空头)                  │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                 │   │
│  │  State:                              │  V2 新增属性:                            │   │
│  │  ├── cash                            │  ├── long_market_value (多头市值)        │   │
│  │  ├── positions: Dict[str, Position]  │  ├── short_market_value (空头市值,负)    │   │
│  │  ├── trade_history                   │  ├── gross_exposure (总敞口)             │   │
│  │  ├── value_history                   │  ├── net_exposure (净敞口)               │   │
│  │  ├── short_borrow_rate: 0.02 (V2)    │  └── short_margin_required              │   │
│  │  └── short_margin_ratio: 0.5 (V2)    │                                          │   │
│  │                                      │                                          │   │
│  │  execute_trade() 交易类型:                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  BUY         │ shares > 0, 无空头  │ 买入/加仓多头                       │   │   │
│  │  │  SELL        │ shares < 0, 有多头  │ 卖出/减仓多头                       │   │   │
│  │  │  SHORT       │ shares < 0, 无仓位  │ 开新空仓 (V2)                       │   │   │
│  │  │  ADD_SHORT   │ shares < 0, 有空头  │ 加空仓 (V2)                         │   │   │
│  │  │  COVER_SHORT │ shares > 0, 有空头  │ 平空仓 (V2)                         │   │   │
│  │  │  SELL_AND_SHORT │ 卖完多头继续做空 │ 先卖后空 (V2)                        │   │   │
│  │  │  COVER_AND_BUY  │ 平完空头继续做多 │ 先平空后做多 (V2)                    │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 V2 做空数据流

```
                              ┌─────────────────────────────┐
                              │      BondExpert 分析        │
                              │  "利率上升，长债承压"         │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │   ExpertRecommendation      │
                              │   symbol: "TLT"             │
                              │   action: SHORT_50%         │
                              │   confidence: 0.70          │
                              │   target_weight: -0.08      │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │     Portfolio Manager       │
                              │   action_multiplier = -0.5  │
                              │   weight = -0.5 × 0.7 × W   │
                              │   target: TLT = -0.08       │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │      Risk Controller        │
                              │   检查: -0.08 < -0.10 ✓     │
                              │   (单一做空上限10%)          │
                              │   通过! adjusted = -0.08    │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │       MultiAssetEnv         │
                              │   target_weight < 0         │
                              │   is_short = True           │
                              │   shares = 负数             │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │  PortfolioState.execute_trade│
                              │  action = "SHORT"           │
                              │  借入TLT → 卖出 → 收现金     │
                              │  Position(shares=-100)      │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │        持仓状态             │
                              │  TLT: shares = -100         │
                              │  avg_cost = $92.45          │
                              │  is_short = True            │
                              │  margin_req = $4,622.50     │
                              └──────────────┬──────────────┘
                                             │
                           ┌─────────────────┴─────────────────┐
                           │                                   │
                           ▼                                   ▼
              ┌─────────────────────┐             ┌─────────────────────┐
              │   价格下跌 $92→$87   │             │   价格上涨 $92→$97   │
              │                     │             │                     │
              │   unrealized_pnl    │             │   unrealized_pnl    │
              │   = 100×(92-87)     │             │   = 100×(92-97)     │
              │   = +$500 盈利      │             │   = -$500 亏损      │
              └─────────────────────┘             └─────────────────────┘
```

### 6.3 13-Action Space 映射

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         13-Action Trading Space                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   做空区间 (V2新增)              │  减仓区间           │  加仓区间            │
│   ◄────────────────────────────│────────────────────│────────────────────► │
│                                │                    │                      │
│   -1.0   -0.75   -0.5   -0.25  │ 0.0  0.25 0.5 0.75│ 1.0  1.25 1.5 1.75 2.0│
│    │       │       │       │   │  │    │    │    │ │  │    │    │    │    │
│    │       │       │       │   │  │    │    │    │ │  │    │    │    │    │
│  SHORT  SHORT  SHORT  SHORT │SELL SELL SELL SELL│HOLD BUY  BUY  BUY  BUY │
│  100%   75%    50%    25%   │100% 75%  50%  25% │     25%  50%  75%  100%│
│                              │                    │                       │
│  ◄─────── 负权重 ──────────────│─────── 0 ─────────│────── 正权重 ────────►│
│                              │                    │                       │
└─────────────────────────────────────────────────────────────────────────────┘

权重计算公式:
  final_weight = base_weight × action_multiplier × confidence

示例:
  股票类别权重 = 0.40
  专家建议 TLT: SHORT_50%, confidence = 0.70

  action_multiplier = -0.5
  final_weight = 0.40 × (-0.5) × 0.70 = -0.14

  → 做空 TLT，目标权重 -14%
```
