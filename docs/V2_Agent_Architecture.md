# FinSage V2 智能体输入输出详解

## 1. 专家智能体 (Expert Agents)

### 1.1 StockExpert (股票专家)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `market_data` | `Dict[str, Any]` | 股票价格数据 (close, open, high, low, volume, change_pct) |
| `news_data` | `List[Dict]` | 新闻列表 (title, published, sentiment, symbols) |
| `technical_indicators` | `Dict[str, Any]` | 技术指标 (rsi_14, macd, sma_20, sma_50, bollinger_upper/lower) |
| `macro_data` | `Dict[str, Any]` | 宏观数据 (vix, sp500_pe, market_breadth) - 可选 |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `ExpertReport.expert_name` | `str` | "Stock Expert" |
| `ExpertReport.asset_class` | `str` | "stocks" |
| `ExpertReport.timestamp` | `str` | ISO格式时间戳 |
| `ExpertReport.recommendations` | `List[ExpertRecommendation]` | 建议列表 |
| `ExpertReport.overall_view` | `str` | "bullish" / "bearish" / "neutral" |
| `ExpertReport.sector_allocation` | `Dict[str, float]` | 细分配置 |
| `ExpertReport.key_factors` | `List[str]` | 关键因素 |

**ExpertRecommendation 结构:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | `str` | 股票代码 (SPY, QQQ, AAPL...) |
| `action` | `Action` | **V2新增SHORT**: BUY_25%~100% / HOLD / SELL_25%~100% / SHORT_25%~100% |
| `confidence` | `float` | 置信度 [0, 1] |
| `target_weight` | `float` | 目标权重 (负数=做空) |
| `reasoning` | `str` | 决策理由 |
| `market_view` | `Dict` | {trend, momentum, sector_preference} |
| `risk_assessment` | `Dict` | {volatility_risk, downside_risk} |

---

### 1.2 BondExpert (债券专家)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `market_data` | `Dict[str, Any]` | 债券ETF价格 + 利率数据 |
| `market_data["rates"]` | `Dict` | {fed_funds, treasury_2y, treasury_10y, treasury_30y, spread_2s10s} |
| `news_data` | `List[Dict]` | FOMC/利率相关新闻 |
| `technical_indicators` | `Dict[str, Any]` | 技术指标 |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `ExpertReport.expert_name` | `str` | "Bond Expert" |
| `ExpertReport.asset_class` | `str` | "bonds" |
| `ExpertReport.recommendations` | `List[ExpertRecommendation]` | 建议列表 |
| `ExpertReport.overall_view` | `str` | "bullish" / "bearish" / "neutral" |

**ExpertRecommendation (债券特有字段):**
| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | `str` | TLT, IEF, SHY, LQD, HYG, AGG |
| `action` | `Action` | **V2关键**: 利率上升时可发出 SHORT_50% 等做空信号 |
| `market_view` | `Dict` | {rate_view, duration_preference, credit_view} |
| `risk_assessment` | `Dict` | {duration_risk, credit_risk, rate_sensitivity} |

---

### 1.3 CommodityExpert (商品专家)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `market_data` | `Dict[str, Any]` | 商品ETF价格 (GLD, SLV, USO, DBA) |
| `market_data["commodities"]` | `Dict` | {gold_spot, silver_spot, crude_wti, natural_gas, copper} |
| `news_data` | `List[Dict]` | OPEC/地缘政治新闻 |
| `macro_data` | `Dict` | {dollar_index, inflation_expectation, real_rates} |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `ExpertReport.recommendations` | `List[ExpertRecommendation]` | 建议列表 |
| `ExpertRecommendation.symbol` | `str` | GLD, SLV, USO, DBA, COPX |
| `ExpertRecommendation.action` | `Action` | 可发出SHORT信号 (如做空原油) |
| `ExpertRecommendation.market_view` | `Dict` | {gold_view, oil_view, dollar_impact} |

---

### 1.4 REITsExpert (房地产信托专家)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `market_data` | `Dict[str, Any]` | REITs ETF价格 (VNQ, IYR, DLR, EQIX) |
| `market_data["rates"]` | `Dict` | {mortgage_30y, treasury_10y} |
| `news_data` | `List[Dict]` | 房地产/利率相关新闻 |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `ExpertReport.recommendations` | `List[ExpertRecommendation]` | 建议列表 |
| `ExpertRecommendation.symbol` | `str` | VNQ, IYR, DLR, EQIX |
| `ExpertRecommendation.action` | `Action` | 可发出SHORT信号 (如做空办公楼REITs) |
| `ExpertRecommendation.market_view` | `Dict` | {sector_view, rate_sensitivity, property_type} |

---

### 1.5 CryptoExpert (加密货币专家)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `market_data` | `Dict[str, Any]` | 加密货币价格 (BTC-USD, ETH-USD) |
| `market_data["market_metrics"]` | `Dict` | {total_market_cap, fear_greed_index, btc_dominance} |
| `news_data` | `List[Dict]` | 监管/ETF/技术升级新闻 |
| `on_chain_data` | `Dict` | {btc_exchange_netflow, eth_staking_ratio, whale_activity} - 可选 |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `ExpertReport.recommendations` | `List[ExpertRecommendation]` | 建议列表 |
| `ExpertRecommendation.symbol` | `str` | BTC-USD, ETH-USD |
| `ExpertRecommendation.action` | `Action` | 可发出SHORT信号 |
| `ExpertRecommendation.market_view` | `Dict` | {trend, catalyst, on_chain, network_health} |
| `ExpertRecommendation.risk_assessment` | `Dict` | {volatility_risk, regulatory_risk, liquidity_risk} |

---

## 2. Portfolio Manager (组合管理器)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `expert_reports` | `Dict[str, ExpertReport]` | 所有专家报告 {stocks, bonds, commodities, reits, crypto} |
| `portfolio_state` | `PortfolioState` | 当前组合状态 |
| `risk_constraints` | `Dict` | 风险约束参数 |
| `allocation_bounds` | `Dict` | 资产配置边界 (**V2: min可为负数**) |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `target_allocation` | `Dict[str, float]` | 目标权重 (**V2: 负权重=做空**) |
| `trade_orders` | `List[TradeOrder]` | 交易指令列表 |
| `optimization_report` | `Dict` | 优化报告 |

**V2 新增输出字段:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `gross_exposure` | `float` | 总敞口 = |多头| + |空头| |
| `net_exposure` | `float` | 净敞口 = 多头 - |空头| |
| `short_exposure` | `float` | 空头敞口 |

---

## 3. Risk Controller (风控控制器)

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `proposed_allocation` | `Dict[str, float]` | 建议配置 (含负权重) |
| `portfolio_state` | `PortfolioState` | 当前组合状态 |
| `historical_returns` | `pd.DataFrame` | 历史收益率 |
| `risk_params` | `Dict` | 风险参数 |

**V2 新增风险参数:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `max_short_single` | `float` | 单一做空上限 (默认0.10) |
| `max_total_short` | `float` | 总做空上限 (默认0.30) |
| `margin_requirement` | `float` | 保证金比例 (默认0.50) |

**输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `adjusted_allocation` | `Dict[str, float]` | 风险调整后配置 |
| `risk_report` | `Dict` | 风险检查报告 |
| `risk_report["short_checks"]` | `Dict` | **V2新增**: 做空风险检查结果 |

---

## 4. Trading Environment (交易环境)

### 4.1 MultiAssetEnv

**输入:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `FinSageConfig` | 系统配置 |
| `market_data_stream` | `Dict` | 市场数据流 {date, prices, volumes, news} |
| `expert_weights` | `Dict` | 专家权重建议 |

**step() 输出:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `observation` | `Dict` | 新的市场状态 |
| `reward` | `float` | 当日收益 |
| `done` | `bool` | 是否结束 |
| `info` | `Dict` | 额外信息 (trades, positions, short_metrics) |

**V2 info 新增字段:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `info["short_metrics"]` | `Dict` | {short_market_value, margin_used, borrowing_cost_daily} |

---

### 4.2 PortfolioState

**Position 数据结构:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | `str` | 资产代码 |
| `shares` | `float` | **V2关键**: 正数=多头, 负数=空头 |
| `avg_cost` | `float` | 平均成本 |
| `current_price` | `float` | 当前价格 |
| `asset_class` | `str` | 资产类别 |
| `is_short` | `bool` | **V2新增**: shares < 0 |
| `market_value` | `float` | 市值 (空头为负数) |
| `unrealized_pnl` | `float` | **V2**: 空头盈亏计算不同 |
| `margin_requirement` | `float` | **V2新增**: 空头保证金 |

**PortfolioState 属性:**
| 属性 | 类型 | 说明 |
|------|------|------|
| `cash` | `float` | 现金余额 |
| `positions` | `Dict[str, Position]` | 持仓字典 |
| `portfolio_value` | `float` | 组合总价值 |
| `long_market_value` | `float` | **V2新增**: 多头总市值 |
| `short_market_value` | `float` | **V2新增**: 空头总市值(负数) |
| `gross_exposure` | `float` | **V2新增**: 总敞口 |
| `net_exposure` | `float` | **V2新增**: 净敞口 |
| `short_margin_required` | `float` | **V2新增**: 空头保证金需求 |
| `short_borrow_rate` | `float` | **V2新增**: 借股费率 (2%年化) |

---

## 5. V2 做空机制详解

### 5.1 Action 枚举 (13-Action Space)

```
做空信号 (负权重):
├── SHORT_100%  → multiplier = -1.0
├── SHORT_75%   → multiplier = -0.75
├── SHORT_50%   → multiplier = -0.5
└── SHORT_25%   → multiplier = -0.25

减仓信号:
├── SELL_100%   → multiplier = 0.0 (清仓)
├── SELL_75%    → multiplier = 0.25
├── SELL_50%    → multiplier = 0.5
└── SELL_25%    → multiplier = 0.75

持有:
└── HOLD        → multiplier = 1.0

加仓信号 (正权重):
├── BUY_25%     → multiplier = 1.25
├── BUY_50%     → multiplier = 1.5
├── BUY_75%     → multiplier = 1.75
└── BUY_100%    → multiplier = 2.0
```

### 5.2 做空交易类型

| 交易类型 | 触发条件 | 说明 |
|----------|----------|------|
| `SHORT` | `shares < 0`, 无现有仓位 | 开新空仓 |
| `ADD_SHORT` | `shares < 0`, 有空头仓位 | 加空仓 |
| `COVER_SHORT` | `shares > 0`, 有空头仓位 | 平空仓 |
| `SELL_AND_SHORT` | `shares < 0`, 有多头仓位, `is_short=True` | 先卖后空 |
| `COVER_AND_BUY` | `shares > 平空所需`, 有空头仓位 | 先平空后做多 |

### 5.3 空头盈亏计算

```
多头 PnL = shares × (current_price - avg_cost)
         价格上涨 → 盈利

空头 PnL = |shares| × (avg_cost - current_price)
         价格下跌 → 盈利
```

### 5.4 保证金和借股成本

| 项目 | 数值 | 说明 |
|------|------|------|
| 保证金比例 | 50% | `margin = |short_value| × 0.5` |
| 借股费率 | 2% 年化 | `daily_cost = |short_value| × 0.02 / 365` |

---

## 6. 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                      Market Data                            │
│  (prices, volumes, news, macro, technical indicators)       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Stock   │   │  Bond    │   │Commodity │  + REITs + Crypto
    │  Expert  │   │  Expert  │   │  Expert  │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         │  ExpertReport (含 SHORT 建议)
         │              │              │
         └──────────────┼──────────────┘
                        ▼
              ┌─────────────────┐
              │    Portfolio    │
              │     Manager     │
              └────────┬────────┘
                       │
              目标配置 (含负权重)
                       │
                       ▼
              ┌─────────────────┐
              │      Risk       │
              │   Controller    │
              └────────┬────────┘
                       │
              风险调整后配置
                       │
                       ▼
              ┌─────────────────┐
              │  MultiAssetEnv  │
              │ PortfolioState  │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
       BUY          SHORT         SELL
     (多头)        (空头)        (平仓)
```
