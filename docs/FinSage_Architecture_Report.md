# FinSage 多智能体金融组合管理系统 - 详细架构报告

---

## 1. 系统概述

FinSage 是一个基于大语言模型 (LLM) 的多智能体金融投资组合管理系统，采用**层次化多智能体架构**，实现从市场分析到交易执行的全流程自动化决策。

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FinSage 系统架构                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    专家智能体层 (Expert Layer)                │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │   │
│  │  │ Stock   │ │  Bond   │ │Commodity│ │  REITs  │ │ Crypto  │ │   │
│  │  │ Expert  │ │ Expert  │ │ Expert  │ │ Expert  │ │ Expert  │ │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │   │
│  └───────┼──────────┼──────────┼──────────┼──────────┼─────────┘   │
│          │          │          │          │          │              │
│          └──────────┴──────────┼──────────┴──────────┘              │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   管理层 (Manager Layer)                      │   │
│  │                                                               │   │
│  │  ┌───────────────────────────────────────────────────────┐   │   │
│  │  │              Manager Coordinator (协调器)              │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  │                            │                                  │   │
│  │              ┌─────────────┼─────────────┐                   │   │
│  │              ▼             ▼             ▼                   │   │
│  │  ┌───────────────┐ ┌─────────────┐ ┌─────────────┐          │   │
│  │  │   Portfolio   │ │  Position   │ │   Hedging   │          │   │
│  │  │   Manager     │ │   Sizing    │ │    Agent    │          │   │
│  │  │               │ │   Agent     │ │             │          │   │
│  │  └───────────────┘ └─────────────┘ └─────────────┘          │   │
│  │         │                 │               │                  │   │
│  │         └─────────────────┼───────────────┘                  │   │
│  │                           ▼                                  │   │
│  │              ┌─────────────────────────┐                    │   │
│  │              │   Integrated Decision   │                    │   │
│  │              │      (整合决策)          │                    │   │
│  │              └─────────────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  风控层 (Risk Control Layer)                  │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │   │
│  │  │ 硬性约束检查 │  │  软性约束优化  │  │   盘中风险监控    │    │   │
│  │  └─────────────┘  └──────────────┘  └──────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                交易执行层 (Trading Environment)               │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │   │
│  │  │  订单生成    │  │   成本计算    │  │    组合更新       │    │   │
│  │  └─────────────┘  └──────────────┘  └──────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
FinSage/
├── finsage/
│   ├── __init__.py
│   ├── config.py                    # 系统配置
│   ├── main_engine.py               # 主引擎
│   │
│   ├── agents/                      # 智能体模块
│   │   ├── base_expert.py           # 专家基类
│   │   ├── portfolio_manager.py     # 组合经理
│   │   ├── position_sizing_agent.py # 仓位管理智能体
│   │   ├── hedging_agent.py         # 对冲策略智能体
│   │   ├── manager_coordinator.py   # 管理层协调器
│   │   └── experts/                 # 5位专家
│   │       ├── stock_expert.py
│   │       ├── bond_expert.py
│   │       ├── commodity_expert.py
│   │       ├── reits_expert.py
│   │       └── crypto_expert.py
│   │
│   ├── hedging/                     # 对冲工具包 (12个工具)
│   │   ├── base_tool.py
│   │   ├── stop_loss.py
│   │   ├── protective_put.py
│   │   ├── covered_call.py
│   │   ├── collar.py
│   │   ├── futures_hedge.py
│   │   ├── pairs_trading.py
│   │   ├── beta_hedge.py
│   │   ├── variance_swap.py
│   │   ├── tail_risk_hedge.py
│   │   ├── dynamic_hedge.py
│   │   ├── cross_asset_hedge.py
│   │   └── correlation_hedge.py
│   │
│   ├── risk/                        # 风险控制
│   │   └── risk_controller.py
│   │
│   ├── trading/                     # 交易环境
│   │   └── environment.py
│   │
│   ├── data/                        # 数据层
│   │   ├── market_data.py
│   │   ├── macro_loader.py
│   │   └── fmp_client.py
│   │
│   └── llm/                         # LLM接口
│       └── provider.py
│
├── main.py                          # 入口文件
├── requirements.txt
└── README.md
```

---

## 3. 专家智能体层 (Expert Layer)

### 3.1 设计原理

每位专家智能体负责**特定资产类别**的深度分析，采用统一的基类 `BaseExpert` 确保接口一致性。

### 3.2 五位专家

| 专家 | 负责资产类别 | 默认标的 | 分析维度 |
|------|-------------|---------|---------|
| **Stock Expert** | 股票 | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, SPY, QQQ, IWM, VTI | 基本面 + 技术面 + 情绪 |
| **Bond Expert** | 债券 | TLT, IEF, SHY, LQD, HYG, AGG | 利率曲线 + 信用利差 |
| **Commodity Expert** | 商品 | GLD, SLV, USO, DBA, COPX | 供需 + 宏观周期 |
| **REITs Expert** | 房地产信托 | VNQ, IYR, DLR, EQIX | Cap Rate + 利率敏感度 |
| **Crypto Expert** | 加密货币 | BTC-USD, ETH-USD | 链上数据 + 市场情绪 |

### 3.3 基类接口 (BaseExpert)

```python
class BaseExpert(ABC):
    """专家基类"""

    def __init__(self, llm_provider, asset_class, symbols, config):
        self.llm = llm_provider
        self.asset_class = asset_class
        self.symbols = symbols  # 固定资产池
        self.config = config

    @abstractmethod
    def analyze(self, market_data, news_data, technical_indicators) -> ExpertReport:
        """生成分析报告"""
        pass

    @abstractmethod
    def _build_analysis_prompt(self, market_data, news_data, indicators) -> str:
        """构建LLM提示词"""
        pass
```

### 3.4 九级动作空间

每位专家的建议采用**9级离散动作空间**：

```
┌─────────────────────────────────────────────────────────────┐
│                     9级动作空间                              │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ SELL_100%│ SELL_75% │ SELL_50% │ SELL_25% │     HOLD       │
│   -100%  │   -75%   │   -50%   │   -25%   │      0%        │
├──────────┼──────────┼──────────┼──────────┼────────────────┤
│ BUY_25%  │ BUY_50%  │ BUY_75%  │ BUY_100% │                │
│   +25%   │   +50%   │   +75%   │  +100%   │                │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

### 3.5 输出格式 (ExpertReport)

```python
@dataclass
class ExpertReport:
    expert_name: str
    asset_class: str
    timestamp: str
    recommendations: List[ExpertRecommendation]
    market_outlook: Dict[str, Any]
    risk_assessment: Dict[str, float]
    confidence: float
```

---

## 4. 管理层 (Manager Layer) - 三智能体并行讨论机制

### 4.1 架构设计

管理层由**3个专门智能体**组成，通过 `ManagerCoordinator` 协调进行**并行讨论**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Manager Coordinator                          │
│                       (协调器)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   阶段1: 并行独立分析 (Phase 1: Parallel Independent Analysis)  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  ThreadPoolExecutor                      │   │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │   │
│   │  │  Portfolio   │ │   Position   │ │   Hedging    │     │   │
│   │  │   Manager    │ │    Sizing    │ │    Agent     │     │   │
│   │  │              │ │    Agent     │ │              │     │   │
│   │  │ 资产配置建议  │ │  仓位大小建议  │ │  对冲策略建议  │     │   │
│   │  └──────────────┘ └──────────────┘ └──────────────┘     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   阶段2: 讨论整合 (Phase 2: Discussion & Integration)           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Round 1:                                                │   │
│   │  ┌──────────────┐      ┌──────────────┐                 │   │
│   │  │   Position   │ ←──→ │   Hedging    │                 │   │
│   │  │    Sizing    │      │    Agent     │                 │   │
│   │  │   修正仓位    │      │   修正对冲    │                 │   │
│   │  └──────────────┘      └──────────────┘                 │   │
│   │         │                     │                          │   │
│   │         └─────────┬───────────┘                          │   │
│   │                   ▼                                      │   │
│   │          ┌────────────────┐                              │   │
│   │          │  共识检查       │                              │   │
│   │          │ sizing_diff<5% │                              │   │
│   │          │ hedge_diff<2%  │                              │   │
│   │          └────────────────┘                              │   │
│   │                   │                                      │   │
│   │        ┌──────────┴──────────┐                           │   │
│   │        ▼                     ▼                           │   │
│   │   [达成共识]            [未达成共识]                      │   │
│   │   结束讨论              进入 Round 2                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   阶段3: 最终整合 (Phase 3: Final Integration)                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  final_allocation = PM配置 × 0.6 + Sizing建议 × 0.4      │   │
│   │  + 对冲调整 (减少风险资产, 增加现金)                       │   │
│   │  → 生成交易指令 (trades)                                 │   │
│   │  → 输出 IntegratedDecision                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 三个管理智能体

#### 4.2.1 Portfolio Manager (组合经理)

**职责**: 整合专家报告，确定资产类别配置

```python
class PortfolioManager:
    """组合管理智能体"""

    def decide(self, expert_reports, market_data,
               current_portfolio, risk_constraints) -> PortfolioDecision:
        """
        1. 解读专家报告
        2. LLM决策资产类别配置
        3. 生成初步配置建议
        """
```

**输出示例**:
```python
PortfolioDecision(
    target_allocation={
        "stocks": 0.40,
        "bonds": 0.25,
        "commodities": 0.15,
        "reits": 0.10,
        "crypto": 0.05,
        "cash": 0.05
    },
    reasoning="基于专家看涨股票、看跌债券的观点...",
    risk_metrics={"expected_vol": 0.12, "var_95": 0.025}
)
```

#### 4.2.2 Position Sizing Agent (仓位规模智能体)

**职责**: 确定每个资产的具体仓位大小

```python
class PositionSizingAgent:
    """仓位规模智能体"""

    # 5种仓位方法
    SIZING_METHODS = {
        "equal_weight": "等权配置",
        "risk_parity": "风险平价",
        "kelly": "Kelly准则",
        "volatility_target": "波动率目标",
        "max_sharpe": "最大夏普",
    }

    def analyze(self, target_allocation, market_data,
                risk_constraints, portfolio_value) -> PositionSizingDecision:
        """
        1. LLM选择最佳仓位方法
        2. 计算各资产仓位
        3. 应用风险约束
        4. 计算风险贡献
        """
```

**仓位方法详解**:

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| **Equal Weight** | w_i = 1/n | 无信息时的基准 |
| **Risk Parity** | w_i ∝ 1/σ_i | 平衡风险贡献 |
| **Kelly Criterion** | f = μ/σ² × 0.5 | 最大化长期增长 |
| **Volatility Target** | scale = σ_target/σ_portfolio | 控制组合波动率 |
| **Max Sharpe** | 优化 Sharpe Ratio | 风险调整收益最大化 |

#### 4.2.3 Hedging Agent (对冲策略智能体)

**职责**: 设计尾部风险对冲策略

```python
class HedgingAgent:
    """对冲策略智能体"""

    # 7种对冲策略
    HEDGING_STRATEGIES = {
        "none": "无对冲",
        "protective_put": "保护性看跌",
        "collar": "领口策略",
        "tail_risk": "尾部风险对冲",
        "dynamic": "动态对冲",
        "cross_asset": "跨资产对冲",
        "correlation": "相关性对冲",
    }

    def analyze(self, target_allocation, position_sizes,
                market_data, risk_constraints) -> HedgingDecision:
        """
        1. 评估尾部风险 (VaR, CVaR, 偏度, 峰度)
        2. LLM选择对冲策略
        3. 确定对冲比例
        4. 选择对冲工具
        """
```

**尾部风险指标**:
```python
tail_risk_metrics = {
    "var_95": 0.025,      # 95% VaR
    "cvar_95": 0.035,     # 95% CVaR (Expected Shortfall)
    "skewness": -0.5,     # 偏度 (负偏表示左尾风险)
    "kurtosis": 4.2,      # 峰度 (>3 表示肥尾)
}
```

### 4.3 并行讨论流程

```python
def coordinate(self, expert_reports, market_data, ...):
    # 阶段1: 并行独立分析
    with ThreadPoolExecutor(max_workers=3) as executor:
        pm_future = executor.submit(self.pm.decide, ...)
        sizing_future = executor.submit(self.sizing_agent.analyze, ...)
        hedging_future = executor.submit(self.hedging_agent.analyze, ...)

        pm_decision = pm_future.result()
        sizing_decision = sizing_future.result()
        hedging_decision = hedging_future.result()

    # 阶段2: 讨论整合 (1-2轮)
    for round_num in range(self.max_discussion_rounds):
        # 构建反馈
        sizing_feedback = {
            "portfolio_manager": pm_decision.to_dict(),
            "hedging_agent": hedging_decision.to_dict(),
        }
        hedging_feedback = {
            "portfolio_manager": pm_decision.to_dict(),
            "sizing_agent": sizing_decision.to_dict(),
        }

        # 并行修正
        new_sizing = self.sizing_agent.revise_based_on_feedback(...)
        new_hedging = self.hedging_agent.revise_based_on_feedback(...)

        # 检查共识
        if self._check_consensus(sizing_decision, new_sizing,
                                  hedging_decision, new_hedging):
            break

    # 阶段3: 最终整合
    return self._integrate_decisions(...)
```

### 4.4 共识检查标准

```python
def _check_consensus(self, old_sizing, new_sizing,
                     old_hedging, new_hedging) -> bool:
    # 仓位变化 < 5%
    sizing_diff = sum(|old[i] - new[i]|) < 0.05

    # 对冲比例变化 < 2%
    hedge_diff = |old_hedge_ratio - new_hedge_ratio| < 0.02

    return sizing_diff and hedge_diff
```

---

## 5. 对冲工具包 (Hedging Toolkit)

### 5.1 12种对冲工具

基于学术文献实现的专业对冲工具：

| 工具 | 类名 | 功能 | 学术依据 |
|------|------|------|---------|
| 1. Stop Loss | `StopLossTool` | 止损保护 | 基础风险管理 |
| 2. Protective Put | `ProtectivePutTool` | 保护性看跌期权 | Black-Scholes |
| 3. Covered Call | `CoveredCallTool` | 备兑看涨期权 | 期权定价理论 |
| 4. Collar | `CollarTool` | 领口策略 | 零成本对冲 |
| 5. Futures Hedge | `FuturesHedgeTool` | 期货对冲 | 最小方差对冲 |
| 6. Pairs Trading | `PairsTradingTool` | 配对交易 | 统计套利 |
| 7. Beta Hedge | `BetaHedgeTool` | Beta中性对冲 | CAPM |
| 8. Variance Swap | `VarianceSwapTool` | 方差互换 | 波动率交易 |
| 9. Tail Risk Hedge | `TailRiskHedgeTool` | 尾部风险对冲 | 极值理论 |
| 10. Dynamic Hedge | `DynamicHedgeTool` | 动态对冲 | Delta对冲 |
| 11. Cross Asset Hedge | `CrossAssetHedgeTool` | 跨资产对冲 | 相关性对冲 |
| 12. Correlation Hedge | `CorrelationHedgeTool` | 相关性崩溃对冲 | 危机期相关性 |

### 5.2 工具接口

```python
class BaseTool(ABC):
    """对冲工具基类"""

    @abstractmethod
    def calculate_hedge(self, portfolio, market_data, params) -> HedgeResult:
        """计算对冲方案"""
        pass

    @abstractmethod
    def estimate_cost(self, hedge_result) -> float:
        """估算对冲成本"""
        pass
```

---

## 6. 风险控制层 (Risk Control Layer)

### 6.1 三层风控机制

```
┌─────────────────────────────────────────────────────────────┐
│                      风险控制层                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  第一层: 硬性约束 (Hard Constraints) - 必须满足              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • 单一资产权重 ≤ 15%                                    │ │
│  │ • 资产类别权重 ≤ 50%                                    │ │
│  │ • 组合VaR(95%) ≤ 3%                                    │ │
│  │ • 最大回撤触发 ≤ 15%                                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  第二层: 软性约束 (Soft Constraints) - 优化目标              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • 目标波动率: 12%                                       │ │
│  │ • 最大相关性聚类: 60%                                   │ │
│  │ • 最小分散化比率: 1.2                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  第三层: 盘中监控 (Intraday Monitoring) - 实时预警           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • 实时损益监控                                          │ │
│  │ • 波动率突变检测                                        │ │
│  │ • 流动性风险监控                                        │ │
│  │ • 自动触发减仓/对冲                                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 风控参数 (RiskConfig)

```python
@dataclass
class RiskConfig:
    # 硬性约束
    max_single_asset: float = 0.15      # 单一资产最大15%
    max_asset_class: float = 0.50       # 资产类别最大50%
    max_drawdown_trigger: float = 0.15  # 回撤触发15%
    max_portfolio_var_95: float = 0.03  # VaR(95%) 最大3%

    # 软性约束
    target_volatility: float = 0.12     # 目标波动率12%
    max_correlation_cluster: float = 0.60
    min_diversification_ratio: float = 1.2
```

---

## 7. 交易环境 (Trading Environment)

### 7.1 交易执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                      交易执行流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 接收决策                                                 │
│     └─→ IntegratedDecision (目标配置 + 对冲指令)            │
│                              │                               │
│  2. 生成交易指令             ▼                               │
│     └─→ 计算当前持仓与目标差异                               │
│     └─→ 过滤小于阈值(2%)的交易                               │
│                              │                               │
│  3. 成本计算                 ▼                               │
│     └─→ 交易成本: 0.1%                                      │
│     └─→ 滑点: 0.05%                                         │
│                              │                               │
│  4. 执行交易                 ▼                               │
│     └─→ 更新持仓                                            │
│     └─→ 更新现金                                            │
│     └─→ 记录交易历史                                        │
│                              │                               │
│  5. 组合估值                 ▼                               │
│     └─→ 计算组合净值                                        │
│     └─→ 计算收益率                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 交易配置

```python
@dataclass
class TradingConfig:
    initial_capital: float = 1_000_000.0  # 初始资金100万
    transaction_cost: float = 0.001       # 交易成本0.1%
    slippage: float = 0.0005              # 滑点0.05%
    min_trade_value: float = 100.0        # 最小交易金额
    rebalance_frequency: str = "daily"    # 调仓频率
    rebalance_threshold: float = 0.02     # 再平衡阈值2%
```

---

## 8. 数据层 (Data Layer)

### 8.1 数据源架构

```
┌─────────────────────────────────────────────────────────────┐
│                        数据层                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              MarketDataProvider                      │    │
│  │              (市场数据提供者)                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│          ┌───────────────┼───────────────┐                  │
│          ▼               ▼               ▼                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ FMP Client  │ │  yfinance   │ │ Local Cache │           │
│  │ (主数据源)   │ │  (备用)     │ │   (缓存)    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              MacroDataLoader                         │    │
│  │              (宏观数据加载器)                         │    │
│  └─────────────────────────────────────────────────────┘    │
│          │                                                   │
│          ├─→ VIX (恐慌指数)                                 │
│          ├─→ DXY (美元指数)                                 │
│          ├─→ Treasury Rates (国债利率)                      │
│          ├─→ Sector Performance (板块表现)                  │
│          ├─→ REITs Cap Rate (房地产Cap Rate)                │
│          └─→ Crypto Data (加密货币数据)                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 FMP API 集成

```python
class FMPClient:
    """FMP数据客户端"""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def get_historical_prices(self, symbol, start, end):
        """获取历史价格"""

    def get_company_profile(self, symbol):
        """获取公司信息"""

    def get_news(self, symbol, limit=50):
        """获取新闻"""

    def get_technical_indicators(self, symbol):
        """获取技术指标"""
```

---

## 9. 资产配置机制

### 9.1 预定义资产池

资产是**预先配置**的，非动态选取：

```python
# config.py
@dataclass
class AssetConfig:
    default_universe: Dict[str, List[str]] = field(default_factory=lambda: {
        "stocks": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG"],
        "commodities": ["GLD", "SLV", "USO", "DBA", "COPX"],
        "reits": ["VNQ", "IYR", "DLR", "EQIX"],
        "crypto": ["BTC-USD", "ETH-USD"],
    })

    allocation_bounds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "stocks": {"min": 0.30, "max": 0.50, "default": 0.40},
        "bonds": {"min": 0.15, "max": 0.35, "default": 0.25},
        "commodities": {"min": 0.10, "max": 0.25, "default": 0.15},
        "reits": {"min": 0.05, "max": 0.15, "default": 0.10},
        "crypto": {"min": 0.00, "max": 0.10, "default": 0.05},
        "cash": {"min": 0.02, "max": 0.15, "default": 0.05},
    })
```

### 9.2 配置模板

系统提供预定义配置模板：

| 模板 | 股票 | 债券 | 商品 | REITs | 加密 | 现金 | 目标波动率 |
|------|------|------|------|-------|------|------|-----------|
| **默认** | 30-50% | 15-35% | 10-25% | 5-15% | 0-10% | 2-15% | 12% |
| **保守** | 20-40% | 30-50% | 5-15% | 5-10% | 0-2% | 5-20% | 8% |
| **激进** | 40-70% | 5-20% | 10-25% | 5-15% | 0-15% | 0-5% | 18% |

---

## 10. 完整执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    FinSage 完整执行流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: 数据获取                                               │
│  └─→ MarketDataProvider.get_data(symbols, date)                 │
│  └─→ MacroDataLoader.get_macro_data()                           │
│                              │                                   │
│                              ▼                                   │
│  Step 2: 专家分析 (并行)                                        │
│  └─→ StockExpert.analyze() ─┐                                   │
│  └─→ BondExpert.analyze()  ─┼─→ Dict[str, ExpertReport]        │
│  └─→ CommodityExpert.analyze()                                  │
│  └─→ REITsExpert.analyze() ─┤                                   │
│  └─→ CryptoExpert.analyze() ┘                                   │
│                              │                                   │
│                              ▼                                   │
│  Step 3: 管理层协调                                             │
│  └─→ ManagerCoordinator.coordinate()                            │
│      ├─→ Phase 1: 并行独立分析                                  │
│      │   ├─→ PortfolioManager.decide()                          │
│      │   ├─→ PositionSizingAgent.analyze()                      │
│      │   └─→ HedgingAgent.analyze()                             │
│      ├─→ Phase 2: 讨论整合 (1-2轮)                              │
│      │   ├─→ revise_based_on_feedback()                         │
│      │   └─→ check_consensus()                                  │
│      └─→ Phase 3: 最终整合                                      │
│          └─→ IntegratedDecision                                 │
│                              │                                   │
│                              ▼                                   │
│  Step 4: 风险控制                                               │
│  └─→ RiskController.validate()                                  │
│      ├─→ check_hard_constraints()                               │
│      ├─→ optimize_soft_constraints()                            │
│      └─→ apply_adjustments()                                    │
│                              │                                   │
│                              ▼                                   │
│  Step 5: 交易执行                                               │
│  └─→ TradingEnvironment.execute()                               │
│      ├─→ generate_orders()                                      │
│      ├─→ calculate_costs()                                      │
│      ├─→ update_portfolio()                                     │
│      └─→ record_history()                                       │
│                              │                                   │
│                              ▼                                   │
│  Step 6: 结果输出                                               │
│  └─→ 组合净值、收益率、风险指标、交易记录                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. 系统特点总结

| 特点 | 描述 |
|------|------|
| **层次化架构** | 专家层 → 管理层 → 风控层 → 执行层 |
| **多智能体协作** | 5位专家 + 3位管理者并行讨论 |
| **LLM驱动决策** | GPT-4/GPT-4o-mini 驱动分析和决策 |
| **9级动作空间** | 精细化的买卖建议 |
| **三层风控** | 硬性约束 + 软性优化 + 实时监控 |
| **12种对冲工具** | 基于学术文献的专业对冲策略 |
| **并行讨论机制** | ThreadPoolExecutor + 共识检查 |
| **多数据源** | FMP + yfinance + 本地缓存 |

---

*报告生成时间: 2024年*
*FinSage 版本: 1.0*
