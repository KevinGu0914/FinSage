#!/usr/bin/env python3
"""
MARFT Training V4 - Full Feature Integration (100% Project Capability)

在V3基础上集成项目中所有未使用的功能模块:
1. VLLMInferenceEngine: 5x推理加速 (PagedAttention, Continuous Batching)
2. ManagerCoordinator: 3个管理层智能体协调 (PM, PositionSizing, Hedging)
3. HedgingToolkit: 11个对冲工具 (CVaR, DCC-GARCH, Black-Litterman等)
4. FactorScorers: 5个因子评分器 (Fama-French 5-factor等)
5. StrategyClasses: 6个策略类 (Strategic, Tactical, CoreSatellite等)
6. DataBridge: 观察/动作格式转换 (ObservationFormatter, ActionConverter)
7. RealCritic: 真正的价值估计网络 (不再硬编码为0)
8. torch.compile: PyTorch 2.0编译优化
9. StaticKVCache: 静态KV缓存预分配
10. DynamicUniverse: 因子驱动的动态资产池

学术基础:
- MARFT: Multi-Agent Reinforcement Fine-Tuning (Liao et al., 2025)
- PPO: Proximal Policy Optimization (Schulman et al., 2017)
- GAE: Generalized Advantage Estimation (Schulman et al., 2015)
- Fama-French: 5-Factor Model (Fama & French, 2015)
- CVaR: Conditional Value-at-Risk (Rockafellar & Uryasev, 2000)

Usage:
    python scripts/train_with_real_data_v4.py --model Qwen/Qwen2.5-14B-Instruct --full_features
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Force unbuffered output
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 详细监控日志系统 (费用很高，及时发现bug)
# ============================================================
class TrainingMonitor:
    """训练监控器 - 实时监控GPU/内存/训练指标"""

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.step_count = 0
        self.start_time = datetime.now()
        self.metrics_history = []
        self.gpu_alerts = []

    def log_gpu_status(self, prefix: str = ""):
        """记录GPU状态"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_used = torch.cuda.memory_allocated(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                mem_pct = mem_used / mem_total * 100

                # 显存警告阈值
                if mem_pct > 90:
                    alert = f"⚠️ GPU{i} 显存危险: {mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.1f}%)"
                    self.gpu_alerts.append(alert)
                    flush_print(f"\n{'='*60}\n{alert}\n{'='*60}")
                elif mem_pct > 80:
                    flush_print(f"[MONITOR] GPU{i} 显存较高: {mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.1f}%)")
                elif prefix:
                    flush_print(f"[MONITOR] {prefix} GPU{i}: {mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.1f}%)")

    def log_step(self, step: int, metrics: Dict[str, float], force: bool = False):
        """记录训练步骤"""
        self.step_count = step
        self.metrics_history.append({"step": step, "time": datetime.now().isoformat(), **metrics})

        if force or step % self.log_interval == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            steps_per_sec = step / elapsed if elapsed > 0 else 0

            flush_print(f"\n[STEP {step}] 耗时: {elapsed/60:.1f}min | 速度: {steps_per_sec:.2f} steps/s")
            for k, v in metrics.items():
                flush_print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
            self.log_gpu_status()

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """记录epoch结束"""
        flush_print(f"\n{'='*70}")
        flush_print(f"[EPOCH {epoch} 完成] 累计步数: {self.step_count}")
        flush_print(f"训练指标:")
        for k, v in train_metrics.items():
            flush_print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        if val_metrics:
            flush_print(f"验证指标:")
            for k, v in val_metrics.items():
                flush_print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        self.log_gpu_status("Epoch结束")
        flush_print(f"{'='*70}\n")

    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        flush_print(f"\n{'!'*70}")
        flush_print(f"[ERROR] {context}")
        flush_print(f"错误类型: {type(error).__name__}")
        flush_print(f"错误信息: {str(error)}")
        flush_print(f"当前步数: {self.step_count}")
        self.log_gpu_status("错误发生时")

        # 保存metrics历史到文件
        error_file = f"error_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(error_file, 'w') as f:
                json.dump(self.metrics_history[-100:], f, indent=2)
            flush_print(f"已保存最近100步metrics到: {error_file}")
        except:
            pass
        flush_print(f"{'!'*70}\n")

    def summary(self):
        """训练总结"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        flush_print(f"\n{'='*70}")
        flush_print(f"[训练总结]")
        flush_print(f"  总耗时: {elapsed/3600:.2f}小时")
        flush_print(f"  总步数: {self.step_count}")
        flush_print(f"  平均速度: {self.step_count/elapsed:.2f} steps/s")
        if self.gpu_alerts:
            flush_print(f"  GPU告警次数: {len(self.gpu_alerts)}")
        flush_print(f"{'='*70}\n")


# 全局监控器
MONITOR = TrainingMonitor(log_interval=5)


# ============================================================
# Feature Availability Flags
# ============================================================

# Reward Functions
try:
    from finsage.rl.reward_functions import (
        CombinedRewardCalculator,
        create_default_reward_calculator,
        RewardComponents,
        ExpertReward,
    )
    HAS_REWARD_FUNCTIONS = True
except ImportError:
    HAS_REWARD_FUNCTIONS = False
    logger.warning("reward_functions module not available")

# ManagerCoordinator (3 Managers)
try:
    from finsage.agents.manager_coordinator import ManagerCoordinator, IntegratedDecision
    from finsage.agents.portfolio_manager import PortfolioManager
    from finsage.agents.position_sizing_agent import PositionSizingAgent
    from finsage.agents.hedging_agent import HedgingAgent
    HAS_MANAGER_COORDINATOR = True
except ImportError:
    HAS_MANAGER_COORDINATOR = False
    logger.warning("ManagerCoordinator not available")

# HedgingToolkit (11 Tools)
try:
    from finsage.hedging.toolkit import HedgingToolkit
    HAS_HEDGING_TOOLKIT = True
except ImportError:
    HAS_HEDGING_TOOLKIT = False
    logger.warning("HedgingToolkit not available")

# Factor Scorers (5 Scorers)
try:
    from finsage.factors import (
        StockFactorScorer,
        BondFactorScorer,
        CommodityFactorScorer,
        REITsFactorScorer,
        CryptoFactorScorer,
    )
    HAS_FACTOR_SCORERS = True
except ImportError:
    HAS_FACTOR_SCORERS = False
    logger.warning("Factor scorers not available")

# Strategy Classes (6 Strategies)
try:
    from finsage.strategies import (
        StrategicAllocationStrategy,
        TacticalAllocationStrategy,
        DynamicRebalancingStrategy,
        CoreSatelliteStrategy,
        StrategyToolkit,
    )
    HAS_STRATEGIES = True
except ImportError:
    HAS_STRATEGIES = False
    logger.warning("Strategy classes not available")

# Data Bridge
try:
    from finsage.rl.data_bridge import (
        ObservationFormatter,
        ActionConverter,
        BatchProcessor,
        MARFTEnvWrapper,
        create_data_bridge,
    )
    HAS_DATA_BRIDGE = True
except ImportError:
    HAS_DATA_BRIDGE = False
    logger.warning("Data bridge not available")

# vLLM Engine
try:
    from finsage.rl.shared_expert_manager import VLLMInferenceEngine
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    logger.warning("VLLMInferenceEngine not available")


# ============================================================
# 1. Complete Action Space (13 Actions)
# ============================================================

class TradeAction(Enum):
    """完整的13动作交易空间"""
    SHORT_100 = "SHORT_100%"
    SHORT_75 = "SHORT_75%"
    SHORT_50 = "SHORT_50%"
    SHORT_25 = "SHORT_25%"
    SELL_100 = "SELL_100%"
    SELL_75 = "SELL_75%"
    SELL_50 = "SELL_50%"
    SELL_25 = "SELL_25%"
    HOLD = "HOLD"
    BUY_25 = "BUY_25%"
    BUY_50 = "BUY_50%"
    BUY_75 = "BUY_75%"
    BUY_100 = "BUY_100%"


ALL_ACTIONS = [a.value for a in TradeAction]


# ============================================================
# 2. PPO Configuration (MARFT Paper Aligned)
# ============================================================

@dataclass
class PPOConfig:
    """PPO超参数配置 (对齐MARFT论文)"""
    # Clipping
    clip_param: float = 0.2
    clip_value_loss: bool = True

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training - High performance settings
    ppo_epochs: int = 4  # Full epochs for better policy updates
    mini_batch_size: int = 8  # Full batch size for better training

    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Constraints
    max_grad_norm: float = 0.5
    kl_threshold: float = 0.5
    target_kl: float = 0.2

    # Learning rate
    policy_lr: float = 5e-6
    critic_lr: float = 1e-4


# ============================================================
# 3. Trajectory Buffer with GAE (MARFT Paper Algorithm 1)
# ============================================================

@dataclass
class TrajectoryStep:
    """单步轨迹数据"""
    observations: List[str]
    actions: List[Dict]
    action_tokens: List[torch.Tensor]
    log_probs: List[float]
    values: List[float]
    reward: float
    done: bool
    individual_rewards: Optional[List[float]] = None
    coordination_reward: float = 0.0
    # V4 新增: 管理层决策和对冲信息
    manager_decision: Optional[Dict] = None
    hedging_info: Optional[Dict] = None


class TrajectoryBuffer:
    """轨迹Buffer - 实现MARFT论文的GAE计算"""

    def __init__(
        self,
        num_agents: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer: List[TrajectoryStep] = []

    def add(self, step: TrajectoryStep):
        self.buffer.append(step)

    def compute_gae(
        self,
        next_values: List[float],
        next_done: bool = False,
        use_individual_rewards: bool = True,
        individual_weight: float = 0.4,
        team_weight: float = 0.4,
        coordination_weight: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算GAE优势估计"""
        T = len(self.buffer)
        advantages = np.zeros((T, self.num_agents), dtype=np.float32)
        returns = np.zeros((T, self.num_agents), dtype=np.float32)

        gae = np.zeros(self.num_agents, dtype=np.float32)

        for t in reversed(range(T)):
            step = self.buffer[t]

            if t == T - 1:
                next_val = np.array(next_values)
                next_non_terminal = 0.0 if next_done else 1.0
            else:
                next_step = self.buffer[t + 1]
                next_val = np.array(next_step.values)
                next_non_terminal = 0.0 if step.done else 1.0

            current_values = np.array(step.values)

            for i in reversed(range(self.num_agents)):
                if use_individual_rewards and step.individual_rewards is not None:
                    individual_r = step.individual_rewards[i]
                    reward_i = (
                        individual_weight * individual_r +
                        team_weight * step.reward +
                        coordination_weight * step.coordination_reward
                    )
                else:
                    reward_i = step.reward

                delta = reward_i + self.gamma * next_val[i] * next_non_terminal - current_values[i]
                gae[i] = delta + self.gamma * self.gae_lambda * next_non_terminal * gae[i]
                advantages[t, i] = gae[i]
                returns[t, i] = advantages[t, i] + current_values[i]

        return advantages, returns

    def get_batch(self) -> Dict[str, Any]:
        return {
            "observations": [s.observations for s in self.buffer],
            "actions": [s.actions for s in self.buffer],
            "action_tokens": [s.action_tokens for s in self.buffer],
            "log_probs": np.array([s.log_probs for s in self.buffer]),
            "values": np.array([s.values for s in self.buffer]),
            "rewards": np.array([s.reward for s in self.buffer]),
            "dones": np.array([s.done for s in self.buffer]),
        }

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 4. Enhanced Critic Network (V4: 真正的价值估计)
# ============================================================

class EnhancedCritic(torch.nn.Module):
    """
    增强版Critic网络 (V4)

    相比V3的改进:
    1. 更深的网络结构
    2. 注意力机制融合
    3. 每个Agent独立的value head
    4. 支持数值特征输入 (来自DataBridge)
    """

    def __init__(
        self,
        num_assets: int = 50,
        hidden_size: int = 512,
        num_agents: int = 9,  # 5 Asset Experts + 4 Meta-Level Agents
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_size = hidden_size

        # 市场特征编码器 (更深)
        input_size = num_assets * 10 + 20  # 10 features per asset + 20 macro

        layers = []
        in_dim = input_size
        for i in range(num_layers):
            out_dim = hidden_size if i < num_layers - 1 else hidden_size
            layers.extend([
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.LayerNorm(out_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.market_encoder = torch.nn.Sequential(*layers)

        # 组合状态编码器
        portfolio_input = 10
        self.portfolio_encoder = torch.nn.Sequential(
            torch.nn.Linear(portfolio_input, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 2),
            torch.nn.GELU(),
        )

        # 注意力融合层
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # 融合层
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

        # 每个Agent独立的Value head
        self.value_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(num_agents)
        ])

    def forward(
        self,
        market_features: torch.Tensor,
        portfolio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            values: [batch, num_agents]
        """
        # 市场特征编码
        market_encoded = self.market_encoder(market_features)  # [batch, hidden]

        # 组合特征编码
        portfolio_encoded = self.portfolio_encoder(portfolio_features)  # [batch, hidden/2]

        # 融合
        fused = self.fusion(torch.cat([market_encoded, portfolio_encoded], dim=-1))

        # 每个Agent的value
        values = []
        for head in self.value_heads:
            v = head(fused).squeeze(-1)  # [batch]
            values.append(v)

        values = torch.stack(values, dim=-1)  # [batch, num_agents]

        return values


# ============================================================
# 5. Factor-Driven Universe Manager (V4: 集成5个因子评分器)
# ============================================================

class FactorDrivenUniverseManager:
    """
    因子驱动的动态资产池管理器 (V4)

    集成5个因子评分器:
    - StockFactorScorer: Fama-French 5-factor
    - BondFactorScorer: 久期/信用利差因子
    - CommodityFactorScorer: 动量/库存因子
    - REITsFactorScorer: Cap Rate/利率敏感度
    - CryptoFactorScorer: 网络/情绪因子
    """

    def __init__(
        self,
        use_factor_screening: bool = True,
        refresh_interval_days: int = 7,
        top_n_per_class: int = 10,
    ):
        self.use_factor_screening = use_factor_screening
        self.refresh_interval_days = refresh_interval_days
        self.top_n_per_class = top_n_per_class
        self._last_refresh_date: Optional[str] = None
        self._current_universe: Optional[Dict[str, List[str]]] = None

        # 初始化因子评分器
        self.factor_scorers = {}
        if use_factor_screening and HAS_FACTOR_SCORERS:
            try:
                self.factor_scorers = {
                    "stocks": StockFactorScorer(),
                    "bonds": BondFactorScorer(),
                    "commodities": CommodityFactorScorer(),
                    "reits": REITsFactorScorer(),
                    "crypto": CryptoFactorScorer(),
                }
                logger.info("All 5 factor scorers initialized!")
            except Exception as e:
                logger.warning(f"Factor scorers init failed: {e}")

    def get_static_universe(self) -> Dict[str, List[str]]:
        """静态资产池 (Fallback)"""
        return {
            "stocks": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                "JPM", "BAC", "GS", "V", "MA",
                "UNH", "JNJ", "PFE", "ABBV",
                "WMT", "COST", "HD", "MCD",
                "SPY", "QQQ", "IWM", "DIA",
            ],
            "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND"],
            "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC"],
            "reits": ["VNQ", "IYR", "SCHH", "DLR", "EQIX", "AMT"],
            "crypto": ["BTC-USD", "ETH-USD", "BITO"],
        }

    def needs_refresh(self, current_date: str) -> bool:
        if self._current_universe is None:
            return True
        if self._last_refresh_date is None:
            return True

        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        last_dt = datetime.strptime(self._last_refresh_date, "%Y-%m-%d")
        days_since = (current_dt - last_dt).days

        return days_since >= self.refresh_interval_days

    def refresh(
        self,
        current_date: str,
        market_data: Optional[pd.DataFrame] = None,
        force: bool = False
    ) -> Dict[str, List[str]]:
        """刷新动态资产池 (使用因子评分)"""
        if not force and not self.needs_refresh(current_date):
            return self._current_universe

        logger.info(f"Refreshing factor-driven universe on {current_date}...")

        universe = {}

        for asset_class in ["stocks", "bonds", "commodities", "reits", "crypto"]:
            static_symbols = self.get_static_universe()[asset_class]

            if asset_class in self.factor_scorers and market_data is not None:
                try:
                    scorer = self.factor_scorers[asset_class]
                    # 计算因子分数并排序
                    scores = {}
                    for symbol in static_symbols:
                        if symbol in market_data.columns:
                            score = scorer.compute_score(
                                symbol=symbol,
                                returns=market_data[symbol].pct_change().dropna(),
                                date=current_date,
                            )
                            scores[symbol] = score

                    # 按分数排序选择top N
                    sorted_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
                    universe[asset_class] = sorted_symbols[:self.top_n_per_class]
                    logger.info(f"  {asset_class}: {len(universe[asset_class])} assets (factor-driven)")
                except Exception as e:
                    logger.warning(f"Factor scoring failed for {asset_class}: {e}")
                    universe[asset_class] = static_symbols
            else:
                universe[asset_class] = static_symbols

        self._current_universe = universe
        self._last_refresh_date = current_date
        total = sum(len(s) for s in universe.values())
        logger.info(f"Factor-driven universe: {total} total assets")

        return universe

    def get_universe(self, current_date: str = None) -> Dict[str, List[str]]:
        if current_date and self.needs_refresh(current_date):
            return self.refresh(current_date)
        if self._current_universe is None:
            return self.get_static_universe()
        return self._current_universe


# ============================================================
# 6. Hedging Integration (V4: 11个对冲工具)
# ============================================================

class HedgingIntegration:
    """
    对冲工具集成 (V4)

    11个可用工具:
    - MinimumVariance: 最小方差组合
    - RiskParity: 风险平价
    - BlackLitterman: Black-Litterman模型
    - MeanVariance: 均值方差优化
    - DCC-GARCH: 动态条件相关
    - HRP: 分层风险平价
    - CVaR: 条件风险价值优化
    - RobustOptimization: 鲁棒优化
    - FactorHedging: 因子对冲
    - RegimeSwitching: 体制转换
    - CopulaHedging: Copula对冲
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.toolkit = None

        if enabled and HAS_HEDGING_TOOLKIT:
            try:
                self.toolkit = HedgingToolkit()
                logger.info(f"HedgingToolkit initialized with {len(self.toolkit.list_tools())} tools")
            except Exception as e:
                logger.warning(f"HedgingToolkit init failed: {e}")
                self.enabled = False

    def compute_hedge_weights(
        self,
        returns: pd.DataFrame,
        tool_name: str = "risk_parity",
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """使用指定工具计算对冲权重"""
        if not self.enabled or self.toolkit is None:
            return {}

        try:
            weights = self.toolkit.call(
                tool_name=tool_name,
                returns=returns,
                expert_views=expert_views,
                constraints=constraints,
            )
            return weights
        except Exception as e:
            logger.warning(f"Hedge computation failed: {e}")
            return {}

    def compare_tools(
        self,
        returns: pd.DataFrame,
        tool_names: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """比较多个对冲工具的输出"""
        if not self.enabled or self.toolkit is None:
            return {}

        return self.toolkit.compare_tools(returns, tool_names=tool_names)


# ============================================================
# 7. Strategy Integration (V4: 6个策略类)
# ============================================================

class StrategyIntegration:
    """
    策略集成 (V4)

    6个可用策略:
    - StrategicAllocation: 战略配置
    - TacticalAllocation: 战术配置
    - DynamicRebalancing: 动态再平衡
    - CoreSatellite: 核心-卫星策略
    - StrategyToolkit: 策略工具箱
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.strategies = {}
        self.toolkit = None

        if enabled and HAS_STRATEGIES:
            try:
                self.strategies = {
                    "strategic": StrategicAllocationStrategy(),
                    "tactical": TacticalAllocationStrategy(),
                    "dynamic": DynamicRebalancingStrategy(),
                    "core_satellite": CoreSatelliteStrategy(),
                }
                self.toolkit = StrategyToolkit()
                logger.info(f"Initialized {len(self.strategies)} strategy classes")
            except Exception as e:
                logger.warning(f"Strategy init failed: {e}")
                self.enabled = False

    def get_strategy_allocation(
        self,
        strategy_name: str,
        market_data: Dict,
        current_portfolio: Dict,
        risk_constraints: Dict,
    ) -> Dict[str, float]:
        """使用指定策略获取配置建议"""
        if not self.enabled or strategy_name not in self.strategies:
            return {}

        try:
            strategy = self.strategies[strategy_name]
            allocation = strategy.compute_allocation(
                market_data=market_data,
                current_portfolio=current_portfolio,
                constraints=risk_constraints,
            )
            return allocation
        except Exception as e:
            logger.warning(f"Strategy {strategy_name} failed: {e}")
            return {}


# ============================================================
# 8. Manager Coordinator Integration (V4: 3个管理层智能体)
# ============================================================

class ManagerIntegration:
    """
    管理层协调器集成 (V4)

    3个管理智能体:
    - PortfolioManager: 组合管理 (资产配置决策)
    - PositionSizingAgent: 仓位管理 (Kelly准则等)
    - HedgingAgent: 对冲管理 (尾部风险)
    """

    def __init__(self, enabled: bool = True, llm_provider: Any = None):
        self.enabled = enabled
        self.coordinator = None

        if enabled and HAS_MANAGER_COORDINATOR and llm_provider:
            try:
                # 创建 HedgingToolkit 实例
                hedging_toolkit = HedgingToolkit() if HAS_HEDGING_TOOLKIT else None
                pm = PortfolioManager(llm_provider=llm_provider, hedging_toolkit=hedging_toolkit)
                sizing = PositionSizingAgent(llm_provider=llm_provider)
                hedging = HedgingAgent(llm_provider=llm_provider)

                self.coordinator = ManagerCoordinator(
                    portfolio_manager=pm,
                    position_sizing_agent=sizing,
                    hedging_agent=hedging,
                    llm_provider=llm_provider,
                    config={
                        "max_discussion_rounds": 2,
                        "consensus_threshold": 0.85,
                        "parallel_execution": True,
                    }
                )
                logger.info("ManagerCoordinator initialized with 3 agents")
            except Exception as e:
                logger.warning(f"ManagerCoordinator init failed: {e}")
                self.enabled = False

    def coordinate_decision(
        self,
        expert_reports: Dict,
        market_data: Dict,
        current_portfolio: Dict,
        risk_constraints: Dict,
        portfolio_value: float,
    ) -> Optional[Dict]:
        """协调3个管理层智能体进行决策"""
        if not self.enabled or self.coordinator is None:
            return None

        try:
            decision = self.coordinator.coordinate(
                expert_reports=expert_reports,
                market_data=market_data,
                current_portfolio=current_portfolio,
                risk_constraints=risk_constraints,
                portfolio_value=portfolio_value,
            )
            return decision.to_dict()
        except Exception as e:
            logger.warning(f"Manager coordination failed: {e}")
            return None


# ============================================================
# 9. vLLM Engine Integration (V4: 5x推理加速)
# ============================================================

class VLLMIntegration:
    """
    vLLM推理引擎集成 (V4)

    特性:
    - PagedAttention: 高效KV Cache管理
    - Continuous Batching: 连续批处理
    - 5x推理加速 (vs HuggingFace)
    """

    def __init__(
        self,
        enabled: bool = True,
        model_path: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
    ):
        self.enabled = enabled
        self.engine = None

        if enabled and HAS_VLLM and model_path:
            try:
                self.engine = VLLMInferenceEngine(
                    model_path=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enable_lora=True,
                )
                logger.info("VLLMInferenceEngine initialized!")
            except Exception as e:
                logger.warning(f"vLLM init failed: {e}")
                self.enabled = False

    def generate(
        self,
        prompts: List[str],
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
    ) -> List[str]:
        """批量生成"""
        if not self.enabled or self.engine is None:
            return [""] * len(prompts)

        return self.engine.generate(prompts, lora_name, temperature)


# ============================================================
# 10. PPO Trainer (V4: 完整功能)
# ============================================================

class MARFTV4PPOTrainer:
    """
    MARFT V4 PPO训练器

    相比V3的改进:
    1. 使用EnhancedCritic进行真正的价值估计
    2. 集成DataBridge进行观察/动作转换
    3. 支持vLLM加速推理
    4. 支持batch推理
    """

    def __init__(
        self,
        manager,  # SharedModelExpertManager
        critic: EnhancedCritic,
        config: PPOConfig,
        device: str = "cuda:0",
        data_bridge: Any = None,
    ):
        self.manager = manager
        self.critic = critic.to(device)
        self.config = config
        self.device = torch.device(device)
        self.data_bridge = data_bridge

        # Expert优化器 (5 Asset Experts + 4 Meta-Level Agents)
        self.expert_optimizers = {}
        expert_roles = [
            # Asset Class Experts
            "Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert",
            # Meta-Level Agents (Coordinators)
            "Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"
        ]
        for role in expert_roles:
            manager.switch_expert(role)
            params = list(manager.parameters(role))
            if params:
                self.expert_optimizers[role] = torch.optim.AdamW(
                    params,
                    lr=config.policy_lr,
                    weight_decay=0.0,
                )

        # Critic优化器
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=config.critic_lr,
            weight_decay=0.01,
        )

        logger.info(f"MARFT V4 PPOTrainer initialized with {len(self.expert_optimizers)} experts")

    def compute_values(
        self,
        market_features: torch.Tensor,
        portfolio_features: torch.Tensor,
    ) -> torch.Tensor:
        """使用Critic计算状态价值"""
        with torch.no_grad():
            values = self.critic(market_features, portfolio_features)
        return values

    def train_step(
        self,
        buffer: TrajectoryBuffer,
        next_values: List[float],
        market_features: Optional[torch.Tensor] = None,
        portfolio_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """执行PPO训练步骤"""
        # 计算GAE
        advantages, returns = buffer.compute_gae(next_values)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 标准化advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        batch = buffer.get_batch()
        T = len(buffer)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0

        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            indices = np.random.permutation(T)

            for mb_start in range(0, T, self.config.mini_batch_size):
                mb_end = min(mb_start + self.config.mini_batch_size, T)
                mb_indices = indices[mb_start:mb_end]

                if len(mb_indices) == 0:
                    continue

                old_log_probs = batch["log_probs"][mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                expert_roles = list(self.expert_optimizers.keys())

                for expert_idx, role in enumerate(expert_roles):
                    optimizer = self.expert_optimizers[role]

                    new_log_probs_list = []
                    entropies_list = []

                    for batch_idx, t_idx in enumerate(mb_indices):
                        obs = batch["observations"][t_idx][expert_idx]
                        action_tokens = batch["action_tokens"][t_idx][expert_idx]

                        predecessor_actions = {}
                        for pred_idx in range(expert_idx):
                            pred_role = expert_roles[pred_idx]
                            predecessor_actions[pred_role] = batch["actions"][t_idx][pred_idx]

                        try:
                            log_prob, entropy = self.manager.get_action_log_prob(
                                role=role,
                                obs=obs,
                                action_tokens=action_tokens.to(self.device),
                                predecessor_actions=predecessor_actions if predecessor_actions else None,
                            )
                            new_log_probs_list.append(log_prob)
                            entropies_list.append(entropy)
                        except Exception as e:
                            logger.warning(f"Error computing log_prob for {role}: {e}")
                            continue

                    if len(new_log_probs_list) == 0:
                        continue

                    new_log_probs = torch.stack(new_log_probs_list)
                    entropies = torch.stack(entropies_list)

                    old_lp = torch.tensor(
                        old_log_probs[:len(new_log_probs), expert_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    adv = mb_advantages[:len(new_log_probs), expert_idx]

                    # PPO Clipped Loss
                    ratio = torch.exp(new_log_probs - old_lp)
                    ratio = torch.clamp(ratio, 0.01, 100.0)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_param,
                        1.0 + self.config.clip_param,
                    ) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    entropy_loss = -self.config.entropy_coef * entropies.mean()

                    total_loss = policy_loss + entropy_loss

                    optimizer.zero_grad()
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        list(self.manager.parameters(role)),
                        self.config.max_grad_norm,
                    )

                    optimizer.step()

                    # Record stats before clearing
                    policy_loss_val = policy_loss.item()
                    entropy_val = entropies.mean().item()
                    with torch.no_grad():
                        kl = (old_lp - new_log_probs).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > self.config.clip_param).float().mean().item()

                    # Clear GPU memory after each expert update to prevent OOM
                    del total_loss, policy_loss, entropy_loss, new_log_probs, entropies, ratio, surr1, surr2
                    torch.cuda.empty_cache()

                    stats["policy_loss"] += policy_loss_val
                    stats["entropy"] += entropy_val
                    stats["kl_divergence"] += abs(kl)
                    stats["clip_fraction"] += clip_frac
                    update_count += 1

                # Value loss (使用真正的Critic)
                # 注意: 检查 features 张量大小与索引范围是否匹配
                if market_features is not None and portfolio_features is not None:
                    # 检查索引是否在有效范围内 (防止 CUDA index out of bounds)
                    max_idx = max(mb_indices) if len(mb_indices) > 0 else 0
                    if max_idx >= market_features.size(0) or max_idx >= portfolio_features.size(0):
                        # 索引越界，跳过这个 mini-batch 的 value loss
                        continue

                    mb_market = market_features[mb_indices]
                    mb_portfolio = portfolio_features[mb_indices]

                    values_pred = self.critic(mb_market, mb_portfolio)
                    value_loss = self.config.value_loss_coef * F.mse_loss(
                        values_pred, mb_returns
                    )

                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.critic_optimizer.step()

                    stats["value_loss"] += value_loss.item()

            # KL早停
            if update_count > 0:
                avg_kl = stats["kl_divergence"] / update_count
                if avg_kl > self.config.kl_threshold:
                    logger.info(f"Early stopping at epoch {epoch} due to KL={avg_kl:.4f}")
                    break

        if update_count > 0:
            stats = {k: v / update_count for k, v in stats.items()}

        return stats


# ============================================================
# 11. Data Fetching
# ============================================================

def fetch_training_data(
    start_date: str,
    end_date: str,
    universe: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """获取训练数据"""
    from finsage.data.fmp_client import FMPClient

    logger.info(f"Fetching data from {start_date} to {end_date}")

    client = FMPClient()
    all_data = {}

    all_symbols = []
    for symbols in universe.values():
        all_symbols.extend(symbols)
    all_symbols = list(set(all_symbols))

    for symbol in all_symbols:
        try:
            df = client.get_historical_price(symbol, start_date, end_date)
            if df is not None and not df.empty:
                col = 'close' if 'close' in df.columns else 'Close'
                if col in df.columns:
                    all_data[symbol] = df[col]
                    logger.info(f"  {symbol}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  {symbol}: failed - {e}")

    if not all_data:
        raise ValueError("No data fetched!")

    prices = pd.DataFrame(all_data).ffill().bfill()
    returns = {
        "1d": prices.pct_change(),
        "5d": prices.pct_change(5),
        "20d": prices.pct_change(20),
    }

    logger.info(f"Got {len(prices)} trading days for {len(all_data)} symbols")
    return prices, returns


def calculate_indicators(
    prices: pd.DataFrame,
    date_idx: int,
    lookback: int = 30,
) -> Dict[str, Dict]:
    """计算技术指标 (包括个股指标和宏观指标)"""
    indicators = {}

    for symbol in prices.columns:
        series = prices[symbol].iloc[max(0, date_idx-lookback):date_idx+1]
        if len(series) < 10:
            continue

        daily_returns = series.pct_change().dropna()

        # RSI
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MA
        ma_5 = series.rolling(5).mean()
        ma_20 = series.rolling(20).mean()

        # Volatility
        volatility = daily_returns.rolling(20).std() * np.sqrt(252)

        indicators[symbol] = {
            "price": series.iloc[-1],
            "returns_1d": daily_returns.iloc[-1] if len(daily_returns) > 0 else 0,
            "rsi": rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            "ma_5": ma_5.iloc[-1] if not pd.isna(ma_5.iloc[-1]) else series.iloc[-1],
            "ma_20": ma_20.iloc[-1] if not pd.isna(ma_20.iloc[-1]) else series.iloc[-1],
            "volatility": volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.2,
        }

    # V4修复: 添加宏观指标计算
    # 使用代理ETF计算宏观指标
    macro_indicators = {}

    # VIX代理 - 使用SPY的波动率估计
    if "SPY" in indicators:
        spy_vol = indicators["SPY"].get("volatility", 0.2)
        # VIX通常是SPY年化波动率的大约100倍的点数
        macro_indicators["vix"] = min(80, max(10, spy_vol * 100))
        macro_indicators["spy_return"] = indicators["SPY"].get("returns_1d", 0)
    else:
        macro_indicators["vix"] = 20
        macro_indicators["spy_return"] = 0

    # 债券收益率代理 - 使用TLT价格变化的反向
    if "TLT" in indicators:
        tlt_return = indicators["TLT"].get("returns_1d", 0)
        # 债券价格下跌 -> 收益率上升
        macro_indicators["bond_yield"] = 0.04 - tlt_return * 10  # 近似
    else:
        macro_indicators["bond_yield"] = 0.04

    # 美元指数代理 - 使用UUP或默认值
    if "UUP" in indicators:
        macro_indicators["usd_index"] = indicators["UUP"].get("price", 100) / 25 * 100  # UUP大约在25左右
    else:
        macro_indicators["usd_index"] = 100

    # 油价代理 - 使用USO
    if "USO" in indicators:
        macro_indicators["oil_price"] = indicators["USO"].get("price", 70)
    else:
        macro_indicators["oil_price"] = 70

    # 黄金价格代理 - 使用GLD
    if "GLD" in indicators:
        macro_indicators["gold_price"] = indicators["GLD"].get("price", 180) * 10  # GLD大约是金价的1/10
    else:
        macro_indicators["gold_price"] = 2000

    # 市场广度 - 基于多个股票的涨跌比例
    up_count = 0
    total_count = 0
    for symbol, ind in indicators.items():
        if "returns_1d" in ind:
            total_count += 1
            if ind["returns_1d"] > 0:
                up_count += 1
    macro_indicators["market_breadth"] = up_count / total_count if total_count > 0 else 0.5

    # Put/Call比率 - 简化估计 (VIX高时P/C高)
    macro_indicators["put_call_ratio"] = 0.8 + (macro_indicators["vix"] - 20) / 100

    # 收益率曲线 - 使用TLT和SHY的差异
    if "TLT" in indicators and "SHY" in indicators:
        tlt_ret = indicators["TLT"].get("returns_1d", 0)
        shy_ret = indicators["SHY"].get("returns_1d", 0)
        # 长期债券表现优于短期 -> 收益率曲线陡峭化
        macro_indicators["yield_curve"] = (shy_ret - tlt_ret) * 100
    else:
        macro_indicators["yield_curve"] = 0

    # 信用利差代理 - 使用HYG和LQD的差异
    if "HYG" in indicators and "LQD" in indicators:
        hyg_ret = indicators["HYG"].get("returns_1d", 0)
        lqd_ret = indicators["LQD"].get("returns_1d", 0)
        # 高收益债表现差于投资级 -> 信用利差扩大
        macro_indicators["credit_spread"] = (lqd_ret - hyg_ret) * 10 + 0.03
    else:
        macro_indicators["credit_spread"] = 0.03

    # 将宏观指标添加到主字典
    indicators["_macro"] = macro_indicators

    return indicators


def create_observation(
    date: str,
    indicators: Dict[str, Dict],
    portfolio_state,
    universe: Dict[str, List[str]],
    asset_class: Optional[str] = None,
) -> str:
    """创建观察"""
    lines = [f"## 市场日期: {date}\n"]

    # 组合状态
    lines.append("## 当前组合状态")
    lines.append(f"- 总价值: ${portfolio_state.portfolio_value:,.2f}")
    lines.append(f"- 现金: ${portfolio_state.cash:,.2f}")
    lines.append(f"- 总收益: {portfolio_state.total_return:.2%}")
    lines.append("")

    # 市场数据
    lines.append("## 市场数据")

    # V4修复: 定义真实资产类别列表
    # Meta-agents (portfolio, hedging, position_sizing, risk) 不在此列表中
    real_asset_classes = {"stocks", "bonds", "commodities", "reits", "crypto"}

    # V4修复: 如果asset_class是meta-agent类型，显示所有资产类别的摘要
    is_meta_agent = asset_class and asset_class not in real_asset_classes

    for cls, symbols in universe.items():
        # 只跳过当：指定了asset_class且不是meta-agent且cls与asset_class不匹配
        if asset_class and not is_meta_agent and cls != asset_class:
            continue

        lines.append(f"\n### {cls.upper()}")
        for symbol in symbols[:5]:
            if symbol not in indicators:
                continue
            ind = indicators[symbol]
            trend = "↑" if ind["price"] > ind["ma_20"] else "↓"
            lines.append(
                f"  - {symbol}: ${ind['price']:.2f} {trend} | "
                f"RSI:{ind['rsi']:.0f} | Vol:{ind['volatility']*100:.1f}%"
            )

    return "\n".join(lines)


# ============================================================
# 12. Checkpoint Resume Utilities
# ============================================================

def find_latest_checkpoint(save_dir: str) -> Optional[Tuple[str, int, int, int]]:
    """
    查找最新的 checkpoint 目录

    Returns:
        Tuple of (checkpoint_dir, epoch, step, total_samples) or None if not found
    """
    if not os.path.exists(save_dir):
        return None

    checkpoints = []

    for name in os.listdir(save_dir):
        ckpt_path = os.path.join(save_dir, name)
        state_file = os.path.join(ckpt_path, "training_state.json")

        if not os.path.isdir(ckpt_path):
            continue

        # 检查是否有 training_state.json
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                saved_epoch = state.get("epoch", 0)
                # V4 Fix: JSON保存的是1-indexed epoch，转换为0-indexed
                epoch = saved_epoch - 1 if saved_epoch >= 1 else saved_epoch
                step = state.get("step", 0)
                total_samples = state.get("total_samples", 0)
                # 使用 epoch * 10000 + step 作为排序键
                sort_key = epoch * 10000 + step
                checkpoints.append((sort_key, ckpt_path, epoch, step, total_samples))
            except (json.JSONDecodeError, IOError):
                pass
        else:
            # 尝试从目录名解析 (e.g., "step_e2_s30" or "epoch_2")
            import re
            step_match = re.match(r"step_e(\d+)_s(\d+)", name)
            epoch_match = re.match(r"epoch_(\d+)", name)

            if step_match:
                epoch = int(step_match.group(1)) - 1  # 0-indexed
                step = int(step_match.group(2))
                sort_key = epoch * 10000 + step
                checkpoints.append((sort_key, ckpt_path, epoch, step, 0))
            elif epoch_match:
                epoch = int(epoch_match.group(1)) - 1  # 0-indexed
                step = 0  # epoch结束时step重置
                sort_key = (epoch + 1) * 10000  # epoch完成后的checkpoint
                checkpoints.append((sort_key, ckpt_path, epoch, step, 0))

    if not checkpoints:
        return None

    # 按 sort_key 排序，取最新的
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    _, ckpt_path, epoch, step, total_samples = checkpoints[0]

    return (ckpt_path, epoch, step, total_samples)


# ============================================================
# 13. Main Training Loop
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_start", default="2023-01-01")
    parser.add_argument("--train_end", default="2024-06-30")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--rebalance_freq", type=int, default=5)
    parser.add_argument("--rollout_length", type=int, default=20)
    parser.add_argument("--save_dir", default="/root/checkpoints/marft_v4")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N rebalance steps (default: 10)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint directory")
    parser.add_argument("--auto_resume", action="store_true", default=True,
                        help="Automatically resume from latest checkpoint (default: True)")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_grad_ckpt", action="store_true")

    # V4 全功能参数
    parser.add_argument("--full_features", action="store_true", default=True,
                        help="Enable all V4 features (default: True)")

    # 推理加速
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for 5x inference acceleration")
    parser.add_argument("--flash_attention", action="store_true", default=True)
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Enable torch.compile (default: False, disabled to save GPU memory)")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--static_kv_cache", action="store_true",
                        help="Enable static KV cache")
    parser.add_argument("--prompt_cache", action="store_true", default=True)
    parser.add_argument("--prompt_cache_size", type=int, default=100)

    # 动态资产池 (因子驱动)
    parser.add_argument("--dynamic_universe", action="store_true", default=True,
                        help="Enable factor-driven dynamic universe (default: True)")
    parser.add_argument("--use_factors", action="store_true", default=True,
                        help="Use 5 factor scorers for asset selection")
    parser.add_argument("--universe_refresh_days", type=int, default=7)

    # 对冲工具
    parser.add_argument("--use_hedging", action="store_true", default=True,
                        help="Enable 11 hedging tools")
    parser.add_argument("--hedge_tool", default="risk_parity",
                        help="Default hedging tool to use")

    # 策略类
    parser.add_argument("--use_strategies", action="store_true", default=True,
                        help="Enable 6 strategy classes")
    parser.add_argument("--strategy", default="tactical",
                        help="Default strategy to use")

    # 管理层协调
    parser.add_argument("--use_managers", action="store_true", default=True,
                        help="Enable 3 manager coordination")

    # Data Bridge
    parser.add_argument("--use_data_bridge", action="store_true", default=True,
                        help="Use data bridge for obs/action conversion")

    # 日志级别
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level (default: INFO)")

    # 奖励方案 (三种策略)
    parser.add_argument("--reward_scheme", default="balanced",
                        choices=["aggressive", "balanced", "adaptive"],
                        help="Reward scheme: aggressive (激进), balanced (平衡), adaptive (自适应)")

    args = parser.parse_args()

    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    if args.log_level == "DEBUG":
        flush_print(f"[DEBUG] 详细日志模式已启用")

    # 加载奖励方案配置
    try:
        from finsage.rl.reward_configs import get_reward_scheme, REWARD_SCHEMES
        reward_config = get_reward_scheme(args.reward_scheme)
        flush_print(f"[INFO] 加载奖励方案: {args.reward_scheme}")
        flush_print(f"[INFO] {reward_config.description[:200]}...")
    except ImportError:
        reward_config = None
        flush_print(f"[WARNING] reward_configs 模块不可用，使用默认配置")

    flush_print("=" * 80)
    flush_print(" MARFT V4 - Full Feature Integration (100% Project Capability)")
    flush_print(f" 奖励方案: {args.reward_scheme.upper()}")
    flush_print("=" * 80)
    flush_print(f" Model: {args.model}")
    flush_print(f" Training Period: {args.train_start} ~ {args.train_end}")
    flush_print(f" Rollout Length: {args.rollout_length}")
    flush_print("")
    flush_print(" Enabled Features:")
    flush_print(f"   - vLLM Acceleration: {args.use_vllm}")
    flush_print(f"   - torch.compile: {args.torch_compile}")
    flush_print(f"   - Factor-Driven Universe: {args.use_factors}")
    flush_print(f"   - Hedging Tools (11): {args.use_hedging}")
    flush_print(f"   - Strategy Classes (6): {args.use_strategies}")
    flush_print(f"   - Manager Coordination (3): {args.use_managers}")
    flush_print(f"   - Data Bridge: {args.use_data_bridge}")
    flush_print("=" * 80)

    if not torch.cuda.is_available():
        flush_print("ERROR: CUDA not available!")
        return

    flush_print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    flush_print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============================================================
    # 初始化所有V4组件
    # ============================================================

    # 1. 因子驱动的动态资产池
    if args.use_factors:
        flush_print("\n>>> Initializing Factor-Driven Universe Manager")
        universe_manager = FactorDrivenUniverseManager(
            use_factor_screening=HAS_FACTOR_SCORERS,
            refresh_interval_days=args.universe_refresh_days,
        )
        universe = universe_manager.refresh(args.train_start, force=True)
    else:
        universe_manager = None
        universe = {
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                       "JPM", "BAC", "GS", "V", "MA", "UNH", "JNJ", "SPY", "QQQ"],
            "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND"],
            "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC"],
            "reits": ["VNQ", "IYR", "SCHH", "DLR", "EQIX", "AMT"],
            "crypto": ["BTC-USD", "ETH-USD", "BITO"],
        }

    total_assets = sum(len(s) for s in universe.values())
    flush_print(f"\nTotal assets: {total_assets}")

    # 获取数据
    prices_df, returns_dict = fetch_training_data(args.train_start, args.train_end, universe)
    trading_days = prices_df.index.tolist()

    if len(trading_days) < 50:
        flush_print("Not enough data!")
        return

    # 2. 对冲工具集成
    hedging_integration = None
    if args.use_hedging:
        flush_print("\n>>> Initializing Hedging Toolkit (11 tools)")
        hedging_integration = HedgingIntegration(enabled=True)

    # 3. 策略集成
    strategy_integration = None
    if args.use_strategies:
        flush_print("\n>>> Initializing Strategy Classes (6 strategies)")
        strategy_integration = StrategyIntegration(enabled=True)

    # 4. Data Bridge
    data_bridge = None
    formatter = None
    converter = None
    if args.use_data_bridge and HAS_DATA_BRIDGE:
        flush_print("\n>>> Initializing Data Bridge")
        # V4修复: num_agents从5改为9 (5 asset experts + 4 meta-level agents)
        formatter, converter, _ = create_data_bridge(universe, num_agents=9)

    # 5. 加载模型
    flush_print("\n" + "=" * 40)
    flush_print(" Loading Model and Creating PPO Trainer")
    flush_print("=" * 40)

    from finsage.rl.shared_expert_manager import SharedModelExpertManager
    from finsage.environment.portfolio_state import PortfolioState

    manager = SharedModelExpertManager(
        model_path=args.model,
        device="cuda:0",
        bf16=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=not args.no_grad_ckpt,
        use_flash_attention=args.flash_attention,
        use_torch_compile=args.torch_compile,
        torch_compile_mode=args.compile_mode,
        use_static_cache=args.static_kv_cache,
        use_prompt_cache=args.prompt_cache,
        prompt_cache_size=args.prompt_cache_size,
    )

    flush_print(f"GPU Memory after model: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # V4修复: 设置模型为训练模式 (启用dropout等)
    manager.train()
    flush_print(">>> Model set to TRAINING mode")

    # 5.5 Checkpoint Resume Logic
    resume_epoch = 0
    resume_step = 0
    resume_total_samples = 0

    # 确定要加载的 checkpoint
    checkpoint_to_load = None
    if args.resume:
        # 用户指定了具体的 checkpoint 目录
        checkpoint_to_load = args.resume
        flush_print(f"\n>>> Resume mode: Loading specified checkpoint: {args.resume}")
    elif args.auto_resume:
        # 自动查找最新的 checkpoint
        latest = find_latest_checkpoint(args.save_dir)
        if latest:
            checkpoint_to_load, resume_epoch, resume_step, resume_total_samples = latest
            flush_print(f"\n>>> Auto-resume: Found checkpoint at {checkpoint_to_load}")
            flush_print(f"    Epoch: {resume_epoch}, Step: {resume_step}, Samples: {resume_total_samples}")

    # 加载 checkpoint (如果有)
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        flush_print(f"\n>>> Loading LoRA adapters from: {checkpoint_to_load}")
        try:
            manager.load_adapters(checkpoint_to_load)
            flush_print(">>> LoRA adapters loaded successfully!")

            # 读取训练状态 (如果存在)
            state_file = os.path.join(checkpoint_to_load, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                saved_epoch = state.get("epoch", 0)
                # V4 Fix: JSON保存的是1-indexed epoch，转换为0-indexed用于训练循环
                resume_epoch = saved_epoch - 1 if saved_epoch >= 1 else saved_epoch
                resume_step = state.get("step", 0)
                resume_total_samples = state.get("total_samples", 0)
                # 显示1-indexed epoch给用户
                flush_print(f">>> Training state loaded: Epoch {resume_epoch + 1}, Step {resume_step}")
        except Exception as e:
            flush_print(f">>> WARNING: Failed to load checkpoint: {e}")
            flush_print(">>> Starting training from scratch...")
            resume_epoch = 0
            resume_step = 0
            resume_total_samples = 0

    # 6. 管理层协调
    manager_integration = None
    if args.use_managers and HAS_MANAGER_COORDINATOR:
        flush_print("\n>>> Initializing Manager Coordinator (3 agents)")
        manager_integration = ManagerIntegration(enabled=True, llm_provider=manager)

    # 7. vLLM引擎 (可选)
    vllm_integration = None
    if args.use_vllm and HAS_VLLM:
        flush_print("\n>>> Initializing vLLM Engine (5x acceleration)")
        vllm_integration = VLLMIntegration(
            enabled=True,
            model_path=args.model,
        )

    # 8. 创建增强版Critic和PPO Trainer
    # V4修复: num_agents从5改为9 (5 asset experts + 4 meta-level agents)
    ppo_config = PPOConfig()
    # V4修复: 固定num_assets=50，与特征构建一致 (lines 2108-2110 pad to 500 = 50*10)
    # 特征构建始终使用50个资产 + 20个宏观指标 = 520维输入
    critic = EnhancedCritic(
        num_assets=50,  # 固定值，不使用total_assets
        hidden_size=512,
        num_agents=9,  # 修复: 5 asset experts + 4 meta-level agents
        num_layers=3,
    )

    # V4修复: 定义device变量供后续使用
    device = torch.device("cuda:0")

    trainer = MARFTV4PPOTrainer(
        manager=manager,
        critic=critic,
        config=ppo_config,
        device="cuda:0",
        data_bridge=data_bridge,
    )

    # Expert配置 (5 Asset Experts + 4 Meta-Level Agents = 9 total)
    expert_order = [
        # Asset Class Experts (first 5, process market data)
        "Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert",
        # Meta-Level Agents (coordinate asset experts)
        "Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"
    ]
    expert_to_class = {
        # Asset Class Experts
        "Stock_Expert": "stocks",
        "Bond_Expert": "bonds",
        "Commodity_Expert": "commodities",
        "REITs_Expert": "reits",
        "Crypto_Expert": "crypto",
        # Meta-Level Agents
        "Portfolio_Manager": "portfolio",
        "Hedging_Agent": "hedging",
        "Position_Sizing_Agent": "position",
        "Risk_Controller": "risk",
    }
    expert_to_reward_key = {
        # Asset Class Experts
        "Stock_Expert": "stock",
        "Bond_Expert": "bond",
        "Commodity_Expert": "commodity",
        "REITs_Expert": "reits",
        "Crypto_Expert": "crypto",
        # Meta-Level Agents (use portfolio-level rewards)
        "Portfolio_Manager": "portfolio",
        "Hedging_Agent": "hedging",
        "Position_Sizing_Agent": "position",
        "Risk_Controller": "risk",
    }

    # 初始化奖励计算器 (使用reward_config中的参数)
    reward_calculator = None
    if HAS_REWARD_FUNCTIONS:
        reward_calculator = create_default_reward_calculator()
        flush_print("\n>>> Specialized Reward Functions enabled!")

    # 导入修改后的奖励计算函数 (使用reward_config)
    compute_modified_reward = None
    detect_regime = None
    if reward_config is not None:
        try:
            from finsage.rl.reward_configs import compute_modified_expert_reward, detect_market_regime
            compute_modified_reward = compute_modified_expert_reward
            detect_regime = detect_market_regime
            flush_print(f">>> 使用奖励方案: {reward_config.name}")
            flush_print(f"    - 错误惩罚缩放: {reward_config.wrong_direction_penalty_scale}")
            flush_print(f"    - 时机惩罚缩放: {reward_config.timing_penalty_scale}")
            flush_print(f"    - 交易激励: {reward_config.trade_bonus}")
            flush_print(f"    - 动量奖励: {reward_config.momentum_bonus}")
            flush_print(f"    - HOLD惩罚: {reward_config.hold_penalty_in_uptrend}")
            flush_print(f"    - 市场自适应: {reward_config.regime_adaptive}")
        except ImportError as e:
            flush_print(f"[WARNING] 无法导入修改后的奖励函数: {e}")

    # ============================================================
    # 训练循环
    # ============================================================
    flush_print("\n" + "=" * 80)
    flush_print(" Starting MARFT V4 Training (Full Feature Integration)")
    flush_print("=" * 80)

    start_time = datetime.now()
    total_samples = resume_total_samples  # 从 checkpoint 恢复

    # 显示恢复信息
    if resume_epoch > 0 or resume_step > 0:
        flush_print(f"\n>>> Resuming from Epoch {resume_epoch + 1}, Step {resume_step + 1}")
        flush_print(f">>> Total samples so far: {total_samples}")

    for epoch in range(resume_epoch, args.num_epochs):  # 从 resume_epoch 开始
        flush_print(f"\n{'='*60}")
        flush_print(f" Epoch {epoch + 1}/{args.num_epochs}")
        flush_print(f"{'='*60}")

        # 初始化组合
        portfolio = PortfolioState(initial_capital=1_000_000.0)

        # 再平衡日
        rebalance_days = trading_days[30::args.rebalance_freq]

        # Trajectory buffer (9 agents: 5 asset experts + 4 meta-level agents)
        buffer = TrajectoryBuffer(
            num_agents=9,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )

        epoch_rewards = []

        # 计算起始步骤 (如果是恢复的 epoch，从 resume_step 开始)
        start_step = 0
        if epoch == resume_epoch and resume_step > 0:
            start_step = resume_step
            flush_print(f">>> Skipping to step {start_step + 1} (already completed)")

        for i, date in enumerate(rebalance_days):
            # 跳过已完成的步骤
            if i < start_step:
                continue

            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
            date_idx = trading_days.index(date)

            # 因子驱动的动态资产池刷新
            if universe_manager is not None and universe_manager.needs_refresh(date_str):
                flush_print(f"  [{date_str}] Refreshing factor-driven universe...")
                new_universe = universe_manager.refresh(date_str, market_data=prices_df)
                for asset_class, symbols in new_universe.items():
                    valid_symbols = [s for s in symbols if s in prices_df.columns]
                    universe[asset_class] = valid_symbols if valid_symbols else universe.get(asset_class, [])

            # 当前价格
            current_prices = prices_df.loc[date]
            price_dict = {s: current_prices[s] for s in current_prices.index if not pd.isna(current_prices[s])}

            # 更新组合
            portfolio.update_prices(price_dict)
            portfolio.record_value(date_str, price_dict)

            # 计算指标
            indicators = calculate_indicators(prices_df, date_idx)

            # 创建观察
            obs = create_observation(date_str, indicators, portfolio, universe)

            value_before = portfolio.portfolio_value

            # ============================================================
            # V4修复: 提前计算 returns_window 以供后续模块使用
            # ============================================================
            returns_window = prices_df.iloc[max(0, date_idx-60):date_idx].pct_change().dropna()

            # ============================================================
            # V4: 使用对冲工具计算对冲权重
            # ============================================================
            hedge_weights = {}
            if hedging_integration is not None and hedging_integration.enabled:
                try:
                    if len(returns_window) >= 20:
                        hedge_weights = hedging_integration.compute_hedge_weights(
                            returns=returns_window,
                            tool_name=args.hedge_tool,
                        )
                except Exception as e:
                    logger.warning(f"Hedging computation failed: {e}")

            # ============================================================
            # V4: 使用策略类获取配置建议
            # ============================================================
            strategy_allocation = {}
            if strategy_integration is not None and strategy_integration.enabled:
                try:
                    # V4修复: 添加 returns 数据以避免 "Empty returns data" 警告
                    strategy_market_data = {
                        "indicators": indicators,
                        "prices": price_dict,
                        "returns": returns_window if not returns_window.empty else pd.DataFrame(),
                    }
                    strategy_allocation = strategy_integration.get_strategy_allocation(
                        strategy_name=args.strategy,
                        market_data=strategy_market_data,
                        current_portfolio=portfolio.get_weights(),
                        risk_constraints={"max_drawdown": 0.15, "max_volatility": 0.20},
                    )
                except Exception as e:
                    logger.warning(f"Strategy computation failed: {e}")

            # 收集所有Expert的动作
            step_observations = []
            step_actions = []
            step_action_tokens = []
            step_log_probs = []
            step_values = []

            torch.cuda.empty_cache()

            all_actions = {}

            for expert_idx, role in enumerate(expert_order):
                asset_class = expert_to_class[role]

                # Expert专属观察
                expert_obs = create_observation(
                    date_str, indicators, portfolio, universe,
                    asset_class=asset_class,
                )
                step_observations.append(expert_obs)

                # 前序动作
                deps = manager.expert_configs[role].get("dependencies", [])
                predecessor_actions = {d: all_actions[d] for d in deps if d in all_actions}

                # 生成动作
                action_dict, tokens, _ = manager.generate_action(
                    role=role,
                    market_obs=expert_obs,
                    predecessor_actions=predecessor_actions if predecessor_actions else None,
                )

                # 计算log_prob
                log_prob, _ = manager.get_action_log_prob(
                    role=role,
                    obs=expert_obs,
                    action_tokens=tokens,
                    predecessor_actions=predecessor_actions if predecessor_actions else None,
                )

                all_actions[role] = action_dict
                step_actions.append(action_dict)
                step_action_tokens.append(tokens.detach().cpu())
                step_log_probs.append(log_prob.item())

                # 🔍 V4调试: 打印每个Expert的动作决策
                flush_print(f"    [DEBUG] {role}: action={action_dict.get('action', 'N/A')}, "
                           f"confidence={action_dict.get('confidence', 0):.3f}, "
                           f"log_prob={log_prob.item():.4f}")

                # V4: 使用EnhancedCritic计算真正的value估计
                # 构建数值特征用于Critic
                try:
                    # Market features: 从indicators和prices构建
                    # EnhancedCritic期望: [batch, num_assets * 10 + 20]
                    market_feat_list = []

                    # 资产特征 (每个资产10个特征)
                    # V4修复: indicators结构是 indicators[symbol]["rsi"]，不是 indicators["rsi"][symbol]
                    for symbol in list(price_dict.keys())[:50]:  # 最多50个资产
                        price = price_dict.get(symbol, 0)
                        sym_ind = indicators.get(symbol, {})  # 获取该symbol的所有指标
                        # 简化特征: 价格归一化 + indicator值
                        asset_feats = [
                            price / 1000.0,  # 归一化价格
                            sym_ind.get("rsi", 50) / 100.0,
                            sym_ind.get("returns_1d", 0) * 10,  # 日收益放大
                            (price / sym_ind.get("ma_20", price) - 1.0) if sym_ind.get("ma_20", 0) > 0 else 0,  # 价格相对MA20
                            (price / sym_ind.get("ma_5", price) - 1.0) if sym_ind.get("ma_5", 0) > 0 else 0,  # 价格相对MA5
                            sym_ind.get("volatility", 0.02),  # 波动率
                            1.0 if price > sym_ind.get("ma_20", price) else -1.0,  # 趋势方向
                            sym_ind.get("volatility", 0.2) * 5,  # 波动率放大
                            0.0,  # 预留
                            0.0,  # 预留
                        ]
                        market_feat_list.extend(asset_feats)

                    # 补齐到50个资产 (500维)
                    while len(market_feat_list) < 500:
                        market_feat_list.extend([0.0] * 10)
                    market_feat_list = market_feat_list[:500]  # 截断

                    # 宏观特征 (20维)
                    # V4修复: 从indicators["_macro"]获取宏观指标
                    macro_ind = indicators.get("_macro", {})
                    macro_feats = [
                        macro_ind.get("vix", 20) / 100.0,
                        macro_ind.get("spy_return", 0),
                        macro_ind.get("bond_yield", 0.04),
                        macro_ind.get("usd_index", 100) / 100.0,
                        macro_ind.get("oil_price", 70) / 100.0,
                        macro_ind.get("gold_price", 2000) / 2000.0,
                        macro_ind.get("market_breadth", 0.5),
                        macro_ind.get("put_call_ratio", 1.0),
                        macro_ind.get("yield_curve", 0),
                        macro_ind.get("credit_spread", 0.01),
                        # 填充剩余维度
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                    ]
                    market_feat_list.extend(macro_feats[:20])

                    # Portfolio features: 从portfolio状态构建
                    # EnhancedCritic期望: [batch, 10]
                    portfolio_feats = [
                        portfolio.cash / portfolio.initial_capital,  # 现金比例
                        portfolio.portfolio_value / portfolio.initial_capital,  # 总价值比例
                        portfolio.total_return,  # 总收益率
                        len(portfolio.positions) / 50.0,  # 持仓数量归一化
                        portfolio.get_metrics().get("sharpe_ratio", 0) / 3.0,  # Sharpe归一化
                        portfolio.get_metrics().get("max_drawdown", 0),  # 最大回撤
                        portfolio.get_metrics().get("volatility", 0.02),  # 波动率
                        portfolio.get_metrics().get("win_rate", 0.5),  # 胜率
                        min(1.0, sum(p.shares * price_dict.get(p.symbol, 0)
                                     for p in portfolio.positions.values()) / portfolio.initial_capital),  # 权益占比
                        0.0,  # 预留
                    ]

                    # 转换为tensor
                    market_features_t = torch.tensor(
                        [market_feat_list], dtype=torch.float32, device=device
                    )
                    portfolio_features_t = torch.tensor(
                        [portfolio_feats[:10]], dtype=torch.float32, device=device
                    )

                    # 使用Critic计算value (9个agents)
                    with torch.no_grad():
                        values = trainer.critic(market_features_t, portfolio_features_t)
                        # values shape: [1, num_agents], 获取当前expert的value
                        agent_idx = len(step_values)  # 当前是第几个agent
                        if agent_idx < values.shape[1]:
                            step_values.append(values[0, agent_idx].item())
                        else:
                            step_values.append(values[0, 0].item())  # fallback
                except Exception as e:
                    # 如果计算失败，使用简单的value估计
                    simple_value = portfolio.total_return * 10  # 基于收益的简单估计
                    step_values.append(simple_value)

            # ============================================================
            # V4: 管理层协调决策
            # ============================================================
            manager_decision = None
            if manager_integration is not None and manager_integration.enabled:
                try:
                    # 将Expert动作转换为报告格式 (V4修复: 使用SimpleNamespace支持属性访问)
                    from types import SimpleNamespace
                    expert_reports = {}
                    for role, action in all_actions.items():
                        reasoning = action.get("reasoning", "")
                        action_str = action.get("action", "HOLD")
                        # 从action推断overall_view: BUY->bullish, SELL->bearish, HOLD->neutral
                        if "BUY" in action_str or "INCREASE" in action_str:
                            overall_view = "bullish"
                        elif "SELL" in action_str or "SHORT" in action_str or "REDUCE" in action_str:
                            overall_view = "bearish"
                        else:
                            overall_view = "neutral"
                        # 使用SimpleNamespace允许属性访问 (包含所有必需属性)
                        expert_reports[role] = SimpleNamespace(
                            action=action_str,
                            confidence=action.get("confidence", 0.5),
                            reasoning=reasoning,
                            overall_view=overall_view,
                            recommendations=[],  # 空列表避免 AttributeError
                            risk_assessment={},  # 空字典
                            market_outlook="neutral",
                        )

                    # V4修复: 添加 returns 数据以避免 "Empty returns data" 警告
                    manager_market_data = {
                        "indicators": indicators,
                        "prices": price_dict,
                        "returns": returns_window if not returns_window.empty else pd.DataFrame(),
                        "macro": {"vix": indicators.get("vix", 20.0)},
                    }
                    manager_decision = manager_integration.coordinate_decision(
                        expert_reports=expert_reports,
                        market_data=manager_market_data,
                        current_portfolio=portfolio.get_weights(),
                        risk_constraints={"max_drawdown": 0.15},
                        portfolio_value=portfolio.portfolio_value,
                    )
                except Exception as e:
                    logger.warning(f"Manager coordination failed: {e}")

            # ============================================================
            # 执行交易 (V4修复: 支持所有动作类型和动态仓位)
            # ============================================================

            # 解析动作百分比的辅助函数
            def parse_action_percentage(action_str: str) -> float:
                """从动作字符串解析百分比 (e.g., 'BUY_50%' -> 0.50)"""
                import re
                match = re.search(r'(\d+)%', action_str)
                if match:
                    return float(match.group(1)) / 100.0
                # 默认百分比
                if "100" in action_str:
                    return 1.0
                elif "75" in action_str:
                    return 0.75
                elif "50" in action_str:
                    return 0.50
                elif "25" in action_str:
                    return 0.25
                return 0.25  # 默认25%

            # 只处理5个资产类Expert的交易 (不处理meta-level agents)
            asset_expert_roles = ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"]

            # 🔍 V4调试: 打印交易前Portfolio状态
            flush_print(f"    [DEBUG] === TRADE EXECUTION START ===")
            flush_print(f"    [DEBUG] Portfolio: cash=${portfolio.cash:,.2f}, value=${portfolio.portfolio_value:,.2f}, "
                       f"positions={len(portfolio.positions)}")

            trade_count = 0  # 跟踪执行的交易数
            for role in asset_expert_roles:
                if role not in all_actions:
                    continue

                action_dict = all_actions[role]
                action = action_dict.get("action", "HOLD")
                confidence = action_dict.get("confidence", 0.5)
                asset_class = expert_to_class[role]
                symbols = universe.get(asset_class, [])

                # 跳过HOLD动作
                if action == "HOLD" or "HOLD" in action:
                    continue

                # 解析仓位百分比
                position_pct = parse_action_percentage(action)

                # 根据confidence调整仓位 (高confidence -> 更大仓位)
                adjusted_pct = position_pct * (0.5 + confidence)  # 0.5x ~ 1.5x

                for symbol in symbols[:5]:  # 增加到5个symbols
                    if symbol not in price_dict:
                        continue

                    price = price_dict[symbol]
                    hedge_adj = hedge_weights.get(symbol, 1.0)

                    try:
                        if "BUY" in action:
                            # BUY: 使用现金的一定比例买入
                            # 每个Expert分配约20%的总配置空间 (5 experts)
                            base_allocation = 0.20  # 每个Expert的基础配置
                            buy_amount = portfolio.cash * base_allocation * adjusted_pct * max(0.5, hedge_adj)
                            shares = int(buy_amount / price)
                            if shares > 0 and buy_amount > 100:  # 最小交易金额$100
                                portfolio.execute_trade(
                                    symbol=symbol,
                                    shares=shares,
                                    price=price,
                                    asset_class=asset_class,
                                    timestamp=date_str,
                                )
                                trade_count += 1
                                flush_print(f"    [DEBUG] TRADE: BUY {shares} {symbol} @${price:.2f} = ${shares*price:,.2f}")

                        elif "SELL" in action:
                            # SELL: 卖出持有的仓位
                            position = portfolio.positions.get(symbol)
                            if position and position.shares > 0:
                                sell_shares = int(position.shares * adjusted_pct)
                                if sell_shares > 0:
                                    portfolio.execute_trade(
                                        symbol=symbol,
                                        shares=-sell_shares,  # 负数表示卖出
                                        price=price,
                                        asset_class=asset_class,
                                        timestamp=date_str,
                                    )
                                    trade_count += 1
                                    flush_print(f"    [DEBUG] TRADE: SELL {sell_shares} {symbol} @${price:.2f} = ${sell_shares*price:,.2f}")

                        elif "SHORT" in action and "COVER" not in action:
                            # SHORT: 做空 - 使用execute_trade的is_short参数
                            short_amount = portfolio.cash * 0.10 * adjusted_pct  # 保守的做空比例
                            shares = int(short_amount / price)
                            if shares > 0:
                                portfolio.execute_trade(
                                    symbol=symbol,
                                    shares=-shares,  # 负数表示做空
                                    price=price,
                                    asset_class=asset_class,
                                    timestamp=date_str,
                                    is_short=True,
                                )
                                trade_count += 1
                                flush_print(f"    [DEBUG] TRADE: SHORT {shares} {symbol} @${price:.2f} = ${shares*price:,.2f}")

                        elif "COVER" in action:
                            # COVER: 平空仓 - 通过position.is_short检查空头持仓
                            position = portfolio.positions.get(symbol)
                            if position and position.is_short and abs(position.shares) > 0:
                                cover_shares = int(abs(position.shares) * adjusted_pct)
                                if cover_shares > 0:
                                    portfolio.execute_trade(
                                        symbol=symbol,
                                        shares=cover_shares,  # 正数表示买回平仓
                                        price=price,
                                        asset_class=asset_class,
                                        timestamp=date_str,
                                    )
                                    trade_count += 1
                                    flush_print(f"    [DEBUG] TRADE: COVER {cover_shares} {symbol} @${price:.2f} = ${cover_shares*price:,.2f}")
                    except Exception as e:
                        logger.debug(f"Trade execution error for {symbol}: {e}")

            # 🔍 V4调试: 打印交易执行汇总
            flush_print(f"    [DEBUG] === TRADE EXECUTION END: {trade_count} trades executed ===")

            # 计算奖励
            if i + 1 < len(rebalance_days):
                next_date = rebalance_days[i + 1]
                next_prices = prices_df.loc[next_date]
                next_price_dict = {s: next_prices[s] for s in next_prices.index if not pd.isna(next_prices[s])}

                portfolio.update_prices(next_price_dict)

                portfolio_return = (portfolio.portfolio_value - value_before) / value_before

                spy_return = 0
                if "SPY" in price_dict and "SPY" in next_price_dict:
                    spy_return = (next_price_dict["SPY"] - price_dict["SPY"]) / price_dict["SPY"]

                team_reward = (portfolio_return - spy_return) * 10
                team_reward = np.clip(team_reward, -2.0, 2.0)

                # 🔍 V4调试: 打印奖励计算详情
                flush_print(f"    [DEBUG] === REWARD CALCULATION ===")
                flush_print(f"    [DEBUG] Portfolio: before=${value_before:,.2f}, after=${portfolio.portfolio_value:,.2f}, return={portfolio_return*100:.4f}%")
                flush_print(f"    [DEBUG] SPY: return={spy_return*100:.4f}%, Alpha={portfolio_return-spy_return:+.4f}")
                flush_print(f"    [DEBUG] Team Reward: {team_reward:.4f}")

                # 计算专业化个体奖励
                individual_rewards = None
                coordination_reward = 0.0

                if reward_calculator is not None:
                    asset_returns = {}
                    for cls, symbols in universe.items():
                        for symbol in symbols:
                            if symbol in price_dict and symbol in next_price_dict:
                                asset_returns[symbol] = (next_price_dict[symbol] - price_dict[symbol]) / price_dict[symbol]

                    individual_rewards = []

                    # V4修复: 计算所有资产的平均收益（用于meta-agents）
                    all_returns = list(asset_returns.values())
                    overall_avg_return = np.mean(all_returns) if all_returns else portfolio_return

                    # V4修复: 定义真实资产类别
                    real_asset_classes = {"stocks", "bonds", "commodities", "reits", "crypto"}

                    for expert_idx, role in enumerate(expert_order):
                        asset_class = expert_to_class[role]
                        reward_key = expert_to_reward_key[role]
                        action_dict = all_actions.get(role, {})

                        class_symbols = universe.get(asset_class, [])
                        class_returns = [asset_returns.get(s, 0) for s in class_symbols if s in asset_returns]

                        # V4修复: 如果是meta-agent（asset_class不在真实资产类别中），使用整体组合收益
                        if asset_class not in real_asset_classes:
                            avg_class_return = overall_avg_return
                        else:
                            avg_class_return = np.mean(class_returns) if class_returns else 0

                        action_str = action_dict.get("action", "HOLD")
                        if "BUY" in action_str:
                            signal = 0.5
                        elif "SELL" in action_str or "SHORT" in action_str:
                            signal = -0.5
                        else:
                            signal = 0.0

                        try:
                            # 🔍 V4: 使用reward_config中的参数计算奖励
                            if compute_modified_reward is not None and reward_config is not None:
                                # 检测市场状态 (用于自适应方案)
                                market_regime = "normal"
                                if detect_regime is not None and reward_config.regime_adaptive:
                                    try:
                                        recent_returns = prices_df.iloc[max(0, date_idx-20):date_idx].pct_change().dropna()
                                        if len(recent_returns) >= 10:
                                            spy_returns = recent_returns.get("SPY", recent_returns.iloc[:, 0])
                                            market_regime = detect_regime(spy_returns.values)
                                    except:
                                        pass

                                # 计算动量 (用于动量奖励)
                                momentum = 0.0
                                try:
                                    if date_idx >= 5:
                                        recent_spy = prices_df["SPY"].iloc[date_idx-5:date_idx]
                                        if len(recent_spy) >= 5:
                                            momentum = (recent_spy.iloc[-1] - recent_spy.iloc[0]) / recent_spy.iloc[0]
                                except:
                                    pass

                                # 使用修改后的奖励函数
                                modified_reward = compute_modified_reward(
                                    signal=signal,
                                    confidence=action_dict.get("confidence", 0.5),
                                    actual_return=avg_class_return,
                                    scheme=reward_config,
                                    market_regime=market_regime,
                                    momentum=momentum,
                                )
                                individual_rewards.append(modified_reward)
                            else:
                                # 回退到原始奖励计算
                                expert_reward_fn = reward_calculator.expert_rewards.get(reward_key) if reward_calculator else None
                                if expert_reward_fn:
                                    reward_result = expert_reward_fn.compute(
                                        signal=signal,
                                        confidence=action_dict.get("confidence", 0.5),
                                        actual_return=avg_class_return,
                                        historical_signals=[],
                                        historical_returns=[],
                                        portfolio_weight=0.2,
                                        asset_contribution=avg_class_return * 0.2,
                                    )
                                    individual_rewards.append(reward_result.total)
                                else:
                                    individual_rewards.append(team_reward)
                        except Exception as e:
                            individual_rewards.append(team_reward)

                    coordination_reward = 0.3 if portfolio_return > spy_return else -0.1

                epoch_rewards.append(team_reward)

                # 添加到buffer
                buffer.add(TrajectoryStep(
                    observations=step_observations,
                    actions=step_actions,
                    action_tokens=step_action_tokens,
                    log_probs=step_log_probs,
                    values=step_values,
                    reward=team_reward,
                    done=False,
                    individual_rewards=individual_rewards,
                    coordination_reward=coordination_reward,
                    manager_decision=manager_decision,
                    hedging_info={"weights": hedge_weights, "tool": args.hedge_tool},
                ))

                total_samples += 1

            # Rollout结束，执行PPO更新
            if len(buffer) >= args.rollout_length:
                flush_print(f"  [{date_str}] PPO Update (buffer={len(buffer)})")

                # V4修复: 使用EnhancedCritic计算next_values和构建特征
                try:
                    # 构建当前状态的market_features和portfolio_features
                    # V4修复: indicators结构是 indicators[symbol]["rsi"]
                    mf_list = []
                    for symbol in list(price_dict.keys())[:50]:
                        price = price_dict.get(symbol, 0)
                        sym_ind = indicators.get(symbol, {})  # 获取该symbol的所有指标
                        asset_feats = [
                            price / 1000.0,
                            sym_ind.get("rsi", 50) / 100.0,
                            sym_ind.get("returns_1d", 0) * 10,
                            (price / sym_ind.get("ma_20", price) - 1.0) if sym_ind.get("ma_20", 0) > 0 else 0,
                            (price / sym_ind.get("ma_5", price) - 1.0) if sym_ind.get("ma_5", 0) > 0 else 0,
                            sym_ind.get("volatility", 0.02),
                            1.0 if price > sym_ind.get("ma_20", price) else -1.0,
                            sym_ind.get("volatility", 0.2) * 5,
                            0.0,
                            0.0,
                        ]
                        mf_list.extend(asset_feats)
                    while len(mf_list) < 500:
                        mf_list.extend([0.0] * 10)
                    mf_list = mf_list[:500]
                    # V4修复: 从indicators["_macro"]获取宏观指标
                    macro_ind2 = indicators.get("_macro", {})
                    macro_feats = [
                        macro_ind2.get("vix", 20) / 100.0,
                        macro_ind2.get("spy_return", 0),
                        macro_ind2.get("bond_yield", 0.04),
                        macro_ind2.get("usd_index", 100) / 100.0,
                        macro_ind2.get("oil_price", 70) / 100.0,
                        macro_ind2.get("gold_price", 2000) / 2000.0,
                        macro_ind2.get("market_breadth", 0.5),
                        macro_ind2.get("put_call_ratio", 1.0),
                        macro_ind2.get("yield_curve", 0),
                        macro_ind2.get("credit_spread", 0.01),
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                    ]
                    mf_list.extend(macro_feats[:20])

                    pf_list = [
                        portfolio.cash / portfolio.initial_capital,
                        portfolio.portfolio_value / portfolio.initial_capital,
                        portfolio.total_return,
                        len(portfolio.positions) / 50.0,
                        portfolio.get_metrics().get("sharpe_ratio", 0) / 3.0,
                        portfolio.get_metrics().get("max_drawdown", 0),
                        portfolio.get_metrics().get("volatility", 0.02),
                        portfolio.get_metrics().get("win_rate", 0.5),
                        min(1.0, sum(p.shares * price_dict.get(p.symbol, 0)
                                     for p in portfolio.positions.values()) / portfolio.initial_capital),
                        0.0,
                    ]

                    market_features_t = torch.tensor([mf_list], dtype=torch.float32, device=device)
                    portfolio_features_t = torch.tensor([pf_list[:10]], dtype=torch.float32, device=device)

                    # 使用Critic计算next_values (9个agents)
                    with torch.no_grad():
                        next_values_tensor = trainer.critic(market_features_t, portfolio_features_t)
                        next_values = next_values_tensor[0].tolist()  # [9] agents
                except Exception as e:
                    logger.warning(f"Failed to compute next_values with Critic: {e}")
                    next_values = [portfolio.total_return * 10] * 9  # 简单估计作为fallback
                    market_features_t = None
                    portfolio_features_t = None

                stats = trainer.train_step(
                    buffer,
                    next_values,
                    market_features=market_features_t,
                    portfolio_features=portfolio_features_t,
                )

                flush_print(
                    f"    Policy Loss: {stats['policy_loss']:.4f} | "
                    f"Value Loss: {stats['value_loss']:.4f} | "
                    f"KL: {stats['kl_divergence']:.4f} | "
                    f"Clip: {stats['clip_fraction']:.2%}"
                )

                # 🔍 详细监控日志
                MONITOR.log_step(total_samples, {
                    "policy_loss": stats['policy_loss'],
                    "value_loss": stats['value_loss'],
                    "kl_divergence": stats['kl_divergence'],
                    "clip_fraction": stats['clip_fraction'],
                    "team_reward": team_reward,
                    "portfolio_return": portfolio_return * 100,
                    "portfolio_value": portfolio.portfolio_value,
                })

                buffer.clear()

            # 日志
            if (i + 1) % 20 == 0:
                flush_print(
                    f"  [{date_str}] Value: ${portfolio.portfolio_value:,.0f} | "
                    f"Return: {portfolio.total_return*100:.1f}% | "
                    f"Samples: {total_samples}"
                )

            # 定期保存检查点 (每 checkpoint_interval 步)
            if (i + 1) % args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(args.save_dir, f"step_e{epoch+1}_s{i+1}")
                manager.save_all_adapters(ckpt_dir)
                # 保存训练状态 (V4 Fix: 使用 epoch+1 保持与目录名一致)
                state = {"epoch": epoch + 1, "step": i + 1, "total_samples": total_samples}
                with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                    json.dump(state, f)
                flush_print(f"  >>> Checkpoint saved: {ckpt_dir}")

        # Epoch统计
        metrics = portfolio.get_metrics()
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0

        flush_print(f"\nEpoch {epoch + 1} Summary:")
        flush_print(f"  Final Value: ${portfolio.portfolio_value:,.0f}")
        flush_print(f"  Total Return: {portfolio.total_return*100:.2f}%")
        flush_print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        flush_print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        flush_print(f"  Avg Reward: {avg_reward:.4f}")

        # 🔍 详细监控日志 - Epoch结束
        MONITOR.log_epoch(epoch + 1, {
            "final_value": portfolio.portfolio_value,
            "total_return": portfolio.total_return * 100,
            "sharpe_ratio": metrics.get('sharpe_ratio', 0),
            "max_drawdown": metrics.get('max_drawdown', 0) * 100,
            "avg_reward": avg_reward,
            "total_samples": total_samples,
        })

        # 保存检查点
        checkpoint_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        manager.save_all_adapters(checkpoint_dir)
        flush_print(f"\n>>> Saved checkpoint to {checkpoint_dir}")

    # 最终保存
    final_dir = os.path.join(args.save_dir, "final")
    manager.save_all_adapters(final_dir)

    total_time = (datetime.now() - start_time).total_seconds()

    flush_print("\n" + "=" * 80)
    flush_print(" MARFT V4 Training Complete!")
    flush_print("=" * 80)
    flush_print(f" Algorithm: PPO + GAE + KL Constraint + Full Feature Integration")
    flush_print(f" Total Epochs: {args.num_epochs}")
    flush_print(f" Total Samples: {total_samples}")
    flush_print(f" Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    flush_print(f" Final Checkpoint: {final_dir}")
    flush_print("")
    flush_print(" Integrated Features:")
    flush_print(f"   - Factor Scorers: {HAS_FACTOR_SCORERS}")
    flush_print(f"   - Hedging Tools: {HAS_HEDGING_TOOLKIT}")
    flush_print(f"   - Strategies: {HAS_STRATEGIES}")
    flush_print(f"   - Manager Coordinator: {HAS_MANAGER_COORDINATOR}")
    flush_print(f"   - Data Bridge: {HAS_DATA_BRIDGE}")
    flush_print(f"   - Reward Functions: {HAS_REWARD_FUNCTIONS}")
    flush_print("=" * 80)

    # 🔍 详细监控日志 - 训练总结
    MONITOR.summary()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 🔍 详细监控日志 - 捕获致命错误
        MONITOR.log_error(e, "训练主循环发生致命错误")
        import traceback
        traceback.print_exc()
        raise
