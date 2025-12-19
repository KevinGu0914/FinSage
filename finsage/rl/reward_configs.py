"""
Reward Function Configurations for MARFT V4 Experiments

全面覆盖所有9个智能体的奖励调整方案:
- 5个专家智能体 (Stock, Bond, Commodity, REITs, Crypto)
- 4个管理层智能体 (PortfolioManager, PositionSizing, Hedging, Coordination)

三个实验方案:
方案A (Aggressive): 激进探索型 - 鼓励大胆交易
方案B (Balanced): 平衡型 - 风险收益平衡
方案C (Adaptive): 自适应型 - 根据市场状态调整

核心问题诊断 (涵盖全部智能体):

【专家层问题】
1. ExpertReward timing_penalty = -confidence * (1 + abs(return) * 10) 惩罚太重
2. 没有交易激励: HOLD是"安全"选择
3. 牛市中不交易不惩罚 (缺少机会成本)

【管理层问题】
4. PortfolioManager consensus_weight=0.25 过于依赖专家共识，不敢独立判断
5. PositionSizing target_volatility=0.12 过于保守，限制收益潜力
6. Hedging 牛市中对冲过多，吃掉收益
7. Coordination consistency_weight过高，抑制智能体独立决策

训练时段特点 (2023-01-01 ~ 2024-06-30):
- 2023年是股市强劲复苏年 (SPY +24%, QQQ +55%)
- REITs因高利率表现较弱
- 积极交易策略有利 (XGBoost达到20%+)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


# =============================================================================
# Expert Agent Configuration
# =============================================================================

@dataclass
class ExpertRewardConfig:
    """专家智能体奖励配置"""
    # 权重分配
    accuracy_weight: float = 0.35
    calibration_weight: float = 0.20
    timing_weight: float = 0.25
    contribution_weight: float = 0.20

    # 惩罚缩放系数 (1.0=原始, <1.0=减轻惩罚)
    wrong_direction_penalty_scale: float = 1.0
    timing_penalty_scale: float = 1.0

    # 交易激励
    trade_bonus: float = 0.0
    momentum_bonus: float = 0.0
    hold_penalty_in_uptrend: float = 0.0


# =============================================================================
# Portfolio Manager Configuration
# =============================================================================

@dataclass
class PortfolioManagerRewardConfig:
    """投资组合经理奖励配置"""
    # 权重分配
    return_weight: float = 0.35
    consensus_weight: float = 0.25  # 降低 = 更独立决策
    quality_weight: float = 0.25
    timing_weight: float = 0.15

    # 收益预期
    target_sharpe: float = 1.5
    bull_market_return_expectation: float = 0.002  # 牛市日收益期望
    bear_market_return_expectation: float = -0.001

    # 独立决策激励
    independent_decision_bonus: float = 0.0  # 敢于偏离共识的奖励
    consensus_deviation_tolerance: float = 0.1  # 允许偏离的阈值


# =============================================================================
# Position Sizing Configuration
# =============================================================================

@dataclass
class PositionSizingRewardConfig:
    """仓位规模智能体奖励配置"""
    # 权重分配
    risk_parity_weight: float = 0.30
    kelly_weight: float = 0.25
    vol_target_weight: float = 0.30
    liquidity_weight: float = 0.15

    # 波动率目标 (年化)
    target_volatility: float = 0.12  # 提高 = 更激进
    vol_deviation_tolerance: float = 0.3  # 允许偏离目标的比例

    # 集中度容忍
    max_single_position: float = 0.25  # 提高 = 允许更集中
    concentration_bonus: float = 0.0  # 高确信度集中投资奖励


# =============================================================================
# Hedging Configuration
# =============================================================================

@dataclass
class HedgingRewardConfig:
    """对冲智能体奖励配置"""
    # 权重分配
    tail_risk_weight: float = 0.35
    cost_efficiency_weight: float = 0.25
    vix_response_weight: float = 0.25
    dynamic_weight: float = 0.15

    # 对冲成本阈值
    max_hedge_cost: float = 0.02
    max_hedge_ratio_bull: float = 0.15  # 牛市最大对冲比例
    max_hedge_ratio_bear: float = 0.40  # 熊市最大对冲比例

    # 牛市惩罚
    overhedge_penalty_in_bull: float = 0.0  # 牛市过度对冲惩罚
    underhedge_penalty_in_bear: float = 0.0  # 熊市对冲不足惩罚


# =============================================================================
# Coordination Configuration
# =============================================================================

@dataclass
class CoordinationRewardConfig:
    """协调奖励配置"""
    # 权重分配
    consistency_weight: float = 0.25  # 降低 = 允许更多分歧
    info_utilization_weight: float = 0.25
    conflict_resolution_weight: float = 0.25
    efficiency_weight: float = 0.25

    # 独立性激励
    diversity_bonus: float = 0.0  # 观点多样性奖励
    max_discussion_rounds: int = 3  # 超过则惩罚


# =============================================================================
# Complete Reward Scheme Configuration
# =============================================================================

@dataclass
class RewardSchemeConfig:
    """完整奖励方案配置 - 覆盖全部9个智能体"""
    name: str
    description: str

    # 专家层配置
    expert_config: ExpertRewardConfig = field(default_factory=ExpertRewardConfig)

    # 管理层配置
    portfolio_manager_config: PortfolioManagerRewardConfig = field(
        default_factory=PortfolioManagerRewardConfig
    )
    position_sizing_config: PositionSizingRewardConfig = field(
        default_factory=PositionSizingRewardConfig
    )
    hedging_config: HedgingRewardConfig = field(
        default_factory=HedgingRewardConfig
    )
    coordination_config: CoordinationRewardConfig = field(
        default_factory=CoordinationRewardConfig
    )

    # 全局PPO超参数调整
    entropy_coef: float = 0.01
    clip_param: float = 0.2

    # 全局风险容忍度
    max_drawdown_tolerance: float = 0.15
    volatility_target: float = 0.12

    # 市场状态自适应 (方案C专用)
    regime_adaptive: bool = False
    bull_aggression_multiplier: float = 1.0
    bear_caution_multiplier: float = 1.0

    # 向后兼容的属性 (映射到expert_config)
    @property
    def accuracy_weight(self) -> float:
        return self.expert_config.accuracy_weight

    @property
    def calibration_weight(self) -> float:
        return self.expert_config.calibration_weight

    @property
    def timing_weight(self) -> float:
        return self.expert_config.timing_weight

    @property
    def contribution_weight(self) -> float:
        return self.expert_config.contribution_weight

    @property
    def wrong_direction_penalty_scale(self) -> float:
        return self.expert_config.wrong_direction_penalty_scale

    @property
    def timing_penalty_scale(self) -> float:
        return self.expert_config.timing_penalty_scale

    @property
    def trade_bonus(self) -> float:
        return self.expert_config.trade_bonus

    @property
    def momentum_bonus(self) -> float:
        return self.expert_config.momentum_bonus

    @property
    def hold_penalty_in_uptrend(self) -> float:
        return self.expert_config.hold_penalty_in_uptrend


# ============================================================
# 方案A: 激进探索型 (Aggressive Exploration)
# ============================================================
SCHEME_A_AGGRESSIVE = RewardSchemeConfig(
    name="aggressive",
    description="""
    激进探索型: 鼓励大胆交易，全部9个智能体协同激进

    【专家层调整】
    1. 错误惩罚减半 (wrong_direction_penalty_scale=0.5)
    2. 时机惩罚大幅降低 (timing_penalty_scale=0.3)
    3. 增加交易激励 (trade_bonus=0.2)
    4. 增加动量跟随奖励 (momentum_bonus=0.3)
    5. 牛市中HOLD有惩罚 (hold_penalty_in_uptrend=0.15)

    【管理层调整】
    6. PM: 降低共识依赖 (consensus_weight=0.15)，鼓励独立决策
    7. PS: 提高波动率目标到18%，允许更集中仓位
    8. Hedging: 牛市最大对冲比例10%，过度对冲惩罚
    9. Coordination: 降低一致性要求，鼓励观点多样性

    预期效果:
    - 全系统更愿意承担风险
    - 牛市中更激进捕捉涨幅
    - 适合单边上涨市场
    """,

    # 专家层配置
    expert_config=ExpertRewardConfig(
        accuracy_weight=0.30,
        calibration_weight=0.15,
        timing_weight=0.20,
        contribution_weight=0.35,
        wrong_direction_penalty_scale=0.5,
        timing_penalty_scale=0.3,
        trade_bonus=0.2,
        momentum_bonus=0.3,
        hold_penalty_in_uptrend=0.15,
    ),

    # Portfolio Manager: 激进配置
    portfolio_manager_config=PortfolioManagerRewardConfig(
        return_weight=0.45,  # 更看重收益
        consensus_weight=0.15,  # 降低共识依赖
        quality_weight=0.25,
        timing_weight=0.15,
        target_sharpe=1.2,  # 降低夏普要求
        bull_market_return_expectation=0.003,  # 提高牛市期望
        independent_decision_bonus=0.15,  # 鼓励独立决策
        consensus_deviation_tolerance=0.15,
    ),

    # Position Sizing: 激进配置
    position_sizing_config=PositionSizingRewardConfig(
        risk_parity_weight=0.20,  # 降低风险平价要求
        kelly_weight=0.35,  # 更看重Kelly
        vol_target_weight=0.25,
        liquidity_weight=0.20,
        target_volatility=0.18,  # 允许更高波动
        vol_deviation_tolerance=0.4,
        max_single_position=0.30,  # 允许更集中
        concentration_bonus=0.1,
    ),

    # Hedging: 激进配置 (减少对冲)
    hedging_config=HedgingRewardConfig(
        tail_risk_weight=0.25,  # 降低尾部风险权重
        cost_efficiency_weight=0.35,  # 更看重成本
        vix_response_weight=0.20,
        dynamic_weight=0.20,
        max_hedge_cost=0.01,  # 降低对冲成本容忍
        max_hedge_ratio_bull=0.10,  # 牛市低对冲
        max_hedge_ratio_bear=0.30,
        overhedge_penalty_in_bull=0.2,  # 牛市过度对冲惩罚
    ),

    # Coordination: 激进配置
    coordination_config=CoordinationRewardConfig(
        consistency_weight=0.15,  # 降低一致性要求
        info_utilization_weight=0.30,
        conflict_resolution_weight=0.25,
        efficiency_weight=0.30,
        diversity_bonus=0.1,  # 鼓励观点多样性
    ),

    # 全局PPO调整
    entropy_coef=0.05,
    clip_param=0.25,
    max_drawdown_tolerance=0.20,
    volatility_target=0.18,
    regime_adaptive=False,
)


# ============================================================
# 方案B: 平衡型 (Balanced)
# ============================================================
SCHEME_B_BALANCED = RewardSchemeConfig(
    name="balanced",
    description="""
    平衡型: 在风险和收益之间取得平衡，全部9个智能体协同稳健

    【专家层调整】
    1. 错误惩罚适度降低 (penalty_scale=0.7)
    2. 适度交易激励 (trade_bonus=0.1)
    3. 适度机会成本惩罚 (hold_penalty=0.1)

    【管理层调整】
    4. PM: 平衡共识与独立 (consensus_weight=0.20)
    5. PS: 中等波动率目标15%
    6. Hedging: 保持适度对冲能力
    7. Coordination: 维持中等一致性要求

    预期效果:
    - 稳健的风险调整后收益
    - 适合震荡或不确定市场
    - 波动控制与收益的平衡
    """,

    # 专家层配置
    expert_config=ExpertRewardConfig(
        accuracy_weight=0.30,
        calibration_weight=0.20,
        timing_weight=0.25,
        contribution_weight=0.25,
        wrong_direction_penalty_scale=0.7,
        timing_penalty_scale=0.6,
        trade_bonus=0.1,
        momentum_bonus=0.15,
        hold_penalty_in_uptrend=0.1,
    ),

    # Portfolio Manager: 平衡配置
    portfolio_manager_config=PortfolioManagerRewardConfig(
        return_weight=0.35,
        consensus_weight=0.20,
        quality_weight=0.25,
        timing_weight=0.20,
        target_sharpe=1.5,
        bull_market_return_expectation=0.002,
        independent_decision_bonus=0.05,
        consensus_deviation_tolerance=0.1,
    ),

    # Position Sizing: 平衡配置
    position_sizing_config=PositionSizingRewardConfig(
        risk_parity_weight=0.25,
        kelly_weight=0.30,
        vol_target_weight=0.30,
        liquidity_weight=0.15,
        target_volatility=0.15,
        vol_deviation_tolerance=0.3,
        max_single_position=0.25,
        concentration_bonus=0.0,
    ),

    # Hedging: 平衡配置
    hedging_config=HedgingRewardConfig(
        tail_risk_weight=0.30,
        cost_efficiency_weight=0.25,
        vix_response_weight=0.25,
        dynamic_weight=0.20,
        max_hedge_cost=0.015,
        max_hedge_ratio_bull=0.15,
        max_hedge_ratio_bear=0.35,
        overhedge_penalty_in_bull=0.1,
    ),

    # Coordination: 平衡配置
    coordination_config=CoordinationRewardConfig(
        consistency_weight=0.20,
        info_utilization_weight=0.25,
        conflict_resolution_weight=0.30,
        efficiency_weight=0.25,
        diversity_bonus=0.05,
    ),

    # 全局PPO调整
    entropy_coef=0.02,
    clip_param=0.2,
    max_drawdown_tolerance=0.15,
    volatility_target=0.15,
    regime_adaptive=False,
)


# ============================================================
# 方案C: 自适应型 (Adaptive/Market-Aware)
# ============================================================
SCHEME_C_ADAPTIVE = RewardSchemeConfig(
    name="adaptive",
    description="""
    自适应型: 根据市场状态动态调整，全部9个智能体智能适应

    【专家层调整】
    1. 启用市场状态感知 (regime_adaptive=True)
    2. 牛市中激进 (bull_aggression_multiplier=1.5)
    3. 熊市中保守 (bear_caution_multiplier=0.7)

    【管理层调整】
    4. PM: 根据市场状态调整收益期望
    5. PS: 动态波动率目标 (牛市高，熊市低)
    6. Hedging: VIX响应更灵敏，牛市低对冲/熊市高对冲
    7. Coordination: 牛市允许分歧，熊市要求一致

    预期效果:
    - 自动适应不同市场环境
    - 牛市中捕捉更多涨幅
    - 熊市中控制回撤
    - 跨多种市场状态表现稳定
    """,

    # 专家层配置
    expert_config=ExpertRewardConfig(
        accuracy_weight=0.30,
        calibration_weight=0.15,
        timing_weight=0.30,
        contribution_weight=0.25,
        wrong_direction_penalty_scale=0.6,
        timing_penalty_scale=0.5,
        trade_bonus=0.15,
        momentum_bonus=0.25,
        hold_penalty_in_uptrend=0.12,
    ),

    # Portfolio Manager: 自适应配置
    portfolio_manager_config=PortfolioManagerRewardConfig(
        return_weight=0.40,
        consensus_weight=0.18,
        quality_weight=0.22,
        timing_weight=0.20,
        target_sharpe=1.3,
        bull_market_return_expectation=0.003,
        bear_market_return_expectation=-0.0005,
        independent_decision_bonus=0.1,
        consensus_deviation_tolerance=0.12,
    ),

    # Position Sizing: 自适应配置
    position_sizing_config=PositionSizingRewardConfig(
        risk_parity_weight=0.22,
        kelly_weight=0.33,
        vol_target_weight=0.28,
        liquidity_weight=0.17,
        target_volatility=0.15,  # 基础值，会动态调整
        vol_deviation_tolerance=0.35,
        max_single_position=0.28,
        concentration_bonus=0.05,
    ),

    # Hedging: 自适应配置
    hedging_config=HedgingRewardConfig(
        tail_risk_weight=0.28,
        cost_efficiency_weight=0.27,
        vix_response_weight=0.30,  # 更高VIX响应权重
        dynamic_weight=0.15,
        max_hedge_cost=0.018,
        max_hedge_ratio_bull=0.12,
        max_hedge_ratio_bear=0.40,
        overhedge_penalty_in_bull=0.15,
        underhedge_penalty_in_bear=0.15,
    ),

    # Coordination: 自适应配置
    coordination_config=CoordinationRewardConfig(
        consistency_weight=0.18,
        info_utilization_weight=0.28,
        conflict_resolution_weight=0.27,
        efficiency_weight=0.27,
        diversity_bonus=0.08,
    ),

    # 全局PPO调整
    entropy_coef=0.03,
    clip_param=0.2,
    max_drawdown_tolerance=0.18,
    volatility_target=0.15,

    # 市场状态感知
    regime_adaptive=True,
    bull_aggression_multiplier=1.5,
    bear_caution_multiplier=0.7,
)


# ============================================================
# 配置字典
# ============================================================
REWARD_SCHEMES = {
    "aggressive": SCHEME_A_AGGRESSIVE,
    "balanced": SCHEME_B_BALANCED,
    "adaptive": SCHEME_C_ADAPTIVE,
    "default": SCHEME_B_BALANCED,  # 默认使用平衡型
}


def get_reward_scheme(name: str) -> RewardSchemeConfig:
    """获取奖励方案配置"""
    if name not in REWARD_SCHEMES:
        raise ValueError(f"Unknown reward scheme: {name}. Available: {list(REWARD_SCHEMES.keys())}")
    return REWARD_SCHEMES[name]


# ============================================================
# 修改后的ExpertReward计算函数
# ============================================================

def compute_modified_expert_reward(
    signal: float,
    confidence: float,
    actual_return: float,
    scheme: RewardSchemeConfig,
    market_regime: str = "normal",
    momentum: float = 0.0,
) -> float:
    """
    使用配置方案计算修改后的Expert奖励

    Args:
        signal: 交易信号 (-1 to 1)
        confidence: 置信度 (0 to 1)
        actual_return: 实际收益率
        scheme: 奖励方案配置
        market_regime: 市场状态 (bull/bear/normal)
        momentum: 市场动量

    Returns:
        修改后的奖励值
    """
    reward = 0.0

    # 1. 准确度奖励/惩罚
    direction_correct = signal * actual_return > 0
    if direction_correct:
        accuracy_reward = min(abs(signal * actual_return * 50), 1.0)
    else:
        # 应用惩罚缩放
        accuracy_reward = -min(abs(signal * actual_return * 50), 1.0) * scheme.wrong_direction_penalty_scale

    reward += scheme.accuracy_weight * accuracy_reward

    # 2. 时机奖励/惩罚
    if direction_correct:
        timing_reward = confidence * (0.5 + abs(actual_return) * 20)
    else:
        # 应用时机惩罚缩放
        timing_reward = -confidence * (0.5 + abs(actual_return) * 20) * scheme.timing_penalty_scale

    timing_reward = np.clip(timing_reward, -1, 1)
    reward += scheme.timing_weight * timing_reward

    # 3. 交易激励
    if abs(signal) > 0.1:  # 有交易信号
        reward += scheme.trade_bonus

    # 4. 动量跟随奖励
    if signal * momentum > 0:  # 信号与动量同向
        reward += scheme.momentum_bonus * min(abs(momentum) * 20, 1.0)

    # 5. HOLD惩罚 (如果在上涨趋势中HOLD)
    if abs(signal) < 0.1 and actual_return > 0.005:  # HOLD但市场上涨
        reward -= scheme.hold_penalty_in_uptrend

    # 6. 市场状态自适应调整 (方案C)
    if scheme.regime_adaptive:
        if market_regime == "bull":
            # 牛市中放大积极奖励
            if reward > 0:
                reward *= scheme.bull_aggression_multiplier
            # 牛市中HOLD额外惩罚
            if abs(signal) < 0.1:
                reward -= 0.1
        elif market_regime == "bear":
            # 熊市中放大保守奖励
            if signal < 0 and actual_return < 0:  # 正确做空
                reward *= scheme.bear_caution_multiplier
            # 熊市中激进交易额外惩罚
            if signal > 0.5 and actual_return < 0:
                reward *= scheme.bear_caution_multiplier

    return np.clip(reward, -2.0, 2.0)


def detect_market_regime(
    returns: np.ndarray,
    lookback: int = 20,
) -> str:
    """
    检测市场状态

    Args:
        returns: 收益率序列
        lookback: 回看天数

    Returns:
        "bull", "bear", or "normal"
    """
    if len(returns) < lookback:
        return "normal"

    recent_returns = returns[-lookback:]
    cumulative_return = (1 + recent_returns).prod() - 1
    volatility = np.std(recent_returns) * np.sqrt(252)

    if cumulative_return > 0.05 and volatility < 0.25:  # 涨幅>5%，波动率<25%
        return "bull"
    elif cumulative_return < -0.05:  # 跌幅>5%
        return "bear"
    else:
        return "normal"


# ============================================================
# 方案对比说明
# ============================================================
"""
三个方案的关键差异对比:

                    方案A(激进)    方案B(平衡)    方案C(自适应)
                    -----------   -----------   -------------
错误惩罚缩放          0.5           0.7           0.6
时机惩罚缩放          0.3           0.6           0.5
交易激励              0.2           0.1           0.15
动量奖励              0.3           0.15          0.25
HOLD惩罚              0.15          0.1           0.12
Entropy系数           0.05          0.02          0.03
最大回撤容忍          20%           15%           18%
波动率目标            18%           15%           15%
市场状态感知          否            否            是

预期收益排序: A > C > B (牛市)
预期风险排序: A > C > B
预期夏普比: C > B > A (理想情况)

建议:
- 如果训练/验证时段主要是牛市: 优先选A
- 如果时段包含震荡期: 优先选B
- 如果时段跨越多种市场状态: 优先选C
"""
