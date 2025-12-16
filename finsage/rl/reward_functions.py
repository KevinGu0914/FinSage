"""
Specialized Reward Functions for FinSage Multi-Agent System
专业化奖励函数模块 - 为每种智能体类型设计独立奖励

Based on MARFT (Multi-Agent Reinforcement Fine-Tuning) paper principles:
- Individual agent rewards for credit assignment
- Coordination bonus for cooperative behavior
- Task-specific metrics for each agent role
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Base Reward Class
# =============================================================================

@dataclass
class RewardComponents:
    """奖励组成部分的详细分解"""
    total: float
    components: Dict[str, float]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "components": self.components,
            "description": self.description
        }


class BaseAgentReward(ABC):
    """智能体奖励函数基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def compute(self, **kwargs) -> RewardComponents:
        """计算奖励"""
        pass

    def _clip_reward(self, reward: float, min_val: float = -10.0, max_val: float = 10.0) -> float:
        """裁剪奖励值防止梯度爆炸"""
        return np.clip(reward, min_val, max_val)


# =============================================================================
# Portfolio Manager Reward
# =============================================================================

class PortfolioManagerReward(BaseAgentReward):
    """
    投资组合经理奖励函数

    核心指标:
    1. 收益贡献 - 组合回报
    2. 专家共识度 - 与专家建议的一致性
    3. 配置质量 - 风险调整后收益
    4. 时机选择 - 市场时机把握能力
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # 权重系数
        self.return_weight = self.config.get("return_weight", 0.35)
        self.consensus_weight = self.config.get("consensus_weight", 0.25)
        self.quality_weight = self.config.get("quality_weight", 0.25)
        self.timing_weight = self.config.get("timing_weight", 0.15)

        # 风险调整参数
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        self.target_sharpe = self.config.get("target_sharpe", 1.5)

    def compute(
        self,
        portfolio_return: float,
        portfolio_volatility: float,
        expert_recommendations: Dict[str, Dict[str, float]],
        actual_allocation: Dict[str, float],
        asset_returns: Dict[str, float],
        market_regime: str = "normal",
        **kwargs
    ) -> RewardComponents:
        """
        计算投资组合经理的奖励

        Args:
            portfolio_return: 组合收益率
            portfolio_volatility: 组合波动率
            expert_recommendations: 专家建议 {expert_name: {asset: weight}}
            actual_allocation: 实际配置
            asset_returns: 各资产实际回报
            market_regime: 市场状态 (bull/bear/normal/volatile)
        """
        components = {}

        # 1. 收益奖励 (风险调整后)
        if portfolio_volatility > 0:
            sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_volatility
            return_reward = sharpe / self.target_sharpe  # 归一化
        else:
            return_reward = portfolio_return * 10  # 低波动时直接用收益
        components["return_reward"] = self._clip_reward(return_reward, -2, 2)

        # 2. 专家共识奖励
        consensus_reward = self._compute_consensus_reward(
            expert_recommendations, actual_allocation
        )
        components["consensus_reward"] = consensus_reward

        # 3. 配置质量奖励
        quality_reward = self._compute_allocation_quality(
            actual_allocation, asset_returns, portfolio_return
        )
        components["quality_reward"] = quality_reward

        # 4. 时机选择奖励
        timing_reward = self._compute_timing_reward(
            portfolio_return, market_regime, portfolio_volatility
        )
        components["timing_reward"] = timing_reward

        # 总奖励
        total = (
            self.return_weight * components["return_reward"] +
            self.consensus_weight * components["consensus_reward"] +
            self.quality_weight * components["quality_reward"] +
            self.timing_weight * components["timing_reward"]
        )

        description = (
            f"PM Reward: return={components['return_reward']:.3f}, "
            f"consensus={components['consensus_reward']:.3f}, "
            f"quality={components['quality_reward']:.3f}, "
            f"timing={components['timing_reward']:.3f}"
        )

        return RewardComponents(
            total=self._clip_reward(total),
            components=components,
            description=description
        )

    def _compute_consensus_reward(
        self,
        expert_recommendations: Dict[str, Dict[str, float]],
        actual_allocation: Dict[str, float]
    ) -> float:
        """计算与专家建议的一致性奖励"""
        if not expert_recommendations:
            return 0.0

        # 计算专家平均建议
        avg_recommendation = {}
        for expert_name, rec in expert_recommendations.items():
            for asset, weight in rec.items():
                if asset not in avg_recommendation:
                    avg_recommendation[asset] = []
                avg_recommendation[asset].append(weight)

        avg_recommendation = {
            asset: np.mean(weights)
            for asset, weights in avg_recommendation.items()
        }

        # 计算与实际配置的偏离度
        all_assets = set(avg_recommendation.keys()) | set(actual_allocation.keys())
        deviations = []
        for asset in all_assets:
            avg_w = avg_recommendation.get(asset, 0)
            actual_w = actual_allocation.get(asset, 0)
            deviations.append(abs(avg_w - actual_w))

        avg_deviation = np.mean(deviations) if deviations else 0

        # 偏离度越小，奖励越高 (使用指数衰减)
        consensus_reward = np.exp(-5 * avg_deviation)  # 偏离10%时奖励约0.6

        return self._clip_reward(consensus_reward, 0, 1)

    def _compute_allocation_quality(
        self,
        allocation: Dict[str, float],
        asset_returns: Dict[str, float],
        portfolio_return: float
    ) -> float:
        """计算配置质量 - 是否选对了资产"""
        if not asset_returns or not allocation:
            return 0.0

        # 计算等权配置的收益作为基准
        equal_return = np.mean(list(asset_returns.values()))

        # 超额收益
        excess_return = portfolio_return - equal_return

        # 配置质量 = 超额收益的函数
        quality = np.tanh(excess_return * 50)  # 归一化到 [-1, 1]

        return self._clip_reward(quality, -1, 1)

    def _compute_timing_reward(
        self,
        portfolio_return: float,
        market_regime: str,
        volatility: float
    ) -> float:
        """计算时机选择奖励"""
        # 根据市场状态调整预期
        regime_expectations = {
            "bull": 0.001,     # 牛市期望正收益
            "bear": -0.001,   # 熊市期望少亏
            "volatile": 0.0,  # 高波动期望保本
            "normal": 0.0005  # 正常期望小正收益
        }

        expected = regime_expectations.get(market_regime, 0.0)

        # 超预期奖励
        if market_regime == "bear":
            # 熊市中少亏就是好的
            timing_reward = 1.0 if portfolio_return > expected else -0.5
        elif market_regime == "volatile":
            # 高波动中控制波动是关键
            timing_reward = 1.0 if volatility < 0.02 else -0.3
        else:
            # 其他情况看超额收益
            timing_reward = np.tanh((portfolio_return - expected) * 100)

        return self._clip_reward(timing_reward, -1, 1)


# =============================================================================
# Position Sizing Reward
# =============================================================================

class PositionSizingReward(BaseAgentReward):
    """
    仓位规模智能体奖励函数

    核心指标:
    1. 风险平价质量 - 各资产风险贡献的均衡度
    2. Kelly效率 - 与理论Kelly仓位的偏离
    3. 波动率目标 - 实际波动率与目标的偏离
    4. 流动性考量 - 考虑流动性约束的合理性
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.risk_parity_weight = self.config.get("risk_parity_weight", 0.30)
        self.kelly_weight = self.config.get("kelly_weight", 0.25)
        self.vol_target_weight = self.config.get("vol_target_weight", 0.30)
        self.liquidity_weight = self.config.get("liquidity_weight", 0.15)

        self.target_volatility = self.config.get("target_volatility", 0.12)

    def compute(
        self,
        position_sizes: Dict[str, float],
        asset_volatilities: Dict[str, float],
        asset_returns: Dict[str, float],
        portfolio_volatility: float,
        risk_contributions: Dict[str, float],
        liquidity_scores: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RewardComponents:
        """
        计算仓位规模智能体的奖励

        Args:
            position_sizes: 各资产仓位
            asset_volatilities: 各资产波动率
            asset_returns: 各资产收益率
            portfolio_volatility: 组合波动率
            risk_contributions: 各资产风险贡献
            liquidity_scores: 流动性评分 (0-1)
        """
        components = {}

        # 1. 风险平价质量
        risk_parity_reward = self._compute_risk_parity_quality(risk_contributions)
        components["risk_parity"] = risk_parity_reward

        # 2. Kelly效率
        kelly_reward = self._compute_kelly_efficiency(
            position_sizes, asset_returns, asset_volatilities
        )
        components["kelly_efficiency"] = kelly_reward

        # 3. 波动率目标达成
        vol_target_reward = self._compute_vol_target_reward(portfolio_volatility)
        components["vol_target"] = vol_target_reward

        # 4. 流动性考量
        liquidity_reward = self._compute_liquidity_reward(
            position_sizes, liquidity_scores
        )
        components["liquidity"] = liquidity_reward

        # 总奖励
        total = (
            self.risk_parity_weight * components["risk_parity"] +
            self.kelly_weight * components["kelly_efficiency"] +
            self.vol_target_weight * components["vol_target"] +
            self.liquidity_weight * components["liquidity"]
        )

        description = (
            f"PS Reward: risk_parity={components['risk_parity']:.3f}, "
            f"kelly={components['kelly_efficiency']:.3f}, "
            f"vol_target={components['vol_target']:.3f}, "
            f"liquidity={components['liquidity']:.3f}"
        )

        return RewardComponents(
            total=self._clip_reward(total),
            components=components,
            description=description
        )

    def _compute_risk_parity_quality(
        self,
        risk_contributions: Dict[str, float]
    ) -> float:
        """计算风险平价质量 - 风险贡献越均匀越好"""
        if not risk_contributions or len(risk_contributions) < 2:
            return 0.0

        contributions = list(risk_contributions.values())
        n = len(contributions)

        # 理想情况每个资产贡献 1/n
        target_contrib = 1.0 / n

        # 计算与理想分布的偏离 (用标准差)
        deviations = [abs(c - target_contrib) for c in contributions]
        avg_deviation = np.mean(deviations)

        # 偏离度越小，奖励越高
        quality = np.exp(-10 * avg_deviation)  # 偏离5%时奖励约0.6

        return self._clip_reward(quality, 0, 1)

    def _compute_kelly_efficiency(
        self,
        position_sizes: Dict[str, float],
        asset_returns: Dict[str, float],
        asset_volatilities: Dict[str, float]
    ) -> float:
        """计算与Kelly最优仓位的偏离"""
        if not position_sizes or not asset_returns or not asset_volatilities:
            return 0.0

        kelly_optimal = {}
        for asset in position_sizes:
            if asset in asset_returns and asset in asset_volatilities:
                ret = asset_returns[asset]
                vol = asset_volatilities[asset]
                if vol > 0:
                    # Half Kelly for safety
                    kelly_f = 0.5 * ret / (vol ** 2) if vol > 0 else 0
                    kelly_optimal[asset] = max(0, kelly_f)
                else:
                    kelly_optimal[asset] = 0

        if not kelly_optimal:
            return 0.0

        # 归一化Kelly仓位
        total_kelly = sum(kelly_optimal.values())
        if total_kelly > 0:
            kelly_optimal = {k: v / total_kelly for k, v in kelly_optimal.items()}

        # 计算与实际仓位的偏离
        deviations = []
        for asset in position_sizes:
            actual = position_sizes[asset]
            optimal = kelly_optimal.get(asset, 0)
            deviations.append(abs(actual - optimal))

        avg_deviation = np.mean(deviations) if deviations else 0

        # 偏离度越小，效率越高
        efficiency = np.exp(-5 * avg_deviation)

        return self._clip_reward(efficiency, 0, 1)

    def _compute_vol_target_reward(self, portfolio_volatility: float) -> float:
        """计算波动率目标达成度"""
        # 年化目标波动率
        target_vol = self.target_volatility / np.sqrt(252)  # 日波动率

        # 与目标的偏离
        deviation = abs(portfolio_volatility - target_vol) / target_vol

        # 偏离度越小，奖励越高
        reward = np.exp(-3 * deviation)  # 偏离20%时奖励约0.55

        return self._clip_reward(reward, 0, 1)

    def _compute_liquidity_reward(
        self,
        position_sizes: Dict[str, float],
        liquidity_scores: Optional[Dict[str, float]]
    ) -> float:
        """计算流动性约束遵守度"""
        if liquidity_scores is None:
            return 0.5  # 无流动性数据时给中等分

        # 低流动性资产应该给小仓位
        weighted_liquidity = 0.0
        total_weight = 0.0

        for asset, size in position_sizes.items():
            if asset in liquidity_scores:
                liq = liquidity_scores[asset]
                # 大仓位+低流动性 = 惩罚
                # 小仓位+低流动性 = OK
                # 大仓位+高流动性 = OK
                if liq < 0.5 and size > 0.15:
                    weighted_liquidity -= (size - 0.15) * (0.5 - liq) * 10
                else:
                    weighted_liquidity += liq * size
                total_weight += size

        if total_weight > 0:
            reward = weighted_liquidity / total_weight
        else:
            reward = 0.5

        return self._clip_reward(reward, -1, 1)


# =============================================================================
# Hedging Reward
# =============================================================================

class HedgingReward(BaseAgentReward):
    """
    对冲智能体奖励函数

    核心指标:
    1. 尾部风险保护 - VaR/CVaR改善程度
    2. 对冲成本效率 - 保护效果与成本的比值
    3. VIX响应 - 对市场波动的反应速度和准确性
    4. 动态调整 - 对冲比例的动态调整能力
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.tail_risk_weight = self.config.get("tail_risk_weight", 0.35)
        self.cost_efficiency_weight = self.config.get("cost_efficiency_weight", 0.25)
        self.vix_response_weight = self.config.get("vix_response_weight", 0.25)
        self.dynamic_weight = self.config.get("dynamic_weight", 0.15)

        self.var_threshold = self.config.get("var_threshold", 0.05)
        self.max_hedge_cost = self.config.get("max_hedge_cost", 0.02)

    def compute(
        self,
        var_before: float,
        var_after: float,
        cvar_before: float,
        cvar_after: float,
        hedge_cost: float,
        vix_level: float,
        vix_change: float,
        hedge_ratio: float,
        hedge_ratio_change: float,
        portfolio_return: float,
        **kwargs
    ) -> RewardComponents:
        """
        计算对冲智能体的奖励

        Args:
            var_before: 对冲前VaR
            var_after: 对冲后VaR
            cvar_before: 对冲前CVaR
            cvar_after: 对冲后CVaR
            hedge_cost: 对冲成本
            vix_level: 当前VIX水平
            vix_change: VIX变化率
            hedge_ratio: 当前对冲比例
            hedge_ratio_change: 对冲比例变化
            portfolio_return: 组合收益
        """
        components = {}

        # 1. 尾部风险保护
        tail_risk_reward = self._compute_tail_risk_protection(
            var_before, var_after, cvar_before, cvar_after
        )
        components["tail_risk"] = tail_risk_reward

        # 2. 成本效率
        cost_efficiency_reward = self._compute_cost_efficiency(
            var_before, var_after, hedge_cost
        )
        components["cost_efficiency"] = cost_efficiency_reward

        # 3. VIX响应
        vix_response_reward = self._compute_vix_response(
            vix_level, vix_change, hedge_ratio, hedge_ratio_change
        )
        components["vix_response"] = vix_response_reward

        # 4. 动态调整
        dynamic_reward = self._compute_dynamic_adjustment(
            hedge_ratio, vix_level, portfolio_return
        )
        components["dynamic_adjustment"] = dynamic_reward

        # 总奖励
        total = (
            self.tail_risk_weight * components["tail_risk"] +
            self.cost_efficiency_weight * components["cost_efficiency"] +
            self.vix_response_weight * components["vix_response"] +
            self.dynamic_weight * components["dynamic_adjustment"]
        )

        description = (
            f"Hedge Reward: tail_risk={components['tail_risk']:.3f}, "
            f"cost_eff={components['cost_efficiency']:.3f}, "
            f"vix_resp={components['vix_response']:.3f}, "
            f"dynamic={components['dynamic_adjustment']:.3f}"
        )

        return RewardComponents(
            total=self._clip_reward(total),
            components=components,
            description=description
        )

    def _compute_tail_risk_protection(
        self,
        var_before: float,
        var_after: float,
        cvar_before: float,
        cvar_after: float
    ) -> float:
        """计算尾部风险改善程度"""
        # VaR改善
        if var_before != 0:
            var_improvement = (abs(var_before) - abs(var_after)) / abs(var_before)
        else:
            var_improvement = 0.0

        # CVaR改善
        if cvar_before != 0:
            cvar_improvement = (abs(cvar_before) - abs(cvar_after)) / abs(cvar_before)
        else:
            cvar_improvement = 0.0

        # CVaR权重更高 (更关注极端风险)
        total_improvement = 0.4 * var_improvement + 0.6 * cvar_improvement

        # 转换为奖励
        reward = np.tanh(total_improvement * 5)  # 改善20%时奖励约0.76

        return self._clip_reward(reward, -1, 1)

    def _compute_cost_efficiency(
        self,
        var_before: float,
        var_after: float,
        hedge_cost: float
    ) -> float:
        """计算对冲成本效率"""
        if hedge_cost <= 0:
            return 1.0  # 无成本的保护是最好的

        # 风险减少量
        risk_reduction = abs(var_before) - abs(var_after)

        # 效率 = 风险减少 / 成本
        efficiency = risk_reduction / hedge_cost if hedge_cost > 0 else 0

        # 归一化 (假设合理效率在5左右)
        reward = np.tanh(efficiency / 5)

        # 如果成本过高，额外惩罚
        if hedge_cost > self.max_hedge_cost:
            reward -= 0.3 * (hedge_cost - self.max_hedge_cost) / self.max_hedge_cost

        return self._clip_reward(reward, -1, 1)

    def _compute_vix_response(
        self,
        vix_level: float,
        vix_change: float,
        hedge_ratio: float,
        hedge_ratio_change: float
    ) -> float:
        """计算对VIX的响应质量"""
        # 期望行为:
        # VIX上升 -> 增加对冲
        # VIX高位 -> 维持高对冲
        # VIX下降 -> 减少对冲

        # 1. 方向一致性
        if vix_change > 0:
            # VIX上升，应该增加对冲
            direction_match = 1 if hedge_ratio_change > 0 else -1
        elif vix_change < 0:
            # VIX下降，可以减少对冲
            direction_match = 1 if hedge_ratio_change <= 0 else 0
        else:
            direction_match = 0.5  # VIX不变，任何调整都可以

        # 2. 幅度适当性
        # VIX高时应该有更高的对冲比例
        if vix_level > 30:
            expected_ratio = 0.3 + (vix_level - 30) * 0.01  # VIX>30时增加对冲
        elif vix_level > 20:
            expected_ratio = 0.15 + (vix_level - 20) * 0.015
        else:
            expected_ratio = 0.05 + vix_level * 0.005

        expected_ratio = min(expected_ratio, 0.5)  # 最高50%对冲

        ratio_deviation = abs(hedge_ratio - expected_ratio)
        magnitude_score = np.exp(-5 * ratio_deviation)

        # 综合得分
        reward = 0.6 * direction_match + 0.4 * magnitude_score

        return self._clip_reward(reward, -1, 1)

    def _compute_dynamic_adjustment(
        self,
        hedge_ratio: float,
        vix_level: float,
        portfolio_return: float
    ) -> float:
        """计算动态调整质量"""
        # 评估对冲比例是否与市场状况匹配

        # 1. 牛市 (正收益) 不需要过多对冲
        if portfolio_return > 0.01:  # 日收益>1%
            if hedge_ratio > 0.3:
                reward = -0.5  # 过度对冲惩罚
            else:
                reward = 0.5
        # 2. 熊市 (负收益) 需要对冲
        elif portfolio_return < -0.01:
            if hedge_ratio < 0.1:
                reward = -0.5  # 对冲不足惩罚
            else:
                reward = 0.5 + hedge_ratio  # 对冲越多越好
        # 3. 震荡市
        else:
            # 根据VIX决定
            if vix_level > 25:
                reward = 0.3 if hedge_ratio > 0.15 else -0.2
            else:
                reward = 0.3 if hedge_ratio < 0.2 else 0.1

        return self._clip_reward(reward, -1, 1)


# =============================================================================
# Expert Agent Reward
# =============================================================================

class ExpertReward(BaseAgentReward):
    """
    专家智能体奖励函数 (Stock, Bond, Commodity, REITs, Crypto)

    核心指标:
    1. 预测准确度 - 信号与实际走势的一致性
    2. 校准质量 - 置信度与准确率的匹配
    3. 短期时机 - 短期信号的有效性
    4. 信息增益 - 对组合决策的贡献
    """

    def __init__(self, expert_type: str, config: Optional[Dict] = None):
        super().__init__(config)
        self.expert_type = expert_type

        self.accuracy_weight = self.config.get("accuracy_weight", 0.35)
        self.calibration_weight = self.config.get("calibration_weight", 0.20)
        self.timing_weight = self.config.get("timing_weight", 0.25)
        self.contribution_weight = self.config.get("contribution_weight", 0.20)

        # 专家类型特定参数
        self.type_configs = {
            "stock": {"volatility_scale": 1.0, "momentum_importance": 0.7},
            "bond": {"volatility_scale": 0.3, "momentum_importance": 0.3},
            "commodity": {"volatility_scale": 1.5, "momentum_importance": 0.5},
            "reits": {"volatility_scale": 0.8, "momentum_importance": 0.5},
            "crypto": {"volatility_scale": 3.0, "momentum_importance": 0.6}
        }
        self.type_config = self.type_configs.get(expert_type, self.type_configs["stock"])

    def compute(
        self,
        signal: float,  # -1 to 1, negative=sell, positive=buy
        confidence: float,  # 0 to 1
        actual_return: float,
        historical_signals: List[float],
        historical_returns: List[float],
        portfolio_weight: float,
        asset_contribution: float,
        **kwargs
    ) -> RewardComponents:
        """
        计算专家智能体的奖励

        Args:
            signal: 当前信号 (-1 to 1)
            confidence: 置信度 (0 to 1)
            actual_return: 实际收益率
            historical_signals: 历史信号列表
            historical_returns: 历史收益列表
            portfolio_weight: 在组合中的权重
            asset_contribution: 对组合收益的贡献
        """
        components = {}

        # 1. 预测准确度
        accuracy_reward = self._compute_accuracy(signal, actual_return)
        components["accuracy"] = accuracy_reward

        # 2. 校准质量
        calibration_reward = self._compute_calibration(
            historical_signals, historical_returns, confidence
        )
        components["calibration"] = calibration_reward

        # 3. 时机选择
        timing_reward = self._compute_timing(signal, actual_return, confidence)
        components["timing"] = timing_reward

        # 4. 组合贡献
        contribution_reward = self._compute_contribution(
            portfolio_weight, asset_contribution, signal
        )
        components["contribution"] = contribution_reward

        # 总奖励
        total = (
            self.accuracy_weight * components["accuracy"] +
            self.calibration_weight * components["calibration"] +
            self.timing_weight * components["timing"] +
            self.contribution_weight * components["contribution"]
        )

        description = (
            f"{self.expert_type.upper()} Expert: acc={components['accuracy']:.3f}, "
            f"calib={components['calibration']:.3f}, "
            f"timing={components['timing']:.3f}, "
            f"contrib={components['contribution']:.3f}"
        )

        return RewardComponents(
            total=self._clip_reward(total),
            components=components,
            description=description
        )

    def _compute_accuracy(self, signal: float, actual_return: float) -> float:
        """计算信号准确度"""
        # 信号方向与收益方向一致
        vol_scale = self.type_config["volatility_scale"]

        # 调整后的收益 (根据资产类型的波动性)
        scaled_return = actual_return / (0.02 * vol_scale)  # 归一化

        # 方向一致性
        if signal * scaled_return > 0:
            # 方向正确，奖励与强度成正比
            accuracy = min(abs(signal * scaled_return), 1.0)
        else:
            # 方向错误，惩罚
            accuracy = -min(abs(signal * scaled_return), 1.0)

        return self._clip_reward(accuracy, -1, 1)

    def _compute_calibration(
        self,
        historical_signals: List[float],
        historical_returns: List[float],
        current_confidence: float
    ) -> float:
        """计算校准质量 - 置信度是否匹配历史准确率"""
        if len(historical_signals) < 5 or len(historical_returns) < 5:
            return 0.0  # 数据不足

        # 计算历史准确率
        correct = 0
        for sig, ret in zip(historical_signals[-20:], historical_returns[-20:]):
            if sig * ret > 0:
                correct += 1

        historical_accuracy = correct / min(len(historical_signals[-20:]), 20)

        # 理想情况: 置信度 ≈ 历史准确率
        calibration_error = abs(current_confidence - historical_accuracy)

        # 校准误差越小越好
        reward = np.exp(-5 * calibration_error)

        return self._clip_reward(reward, 0, 1)

    def _compute_timing(
        self,
        signal: float,
        actual_return: float,
        confidence: float
    ) -> float:
        """计算时机选择质量"""
        # 高置信度+正确方向 = 高奖励
        # 高置信度+错误方向 = 高惩罚
        # 低置信度 = 中性

        direction_correct = signal * actual_return > 0

        if direction_correct:
            # 正确时，奖励与置信度正相关
            reward = confidence * (1 + abs(actual_return) * 10)
        else:
            # 错误时，惩罚与置信度正相关
            reward = -confidence * (1 + abs(actual_return) * 10)

        return self._clip_reward(reward, -1, 1)

    def _compute_contribution(
        self,
        portfolio_weight: float,
        asset_contribution: float,
        signal: float
    ) -> float:
        """计算对组合收益的贡献"""
        # 贡献 = 权重 × 资产收益贡献

        # 如果信号方向正确且贡献为正
        if signal * asset_contribution > 0:
            reward = asset_contribution * 100  # 放大
        else:
            reward = asset_contribution * 50  # 负贡献惩罚较轻

        return self._clip_reward(reward, -1, 1)


# =============================================================================
# Coordination Reward
# =============================================================================

class CoordinationReward(BaseAgentReward):
    """
    协调奖励函数 - 评估多智能体协作质量

    核心指标:
    1. 决策一致性 - 各智能体决策的协调程度
    2. 信息利用 - 是否有效利用其他智能体的信息
    3. 冲突解决 - 如何处理意见分歧
    4. 整体效率 - 协作是否提升整体表现
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.consistency_weight = self.config.get("consistency_weight", 0.25)
        self.info_utilization_weight = self.config.get("info_utilization_weight", 0.25)
        self.conflict_resolution_weight = self.config.get("conflict_resolution_weight", 0.25)
        self.efficiency_weight = self.config.get("efficiency_weight", 0.25)

    def compute(
        self,
        manager_decisions: Dict[str, Dict[str, float]],
        expert_signals: Dict[str, float],
        final_allocation: Dict[str, float],
        individual_returns: Dict[str, float],
        portfolio_return: float,
        discussion_rounds: int = 1,
        **kwargs
    ) -> RewardComponents:
        """
        计算协调奖励

        Args:
            manager_decisions: 各经理的决策 {manager: {asset: weight}}
            expert_signals: 专家信号 {expert: signal}
            final_allocation: 最终配置
            individual_returns: 各智能体独立决策的假设收益
            portfolio_return: 实际组合收益
            discussion_rounds: 讨论轮数
        """
        components = {}

        # 1. 决策一致性
        consistency_reward = self._compute_consistency(
            manager_decisions, final_allocation
        )
        components["consistency"] = consistency_reward

        # 2. 信息利用
        info_utilization_reward = self._compute_info_utilization(
            expert_signals, final_allocation
        )
        components["info_utilization"] = info_utilization_reward

        # 3. 冲突解决
        conflict_resolution_reward = self._compute_conflict_resolution(
            manager_decisions, final_allocation, discussion_rounds
        )
        components["conflict_resolution"] = conflict_resolution_reward

        # 4. 整体效率
        efficiency_reward = self._compute_coordination_efficiency(
            individual_returns, portfolio_return
        )
        components["efficiency"] = efficiency_reward

        # 总奖励
        total = (
            self.consistency_weight * components["consistency"] +
            self.info_utilization_weight * components["info_utilization"] +
            self.conflict_resolution_weight * components["conflict_resolution"] +
            self.efficiency_weight * components["efficiency"]
        )

        description = (
            f"Coord Reward: consist={components['consistency']:.3f}, "
            f"info_util={components['info_utilization']:.3f}, "
            f"conflict={components['conflict_resolution']:.3f}, "
            f"efficiency={components['efficiency']:.3f}"
        )

        return RewardComponents(
            total=self._clip_reward(total),
            components=components,
            description=description
        )

    def _compute_consistency(
        self,
        manager_decisions: Dict[str, Dict[str, float]],
        final_allocation: Dict[str, float]
    ) -> float:
        """计算决策一致性"""
        if not manager_decisions:
            return 0.5

        # 计算各经理与最终决策的偏离
        deviations = []
        for manager, decision in manager_decisions.items():
            for asset in final_allocation:
                manager_weight = decision.get(asset, 0)
                final_weight = final_allocation.get(asset, 0)
                deviations.append(abs(manager_weight - final_weight))

        avg_deviation = np.mean(deviations) if deviations else 0

        # 偏离越小，一致性越高
        consistency = np.exp(-5 * avg_deviation)

        return self._clip_reward(consistency, 0, 1)

    def _compute_info_utilization(
        self,
        expert_signals: Dict[str, float],
        final_allocation: Dict[str, float]
    ) -> float:
        """计算专家信息的利用程度"""
        if not expert_signals:
            return 0.5

        # 检查信号与配置方向是否一致
        aligned = 0
        total = 0

        for expert, signal in expert_signals.items():
            # 假设专家名称包含资产类型
            asset_type = expert.lower().replace("_expert", "")

            # 找到对应资产的配置
            for asset, weight in final_allocation.items():
                if asset_type in asset.lower():
                    # 正信号应该有较高权重
                    if signal > 0 and weight > 0.1:
                        aligned += 1
                    elif signal < 0 and weight < 0.1:
                        aligned += 1
                    total += 1

        utilization = aligned / total if total > 0 else 0.5

        return self._clip_reward(utilization, 0, 1)

    def _compute_conflict_resolution(
        self,
        manager_decisions: Dict[str, Dict[str, float]],
        final_allocation: Dict[str, float],
        discussion_rounds: int
    ) -> float:
        """计算冲突解决质量"""
        if len(manager_decisions) < 2:
            return 0.5

        # 计算经理之间的分歧程度
        decisions_list = list(manager_decisions.values())
        all_assets = set()
        for d in decisions_list:
            all_assets.update(d.keys())

        disagreements = []
        for asset in all_assets:
            weights = [d.get(asset, 0) for d in decisions_list]
            disagreements.append(np.std(weights))

        avg_disagreement = np.mean(disagreements) if disagreements else 0

        # 最终决策与平均决策的偏离
        avg_decision = {}
        for asset in all_assets:
            avg_decision[asset] = np.mean([d.get(asset, 0) for d in decisions_list])

        resolution_deviations = []
        for asset in all_assets:
            resolution_deviations.append(
                abs(final_allocation.get(asset, 0) - avg_decision.get(asset, 0))
            )
        avg_resolution_dev = np.mean(resolution_deviations) if resolution_deviations else 0

        # 高分歧+低偏离 = 好的冲突解决
        # 低分歧 = 无需解决
        if avg_disagreement > 0.1:
            # 有明显分歧时，看解决质量
            resolution_quality = 1 - avg_resolution_dev / avg_disagreement
        else:
            resolution_quality = 0.8  # 无分歧时给高分

        # 讨论轮数的影响 (太多轮不好)
        round_penalty = max(0, (discussion_rounds - 2) * 0.1)

        reward = resolution_quality - round_penalty

        return self._clip_reward(reward, 0, 1)

    def _compute_coordination_efficiency(
        self,
        individual_returns: Dict[str, float],
        portfolio_return: float
    ) -> float:
        """计算协调效率 - 协作是否优于独立"""
        if not individual_returns:
            return 0.5

        # 各智能体独立决策的平均收益
        avg_individual = np.mean(list(individual_returns.values()))

        # 协调提升 = 实际收益 - 平均独立收益
        coordination_gain = portfolio_return - avg_individual

        # 转换为奖励
        reward = np.tanh(coordination_gain * 50)  # 放大效果

        return self._clip_reward(reward, -1, 1)


# =============================================================================
# Combined Reward Calculator
# =============================================================================

class CombinedRewardCalculator:
    """
    综合奖励计算器 - 整合所有智能体的奖励

    实现 MARFT 风格的奖励分配:
    1. 个体奖励 (Individual Reward)
    2. 团队奖励 (Team Reward)
    3. 协调奖励 (Coordination Bonus)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # 权重配置
        self.individual_weight = self.config.get("individual_weight", 0.4)
        self.team_weight = self.config.get("team_weight", 0.4)
        self.coordination_weight = self.config.get("coordination_weight", 0.2)

        # 初始化各类奖励函数
        self.pm_reward = PortfolioManagerReward(
            self.config.get("portfolio_manager", {})
        )
        self.ps_reward = PositionSizingReward(
            self.config.get("position_sizing", {})
        )
        self.hedge_reward = HedgingReward(
            self.config.get("hedging", {})
        )
        self.coord_reward = CoordinationReward(
            self.config.get("coordination", {})
        )

        # 各专家奖励
        self.expert_rewards = {
            "stock": ExpertReward("stock", self.config.get("expert_stock", {})),
            "bond": ExpertReward("bond", self.config.get("expert_bond", {})),
            "commodity": ExpertReward("commodity", self.config.get("expert_commodity", {})),
            "reits": ExpertReward("reits", self.config.get("expert_reits", {})),
            "crypto": ExpertReward("crypto", self.config.get("expert_crypto", {})),
        }

        logger.info("CombinedRewardCalculator initialized")

    def compute_all_rewards(
        self,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[str, RewardComponents]:
        """
        计算所有智能体的奖励

        Args:
            state: 当前状态
            actions: 各智能体的动作
            next_state: 下一状态
            info: 环境额外信息

        Returns:
            Dict[str, RewardComponents]: 各智能体的奖励
        """
        rewards = {}

        # 提取必要信息
        portfolio_return = info.get("portfolio_return", 0)
        portfolio_volatility = info.get("portfolio_volatility", 0.02)

        # 1. Portfolio Manager 奖励
        pm_reward = self._compute_pm_reward(state, actions, info)
        rewards["portfolio_manager"] = pm_reward

        # 2. Position Sizing 奖励
        ps_reward = self._compute_ps_reward(state, actions, info)
        rewards["position_sizing"] = ps_reward

        # 3. Hedging 奖励
        hedge_reward = self._compute_hedge_reward(state, actions, info)
        rewards["hedging"] = hedge_reward

        # 4. Expert 奖励
        for expert_name, expert_reward_fn in self.expert_rewards.items():
            expert_reward = self._compute_expert_reward(
                expert_name, expert_reward_fn, state, actions, info
            )
            rewards[f"expert_{expert_name}"] = expert_reward

        # 5. 协调奖励 (应用于所有经理智能体)
        coord_reward = self._compute_coordination_reward(state, actions, info)
        rewards["coordination"] = coord_reward

        return rewards

    def compute_agent_total_reward(
        self,
        agent_name: str,
        individual_reward: RewardComponents,
        team_reward: float,
        coordination_reward: RewardComponents
    ) -> float:
        """
        计算单个智能体的总奖励

        总奖励 = w1 * 个体奖励 + w2 * 团队奖励 + w3 * 协调奖励
        """
        total = (
            self.individual_weight * individual_reward.total +
            self.team_weight * team_reward +
            self.coordination_weight * coordination_reward.total
        )

        return np.clip(total, -10, 10)

    def _compute_pm_reward(
        self,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        info: Dict[str, Any]
    ) -> RewardComponents:
        """计算 Portfolio Manager 奖励"""
        return self.pm_reward.compute(
            portfolio_return=info.get("portfolio_return", 0),
            portfolio_volatility=info.get("portfolio_volatility", 0.02),
            expert_recommendations=info.get("expert_recommendations", {}),
            actual_allocation=actions.get("portfolio_manager", {}).get("allocation", {}),
            asset_returns=info.get("asset_returns", {}),
            market_regime=info.get("market_regime", "normal")
        )

    def _compute_ps_reward(
        self,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        info: Dict[str, Any]
    ) -> RewardComponents:
        """计算 Position Sizing 奖励"""
        return self.ps_reward.compute(
            position_sizes=actions.get("position_sizing", {}).get("sizes", {}),
            asset_volatilities=info.get("asset_volatilities", {}),
            asset_returns=info.get("asset_returns", {}),
            portfolio_volatility=info.get("portfolio_volatility", 0.02),
            risk_contributions=info.get("risk_contributions", {}),
            liquidity_scores=info.get("liquidity_scores", None)
        )

    def _compute_hedge_reward(
        self,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        info: Dict[str, Any]
    ) -> RewardComponents:
        """计算 Hedging 奖励"""
        return self.hedge_reward.compute(
            var_before=info.get("var_before", -0.05),
            var_after=info.get("var_after", -0.04),
            cvar_before=info.get("cvar_before", -0.07),
            cvar_after=info.get("cvar_after", -0.06),
            hedge_cost=info.get("hedge_cost", 0.001),
            vix_level=info.get("vix", 20),
            vix_change=info.get("vix_change", 0),
            hedge_ratio=actions.get("hedging", {}).get("ratio", 0.1),
            hedge_ratio_change=info.get("hedge_ratio_change", 0),
            portfolio_return=info.get("portfolio_return", 0)
        )

    def _compute_expert_reward(
        self,
        expert_name: str,
        expert_reward_fn: ExpertReward,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        info: Dict[str, Any]
    ) -> RewardComponents:
        """计算专家奖励"""
        expert_key = f"expert_{expert_name}"
        expert_action = actions.get(expert_key, {})

        return expert_reward_fn.compute(
            signal=expert_action.get("signal", 0),
            confidence=expert_action.get("confidence", 0.5),
            actual_return=info.get("asset_returns", {}).get(expert_name, 0),
            historical_signals=info.get(f"{expert_name}_historical_signals", []),
            historical_returns=info.get(f"{expert_name}_historical_returns", []),
            portfolio_weight=info.get("portfolio_weights", {}).get(expert_name, 0),
            asset_contribution=info.get("asset_contributions", {}).get(expert_name, 0)
        )

    def _compute_coordination_reward(
        self,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        info: Dict[str, Any]
    ) -> RewardComponents:
        """计算协调奖励"""
        # 收集经理决策
        manager_decisions = {
            "portfolio_manager": actions.get("portfolio_manager", {}).get("allocation", {}),
            "position_sizing": actions.get("position_sizing", {}).get("sizes", {}),
        }

        # 专家信号
        expert_signals = {}
        for expert_name in self.expert_rewards:
            expert_key = f"expert_{expert_name}"
            expert_signals[expert_name] = actions.get(expert_key, {}).get("signal", 0)

        return self.coord_reward.compute(
            manager_decisions=manager_decisions,
            expert_signals=expert_signals,
            final_allocation=info.get("final_allocation", {}),
            individual_returns=info.get("individual_returns", {}),
            portfolio_return=info.get("portfolio_return", 0),
            discussion_rounds=info.get("discussion_rounds", 1)
        )


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_reward_calculator() -> CombinedRewardCalculator:
    """创建默认配置的奖励计算器"""
    default_config = {
        "individual_weight": 0.4,
        "team_weight": 0.4,
        "coordination_weight": 0.2,
        "portfolio_manager": {
            "return_weight": 0.35,
            "consensus_weight": 0.25,
            "quality_weight": 0.25,
            "timing_weight": 0.15,
        },
        "position_sizing": {
            "risk_parity_weight": 0.30,
            "kelly_weight": 0.25,
            "vol_target_weight": 0.30,
            "liquidity_weight": 0.15,
        },
        "hedging": {
            "tail_risk_weight": 0.35,
            "cost_efficiency_weight": 0.25,
            "vix_response_weight": 0.25,
            "dynamic_weight": 0.15,
        },
    }
    return CombinedRewardCalculator(default_config)


def compute_gae_with_individual_rewards(
    rewards_dict: Dict[str, List[float]],
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用个体奖励计算 GAE

    Args:
        rewards_dict: 各智能体的奖励序列
        values: 价值函数估计
        dones: 终止标志
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数

    Returns:
        advantages: GAE 优势值
        returns: 目标回报
    """
    # 合并为总奖励
    total_rewards = np.zeros(len(dones))
    for agent_rewards in rewards_dict.values():
        total_rewards += np.array(agent_rewards)
    total_rewards /= len(rewards_dict)  # 平均

    n_steps = len(total_rewards)
    advantages = np.zeros(n_steps)
    last_gae = 0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = total_rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values

    return advantages, returns


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    # 创建奖励计算器
    calculator = create_default_reward_calculator()

    # 模拟数据
    state = {}
    actions = {
        "portfolio_manager": {"allocation": {"stock": 0.4, "bond": 0.3, "commodity": 0.3}},
        "position_sizing": {"sizes": {"stock": 0.35, "bond": 0.35, "commodity": 0.30}},
        "hedging": {"ratio": 0.15},
        "expert_stock": {"signal": 0.5, "confidence": 0.7},
        "expert_bond": {"signal": 0.2, "confidence": 0.6},
        "expert_commodity": {"signal": -0.3, "confidence": 0.5},
        "expert_reits": {"signal": 0.1, "confidence": 0.5},
        "expert_crypto": {"signal": 0.4, "confidence": 0.4},
    }
    info = {
        "portfolio_return": 0.005,
        "portfolio_volatility": 0.015,
        "asset_returns": {"stock": 0.008, "bond": 0.002, "commodity": -0.003},
        "expert_recommendations": {"stock_expert": {"stock": 0.45}},
        "market_regime": "normal",
        "vix": 18,
        "var_before": -0.05,
        "var_after": -0.04,
        "cvar_before": -0.07,
        "cvar_after": -0.055,
        "hedge_cost": 0.001,
        "final_allocation": {"stock": 0.4, "bond": 0.3, "commodity": 0.3},
    }

    # 计算奖励
    rewards = calculator.compute_all_rewards(state, actions, {}, info)

    print("\n=== Reward Calculation Results ===\n")
    for agent, reward in rewards.items():
        print(f"{agent}: {reward.total:.4f}")
        print(f"  {reward.description}")
        print()
