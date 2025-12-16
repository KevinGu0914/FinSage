"""
Dynamic Hedge Selector
动态对冲资产选择器 - 从全市场筛选最优对冲工具

核心功能:
1. 分析当前组合的风险敞口
2. 识别对冲需求和目标
3. 从全市场筛选最优对冲工具
4. 计算最优对冲配置
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
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
class PortfolioExposure:
    """组合敞口分析结果"""
    beta: float = 1.0                       # 市场Beta
    volatility: float = 0.12                # 年化波动率
    concentration_hhi: float = 0.1          # HHI集中度指数
    sector_exposure: Dict[str, float] = field(default_factory=dict)  # 行业敞口
    top_holdings: List[Dict[str, Any]] = field(default_factory=list)  # 前N大持仓
    correlation_with_spy: float = 0.8       # 与SPY相关性
    var_95: float = -0.02                   # 95% VaR

    def to_dict(self) -> Dict:
        return {
            "beta": round(self.beta, 3),
            "volatility": round(self.volatility, 4),
            "concentration_hhi": round(self.concentration_hhi, 4),
            "sector_exposure": self.sector_exposure,
            "top_holdings": self.top_holdings,
            "correlation_with_spy": round(self.correlation_with_spy, 3),
            "var_95": round(self.var_95, 4),
        }


@dataclass
class HedgeCandidate:
    """对冲候选资产评分"""
    asset: HedgeAsset
    correlation_score: float = 0.0      # 相关性得分 (越负越好)
    liquidity_score: float = 0.0        # 流动性得分
    cost_score: float = 0.0             # 成本得分 (越低越好)
    efficiency_score: float = 0.0       # 对冲效率得分
    total_score: float = 0.0            # 综合得分
    raw_correlation: float = 0.0        # 原始相关系数

    def to_dict(self) -> Dict:
        return {
            "symbol": self.asset.symbol,
            "name": self.asset.name,
            "category": self.asset.category.value,
            "leverage": self.asset.leverage,
            "expense_ratio": self.asset.expense_ratio,
            "correlation_score": round(self.correlation_score, 4),
            "liquidity_score": round(self.liquidity_score, 4),
            "cost_score": round(self.cost_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "total_score": round(self.total_score, 4),
            "raw_correlation": round(self.raw_correlation, 4),
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
    exposure_analysis: Optional[PortfolioExposure] = None

    def to_dict(self) -> Dict:
        return {
            "objective": self.objective.value,
            "top_candidates": [c.to_dict() for c in self.candidates[:5]],
            "recommended_allocation": {k: round(v, 4) for k, v in self.recommended_allocation.items()},
            "expected_correlation_reduction": round(self.expected_correlation_reduction, 4),
            "expected_cost": round(self.expected_cost, 4),
            "reasoning": self.reasoning,
            "exposure_analysis": self.exposure_analysis.to_dict() if self.exposure_analysis else None,
        }

    def get_instruments_for_agent(self) -> List[Dict]:
        """
        转换为 HedgingAgent 兼容的工具格式
        """
        instruments = []
        for symbol, allocation in self.recommended_allocation.items():
            # 查找对应的候选资产
            candidate = next((c for c in self.candidates if c.asset.symbol == symbol), None)
            if candidate:
                instruments.append({
                    "symbol": symbol,
                    "name": candidate.asset.name,
                    "type": "etf",
                    "allocation": allocation,
                    "cost_rate": candidate.asset.expense_ratio,
                    "leverage": candidate.asset.leverage,
                    "category": candidate.asset.category.value,
                    "source": "dynamic",  # 标记来源为动态选择
                })
        return instruments


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
        config: Optional[Dict] = None
    ):
        """
        初始化选择器

        Args:
            universe: 对冲资产全集
            config: 配置参数
        """
        self.universe = universe or HedgeAssetUniverse()
        self.config = config or {}

        # 配置参数
        self.min_liquidity = self.config.get("min_daily_volume", 1e5)  # 最小日均成交量
        self.max_expense = self.config.get("max_expense_ratio", 0.02)  # 最大费率
        self.lookback_days = self.config.get("correlation_lookback", 60)  # 相关性计算回溯期
        self.top_k = self.config.get("top_candidates", 5)  # 返回前K个候选

        # 评分权重 (可配置)
        self.weights = self.config.get("scoring_weights", {
            "correlation": 0.35,   # 相关性权重 (负相关越好)
            "liquidity": 0.20,     # 流动性权重
            "cost": 0.20,          # 成本权重
            "efficiency": 0.25,    # 对冲效率权重 (杠杆等)
        })

        logger.info(f"DynamicHedgeSelector initialized with {len(self.universe.get_all_symbols())} assets in universe")

    def analyze_portfolio_exposure(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        market_data: Optional[Dict[str, Any]] = None
    ) -> PortfolioExposure:
        """
        分析组合的风险敞口

        Args:
            portfolio_weights: 组合权重 {symbol: weight}
            returns_data: 历史收益率数据
            market_data: 市场数据

        Returns:
            PortfolioExposure: 敞口分析结果
        """
        exposure = PortfolioExposure()

        if returns_data.empty or not portfolio_weights:
            logger.warning("Empty returns data or portfolio weights")
            return exposure

        # 获取可用资产
        available = [a for a in portfolio_weights if a in returns_data.columns]
        if not available:
            logger.warning("No overlapping assets between portfolio and returns data")
            return exposure

        # 归一化权重
        weights = np.array([portfolio_weights.get(a, 0) for a in available])
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            return exposure

        # 计算组合收益率
        portfolio_returns = returns_data[available].values @ weights

        # 1. 计算波动率 (年化)
        exposure.volatility = float(np.std(portfolio_returns) * np.sqrt(252))

        # 2. 计算 Beta (相对于 SPY)
        if "SPY" in returns_data.columns:
            spy_returns = returns_data["SPY"].values
            cov = np.cov(portfolio_returns, spy_returns)[0, 1]
            spy_var = np.var(spy_returns)
            if spy_var > 0:
                exposure.beta = float(cov / spy_var)
            # 相关性
            corr = np.corrcoef(portfolio_returns, spy_returns)[0, 1]
            exposure.correlation_with_spy = float(corr) if not np.isnan(corr) else 0.8

        # 3. 计算集中度 (HHI)
        exposure.concentration_hhi = float(np.sum(weights ** 2))

        # 4. 计算 VaR
        if len(portfolio_returns) >= 20:
            exposure.var_95 = float(np.percentile(portfolio_returns, 5))

        # 5. Top holdings
        sorted_holdings = sorted(zip(available, weights), key=lambda x: -x[1])
        exposure.top_holdings = [
            {"symbol": s, "weight": float(w)}
            for s, w in sorted_holdings[:5]
        ]

        logger.info(f"Portfolio exposure: beta={exposure.beta:.2f}, vol={exposure.volatility:.1%}, HHI={exposure.concentration_hhi:.3f}")
        return exposure

    def identify_hedge_objective(
        self,
        exposure: PortfolioExposure,
        hedge_strategy: str,
        risk_constraints: Optional[Dict[str, float]] = None
    ) -> HedgeObjective:
        """
        识别对冲目标

        Args:
            exposure: 组合敞口分析
            hedge_strategy: HedgingAgent建议的策略
            risk_constraints: 风控约束

        Returns:
            HedgeObjective: 对冲目标
        """
        # 策略名到对冲目标的映射
        strategy_map = {
            "put_protection": HedgeObjective.TAIL_RISK,
            "collar": HedgeObjective.TAIL_RISK,
            "tail_hedge": HedgeObjective.TAIL_RISK,
            "dynamic_hedge": HedgeObjective.BETA_NEUTRAL,
            "diversification": HedgeObjective.DIVERSIFICATION,
            "safe_haven": HedgeObjective.CORRELATION_HEDGE,
            "none": HedgeObjective.DIVERSIFICATION,
        }

        # 首先检查策略映射
        if hedge_strategy in strategy_map:
            objective = strategy_map[hedge_strategy]
            logger.info(f"Hedge objective from strategy '{hedge_strategy}': {objective.value}")
            return objective

        # 如果没有明确策略，根据敞口分析自动判断
        beta = exposure.beta
        volatility = exposure.volatility
        concentration = exposure.concentration_hhi

        # 高 Beta (>1.2) 需要 Beta 对冲
        if abs(beta) > 1.2:
            logger.info(f"High beta ({beta:.2f}) detected, recommending beta neutral hedge")
            return HedgeObjective.BETA_NEUTRAL

        # 高波动率 (>18%) 需要尾部风险对冲
        if volatility > 0.18:
            logger.info(f"High volatility ({volatility:.1%}) detected, recommending tail risk hedge")
            return HedgeObjective.TAIL_RISK

        # 高集中度 (HHI > 0.25) 需要分散化
        if concentration > 0.25:
            logger.info(f"High concentration (HHI={concentration:.3f}) detected, recommending diversification")
            return HedgeObjective.DIVERSIFICATION

        # 默认：相关性对冲
        logger.info("Using default correlation hedge objective")
        return HedgeObjective.CORRELATION_HEDGE

    def select_candidates(
        self,
        objective: HedgeObjective,
        exposure: PortfolioExposure,
    ) -> List[HedgeAsset]:
        """
        根据对冲目标筛选候选资产

        Args:
            objective: 对冲目标
            exposure: 组合敞口

        Returns:
            List[HedgeAsset]: 候选资产列表
        """
        candidates = []

        # 根据对冲目标选择资产类别
        if objective == HedgeObjective.BETA_NEUTRAL:
            # Beta 对冲：反向股票ETF
            candidates.extend(self.universe.get_by_category(HedgeCategory.INVERSE_EQUITY))
            candidates.extend(self.universe.get_by_category(HedgeCategory.INVERSE_SECTOR))

        elif objective == HedgeObjective.TAIL_RISK:
            # 尾部风险对冲：波动率工具 + 反向ETF
            candidates.extend(self.universe.get_by_category(HedgeCategory.VOLATILITY))
            candidates.extend(self.universe.get_by_category(HedgeCategory.INVERSE_EQUITY))
            # 添加避险资产
            candidates.extend(self.universe.get_by_category(HedgeCategory.SAFE_HAVEN))

        elif objective == HedgeObjective.SECTOR_HEDGE:
            # 行业对冲：反向行业ETF + 行业ETF (配对)
            candidates.extend(self.universe.get_by_category(HedgeCategory.INVERSE_SECTOR))
            candidates.extend(self.universe.get_by_category(HedgeCategory.SECTOR_ETF))

        elif objective == HedgeObjective.CORRELATION_HEDGE:
            # 相关性对冲：避险资产 + 固定收益
            candidates.extend(self.universe.get_by_category(HedgeCategory.SAFE_HAVEN))
            candidates.extend(self.universe.get_by_category(HedgeCategory.FIXED_INCOME))
            # 排除反向债券
            candidates = [c for c in candidates if not c.is_inverse]

        elif objective == HedgeObjective.VOLATILITY_HEDGE:
            # 波动率对冲：波动率工具
            candidates.extend(self.universe.get_by_category(HedgeCategory.VOLATILITY))

        elif objective == HedgeObjective.RATE_HEDGE:
            # 利率对冲：反向债券ETF
            candidates.extend(self.universe.filter(
                category=HedgeCategory.FIXED_INCOME,
                tags=["inverse", "rate_hedge"]
            ))

        elif objective == HedgeObjective.CURRENCY_HEDGE:
            # 货币对冲
            candidates.extend(self.universe.get_by_category(HedgeCategory.CURRENCY))

        elif objective == HedgeObjective.DIVERSIFICATION:
            # 分散化：多类资产
            candidates.extend(self.universe.get_by_category(HedgeCategory.SAFE_HAVEN))
            candidates.extend(self.universe.filter(
                category=HedgeCategory.FIXED_INCOME,
                inverse_only=False
            ))
            candidates.extend(self.universe.get_by_category(HedgeCategory.INTERNATIONAL))

        else:
            # 默认：避险资产
            candidates.extend(self.universe.get_by_category(HedgeCategory.SAFE_HAVEN))
            candidates.extend(self.universe.get_by_category(HedgeCategory.FIXED_INCOME))

        # 应用过滤条件
        # 1. 流动性过滤
        candidates = [c for c in candidates if c.avg_daily_volume >= self.min_liquidity]

        # 2. 成本过滤
        candidates = [c for c in candidates if c.expense_ratio <= self.max_expense]

        # 去重
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.symbol not in seen:
                seen.add(c.symbol)
                unique_candidates.append(c)

        logger.info(f"Selected {len(unique_candidates)} candidates for {objective.value} objective")
        return unique_candidates

    def score_candidates(
        self,
        candidates: List[HedgeAsset],
        portfolio_returns: np.ndarray,
        returns_data: pd.DataFrame,
    ) -> List[HedgeCandidate]:
        """
        对候选资产进行多因子评分

        Scoring factors:
        1. 相关性 (越负越好，对于对冲)
        2. 流动性 (越高越好)
        3. 成本 (越低越好)
        4. 对冲效率 (考虑杠杆等)

        Args:
            candidates: 候选资产列表
            portfolio_returns: 组合收益率序列
            returns_data: 收益率数据

        Returns:
            List[HedgeCandidate]: 评分后的候选列表
        """
        scored = []

        for asset in candidates:
            # 1. 计算相关性得分
            raw_correlation = 0.0
            if asset.symbol in returns_data.columns and len(portfolio_returns) > 0:
                asset_returns = returns_data[asset.symbol].values
                if len(asset_returns) == len(portfolio_returns):
                    corr = np.corrcoef(portfolio_returns, asset_returns)[0, 1]
                    if not np.isnan(corr):
                        raw_correlation = float(corr)

            # 相关性得分: -1 对应 1.0 (最好), 0 对应 0.5, 1 对应 0.0 (最差)
            # 对于对冲，我们希望负相关
            correlation_score = (1 - raw_correlation) / 2

            # 2. 流动性得分 (对数归一化)
            # log10(1e9) = 9, log10(1e5) = 5
            volume_log = np.log10(max(asset.avg_daily_volume, 1))
            liquidity_score = min(1.0, max(0.0, (volume_log - 5) / 4))  # 5-9 映射到 0-1

            # 3. 成本得分 (越低越好)
            # 0% = 1.0, 1% = 0.9, 2% = 0.8
            cost_score = max(0.0, 1 - asset.total_cost_estimate * 10)

            # 4. 对冲效率得分 (考虑杠杆)
            # 杠杆越高，对冲效率越高 (但也有更高风险)
            leverage_abs = abs(asset.leverage) if asset.leverage else 1.0
            efficiency_score = min(1.0, leverage_abs / 3)  # 3x = 满分

            # 计算综合得分
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
                raw_correlation=raw_correlation,
            ))

        # 按总分降序排序
        scored.sort(key=lambda x: -x.total_score)

        logger.info(f"Scored {len(scored)} candidates, top score: {scored[0].total_score:.3f}" if scored else "No candidates scored")
        return scored

    def compute_optimal_allocation(
        self,
        candidates: List[HedgeCandidate],
        hedge_ratio: float,
        portfolio_returns: np.ndarray,
        returns_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        计算最优对冲配置

        使用得分加权的方式分配对冲比例

        Args:
            candidates: 评分后的候选列表
            hedge_ratio: 总对冲比例
            portfolio_returns: 组合收益率
            returns_data: 收益率数据

        Returns:
            Dict[str, float]: {symbol: allocation}
        """
        if not candidates or hedge_ratio <= 0:
            return {}

        # 取前 top_k 个候选
        top_candidates = candidates[:self.top_k]

        # 计算得分加权的配置
        total_score = sum(c.total_score for c in top_candidates)

        if total_score <= 0:
            # 如果所有得分为0，等权分配
            return {c.asset.symbol: hedge_ratio / len(top_candidates) for c in top_candidates}

        allocation = {}
        for candidate in top_candidates:
            weight = (candidate.total_score / total_score) * hedge_ratio
            allocation[candidate.asset.symbol] = weight

        # 确保总和等于 hedge_ratio
        actual_sum = sum(allocation.values())
        if actual_sum > 0:
            scale = hedge_ratio / actual_sum
            allocation = {k: v * scale for k, v in allocation.items()}

        logger.info(f"Computed allocation for {len(allocation)} assets, total ratio: {sum(allocation.values()):.2%}")
        return allocation

    def recommend(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        hedge_strategy: str,
        hedge_ratio: float,
        market_data: Optional[Dict[str, Any]] = None,
        risk_constraints: Optional[Dict[str, float]] = None,
    ) -> HedgeRecommendation:
        """
        生成对冲推荐 (主入口函数)

        Args:
            portfolio_weights: 当前组合权重 {symbol: weight}
            returns_data: 历史收益率数据
            hedge_strategy: HedgingAgent 建议的策略名
            hedge_ratio: 建议的对冲比例
            market_data: 市场数据
            risk_constraints: 风控约束

        Returns:
            HedgeRecommendation: 对冲推荐结果
        """
        logger.info(f"Generating hedge recommendation: strategy={hedge_strategy}, ratio={hedge_ratio:.1%}")

        # Step 1: 分析组合敞口
        exposure = self.analyze_portfolio_exposure(
            portfolio_weights, returns_data, market_data
        )

        # Step 2: 识别对冲目标
        objective = self.identify_hedge_objective(
            exposure, hedge_strategy, risk_constraints
        )

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
                exposure_analysis=exposure,
            )

        # Step 4: 计算组合收益率
        available = [a for a in portfolio_weights if a in returns_data.columns]
        if available:
            weights = np.array([portfolio_weights.get(a, 0) for a in available])
            weights = weights / weights.sum() if weights.sum() > 0 else weights
            portfolio_returns = returns_data[available].values @ weights
        else:
            portfolio_returns = np.array([])

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
        if scored_candidates:
            top_k_candidates = [c for c in scored_candidates[:self.top_k] if c.asset.symbol in allocation]
            if top_k_candidates:
                avg_corr_score = np.mean([c.correlation_score for c in top_k_candidates])
                expected_corr_reduction = avg_corr_score * hedge_ratio
            else:
                expected_corr_reduction = 0.0
        else:
            expected_corr_reduction = 0.0

        # 生成推荐理由
        reasoning = self._generate_reasoning(
            objective, exposure, scored_candidates, allocation, hedge_ratio
        )

        recommendation = HedgeRecommendation(
            objective=objective,
            candidates=scored_candidates,
            recommended_allocation=allocation,
            expected_correlation_reduction=expected_corr_reduction,
            expected_cost=expected_cost,
            reasoning=reasoning,
            exposure_analysis=exposure,
        )

        logger.info(f"Generated recommendation: {len(allocation)} assets, cost={expected_cost:.2%}")
        return recommendation

    def _generate_reasoning(
        self,
        objective: HedgeObjective,
        exposure: PortfolioExposure,
        candidates: List[HedgeCandidate],
        allocation: Dict[str, float],
        hedge_ratio: float,
    ) -> str:
        """生成推荐理由"""
        if not candidates or not allocation:
            return "未能生成有效的对冲推荐。"

        # 目标描述
        objective_desc = {
            HedgeObjective.BETA_NEUTRAL: "降低市场Beta敞口",
            HedgeObjective.TAIL_RISK: "对冲尾部风险",
            HedgeObjective.SECTOR_HEDGE: "行业风险对冲",
            HedgeObjective.CORRELATION_HEDGE: "降低组合相关性",
            HedgeObjective.VOLATILITY_HEDGE: "波动率对冲",
            HedgeObjective.RATE_HEDGE: "利率风险对冲",
            HedgeObjective.CURRENCY_HEDGE: "货币风险对冲",
            HedgeObjective.DIVERSIFICATION: "提升组合分散化",
        }

        top_symbols = list(allocation.keys())[:3]
        top_candidates_info = [
            f"{c.asset.symbol}(得分{c.total_score:.2f})"
            for c in candidates[:3] if c.asset.symbol in allocation
        ]

        reasoning = f"基于{objective_desc.get(objective, objective.value)}目标，"
        reasoning += f"推荐使用 {', '.join(top_symbols)} 进行对冲。"
        reasoning += f"当前组合Beta={exposure.beta:.2f}，波动率={exposure.volatility:.1%}，"
        reasoning += f"HHI集中度={exposure.concentration_hhi:.3f}。"
        reasoning += f"对冲比例{hedge_ratio:.1%}，"
        reasoning += f"Top候选: {', '.join(top_candidates_info)}。"

        return reasoning

    def get_universe_summary(self) -> Dict[str, Any]:
        """获取资产全集摘要"""
        return {
            "total_assets": len(self.universe.get_all_symbols()),
            "categories": self.universe.summary(),
            "config": {
                "min_liquidity": self.min_liquidity,
                "max_expense": self.max_expense,
                "top_k": self.top_k,
                "weights": self.weights,
            }
        }
