"""
Bond Factor Scorer - Corporate Bond Factor Investing
债券四因子评分器

学术基础:
- Houweling, P. & van Zundert, J. (2017).
  "Factor Investing in the Corporate Bond Market"
  Financial Analysts Journal, 73(2), 100-115.

- Bai, J., Bali, T.G., & Wen, Q. (2019).
  "Common Risk Factors in the Cross-Section of Corporate Bond Returns"
  Journal of Financial Economics.

- Israel, R., Palhares, D., & Richardson, S. (2018).
  "Common Factors in Corporate Bond Returns"
  The Journal of Portfolio Management, 44(2), 17-34.

债券四因子:
1. Value (价值): 信用利差相对历史分位数
   - 高利差 (被低估) → 做多
   - 低利差 (被高估) → 做空或避免

2. Momentum (动量): 过去6-12个月超额收益
   - 债券市场也有动量效应

3. Carry (套息): YTM vs 无风险利率
   - 高carry债券有溢价

4. Low-Risk (低风险): 评级 + 波动率
   - 低风险债券风险调整后收益更高
   - 质量因子

额外因子:
- Duration (久期): 利率敏感度
- Liquidity (流动性): 买卖价差、成交量
- Credit Quality (信用质量): 违约距离、评级
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

from finsage.factors.base_factor import (
    BaseFactorScorer,
    FactorType,
    FactorExposure,
    FactorScore,
)

logger = logging.getLogger(__name__)


class BondFactorScorer(BaseFactorScorer):
    """
    债券四因子评分器

    基于价值、动量、套息和低风险因子评估债券的投资价值。

    适用范围:
    - 公司债券 (Investment Grade, High Yield)
    - 债券ETF (LQD, HYG, AGG等)
    - 国债ETF (TLT, IEF, SHY)

    使用方法:
    ```python
    scorer = BondFactorScorer()
    score = scorer.score("LQD", bond_data, returns)
    print(score.factor_exposures["value"])  # 价值因子
    print(score.factor_exposures["carry"])  # 套息因子
    ```
    """

    # 评级映射到数值 (用于计算)
    RATING_SCORES = {
        "AAA": 1.0, "AA+": 0.95, "AA": 0.90, "AA-": 0.85,
        "A+": 0.80, "A": 0.75, "A-": 0.70,
        "BBB+": 0.65, "BBB": 0.60, "BBB-": 0.55,  # 投资级底线
        "BB+": 0.45, "BB": 0.40, "BB-": 0.35,
        "B+": 0.30, "B": 0.25, "B-": 0.20,
        "CCC+": 0.15, "CCC": 0.10, "CCC-": 0.05,
        "CC": 0.03, "C": 0.02, "D": 0.0,
    }

    # 典型久期范围
    DURATION_BENCHMARKS = {
        "short": {"min": 0, "max": 3, "benchmark": "SHY"},
        "intermediate": {"min": 3, "max": 7, "benchmark": "IEF"},
        "long": {"min": 7, "max": 30, "benchmark": "TLT"},
    }

    @property
    def asset_class(self) -> str:
        return "bonds"

    @property
    def supported_factors(self) -> List[FactorType]:
        return [
            FactorType.VALUE,
            FactorType.MOMENTUM,
            FactorType.CARRY,
            FactorType.LOW_RISK,
            FactorType.DURATION,
            FactorType.CREDIT,
        ]

    def _default_weights(self) -> Dict[str, float]:
        """
        默认因子权重

        基于学术文献的因子溢价:
        - Carry: 25% - 债券市场的核心因子
        - Value: 25% - 信用利差均值回归
        - Low-Risk: 20% - 质量因子
        - Momentum: 15% - 债券动量较弱但存在
        - Duration: 10% - 利率风险管理
        - Credit: 5% - 辅助因子
        """
        return {
            "value": 0.25,
            "momentum": 0.15,
            "carry": 0.25,
            "low_risk": 0.20,
            "duration": 0.10,
            "credit": 0.05,
        }

    def _compute_factor_exposures(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, FactorExposure]:
        """
        计算债券四因子暴露

        Args:
            symbol: 债券/ETF代码
            data: 债券数据，应包含:
                - ytm: 到期收益率 (Yield to Maturity)
                - duration: 修正久期
                - credit_spread: 信用利差 (vs 同久期国债)
                - spread_percentile: 利差历史分位数
                - rating: 信用评级
                - volatility: 波动率
                - excess_return_6m: 6个月超额收益
                - treasury_rate: 同久期国债收益率
            returns: 历史收益率序列

        Returns:
            Dict[factor_name, FactorExposure]
        """
        exposures = {}

        # 1. Value Factor (价值因子)
        exposures["value"] = self._compute_value_exposure(data)

        # 2. Momentum Factor (动量因子)
        exposures["momentum"] = self._compute_momentum_exposure(data, returns)

        # 3. Carry Factor (套息因子)
        exposures["carry"] = self._compute_carry_exposure(data)

        # 4. Low-Risk Factor (低风险因子)
        exposures["low_risk"] = self._compute_low_risk_exposure(data)

        # 5. Duration Factor (久期因子)
        exposures["duration"] = self._compute_duration_exposure(data)

        # 6. Credit Factor (信用因子)
        exposures["credit"] = self._compute_credit_exposure(data)

        return exposures

    def _compute_value_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算价值因子暴露

        债券价值因子: 信用利差相对历史分位数

        高利差分位数 (被低估):
        - 利差高于历史均值
        - 市场过度悲观
        - 信号: LONG

        低利差分位数 (被高估):
        - 利差低于历史均值
        - 市场过度乐观
        - 信号: SHORT 或 NEUTRAL
        """
        credit_spread = data.get("credit_spread", 0.02)  # 200bps默认
        spread_percentile = data.get("spread_percentile", 50)  # 历史分位数

        # 利差分位数转换为价值信号
        # 高分位数 (>70) = 高价值, 低分位数 (<30) = 低价值
        # 转换: 0% -> -1, 50% -> 0, 100% -> +1
        exposure = self.clip_exposure((spread_percentile - 50) / 50, -1, 1)

        z_score = self.normalize_to_zscore(spread_percentile, mean=50, std=25)

        # 信号
        if spread_percentile > 75:
            signal = "CHEAP"  # 利差高，被低估
        elif spread_percentile < 25:
            signal = "EXPENSIVE"  # 利差低，被高估
        else:
            signal = "FAIR_VALUE"

        return FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=exposure,
            z_score=z_score,
            percentile=spread_percentile,
            signal=signal,
            confidence=0.75,
        )

    def _compute_momentum_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算动量因子暴露

        债券动量效应弱于股票，但仍然存在。
        典型回溯期: 6个月超额收益 (vs 同久期国债)
        """
        # 优先使用历史收益
        if returns is not None and len(returns) >= 126:
            momentum_6m = returns.iloc[-126:].sum()
        else:
            momentum_6m = data.get("excess_return_6m", 0.0)

        # 转换: -5% -> -1, 0% -> 0, +5% -> +1
        exposure = self.clip_exposure(momentum_6m / 0.05, -1, 1)

        z_score = self.normalize_to_zscore(momentum_6m, mean=0.01, std=0.03)

        # 信号
        if momentum_6m > 0.03:
            signal = "POSITIVE_MOMENTUM"
        elif momentum_6m < -0.02:
            signal = "NEGATIVE_MOMENTUM"
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.60,  # 债券动量信心较低
        )

    def _compute_carry_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算套息因子暴露

        Carry = YTM - 无风险利率 (同久期国债)

        高Carry债券:
        - 提供更高的当期收益
        - 承担更高信用风险
        - 在风险偏好高时表现好

        计算方式:
        1. 简单版: YTM - Treasury Rate
        2. 精确版: YTM - Duration-matched Treasury - Expected Default Loss
        """
        ytm = data.get("ytm", 0.05)  # 到期收益率
        treasury_rate = data.get("treasury_rate", 0.04)  # 同久期国债
        credit_spread = data.get("credit_spread", ytm - treasury_rate)

        # Carry近似等于信用利差 (假设无违约)
        carry = credit_spread

        # 转换: 0% -> -0.5, 2% -> 0, 5% -> +1
        exposure = self.clip_exposure((carry - 0.02) / 0.03, -1, 1)

        z_score = self.normalize_to_zscore(carry, mean=0.02, std=0.015)

        # 信号
        if carry > 0.04:
            signal = "HIGH_CARRY"
        elif carry < 0.01:
            signal = "LOW_CARRY"
        else:
            signal = "MODERATE_CARRY"

        return FactorExposure(
            factor_type=FactorType.CARRY,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.80,
        )

    def _compute_low_risk_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算低风险因子暴露

        低风险因子结合:
        1. 信用评级 (高评级 = 低风险)
        2. 波动率 (低波动 = 低风险)

        低风险债券:
        - 风险调整后收益更高 (Low-Risk Anomaly)
        - 在市场下跌时表现更好
        """
        rating = data.get("rating", "BBB")
        volatility = data.get("volatility", 0.05)

        # 评级分数
        rating_score = self.RATING_SCORES.get(rating, 0.5)

        # 波动率分数: 低波动 -> 高分
        # 2% -> 1.0, 5% -> 0.5, 10% -> 0.0
        vol_score = self.clip_exposure((0.10 - volatility) / 0.08, 0, 1)

        # 综合低风险分数 (评级60%, 波动率40%)
        low_risk_score = 0.6 * rating_score + 0.4 * vol_score

        # 转换为exposure: 0 -> -1, 0.5 -> 0, 1 -> +1
        exposure = self.clip_exposure((low_risk_score - 0.5) * 2, -1, 1)

        z_score = self.normalize_to_zscore(low_risk_score, mean=0.5, std=0.2)

        # 信号
        if rating_score >= 0.7 and volatility < 0.04:
            signal = "DEFENSIVE"  # 高质量防御
        elif rating_score < 0.4 or volatility > 0.08:
            signal = "HIGH_RISK"
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.LOW_RISK,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.75,
        )

    def _compute_duration_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算久期因子暴露

        久期衡量利率敏感度:
        - 长久期: 利率下降时收益高，但利率风险大
        - 短久期: 利率风险低，但收益也低

        久期决策取决于利率预期:
        - 预期降息: 加久期
        - 预期加息: 减久期
        """
        duration = data.get("duration", 5.0)

        # 分类
        if duration < 3:
            duration_category = "short"
            signal = "LOW_RATE_RISK"
        elif duration < 7:
            duration_category = "intermediate"
            signal = "MODERATE_RATE_RISK"
        else:
            duration_category = "long"
            signal = "HIGH_RATE_RISK"

        # 转换: 0 -> -1, 7 -> 0, 15 -> +1
        exposure = self.clip_exposure((duration - 7) / 8, -1, 1)

        z_score = self.normalize_to_zscore(duration, mean=6, std=4)

        return FactorExposure(
            factor_type=FactorType.DURATION,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.90,  # 久期是确定性指标
        )

    def _compute_credit_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算信用因子暴露

        信用因子关注违约风险:
        - 违约距离 (Distance to Default)
        - 评级变化趋势
        - 财务杠杆
        """
        rating = data.get("rating", "BBB")
        rating_score = self.RATING_SCORES.get(rating, 0.5)

        # 评级趋势
        rating_trend = data.get("rating_trend", "stable")
        if rating_trend == "upgrade":
            trend_adjustment = 0.1
        elif rating_trend == "downgrade":
            trend_adjustment = -0.1
        else:
            trend_adjustment = 0.0

        credit_score = rating_score + trend_adjustment
        credit_score = max(0, min(1, credit_score))

        # 转换
        exposure = self.clip_exposure((credit_score - 0.5) * 2, -1, 1)

        # 信号
        if rating_score >= 0.55:  # BBB-及以上
            signal = "INVESTMENT_GRADE"
        else:
            signal = "HIGH_YIELD"

        return FactorExposure(
            factor_type=FactorType.CREDIT,
            exposure=exposure,
            z_score=self.normalize_to_zscore(credit_score, 0.5, 0.2),
            percentile=credit_score * 100,
            signal=signal,
            confidence=0.70,
        )

    def _get_factor_premiums(self) -> Dict[str, float]:
        """
        获取历史因子溢价 (年化)

        基于学术文献 (FAJ 2017, JPM 2018):
        - Carry: ~2% (套息溢价)
        - Value: ~1.5% (价值溢价)
        - Momentum: ~1% (动量溢价，弱于股票)
        - Low-Risk: ~1% (低风险异象)
        """
        return {
            "value": 0.015,
            "momentum": 0.01,
            "carry": 0.02,
            "low_risk": 0.01,
            "duration": 0.005,
            "credit": 0.01,
        }

    def get_duration_recommendation(
        self,
        rate_view: str,
        current_duration: float
    ) -> Dict[str, Any]:
        """
        基于利率预期的久期建议

        Args:
            rate_view: 利率预期 ("rising", "falling", "stable")
            current_duration: 当前组合久期

        Returns:
            久期调整建议
        """
        if rate_view == "rising":
            target_duration = max(1, current_duration * 0.7)
            action = "REDUCE_DURATION"
            reasoning = "利率上升预期，减少久期以降低利率风险"
        elif rate_view == "falling":
            target_duration = min(15, current_duration * 1.3)
            action = "EXTEND_DURATION"
            reasoning = "利率下降预期，增加久期以获取资本利得"
        else:
            target_duration = current_duration
            action = "MAINTAIN"
            reasoning = "利率稳定预期，维持当前久期"

        return {
            "current_duration": current_duration,
            "target_duration": target_duration,
            "action": action,
            "reasoning": reasoning,
        }

    def get_credit_allocation(
        self,
        risk_appetite: str,
        scores: Dict[str, FactorScore]
    ) -> Dict[str, float]:
        """
        基于风险偏好的信用配置建议

        Args:
            risk_appetite: 风险偏好 ("conservative", "moderate", "aggressive")
            scores: 各债券的因子评分

        Returns:
            信用等级配置权重
        """
        if risk_appetite == "conservative":
            # 保守: 偏向高评级
            allocation = {
                "treasury": 0.30,
                "aaa_aa": 0.30,
                "a_bbb": 0.30,
                "high_yield": 0.10,
            }
        elif risk_appetite == "aggressive":
            # 激进: 追逐收益
            allocation = {
                "treasury": 0.10,
                "aaa_aa": 0.15,
                "a_bbb": 0.35,
                "high_yield": 0.40,
            }
        else:
            # 中性
            allocation = {
                "treasury": 0.20,
                "aaa_aa": 0.25,
                "a_bbb": 0.35,
                "high_yield": 0.20,
            }

        return allocation
