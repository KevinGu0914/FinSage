"""
Factor-Enhanced Expert Mixin
因子增强专家混入类

将学术因子评分模块集成到现有Expert系统中

核心思想:
- 不替换现有Expert的LLM分析能力
- 增加基于学术文献的因子评分
- 因子评分作为LLM分析的补充输入
- 综合LLM判断和因子信号生成最终建议
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
import pandas as pd
import logging

from finsage.agents.base_expert import (
    BaseExpert,
    ExpertRecommendation,
    ExpertReport,
    Action,
)
from finsage.factors.base_factor import (
    BaseFactorScorer,
    FactorScore,
    FactorExposure,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRecommendation(ExpertRecommendation):
    """增强版建议 - 包含因子评分"""
    factor_score: Optional[FactorScore] = None
    factor_signal: Optional[str] = None
    factor_alpha: Optional[float] = None

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        if self.factor_score:
            base_dict["factor_score"] = self.factor_score.to_dict()
        base_dict["factor_signal"] = self.factor_signal
        base_dict["factor_alpha"] = self.factor_alpha
        return base_dict


class FactorEnhancedExpertMixin:
    """
    因子增强专家混入类

    为任何BaseExpert子类添加因子评分能力

    使用方法:
    ```python
    class EnhancedStockExpert(FactorEnhancedExpertMixin, StockExpert):
        pass

    expert = EnhancedStockExpert(llm_provider, factor_scorer=StockFactorScorer())
    report = expert.analyze_with_factors(market_data, returns_df)
    ```
    """

    def __init__(
        self,
        *args,
        factor_scorer: Optional[BaseFactorScorer] = None,
        factor_weight: float = 0.4,  # 因子信号在最终决策中的权重
        **kwargs
    ):
        """
        初始化因子增强专家

        Args:
            factor_scorer: 因子评分器实例
            factor_weight: 因子评分在最终决策中的权重 [0, 1]
                          0 = 完全依赖LLM
                          1 = 完全依赖因子
                          0.4 = 推荐值 (LLM 60% + 因子 40%)
        """
        super().__init__(*args, **kwargs)
        self.factor_scorer = factor_scorer
        self.factor_weight = factor_weight
        self._factor_scores: Dict[str, FactorScore] = {}

        if factor_scorer:
            logger.info(
                f"Factor-enhanced {self.__class__.__name__} initialized with "
                f"{factor_scorer.__class__.__name__}, weight={factor_weight}"
            )

    def analyze_with_factors(
        self,
        market_data: Dict[str, Any],
        returns: Optional[pd.DataFrame] = None,
        news_data: Optional[List[Dict]] = None,
        technical_indicators: Optional[Dict[str, Any]] = None,
        macro_data: Optional[Dict[str, Any]] = None,
        market_regime: Optional[str] = None,
    ) -> ExpertReport:
        """
        执行因子增强分析

        流程:
        1. 计算所有资产的因子评分
        2. 将因子信息注入LLM Prompt
        3. 调用原始analyze方法
        4. 综合LLM和因子信号生成最终建议

        Args:
            market_data: 市场数据
            returns: 历史收益率DataFrame (列为symbol)
            news_data: 新闻数据
            technical_indicators: 技术指标
            macro_data: 宏观数据
            market_regime: 市场体制 ("bull", "bear", "neutral")

        Returns:
            EnhancedExpertReport
        """
        # Step 1: 计算因子评分
        if self.factor_scorer:
            self._compute_factor_scores(market_data, returns, market_regime)

        # Step 2: 增强市场数据 (注入因子信息)
        enhanced_market_data = self._enhance_market_data_with_factors(market_data)

        # Step 3: 调用原始分析
        base_report = self.analyze(
            market_data=enhanced_market_data,
            news_data=news_data,
            technical_indicators=technical_indicators,
            macro_data=macro_data,
        )

        # Step 4: 综合因子信号调整建议
        enhanced_recommendations = self._enhance_recommendations_with_factors(
            base_report.recommendations
        )

        # 更新报告
        base_report.recommendations = enhanced_recommendations

        # 添加因子摘要到key_factors
        factor_summary = self._generate_factor_summary()
        if factor_summary:
            base_report.key_factors.extend(factor_summary)

        return base_report

    def _compute_factor_scores(
        self,
        market_data: Dict[str, Any],
        returns: Optional[pd.DataFrame],
        market_regime: Optional[str]
    ):
        """计算所有资产的因子评分"""
        self._factor_scores = {}

        for symbol in self.symbols:
            if symbol not in market_data:
                continue

            symbol_data = market_data[symbol]
            symbol_returns = None
            if returns is not None and symbol in returns.columns:
                symbol_returns = returns[symbol]

            try:
                score = self.factor_scorer.score(
                    symbol=symbol,
                    data=symbol_data,
                    returns=symbol_returns,
                    market_regime=market_regime,
                )
                self._factor_scores[symbol] = score
                logger.debug(
                    f"Factor score for {symbol}: {score.composite_score:.2f} ({score.signal})"
                )
            except Exception as e:
                logger.warning(f"Failed to compute factor score for {symbol}: {e}")

    def _enhance_market_data_with_factors(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将因子评分注入市场数据"""
        enhanced_data = market_data.copy()

        for symbol, score in self._factor_scores.items():
            if symbol in enhanced_data:
                # 添加因子信息到市场数据
                enhanced_data[symbol]["factor_score"] = score.composite_score
                enhanced_data[symbol]["factor_signal"] = score.signal
                enhanced_data[symbol]["factor_alpha"] = score.expected_alpha
                enhanced_data[symbol]["factor_reasoning"] = score.reasoning

                # 添加各因子暴露
                for factor_name, exposure in score.factor_exposures.items():
                    enhanced_data[symbol][f"factor_{factor_name}"] = exposure.exposure
                    enhanced_data[symbol][f"factor_{factor_name}_signal"] = exposure.signal

        return enhanced_data

    def _enhance_recommendations_with_factors(
        self,
        recommendations: List[ExpertRecommendation]
    ) -> List[EnhancedRecommendation]:
        """综合因子信号调整建议"""
        enhanced = []

        for rec in recommendations:
            factor_score = self._factor_scores.get(rec.symbol)

            if factor_score and self.factor_weight > 0:
                # 综合LLM和因子信号
                adjusted_confidence = self._adjust_confidence(rec, factor_score)
                adjusted_action = self._adjust_action(rec, factor_score)
                adjusted_weight = self._adjust_target_weight(rec, factor_score)

                enhanced_rec = EnhancedRecommendation(
                    asset_class=rec.asset_class,
                    symbol=rec.symbol,
                    action=adjusted_action,
                    confidence=adjusted_confidence,
                    target_weight=adjusted_weight,
                    reasoning=self._enhance_reasoning(rec.reasoning, factor_score),
                    market_view=rec.market_view,
                    risk_assessment=rec.risk_assessment,
                    factor_score=factor_score,
                    factor_signal=factor_score.signal,
                    factor_alpha=factor_score.expected_alpha,
                )
            else:
                # 没有因子评分，保持原建议
                enhanced_rec = EnhancedRecommendation(
                    asset_class=rec.asset_class,
                    symbol=rec.symbol,
                    action=rec.action,
                    confidence=rec.confidence,
                    target_weight=rec.target_weight,
                    reasoning=rec.reasoning,
                    market_view=rec.market_view,
                    risk_assessment=rec.risk_assessment,
                    factor_score=None,
                    factor_signal=None,
                    factor_alpha=None,
                )

            enhanced.append(enhanced_rec)

        return enhanced

    def _adjust_confidence(
        self,
        rec: ExpertRecommendation,
        factor_score: FactorScore
    ) -> float:
        """综合调整置信度"""
        llm_conf = rec.confidence
        factor_conf = factor_score.composite_score

        # 加权平均
        combined = (
            llm_conf * (1 - self.factor_weight) +
            factor_conf * self.factor_weight
        )

        # 如果信号一致，提高置信度
        llm_bullish = "BUY" in rec.action.value
        factor_bullish = factor_score.signal in ["STRONG_BUY", "BUY"]

        if llm_bullish == factor_bullish:
            combined = min(combined * 1.1, 1.0)  # 最多提高10%
        else:
            combined = max(combined * 0.9, 0.1)  # 最多降低10%

        return round(combined, 3)

    def _adjust_action(
        self,
        rec: ExpertRecommendation,
        factor_score: FactorScore
    ) -> Action:
        """综合调整动作"""
        # 映射因子信号到动作强度
        factor_action_map = {
            "STRONG_BUY": 2,
            "BUY": 1,
            "HOLD": 0,
            "SELL": -1,
            "STRONG_SELL": -2,
        }

        llm_action_map = {
            Action.BUY_100: 2,
            Action.BUY_75: 1.5,
            Action.BUY_50: 1,
            Action.BUY_25: 0.5,
            Action.HOLD: 0,
            Action.SELL_25: -0.5,
            Action.SELL_50: -1,
            Action.SELL_75: -1.5,
            Action.SELL_100: -2,
        }

        llm_strength = llm_action_map.get(rec.action, 0)
        factor_strength = factor_action_map.get(factor_score.signal, 0)

        # 加权平均
        combined_strength = (
            llm_strength * (1 - self.factor_weight) +
            factor_strength * self.factor_weight
        )

        # 映射回动作
        if combined_strength >= 1.5:
            return Action.BUY_75
        elif combined_strength >= 1.0:
            return Action.BUY_50
        elif combined_strength >= 0.5:
            return Action.BUY_25
        elif combined_strength > -0.5:
            return Action.HOLD
        elif combined_strength > -1.0:
            return Action.SELL_25
        elif combined_strength > -1.5:
            return Action.SELL_50
        else:
            return Action.SELL_75

    def _adjust_target_weight(
        self,
        rec: ExpertRecommendation,
        factor_score: FactorScore
    ) -> float:
        """综合调整目标权重"""
        llm_weight = rec.target_weight

        # 因子评分转换为权重调整因子
        # composite_score 0.5 为中性, >0.5 增加权重, <0.5 减少权重
        factor_multiplier = 0.5 + factor_score.composite_score

        # 加权调整
        adjusted = llm_weight * (
            (1 - self.factor_weight) +
            self.factor_weight * factor_multiplier
        )

        return round(max(0.0, min(adjusted, 0.25)), 4)  # 限制单一资产最大25%

    def _enhance_reasoning(self, original_reasoning: str, factor_score: FactorScore) -> str:
        """增强决策理由"""
        factor_info = f"\n[因子分析] {factor_score.reasoning}"

        # 添加最重要的因子信息
        exposures = factor_score.factor_exposures
        if exposures:
            top_factors = sorted(
                exposures.items(),
                key=lambda x: abs(x[1].exposure),
                reverse=True
            )[:2]

            for name, exp in top_factors:
                factor_info += f"\n  - {name}: {exp.exposure:+.2f} ({exp.signal})"

        return original_reasoning + factor_info

    def _generate_factor_summary(self) -> List[str]:
        """生成因子摘要"""
        if not self._factor_scores:
            return []

        summaries = []

        # 统计信号分布
        signals = [s.signal for s in self._factor_scores.values()]
        buy_count = sum(1 for s in signals if "BUY" in s)
        sell_count = sum(1 for s in signals if "SELL" in s)

        summaries.append(
            f"[因子信号] 买入:{buy_count} 卖出:{sell_count} 持有:{len(signals)-buy_count-sell_count}"
        )

        # 最高和最低评分
        sorted_scores = sorted(
            self._factor_scores.items(),
            key=lambda x: x[1].composite_score,
            reverse=True
        )

        if sorted_scores:
            best = sorted_scores[0]
            worst = sorted_scores[-1]
            summaries.append(
                f"[因子最优] {best[0]}: {best[1].composite_score:.2f}"
            )
            summaries.append(
                f"[因子最差] {worst[0]}: {worst[1].composite_score:.2f}"
            )

        return summaries

    def get_factor_scores(self) -> Dict[str, FactorScore]:
        """获取所有因子评分"""
        return self._factor_scores.copy()

    def get_factor_report(self, symbol: str) -> Optional[str]:
        """获取单个资产的详细因子报告"""
        score = self._factor_scores.get(symbol)
        if not score:
            return None

        report = f"""
=== {symbol} 因子分析报告 ===
资产类别: {score.asset_class}
时间: {score.timestamp}

综合评分: {score.composite_score:.2f}
交易信号: {score.signal}
预期Alpha: {score.expected_alpha:.2%}
风险贡献: {score.risk_contribution:.2f}

因子暴露:
"""
        for name, exp in score.factor_exposures.items():
            report += f"  {name}:\n"
            report += f"    暴露度: {exp.exposure:+.3f}\n"
            report += f"    Z-Score: {exp.z_score:+.2f}\n"
            report += f"    百分位: {exp.percentile:.1f}%\n"
            report += f"    信号: {exp.signal}\n"
            report += f"    置信度: {exp.confidence:.2f}\n"

        report += f"\n分析理由:\n{score.reasoning}"

        return report


# 便捷函数: 创建增强版专家
def create_factor_enhanced_expert(
    base_expert_class: Type[BaseExpert],
    factor_scorer: BaseFactorScorer,
    **kwargs
) -> BaseExpert:
    """
    便捷函数: 为任意Expert类创建因子增强版本

    Args:
        base_expert_class: 原始Expert类
        factor_scorer: 因子评分器
        **kwargs: 传递给Expert的参数

    Returns:
        因子增强版Expert实例

    Example:
        ```python
        from finsage.agents.experts.stock_expert import StockExpert
        from finsage.factors import StockFactorScorer

        enhanced_expert = create_factor_enhanced_expert(
            StockExpert,
            StockFactorScorer(),
            llm_provider=my_llm,
            symbols=["AAPL", "MSFT"]
        )
        ```
    """
    # 动态创建增强类
    EnhancedClass = type(
        f"Enhanced{base_expert_class.__name__}",
        (FactorEnhancedExpertMixin, base_expert_class),
        {}
    )

    return EnhancedClass(factor_scorer=factor_scorer, **kwargs)


# 预定义的增强专家类
def get_enhanced_experts():
    """
    获取所有预定义的增强专家类

    Returns:
        Dict[asset_class, (EnhancedExpertClass, FactorScorerClass)]
    """
    from finsage.agents.experts.stock_expert import StockExpert
    from finsage.agents.experts.bond_expert import BondExpert
    from finsage.agents.experts.commodity_expert import CommodityExpert
    from finsage.agents.experts.reits_expert import REITsExpert
    from finsage.agents.experts.crypto_expert import CryptoExpert

    from finsage.factors import (
        StockFactorScorer,
        BondFactorScorer,
        CommodityFactorScorer,
        REITsFactorScorer,
        CryptoFactorScorer,
    )

    # 创建增强类
    class EnhancedStockExpert(FactorEnhancedExpertMixin, StockExpert):
        """因子增强股票专家"""
        pass

    class EnhancedBondExpert(FactorEnhancedExpertMixin, BondExpert):
        """因子增强债券专家"""
        pass

    class EnhancedCommodityExpert(FactorEnhancedExpertMixin, CommodityExpert):
        """因子增强商品专家"""
        pass

    class EnhancedREITsExpert(FactorEnhancedExpertMixin, REITsExpert):
        """因子增强REITs专家"""
        pass

    class EnhancedCryptoExpert(FactorEnhancedExpertMixin, CryptoExpert):
        """因子增强加密货币专家"""
        pass

    return {
        "stocks": (EnhancedStockExpert, StockFactorScorer),
        "bonds": (EnhancedBondExpert, BondFactorScorer),
        "commodities": (EnhancedCommodityExpert, CommodityFactorScorer),
        "reits": (EnhancedREITsExpert, REITsFactorScorer),
        "crypto": (EnhancedCryptoExpert, CryptoFactorScorer),
    }
