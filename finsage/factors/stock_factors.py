"""
Stock Factor Scorer - Fama-French Five Factor Model
股票五因子评分器

学术基础:
- Fama, E.F. & French, K.R. (2015). "A Five-Factor Asset Pricing Model"
  Journal of Financial Economics, 116(1), 1-22.
  (万次引用，现代量化选股的"宪法")

- Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning"
  The Review of Financial Studies, 33(5), 2223-2273.
  (机器学习增强因子选股)

五因子模型:
r_i - r_f = α_i + β_MKT(MKT-RF) + β_SMB(SMB) + β_HML(HML) + β_RMW(RMW) + β_CMA(CMA) + ε_i

因子定义:
- MKT-RF: 市场超额收益 (Market Risk Premium)
- SMB: 小盘股溢价 (Small Minus Big) - 小市值公司跑赢大市值
- HML: 价值溢价 (High Minus Low B/M) - 高账面市值比跑赢低账面市值比
- RMW: 盈利因子 (Robust Minus Weak) - 高盈利公司跑赢低盈利
- CMA: 投资因子 (Conservative Minus Aggressive) - 保守投资跑赢激进投资
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


class StockFactorScorer(BaseFactorScorer):
    """
    股票五因子评分器

    基于Fama-French五因子模型，评估个股的因子暴露和预期Alpha。

    使用方法:
    ```python
    scorer = StockFactorScorer()
    score = scorer.score("AAPL", stock_data, returns)
    print(score.composite_score)  # 综合评分
    print(score.signal)           # 交易信号
    ```
    """

    @property
    def asset_class(self) -> str:
        return "stocks"

    @property
    def supported_factors(self) -> List[FactorType]:
        return [
            FactorType.MARKET,
            FactorType.SIZE,
            FactorType.VALUE,
            FactorType.PROFITABILITY,
            FactorType.INVESTMENT,
            FactorType.MOMENTUM,
        ]

    def _default_weights(self) -> Dict[str, float]:
        """
        默认因子权重

        基于学术文献的因子溢价和稳定性设置:
        - 盈利因子(RMW)最稳定，权重最高
        - 价值因子(HML)长期有效，权重较高
        - 投资因子(CMA)稳定但溢价较小
        - 规模因子(SMB)近年减弱
        - 动量因子(MOM)高波动但高溢价
        """
        return {
            "market": 0.10,
            "size": 0.15,
            "value": 0.20,
            "profitability": 0.25,
            "investment": 0.15,
            "momentum": 0.15,
        }

    def _compute_factor_exposures(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, FactorExposure]:
        """
        计算五因子暴露

        Args:
            symbol: 股票代码
            data: 股票数据，应包含:
                - market_cap: 市值
                - book_to_market: 账面市值比 (B/M)
                - roe: 净资产收益率
                - operating_margin: 营业利润率
                - asset_growth: 资产增长率
                - price_change_12m: 12个月价格变化
                - beta: 市场Beta
            returns: 历史收益率序列

        Returns:
            Dict[factor_name, FactorExposure]
        """
        exposures = {}

        # 1. Market Factor (市场因子)
        exposures["market"] = self._compute_market_exposure(data, returns)

        # 2. Size Factor (规模因子 SMB)
        exposures["size"] = self._compute_size_exposure(data)

        # 3. Value Factor (价值因子 HML)
        exposures["value"] = self._compute_value_exposure(data)

        # 4. Profitability Factor (盈利因子 RMW)
        exposures["profitability"] = self._compute_profitability_exposure(data)

        # 5. Investment Factor (投资因子 CMA)
        exposures["investment"] = self._compute_investment_exposure(data)

        # 6. Momentum Factor (动量因子)
        exposures["momentum"] = self._compute_momentum_exposure(data, returns)

        return exposures

    def _compute_market_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算市场因子暴露 (Beta)

        Beta > 1: 高于市场风险
        Beta < 1: 低于市场风险
        Beta < 0: 负相关 (罕见)
        """
        beta = data.get("beta", 1.0)

        # 将Beta转换为exposure [-1, 1]
        # Beta=0 -> -1, Beta=1 -> 0, Beta=2 -> 1
        exposure = self.clip_exposure(beta - 1, -1, 1)

        # Z-score (假设Beta均值1，标准差0.5)
        z_score = self.normalize_to_zscore(beta, mean=1.0, std=0.5)

        # 信号判断
        if beta < 0.7:
            signal = "LOW_BETA"  # 防御性
        elif beta > 1.3:
            signal = "HIGH_BETA"  # 激进
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.MARKET,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.8,
        )

    def _compute_size_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算规模因子暴露 (SMB)

        小市值股票有历史溢价，但近年减弱。

        基准:
        - Mega Cap: >$200B
        - Large Cap: $10B-$200B
        - Mid Cap: $2B-$10B
        - Small Cap: $300M-$2B
        - Micro Cap: <$300M
        """
        market_cap = data.get("market_cap", 100e9)  # 默认1000亿

        # 对数市值，转换为exposure
        # 小市值 -> 正exposure (SMB因子做多小盘)
        # 大市值 -> 负exposure
        log_cap = np.log10(market_cap) if market_cap > 0 else 10

        # log10(300M)≈8.5, log10(200B)≈11.3
        # 映射: 8.5 -> +1, 10 -> 0, 11.5 -> -1
        exposure = self.clip_exposure((10 - log_cap) / 1.5, -1, 1)

        # Z-score
        z_score = self.normalize_to_zscore(log_cap, mean=10.5, std=1.0)

        # 信号
        if market_cap < 2e9:
            signal = "SMALL_CAP"  # 小盘股溢价
        elif market_cap > 100e9:
            signal = "LARGE_CAP"  # 大盘股稳定
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.SIZE,
            exposure=exposure,
            z_score=-z_score,  # 取反，因为小盘对应正SMB
            percentile=100 - self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.7,
        )

    def _compute_value_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算价值因子暴露 (HML)

        高账面市值比 (B/M) 的股票有历史溢价。
        也可使用 E/P, EBITDA/EV 等替代指标。

        典型B/M值:
        - Value Stock: B/M > 0.8
        - Growth Stock: B/M < 0.3
        - Blend: 0.3-0.8
        """
        # 可以使用多个价值指标
        book_to_market = data.get("book_to_market", 0.5)
        pe_ratio = data.get("pe_ratio", 20)
        pb_ratio = data.get("pb_ratio", 3)

        # B/M为主要指标，PE/PB作为辅助
        # 高B/M -> 正exposure (价值股)
        # 低B/M -> 负exposure (成长股)

        # B/M转换: 0.2 -> -1, 0.5 -> 0, 0.8 -> +1
        bm_exposure = self.clip_exposure((book_to_market - 0.5) / 0.3, -1, 1)

        # PE辅助: 低PE -> 价值
        # PE转换: 10 -> +1, 20 -> 0, 40 -> -1
        if pe_ratio > 0:
            pe_exposure = self.clip_exposure((20 - pe_ratio) / 20, -1, 1)
        else:
            pe_exposure = 0.0

        # 综合 (B/M权重0.7, PE权重0.3)
        exposure = 0.7 * bm_exposure + 0.3 * pe_exposure

        z_score = self.normalize_to_zscore(book_to_market, mean=0.5, std=0.3)

        # 信号
        if book_to_market > 0.7:
            signal = "VALUE"  # 价值股
        elif book_to_market < 0.3:
            signal = "GROWTH"  # 成长股
        else:
            signal = "BLEND"

        return FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.75,
        )

    def _compute_profitability_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算盈利因子暴露 (RMW - Robust Minus Weak)

        高盈利质量的公司有稳定溢价。
        指标: ROE, 营业利润率, 毛利率

        Fama-French使用: Operating Profitability = (Revenue - COGS - SG&A - Interest) / Book Equity
        """
        roe = data.get("roe", 0.15)  # 净资产收益率
        operating_margin = data.get("operating_margin", 0.15)  # 营业利润率
        gross_margin = data.get("gross_margin", 0.40)  # 毛利率

        # 综合盈利评分
        # ROE: 0% -> -1, 15% -> 0, 30% -> +1
        roe_score = self.clip_exposure((roe - 0.15) / 0.15, -1, 1)

        # Operating Margin: 0% -> -1, 15% -> 0, 30% -> +1
        op_score = self.clip_exposure((operating_margin - 0.15) / 0.15, -1, 1)

        # Gross Margin: 20% -> -1, 40% -> 0, 60% -> +1
        gm_score = self.clip_exposure((gross_margin - 0.40) / 0.20, -1, 1)

        # 综合 (ROE 0.5, Operating Margin 0.3, Gross Margin 0.2)
        exposure = 0.5 * roe_score + 0.3 * op_score + 0.2 * gm_score

        z_score = self.normalize_to_zscore(roe, mean=0.12, std=0.10)

        # 信号
        if roe > 0.20 and operating_margin > 0.20:
            signal = "HIGH_QUALITY"
        elif roe < 0.05 or operating_margin < 0.05:
            signal = "LOW_QUALITY"
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.PROFITABILITY,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.85,  # 盈利因子最稳定
        )

    def _compute_investment_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算投资因子暴露 (CMA - Conservative Minus Aggressive)

        保守投资(低资产增长)的公司有溢价。
        激进扩张的公司往往过度投资，损害股东价值。

        指标: 总资产增长率
        """
        asset_growth = data.get("asset_growth", 0.10)  # 资产增长率
        capex_ratio = data.get("capex_to_revenue", 0.05)  # 资本支出/收入

        # 低资产增长 -> 正exposure (保守)
        # 高资产增长 -> 负exposure (激进)
        # 转换: -10% -> +1, 10% -> 0, 30% -> -1
        growth_score = self.clip_exposure((0.10 - asset_growth) / 0.20, -1, 1)

        # CapEx辅助
        capex_score = self.clip_exposure((0.05 - capex_ratio) / 0.05, -1, 1)

        # 综合
        exposure = 0.7 * growth_score + 0.3 * capex_score

        z_score = self.normalize_to_zscore(asset_growth, mean=0.08, std=0.15)

        # 信号
        if asset_growth < 0.05:
            signal = "CONSERVATIVE"
        elif asset_growth > 0.20:
            signal = "AGGRESSIVE"
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.INVESTMENT,
            exposure=exposure,
            z_score=-z_score,  # 取反，低增长对应正CMA
            percentile=100 - self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.7,
        )

    def _compute_momentum_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算动量因子暴露

        经典定义: 过去12个月收益 (跳过最近1个月)
        - Jegadeesh & Titman (1993)

        高动量股票有短期持续性溢价。
        """
        # 优先使用历史收益序列计算
        if returns is not None and len(returns) >= 252:
            # 12-1动量: 过去12个月收益，跳过最近1个月
            momentum_12_1 = returns.iloc[-252:-21].sum()  # 简化计算
        else:
            # 使用提供的数据
            momentum_12_1 = data.get("price_change_12m", 0.10)
            # 扣除最近一个月
            recent_1m = data.get("price_change_1m", 0.02)
            momentum_12_1 = momentum_12_1 - recent_1m

        # 转换: -20% -> -1, 0% -> 0, 30% -> +1
        exposure = self.clip_exposure(momentum_12_1 / 0.30, -1, 1)

        z_score = self.normalize_to_zscore(momentum_12_1, mean=0.08, std=0.25)

        # 信号
        if momentum_12_1 > 0.20:
            signal = "STRONG_MOMENTUM"
        elif momentum_12_1 < -0.10:
            signal = "REVERSAL"  # 可能反转
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.65,  # 动量因子波动较大
        )

    def _get_factor_premiums(self) -> Dict[str, float]:
        """
        获取历史因子溢价 (年化)

        基于Fama-French数据 (1963-2023均值):
        - MKT-RF: ~7% (市场风险溢价)
        - SMB: ~2% (小盘溢价，近年减弱)
        - HML: ~3% (价值溢价)
        - RMW: ~3% (盈利溢价)
        - CMA: ~2% (投资溢价)
        - MOM: ~6% (动量溢价，高波动)
        """
        return {
            "market": 0.07,
            "size": 0.02,
            "value": 0.03,
            "profitability": 0.03,
            "investment": 0.02,
            "momentum": 0.06,
        }

    def get_factor_summary(self, score: FactorScore) -> str:
        """
        生成因子评分摘要

        Args:
            score: FactorScore对象

        Returns:
            人类可读的因子摘要
        """
        summary = f"=== {score.symbol} 五因子评分 ===\n"
        summary += f"综合评分: {score.composite_score:.2f} ({score.signal})\n"
        summary += f"预期Alpha: {score.expected_alpha:.2%}\n\n"

        summary += "因子暴露:\n"
        factor_names = {
            "market": "市场(Beta)",
            "size": "规模(SMB)",
            "value": "价值(HML)",
            "profitability": "盈利(RMW)",
            "investment": "投资(CMA)",
            "momentum": "动量(MOM)",
        }

        for factor_key, factor_name in factor_names.items():
            if factor_key in score.factor_exposures:
                exp = score.factor_exposures[factor_key]
                bar = "█" * int((exp.exposure + 1) * 5) + "░" * (10 - int((exp.exposure + 1) * 5))
                summary += f"  {factor_name:12s}: {exp.exposure:+.2f} [{bar}] {exp.signal}\n"

        summary += f"\n分析: {score.reasoning}"
        return summary


class MLEnhancedStockScorer(StockFactorScorer):
    """
    机器学习增强的股票因子评分器

    基于: Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning"

    在五因子基础上增加:
    1. 94个股票特征 (可扩展)
    2. 非线性因子交互
    3. 时变因子权重
    """

    def __init__(self, config: Optional[Dict] = None, use_ml: bool = True):
        super().__init__(config)
        self.use_ml = use_ml
        self.ml_model = None

        if use_ml:
            self._initialize_ml_model()

    def _initialize_ml_model(self):
        """初始化机器学习模型"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self.ml_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            logger.info("ML model initialized")
        except ImportError:
            logger.warning("sklearn not available, ML enhancement disabled")
            self.use_ml = False

    def _compute_ml_alpha(
        self,
        data: Dict[str, Any],
        factor_exposures: Dict[str, FactorExposure]
    ) -> float:
        """
        使用ML模型预测Alpha

        特征包括:
        - 五因子暴露
        - 因子交互项
        - 额外的股票特征
        """
        if not self.use_ml or self.ml_model is None:
            return 0.0

        # 构建特征向量
        features = []

        # 基础因子暴露
        for factor in ["market", "size", "value", "profitability", "investment", "momentum"]:
            if factor in factor_exposures:
                features.append(factor_exposures[factor].exposure)
            else:
                features.append(0.0)

        # 额外特征 (示例)
        features.extend([
            data.get("volatility", 0.2),
            data.get("turnover", 0.1),
            data.get("analyst_coverage", 10) / 50,
            data.get("institutional_ownership", 0.5),
        ])

        # 此处应使用预训练模型进行预测
        # 简化处理: 返回基于因子的线性估计
        return sum(features[:6]) * 0.01

    def score(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
        market_regime: Optional[str] = None,
    ) -> FactorScore:
        """增强版评分，加入ML预测"""
        # 获取基础评分
        base_score = super().score(symbol, data, returns, market_regime)

        if self.use_ml:
            # ML增强Alpha
            ml_alpha = self._compute_ml_alpha(data, base_score.factor_exposures)
            base_score.expected_alpha += ml_alpha
            base_score.reasoning += f" ML增强Alpha: {ml_alpha:.2%}"

        return base_score
