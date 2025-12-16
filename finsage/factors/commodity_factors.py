"""
Commodity Factor Scorer - Term Structure & Factor Premiums
商品三因子评分器

学术基础:
- Miffre, J. & Rallis, G. (2007). "Momentum Strategies in Commodity Futures Markets"
  Journal of Banking & Finance.

- Erb, C.B. & Harvey, C.R. (2006). "The Strategic and Tactical Value of Commodity Futures"
  Financial Analysts Journal.

- Szymanowska, M. et al. (2014). "An Anatomy of Commodity Futures Risk Premia"
  Journal of Finance.

- 核心文献: The Journal of Portfolio Management (2013)
  "Strategic Allocation to Commodity Factor Premiums"

  核心观点: "傻傻地买入并持有一个大宗商品指数(如GSCI)不仅风险巨大，
            而且长期收益很低。真正的金矿在于大宗商品内部的因子(Factors)。"

商品三因子:
1. Term Structure (期限结构): Backwardation vs Contango
   - Backwardation (现货>期货): 正展期收益，做多溢价
   - Contango (现货<期货): 负展期收益，避免或做空

2. Momentum (动量): 12个月价格趋势
   - 商品动量效应强于股票

3. Basis (基差): 现货与期货价差
   - 反映供需紧张程度

额外因子 (扩展):
- Hedging Pressure: 套保者vs投机者净头寸
- Seasonality: 季节性模式
- Inventory: 库存水平
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


class CommodityFactorScorer(BaseFactorScorer):
    """
    商品三因子评分器

    基于期限结构、动量和基差因子评估商品的投资价值。

    关键洞察 (JPM 2013):
    - 买入持有商品指数长期收益很低
    - 因子策略能获取稳定的风险溢价
    - 期限结构因子最为重要

    使用方法:
    ```python
    scorer = CommodityFactorScorer()
    score = scorer.score("GLD", commodity_data, returns)
    print(score.factor_exposures["term_structure"])  # 期限结构信号
    ```
    """

    # 商品分类
    COMMODITY_SECTORS = {
        "energy": ["CL", "NG", "HO", "RB", "USO", "UNG", "XLE"],
        "precious_metals": ["GC", "SI", "PA", "PL", "GLD", "SLV", "IAU"],
        "industrial_metals": ["HG", "AL", "ZN", "NI", "COPX"],
        "agriculture": ["ZC", "ZS", "ZW", "KC", "CT", "DBA"],
        "livestock": ["LC", "LH", "FC"],
    }

    @property
    def asset_class(self) -> str:
        return "commodities"

    @property
    def supported_factors(self) -> List[FactorType]:
        return [
            FactorType.TERM_STRUCTURE,
            FactorType.MOMENTUM,
            FactorType.BASIS,
            FactorType.CARRY,
        ]

    def _default_weights(self) -> Dict[str, float]:
        """
        默认因子权重

        期限结构最重要 (决定展期收益):
        - Term Structure: 40% - 核心因子，决定持有成本/收益
        - Momentum: 30% - 商品动量效应显著
        - Basis: 20% - 供需信号
        - Carry: 10% - 辅助因子
        """
        return {
            "term_structure": 0.40,
            "momentum": 0.30,
            "basis": 0.20,
            "carry": 0.10,
        }

    def _compute_factor_exposures(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, FactorExposure]:
        """
        计算商品三因子暴露

        Args:
            symbol: 商品代码
            data: 商品数据，应包含:
                - front_price: 近月期货价格
                - back_price: 远月期货价格
                - spot_price: 现货价格
                - price_change_12m: 12个月价格变化
                - inventory_level: 库存水平 (可选)
                - days_to_expiry: 近月合约到期天数
            returns: 历史收益率序列

        Returns:
            Dict[factor_name, FactorExposure]
        """
        exposures = {}

        # 1. Term Structure Factor (期限结构因子) - 最重要!
        exposures["term_structure"] = self._compute_term_structure_exposure(data)

        # 2. Momentum Factor (动量因子)
        exposures["momentum"] = self._compute_momentum_exposure(data, returns)

        # 3. Basis Factor (基差因子)
        exposures["basis"] = self._compute_basis_exposure(data)

        # 4. Carry Factor (套息因子)
        exposures["carry"] = self._compute_carry_exposure(data)

        return exposures

    def _compute_term_structure_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算期限结构因子暴露

        这是商品投资最关键的因子!

        Backwardation (现货/近月 > 远月):
        - Roll Yield > 0
        - 持有多头获得正展期收益
        - 信号: LONG

        Contango (现货/近月 < 远月):
        - Roll Yield < 0
        - 持有多头承受负展期损失
        - 信号: SHORT 或 AVOID

        计算: Roll Yield = (近月 - 远月) / 远月 × 年化
        """
        front_price = data.get("front_price", data.get("price", 100))
        back_price = data.get("back_price", front_price)
        days_to_roll = data.get("days_to_roll", 30)  # 到展期的天数

        # 防止除零
        if back_price <= 0:
            back_price = front_price

        # 计算展期收益 (年化)
        if days_to_roll > 0:
            roll_yield = ((front_price - back_price) / back_price) * (365 / days_to_roll)
        else:
            roll_yield = (front_price - back_price) / back_price * 12  # 假设月度展期

        # 判断结构类型
        if roll_yield > 0.02:  # 年化 > 2%
            structure_type = "BACKWARDATION"
            signal = "LONG"  # 做多获得正展期
        elif roll_yield < -0.02:  # 年化 < -2%
            structure_type = "CONTANGO"
            signal = "AVOID"  # 避免或做空
        else:
            structure_type = "FLAT"
            signal = "NEUTRAL"

        # 转换为exposure [-1, 1]
        # Backwardation +10% -> +1, Contango -10% -> -1
        exposure = self.clip_exposure(roll_yield / 0.10, -1, 1)

        # Z-score (假设roll yield均值0，标准差5%)
        z_score = self.normalize_to_zscore(roll_yield, mean=0.0, std=0.05)

        return FactorExposure(
            factor_type=FactorType.TERM_STRUCTURE,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.85,  # 期限结构是确定性信号
        )

    def _compute_momentum_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算动量因子暴露

        商品动量效应比股票更强且更持久。
        典型回溯期: 12个月

        策略: 做多过去表现好的商品，做空表现差的
        """
        # 优先使用收益序列
        if returns is not None and len(returns) >= 252:
            momentum_12m = returns.iloc[-252:].sum()
        else:
            momentum_12m = data.get("price_change_12m", 0.0)

        # 转换: -30% -> -1, 0% -> 0, +30% -> +1
        exposure = self.clip_exposure(momentum_12m / 0.30, -1, 1)

        z_score = self.normalize_to_zscore(momentum_12m, mean=0.02, std=0.20)

        # 信号
        if momentum_12m > 0.15:
            signal = "STRONG_MOMENTUM"
        elif momentum_12m > 0:
            signal = "LONG"
        elif momentum_12m > -0.15:
            signal = "NEUTRAL"
        else:
            signal = "SHORT"

        return FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.75,
        )

    def _compute_basis_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算基差因子暴露

        Basis = (现货价格 - 期货价格) / 期货价格

        高基差 (现货溢价):
        - 供应紧张
        - 库存低
        - 信号: LONG

        低基差 (期货溢价):
        - 供应充裕
        - 库存高
        - 信号: SHORT 或 NEUTRAL
        """
        spot_price = data.get("spot_price", data.get("price", 100))
        futures_price = data.get("front_price", spot_price)

        # 确保价格非零
        if futures_price <= 0:
            futures_price = spot_price if spot_price > 0 else 100

        # 计算基差 (安全除法)
        basis = (spot_price - futures_price) / futures_price if futures_price > 1e-10 else 0

        # 转换: -5% -> -1, 0% -> 0, +5% -> +1
        exposure = self.clip_exposure(basis / 0.05, -1, 1)

        z_score = self.normalize_to_zscore(basis, mean=0.0, std=0.03)

        # 信号
        if basis > 0.03:
            signal = "TIGHT_SUPPLY"  # 供应紧张，做多
        elif basis < -0.03:
            signal = "OVERSUPPLY"  # 供应过剩
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.BASIS,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.70,
        )

    def _compute_carry_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算套息因子暴露

        Carry = Roll Yield + Collateral Yield - Storage Cost

        对于商品:
        - Collateral Yield: 保证金收益 (近似无风险利率)
        - Storage Cost: 存储成本 (因商品而异)

        简化计算: Carry ≈ Roll Yield + Risk-Free Rate
        """
        # 获取展期收益
        front_price = data.get("front_price", data.get("price", 100))
        back_price = data.get("back_price", front_price)
        days_to_roll = data.get("days_to_roll", 30)

        if back_price > 0 and days_to_roll > 0:
            roll_yield = ((front_price - back_price) / back_price) * (365 / days_to_roll)
        else:
            roll_yield = 0.0

        # 抵押品收益 (假设3%无风险利率)
        collateral_yield = data.get("risk_free_rate", 0.03)

        # 存储成本 (因商品而异)
        storage_cost = data.get("storage_cost", 0.02)  # 默认2%

        # 总carry
        carry = roll_yield + collateral_yield - storage_cost

        # 转换
        exposure = self.clip_exposure(carry / 0.10, -1, 1)

        z_score = self.normalize_to_zscore(carry, mean=0.02, std=0.05)

        # 信号
        if carry > 0.05:
            signal = "HIGH_CARRY"
        elif carry < -0.02:
            signal = "NEGATIVE_CARRY"
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.CARRY,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.65,
        )

    def _get_factor_premiums(self) -> Dict[str, float]:
        """
        获取历史因子溢价 (年化)

        基于学术文献:
        - Term Structure: ~4% (Backwardation vs Contango差异)
        - Momentum: ~5% (商品动量溢价)
        - Basis: ~2%
        - Carry: ~3%
        """
        return {
            "term_structure": 0.04,
            "momentum": 0.05,
            "basis": 0.02,
            "carry": 0.03,
        }

    def get_term_structure_summary(self, symbol: str, data: Dict[str, Any]) -> str:
        """
        生成期限结构分析摘要

        这是商品投资决策的核心信息
        """
        front = data.get("front_price", 100)
        back = data.get("back_price", 100)
        days = data.get("days_to_roll", 30)

        if back > 0 and days > 0:
            roll_yield = ((front - back) / back) * (365 / days)
        else:
            roll_yield = 0

        if roll_yield > 0.02:
            structure = "Backwardation (现货溢价)"
            recommendation = "做多有正展期收益"
        elif roll_yield < -0.02:
            structure = "Contango (期货溢价)"
            recommendation = "持有多头将承受展期损失，考虑避免或做空"
        else:
            structure = "Flat (平坦)"
            recommendation = "无明显期限结构优势"

        summary = f"""
=== {symbol} 期限结构分析 ===
近月价格: {front:.2f}
远月价格: {back:.2f}
展期天数: {days}天
年化Roll Yield: {roll_yield:.2%}

结构类型: {structure}
建议: {recommendation}

注意: 期限结构是商品投资最重要的因子!
      "买入持有"策略在Contango市场会持续亏损。
"""
        return summary

    def get_sector_allocation(
        self,
        scores: Dict[str, FactorScore]
    ) -> Dict[str, float]:
        """
        基于因子评分的商品板块配置建议

        Args:
            scores: 各商品的因子评分

        Returns:
            板块配置权重
        """
        sector_scores = {}

        for sector, symbols in self.COMMODITY_SECTORS.items():
            sector_sum = 0.0
            count = 0
            for symbol in symbols:
                if symbol in scores:
                    sector_sum += scores[symbol].composite_score
                    count += 1
            if count > 0:
                sector_scores[sector] = sector_sum / count

        # 归一化
        total = sum(sector_scores.values())
        if total > 0:
            return {k: v / total for k, v in sector_scores.items()}
        else:
            # 等权
            return {k: 1.0 / len(self.COMMODITY_SECTORS) for k in self.COMMODITY_SECTORS}
