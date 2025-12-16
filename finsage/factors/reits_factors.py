"""
REITs Factor Scorer - Asset-Level Risk and NAV Analysis
REITs因子评分器

学术基础:
- Sagi, J.S. (2021). "Asset-Level Risk and Return in Real Estate Investments"
  The Review of Financial Studies, 34(8), 3877-3919.

  核心发现: "时间无法消除风险" - 单体物业的特质风险不会随持有期延长而下降，
           这与传统资产定价理论的时间分散化假设相矛盾。

- Giacoletti, M. (2021). "Idiosyncratic Risk in Housing Markets"
  The Review of Financial Studies, 34(8), 3695-3741.

  核心发现: "我们看到的房地产指数波动率是假的，真实的单体风险要大得多，
           而且这种风险是可以预测的。"

- JREFE (2021). "Betting Against the Sentiment in REIT NAV Premiums"

REITs因子:
1. NAV Premium/Discount (净资产折溢价):
   - NAV Premium (市价>NAV): 看空信号
   - NAV Discount (市价<NAV): 潜在价值

2. Idiosyncratic Risk (特质风险):
   - 基于房价分位数和地理位置可预测
   - 高特质风险需要更高风险溢价

3. Sector Outlook (行业前景):
   - 数据中心: 高增长
   - 物流仓储: 电商受益
   - 住宅: 防御性
   - 零售/办公: 结构性挑战

4. Interest Rate Sensitivity (利率敏感度):
   - REITs对利率高度敏感
   - 利率上升时表现差
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


class REITsFactorScorer(BaseFactorScorer):
    """
    REITs因子评分器

    基于NAV折溢价、特质风险和行业前景评估REITs投资价值。

    关键洞察:
    1. NAV折溢价是REITs的核心估值指标
    2. 单体物业风险远大于指数波动率
    3. 行业分化严重 (数据中心 vs 零售)
    4. 利率敏感度高于一般股票

    适用范围:
    - REITs ETF: VNQ, IYR, SCHH
    - 个股REITs: DLR, EQIX, PLD, EQR, AVB等

    使用方法:
    ```python
    scorer = REITsFactorScorer()
    score = scorer.score("VNQ", reit_data, returns)
    print(score.factor_exposures["nav"])  # NAV因子
    ```
    """

    # REITs行业分类及前景评分
    SECTOR_OUTLOOK = {
        "data_center": {
            "outlook": "positive",
            "score": 0.85,
            "drivers": ["cloud computing", "AI", "digital transformation"],
            "examples": ["DLR", "EQIX", "COR"],
        },
        "logistics": {
            "outlook": "positive",
            "score": 0.80,
            "drivers": ["e-commerce", "supply chain", "last-mile delivery"],
            "examples": ["PLD", "STAG", "REXR"],
        },
        "residential": {
            "outlook": "neutral_positive",
            "score": 0.70,
            "drivers": ["demographics", "affordability crisis", "rent growth"],
            "examples": ["EQR", "AVB", "ESS", "MAA"],
        },
        "healthcare": {
            "outlook": "neutral",
            "score": 0.60,
            "drivers": ["aging population", "healthcare spending"],
            "examples": ["WELL", "VTR", "PEAK"],
        },
        "retail": {
            "outlook": "negative",
            "score": 0.35,
            "drivers": ["e-commerce disruption", "mall decline"],
            "examples": ["SPG", "O", "NNN"],
        },
        "office": {
            "outlook": "negative",
            "score": 0.30,
            "drivers": ["remote work", "hybrid models", "vacancy rates"],
            "examples": ["BXP", "ARE", "KRC"],
        },
        "diversified": {
            "outlook": "neutral",
            "score": 0.55,
            "drivers": ["diversification"],
            "examples": ["VNQ", "IYR", "SCHH"],
        },
    }

    @property
    def asset_class(self) -> str:
        return "reits"

    @property
    def supported_factors(self) -> List[FactorType]:
        return [
            FactorType.NAV,
            FactorType.IDIOSYNCRATIC,
            FactorType.SECTOR,
            FactorType.VALUE,
            FactorType.MOMENTUM,
        ]

    def _default_weights(self) -> Dict[str, float]:
        """
        默认因子权重

        NAV是REITs的核心估值指标:
        - NAV: 35% - 核心估值因子
        - Sector: 25% - 行业前景关键
        - Idiosyncratic: 20% - 特质风险定价
        - Value: 10% - 其他估值指标
        - Momentum: 10% - 价格动量
        """
        return {
            "nav": 0.35,
            "idiosyncratic": 0.20,
            "sector": 0.25,
            "value": 0.10,
            "momentum": 0.10,
        }

    def _compute_factor_exposures(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, FactorExposure]:
        """
        计算REITs因子暴露

        Args:
            symbol: REITs代码
            data: REITs数据，应包含:
                - price: 当前股价
                - nav: 净资产价值
                - nav_premium: NAV溢价率
                - sector: 行业分类
                - dividend_yield: 股息率
                - ffo_multiple: P/FFO倍数
                - occupancy_rate: 入住率
                - cap_rate: 资本化率
                - interest_rate_sensitivity: 利率敏感度
            returns: 历史收益率序列

        Returns:
            Dict[factor_name, FactorExposure]
        """
        exposures = {}

        # 1. NAV Factor (NAV因子) - 最重要!
        exposures["nav"] = self._compute_nav_exposure(data)

        # 2. Idiosyncratic Risk Factor (特质风险因子)
        exposures["idiosyncratic"] = self._compute_idiosyncratic_exposure(data, returns)

        # 3. Sector Factor (行业因子)
        exposures["sector"] = self._compute_sector_exposure(symbol, data)

        # 4. Value Factor (估值因子)
        exposures["value"] = self._compute_value_exposure(data)

        # 5. Momentum Factor (动量因子)
        exposures["momentum"] = self._compute_momentum_exposure(data, returns)

        return exposures

    def _compute_nav_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算NAV因子暴露

        NAV Premium/Discount是REITs最重要的估值信号:

        NAV Premium (P > NAV):
        - 市场对未来增长乐观
        - 但历史上高溢价往往预示未来低收益
        - 信号: SELL (逆向)

        NAV Discount (P < NAV):
        - 市场悲观或流动性差
        - 潜在价值投资机会
        - 信号: BUY (逆向)

        基于: JREFE 2021 "Betting Against the Sentiment in REIT NAV Premiums"
        """
        price = data.get("price", 100)
        nav = data.get("nav", data.get("nav_per_share", 100))

        # 直接使用提供的溢价率，或计算
        if "nav_premium" in data:
            nav_premium = data["nav_premium"]
        else:
            nav_premium = (price - nav) / nav if nav > 0 else 0

        # NAV策略是逆向的: 折价做多，溢价做空
        # 折价20% -> +1, 平价 -> 0, 溢价20% -> -1
        exposure = self.clip_exposure(-nav_premium / 0.20, -1, 1)

        z_score = self.normalize_to_zscore(nav_premium, mean=0.0, std=0.15)

        # 信号 (逆向逻辑)
        if nav_premium < -0.10:
            signal = "DEEP_DISCOUNT"  # 深度折价，买入
        elif nav_premium < 0:
            signal = "DISCOUNT"  # 折价
        elif nav_premium > 0.15:
            signal = "HIGH_PREMIUM"  # 高溢价，卖出
        elif nav_premium > 0:
            signal = "PREMIUM"  # 溢价
        else:
            signal = "FAIR_VALUE"

        return FactorExposure(
            factor_type=FactorType.NAV,
            exposure=exposure,
            z_score=-z_score,  # 取反，折价对应正分
            percentile=100 - self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.80,
        )

    def _compute_idiosyncratic_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算特质风险因子暴露

        基于: RFS 2021 Sagi & Giacoletti

        关键发现:
        1. 单体物业风险是指数波动率的2-3倍
        2. 特质风险可以预测 (基于房价分位数、地理位置)
        3. 高特质风险需要更高风险补偿

        预测变量:
        - 房价分位数 (高价物业风险更高)
        - 地理集中度
        - 物业类型多样性
        """
        # 计算实现波动率
        if returns is not None and len(returns) >= 60:
            realized_vol = returns.std() * np.sqrt(252)
        else:
            realized_vol = data.get("volatility", 0.20)

        # 特质风险估计 (简化)
        # 因子: 波动率、地理集中度、物业多样性
        geographic_concentration = data.get("geographic_concentration", 0.5)
        property_diversity = data.get("property_diversity", 0.5)

        # 高波动 + 高集中 + 低多样性 = 高特质风险
        idio_risk = (
            0.5 * realized_vol / 0.25 +  # 波动率贡献
            0.3 * geographic_concentration +  # 地理集中贡献
            0.2 * (1 - property_diversity)  # 缺乏多样性贡献
        )

        # 归一化到[0, 1]
        idio_risk = min(1.0, max(0.0, idio_risk))

        # 低特质风险是好的 -> 转换为正向因子
        # 低风险 -> 正exposure
        exposure = self.clip_exposure((0.5 - idio_risk) * 2, -1, 1)

        # 信号
        if idio_risk < 0.3:
            signal = "LOW_IDIO_RISK"
        elif idio_risk > 0.6:
            signal = "HIGH_IDIO_RISK"
        else:
            signal = "MODERATE_RISK"

        return FactorExposure(
            factor_type=FactorType.IDIOSYNCRATIC,
            exposure=exposure,
            z_score=self.normalize_to_zscore(idio_risk, 0.4, 0.2),
            percentile=(1 - idio_risk) * 100,  # 低风险=高分位
            signal=signal,
            confidence=0.65,
        )

    def _compute_sector_exposure(self, symbol: str, data: Dict[str, Any]) -> FactorExposure:
        """
        计算行业因子暴露

        REITs行业分化严重:
        - 数据中心/物流: 结构性受益
        - 住宅: 防御性
        - 零售/办公: 结构性挑战
        """
        # 获取行业
        sector = data.get("sector", self._infer_sector(symbol))

        # 获取行业展望
        sector_info = self.SECTOR_OUTLOOK.get(sector, self.SECTOR_OUTLOOK["diversified"])
        sector_score = sector_info["score"]

        # 转换: 0.3 -> -0.4, 0.5 -> 0, 0.85 -> +0.7
        exposure = self.clip_exposure((sector_score - 0.5) * 2, -1, 1)

        # 信号
        outlook = sector_info["outlook"]
        if outlook == "positive":
            signal = "GROWTH_SECTOR"
        elif outlook == "negative":
            signal = "CHALLENGED_SECTOR"
        else:
            signal = "STABLE_SECTOR"

        return FactorExposure(
            factor_type=FactorType.SECTOR,
            exposure=exposure,
            z_score=self.normalize_to_zscore(sector_score, 0.55, 0.2),
            percentile=sector_score * 100,
            signal=signal,
            confidence=0.75,
        )

    def _compute_value_exposure(self, data: Dict[str, Any]) -> FactorExposure:
        """
        计算估值因子暴露

        REITs估值指标:
        - P/FFO (类似P/E)
        - 股息率
        - Cap Rate vs 无风险利率
        """
        p_ffo = data.get("p_ffo", data.get("ffo_multiple", 15))
        dividend_yield = data.get("dividend_yield", 0.04)
        cap_rate = data.get("cap_rate", 0.05)
        risk_free = data.get("risk_free_rate", 0.04)

        # P/FFO估值: 10 -> +1, 15 -> 0, 25 -> -1
        ffo_score = self.clip_exposure((15 - p_ffo) / 10, -1, 1)

        # 股息率: 2% -> -0.5, 4% -> 0, 6% -> +0.5
        yield_score = self.clip_exposure((dividend_yield - 0.04) / 0.04, -1, 1)

        # Cap Rate Spread: 高于无风险利率越多越好
        cap_spread = cap_rate - risk_free
        spread_score = self.clip_exposure(cap_spread / 0.03, -1, 1)

        # 综合
        value_score = 0.4 * ffo_score + 0.3 * yield_score + 0.3 * spread_score

        # 信号
        if value_score > 0.3:
            signal = "UNDERVALUED"
        elif value_score < -0.3:
            signal = "OVERVALUED"
        else:
            signal = "FAIR_VALUE"

        return FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=value_score,
            z_score=self.normalize_to_zscore(value_score, 0, 0.3),
            percentile=self.zscore_to_percentile(value_score / 0.3),
            signal=signal,
            confidence=0.70,
        )

    def _compute_momentum_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """计算动量因子暴露"""
        if returns is not None and len(returns) >= 252:
            momentum_12m = returns.iloc[-252:].sum()
        else:
            momentum_12m = data.get("price_change_12m", 0.0)

        exposure = self.clip_exposure(momentum_12m / 0.25, -1, 1)

        if momentum_12m > 0.15:
            signal = "STRONG_MOMENTUM"
        elif momentum_12m < -0.10:
            signal = "NEGATIVE_MOMENTUM"
        else:
            signal = "NEUTRAL"

        return FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=exposure,
            z_score=self.normalize_to_zscore(momentum_12m, 0.05, 0.15),
            percentile=self.zscore_to_percentile(momentum_12m / 0.15),
            signal=signal,
            confidence=0.60,
        )

    def _infer_sector(self, symbol: str) -> str:
        """根据代码推断行业"""
        symbol_upper = symbol.upper()

        for sector, info in self.SECTOR_OUTLOOK.items():
            if symbol_upper in info.get("examples", []):
                return sector

        # 默认为diversified
        return "diversified"

    def _get_factor_premiums(self) -> Dict[str, float]:
        """
        获取历史因子溢价 (年化)

        REITs因子溢价:
        - NAV折溢价: ~2% (逆向策略)
        - 行业因子: ~3% (增长行业vs衰退行业)
        - 特质风险: ~1.5%
        """
        return {
            "nav": 0.02,
            "idiosyncratic": 0.015,
            "sector": 0.03,
            "value": 0.01,
            "momentum": 0.01,
        }

    def get_nav_analysis(self, symbol: str, data: Dict[str, Any]) -> str:
        """
        生成NAV折溢价分析报告
        """
        price = data.get("price", 100)
        nav = data.get("nav", 100)
        nav_premium = (price - nav) / nav if nav > 0 else 0

        analysis = f"""
=== {symbol} NAV分析 ===
当前股价: ${price:.2f}
估算NAV: ${nav:.2f}
NAV溢价/折价: {nav_premium:.1%}

解读:
"""
        if nav_premium < -0.15:
            analysis += """
深度折价 (>15%):
- 可能存在价值投资机会
- 需关注: 是否有基本面恶化
- 策略: 逐步建仓，等待均值回归
"""
        elif nav_premium > 0.20:
            analysis += """
高度溢价 (>20%):
- 市场过度乐观
- 历史上高溢价往往预示低未来收益
- 策略: 考虑减持或避免
"""
        else:
            analysis += """
合理范围:
- 估值相对中性
- 关注其他因子 (行业、增长)
"""

        analysis += f"""
学术依据:
- RFS 2021: NAV溢价是情绪指标
- 逆向策略: Betting Against NAV Premium
"""
        return analysis

    def get_sector_allocation(
        self,
        scores: Dict[str, FactorScore],
        market_regime: str = "neutral"
    ) -> Dict[str, float]:
        """
        基于因子评分的REITs行业配置建议

        Args:
            scores: 各REITs的因子评分
            market_regime: 市场体制

        Returns:
            行业配置权重
        """
        # 基础配置
        base_allocation = {
            "data_center": 0.20,
            "logistics": 0.20,
            "residential": 0.25,
            "healthcare": 0.15,
            "retail": 0.10,
            "office": 0.05,
            "diversified": 0.05,
        }

        # 根据市场体制调整
        if market_regime == "risk_off":
            # 避险: 增加防御性行业
            base_allocation["residential"] *= 1.3
            base_allocation["healthcare"] *= 1.2
            base_allocation["retail"] *= 0.7
            base_allocation["office"] *= 0.5
        elif market_regime == "risk_on":
            # 激进: 增加成长性行业
            base_allocation["data_center"] *= 1.3
            base_allocation["logistics"] *= 1.2

        # 归一化
        total = sum(base_allocation.values())
        return {k: v / total for k, v in base_allocation.items()}
