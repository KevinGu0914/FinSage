"""
Base Factor Scorer
因子评分基类

所有资产类别因子评分器的抽象基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """因子类型"""
    # 股票因子
    MARKET = "market"           # 市场因子
    SIZE = "size"               # 规模因子
    VALUE = "value"             # 价值因子
    PROFITABILITY = "profitability"  # 盈利因子
    INVESTMENT = "investment"   # 投资因子
    MOMENTUM = "momentum"       # 动量因子

    # 债券因子
    CARRY = "carry"             # 套息因子
    LOW_RISK = "low_risk"       # 低风险因子
    CREDIT = "credit"           # 信用因子
    DURATION = "duration"       # 久期因子

    # 商品因子
    TERM_STRUCTURE = "term_structure"  # 期限结构
    BASIS = "basis"             # 基差

    # REITs因子
    NAV = "nav"                 # 净资产价值
    IDIOSYNCRATIC = "idiosyncratic"  # 特质风险
    SECTOR = "sector"           # 行业因子

    # 加密货币因子
    NETWORK = "network"         # 网络效应
    ADOPTION = "adoption"       # 采纳度
    CRASH_RISK = "crash_risk"   # 崩盘风险


@dataclass
class FactorExposure:
    """单个因子暴露"""
    factor_type: FactorType
    exposure: float             # 因子暴露值 [-1, 1]
    z_score: float              # 标准化分数
    percentile: float           # 百分位数 [0, 100]
    signal: str                 # 交易信号: "LONG", "SHORT", "NEUTRAL"
    confidence: float           # 置信度 [0, 1]

    def to_dict(self) -> Dict:
        return {
            "factor": self.factor_type.value,
            "exposure": round(self.exposure, 4),
            "z_score": round(self.z_score, 4),
            "percentile": round(self.percentile, 2),
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class FactorScore:
    """资产的完整因子评分"""
    symbol: str
    asset_class: str
    timestamp: str
    factor_exposures: Dict[str, FactorExposure]
    composite_score: float      # 综合因子评分 [0, 1]
    expected_alpha: float       # 预期超额收益 (年化)
    risk_contribution: float    # 风险贡献度
    signal: str                 # 综合信号: "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    reasoning: str              # 评分理由

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "timestamp": self.timestamp,
            "factor_exposures": {k: v.to_dict() for k, v in self.factor_exposures.items()},
            "composite_score": round(self.composite_score, 4),
            "expected_alpha": round(self.expected_alpha, 4),
            "risk_contribution": round(self.risk_contribution, 4),
            "signal": self.signal,
            "reasoning": self.reasoning,
        }

    def get_exposure(self, factor_type: FactorType) -> Optional[FactorExposure]:
        """获取特定因子的暴露"""
        return self.factor_exposures.get(factor_type.value)


class BaseFactorScorer(ABC):
    """
    因子评分器基类

    所有资产类别的因子评分器都继承此类
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化因子评分器

        Args:
            config: 配置参数
                - lookback_period: 回溯期 (默认252天)
                - factor_weights: 各因子权重
                - signal_thresholds: 信号阈值
        """
        self.config = config or {}
        self.lookback_period = self.config.get("lookback_period", 252)
        self.factor_weights = self.config.get("factor_weights", self._default_weights())
        self.signal_thresholds = self.config.get("signal_thresholds", {
            "strong_buy": 0.8,
            "buy": 0.6,
            "hold_upper": 0.55,
            "hold_lower": 0.45,
            "sell": 0.4,
            "strong_sell": 0.2,
        })

        logger.info(f"Initialized {self.__class__.__name__}")

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """资产类别"""
        pass

    @property
    @abstractmethod
    def supported_factors(self) -> List[FactorType]:
        """支持的因子类型"""
        pass

    @abstractmethod
    def _default_weights(self) -> Dict[str, float]:
        """默认因子权重"""
        pass

    @abstractmethod
    def _compute_factor_exposures(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, FactorExposure]:
        """计算因子暴露"""
        pass

    def score(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
        market_regime: Optional[str] = None,
    ) -> FactorScore:
        """
        计算资产的因子评分

        Args:
            symbol: 资产代码
            data: 资产数据 (价格、基本面等)
            returns: 历史收益率序列
            market_regime: 市场体制 ("bull", "bear", "neutral")

        Returns:
            FactorScore: 完整的因子评分
        """
        from datetime import datetime

        # 计算因子暴露
        factor_exposures = self._compute_factor_exposures(symbol, data, returns)

        # 根据市场体制调整权重
        adjusted_weights = self._adjust_weights_for_regime(market_regime)

        # 计算综合分数
        composite_score = self._compute_composite_score(factor_exposures, adjusted_weights)

        # 计算预期Alpha
        expected_alpha = self._estimate_expected_alpha(factor_exposures, adjusted_weights)

        # 计算风险贡献
        risk_contribution = self._compute_risk_contribution(factor_exposures)

        # 生成交易信号
        signal = self._generate_signal(composite_score)

        # 生成评分理由
        reasoning = self._generate_reasoning(symbol, factor_exposures, composite_score)

        return FactorScore(
            symbol=symbol,
            asset_class=self.asset_class,
            timestamp=datetime.now().isoformat(),
            factor_exposures=factor_exposures,
            composite_score=composite_score,
            expected_alpha=expected_alpha,
            risk_contribution=risk_contribution,
            signal=signal,
            reasoning=reasoning,
        )

    def score_portfolio(
        self,
        symbols: List[str],
        data: Dict[str, Dict[str, Any]],
        returns: Optional[pd.DataFrame] = None,
        market_regime: Optional[str] = None,
    ) -> Dict[str, FactorScore]:
        """
        批量计算组合的因子评分

        Args:
            symbols: 资产代码列表
            data: 各资产数据
            returns: 收益率DataFrame
            market_regime: 市场体制

        Returns:
            Dict[symbol, FactorScore]
        """
        scores = {}
        for symbol in symbols:
            if symbol in data:
                symbol_returns = returns[symbol] if returns is not None and symbol in returns.columns else None
                try:
                    scores[symbol] = self.score(symbol, data[symbol], symbol_returns, market_regime)
                except Exception as e:
                    logger.warning(f"Failed to score {symbol}: {e}")

        return scores

    def _adjust_weights_for_regime(self, regime: Optional[str]) -> Dict[str, float]:
        """根据市场体制调整因子权重"""
        weights = self.factor_weights.copy()

        if regime == "bull":
            # 牛市: 增加动量权重，减少价值权重
            if "momentum" in weights:
                weights["momentum"] *= 1.2
            if "value" in weights:
                weights["value"] *= 0.8
        elif regime == "bear":
            # 熊市: 增加低风险和价值权重
            if "low_risk" in weights:
                weights["low_risk"] *= 1.3
            if "value" in weights:
                weights["value"] *= 1.2
            if "momentum" in weights:
                weights["momentum"] *= 0.7

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _compute_composite_score(
        self,
        exposures: Dict[str, FactorExposure],
        weights: Dict[str, float],
    ) -> float:
        """计算综合因子分数"""
        score = 0.0
        total_weight = 0.0

        for factor_name, exposure in exposures.items():
            weight = weights.get(factor_name, 0.0)
            if weight > 0:
                # 将exposure转换为[0,1]分数
                # exposure在[-1,1]，转换为[0,1]
                normalized_score = (exposure.exposure + 1) / 2
                score += weight * normalized_score
                total_weight += weight

        if total_weight > 0:
            score /= total_weight

        return max(0.0, min(1.0, score))

    def _estimate_expected_alpha(
        self,
        exposures: Dict[str, FactorExposure],
        weights: Dict[str, float],
    ) -> float:
        """估算预期超额收益 (年化)"""
        # 基于因子暴露和历史因子溢价估算
        # 简化模型: alpha = sum(exposure_i * premium_i)

        factor_premiums = self._get_factor_premiums()
        alpha = 0.0

        for factor_name, exposure in exposures.items():
            premium = factor_premiums.get(factor_name, 0.0)
            alpha += exposure.exposure * premium

        return alpha

    def _get_factor_premiums(self) -> Dict[str, float]:
        """获取历史因子溢价 (年化)"""
        # 默认因子溢价 (基于学术文献)
        return {
            "size": 0.02,           # SMB ~2%
            "value": 0.03,          # HML ~3%
            "profitability": 0.03,  # RMW ~3%
            "investment": 0.02,     # CMA ~2%
            "momentum": 0.06,       # MOM ~6%
            "carry": 0.02,          # Carry ~2%
            "low_risk": 0.01,       # Low-risk ~1%
            "term_structure": 0.04, # 商品期限结构 ~4%
        }

    def _compute_risk_contribution(self, exposures: Dict[str, FactorExposure]) -> float:
        """计算风险贡献度"""
        # 简化: 基于各因子暴露的绝对值
        total_exposure = sum(abs(e.exposure) for e in exposures.values())
        return total_exposure / len(exposures) if exposures else 0.0

    def _generate_signal(self, composite_score: float) -> str:
        """生成交易信号"""
        thresholds = self.signal_thresholds

        if composite_score >= thresholds["strong_buy"]:
            return "STRONG_BUY"
        elif composite_score >= thresholds["buy"]:
            return "BUY"
        elif composite_score >= thresholds["hold_upper"]:
            return "HOLD"
        elif composite_score >= thresholds["hold_lower"]:
            return "HOLD"
        elif composite_score >= thresholds["sell"]:
            return "SELL"
        else:
            return "STRONG_SELL"

    def _generate_reasoning(
        self,
        symbol: str,
        exposures: Dict[str, FactorExposure],
        composite_score: float,
    ) -> str:
        """生成评分理由"""
        # 找出最强和最弱因子
        sorted_exposures = sorted(
            exposures.items(),
            key=lambda x: x[1].exposure,
            reverse=True
        )

        if not sorted_exposures:
            return f"{symbol}: 无因子数据"

        strongest = sorted_exposures[0]
        weakest = sorted_exposures[-1]

        reasoning = f"{symbol}: 综合评分{composite_score:.2f}。"
        reasoning += f"最强因子: {strongest[0]}({strongest[1].exposure:+.2f})。"
        reasoning += f"最弱因子: {weakest[0]}({weakest[1].exposure:+.2f})。"

        return reasoning

    @staticmethod
    def normalize_to_zscore(value: float, mean: float, std: float) -> float:
        """标准化为Z-score"""
        if std <= 0:
            return 0.0
        return (value - mean) / std

    @staticmethod
    def zscore_to_percentile(zscore: float) -> float:
        """Z-score转换为百分位数"""
        from scipy import stats
        return stats.norm.cdf(zscore) * 100

    @staticmethod
    def clip_exposure(exposure: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """裁剪暴露值到合理范围"""
        return max(min_val, min(max_val, exposure))

    def compute_score(
        self,
        symbol: str,
        returns: Optional[pd.Series] = None,
        date: Optional[str] = None,
        market_regime: Optional[str] = None,
    ) -> float:
        """
        计算简化的因子分数 (用于资产排序)

        这是 score() 方法的简化版本，返回单个数值用于排序。

        Args:
            symbol: 资产代码
            returns: 收益率序列
            date: 日期 (用于构建数据)
            market_regime: 市场体制

        Returns:
            float: 综合因子分数 [0, 1]
        """
        # 构建简化的数据字典
        data = {}
        if returns is not None and len(returns) > 0:
            data["returns"] = returns
            price_series = (1 + returns).cumprod()
            # 提取标量值，避免 Series 布尔歧义错误
            data["price"] = float(price_series.iloc[-1]) if len(price_series) > 0 else 100.0
            data["volatility"] = float(returns.std() * np.sqrt(252))
            data["mean_return"] = float(returns.mean() * 252)

        try:
            factor_score = self.score(
                symbol=symbol,
                data=data,
                returns=returns,
                market_regime=market_regime,
            )
            return factor_score.composite_score
        except Exception as e:
            logger.warning(f"compute_score failed for {symbol}: {e}")
            return 0.5  # 返回中性分数
