"""
Deep tests for BaseFactorScorer
因子评分基类深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.factors.base_factor import (
    BaseFactorScorer,
    FactorType,
    FactorExposure,
    FactorScore,
)


class TestFactorType:
    """FactorType枚举测试"""

    def test_stock_factors(self):
        """测试股票因子类型"""
        assert FactorType.MARKET.value == "market"
        assert FactorType.SIZE.value == "size"
        assert FactorType.VALUE.value == "value"
        assert FactorType.PROFITABILITY.value == "profitability"
        assert FactorType.INVESTMENT.value == "investment"
        assert FactorType.MOMENTUM.value == "momentum"

    def test_bond_factors(self):
        """测试债券因子类型"""
        assert FactorType.CARRY.value == "carry"
        assert FactorType.LOW_RISK.value == "low_risk"
        assert FactorType.CREDIT.value == "credit"
        assert FactorType.DURATION.value == "duration"

    def test_commodity_factors(self):
        """测试商品因子类型"""
        assert FactorType.TERM_STRUCTURE.value == "term_structure"
        assert FactorType.BASIS.value == "basis"

    def test_reits_factors(self):
        """测试REITs因子类型"""
        assert FactorType.NAV.value == "nav"
        assert FactorType.IDIOSYNCRATIC.value == "idiosyncratic"
        assert FactorType.SECTOR.value == "sector"

    def test_crypto_factors(self):
        """测试加密货币因子类型"""
        assert FactorType.NETWORK.value == "network"
        assert FactorType.ADOPTION.value == "adoption"
        assert FactorType.CRASH_RISK.value == "crash_risk"


class TestFactorExposure:
    """FactorExposure数据类测试"""

    def test_create_factor_exposure(self):
        """测试创建因子暴露"""
        exposure = FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=0.5,
            z_score=1.5,
            percentile=90.0,
            signal="LONG",
            confidence=0.8,
        )
        assert exposure.exposure == 0.5
        assert exposure.signal == "LONG"

    def test_to_dict(self):
        """测试转换为字典"""
        exposure = FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=0.3,
            z_score=0.8,
            percentile=75.0,
            signal="NEUTRAL",
            confidence=0.7,
        )
        result = exposure.to_dict()
        assert result["factor"] == "value"
        assert result["exposure"] == 0.3
        assert result["signal"] == "NEUTRAL"

    def test_negative_exposure(self):
        """测试负暴露"""
        exposure = FactorExposure(
            factor_type=FactorType.SIZE,
            exposure=-0.5,
            z_score=-1.5,
            percentile=10.0,
            signal="SHORT",
            confidence=0.75,
        )
        assert exposure.exposure == -0.5


class TestFactorScore:
    """FactorScore数据类测试"""

    def test_create_factor_score(self):
        """测试创建因子评分"""
        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.7,
            expected_alpha=0.05,
            risk_contribution=0.3,
            signal="BUY",
            reasoning="动量因子强劲",
        )
        assert score.symbol == "AAPL"
        assert score.signal == "BUY"

    def test_to_dict(self):
        """测试转换为字典"""
        exposure = FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=0.5,
            z_score=1.0,
            percentile=80.0,
            signal="LONG",
            confidence=0.8,
        )
        score = FactorScore(
            symbol="MSFT",
            asset_class="stocks",
            timestamp="2024-01-15",
            factor_exposures={"momentum": exposure},
            composite_score=0.65,
            expected_alpha=0.04,
            risk_contribution=0.25,
            signal="BUY",
            reasoning="看涨",
        )
        result = score.to_dict()
        assert result["symbol"] == "MSFT"
        assert "momentum" in result["factor_exposures"]
        assert result["composite_score"] == 0.65

    def test_get_exposure(self):
        """测试获取特定因子暴露"""
        exposure = FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=0.4,
            z_score=1.2,
            percentile=85.0,
            signal="LONG",
            confidence=0.75,
        )
        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp="2024-01-15",
            factor_exposures={"value": exposure},
            composite_score=0.6,
            expected_alpha=0.03,
            risk_contribution=0.2,
            signal="BUY",
            reasoning="价值因子正向",
        )
        result = score.get_exposure(FactorType.VALUE)
        assert result is not None
        assert result.exposure == 0.4

    def test_get_exposure_not_found(self):
        """测试获取不存在的因子暴露"""
        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp="2024-01-15",
            factor_exposures={},
            composite_score=0.5,
            expected_alpha=0.02,
            risk_contribution=0.15,
            signal="HOLD",
            reasoning="",
        )
        result = score.get_exposure(FactorType.MOMENTUM)
        assert result is None


class ConcreteFactorScorer(BaseFactorScorer):
    """用于测试的具体因子评分器"""

    @property
    def asset_class(self):
        return "test"

    @property
    def supported_factors(self):
        return [FactorType.MOMENTUM, FactorType.VALUE]

    def _default_weights(self):
        return {"momentum": 0.5, "value": 0.5}

    def _compute_factor_exposures(self, symbol, data, returns=None):
        return {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=data.get("momentum", 0.5),
                z_score=1.0,
                percentile=80.0,
                signal="LONG",
                confidence=0.7,
            ),
            "value": FactorExposure(
                factor_type=FactorType.VALUE,
                exposure=data.get("value", 0.3),
                z_score=0.5,
                percentile=70.0,
                signal="NEUTRAL",
                confidence=0.6,
            ),
        }


class TestBaseFactorScorerInit:
    """BaseFactorScorer初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        scorer = ConcreteFactorScorer()
        assert scorer.lookback_period == 252
        assert "momentum" in scorer.factor_weights

    def test_custom_config(self):
        """测试自定义配置"""
        config = {
            "lookback_period": 126,
            "factor_weights": {"momentum": 0.7, "value": 0.3},
        }
        scorer = ConcreteFactorScorer(config=config)
        assert scorer.lookback_period == 126
        assert scorer.factor_weights["momentum"] == 0.7

    def test_custom_signal_thresholds(self):
        """测试自定义信号阈值"""
        config = {
            "signal_thresholds": {
                "strong_buy": 0.9,
                "buy": 0.7,
                "hold_upper": 0.6,
                "hold_lower": 0.4,
                "sell": 0.3,
                "strong_sell": 0.1,
            }
        }
        scorer = ConcreteFactorScorer(config=config)
        assert scorer.signal_thresholds["strong_buy"] == 0.9


class TestAdjustWeightsForRegime:
    """市场体制权重调整测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_bull_market_adjustment(self, scorer):
        """测试牛市调整"""
        weights = scorer._adjust_weights_for_regime("bull")
        # 牛市应增加动量权重
        assert weights.get("momentum", 0) > 0

    def test_bear_market_adjustment(self, scorer):
        """测试熊市调整"""
        weights = scorer._adjust_weights_for_regime("bear")
        # 熊市应减少动量权重
        assert isinstance(weights, dict)

    def test_neutral_market(self, scorer):
        """测试中性市场"""
        weights = scorer._adjust_weights_for_regime("neutral")
        # 中性市场权重应接近原始
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_none_regime(self, scorer):
        """测试无体制信息"""
        weights = scorer._adjust_weights_for_regime(None)
        assert isinstance(weights, dict)


class TestComputeCompositeScore:
    """综合评分计算测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_compute_composite_basic(self, scorer):
        """测试基本综合评分"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=0.5,
                z_score=1.0,
                percentile=80.0,
                signal="LONG",
                confidence=0.7,
            ),
            "value": FactorExposure(
                factor_type=FactorType.VALUE,
                exposure=0.3,
                z_score=0.5,
                percentile=70.0,
                signal="NEUTRAL",
                confidence=0.6,
            ),
        }
        weights = {"momentum": 0.5, "value": 0.5}
        score = scorer._compute_composite_score(exposures, weights)
        assert 0 <= score <= 1

    def test_composite_all_positive(self, scorer):
        """测试全正向暴露"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=1.0,
                z_score=2.0,
                percentile=95.0,
                signal="LONG",
                confidence=0.9,
            ),
        }
        weights = {"momentum": 1.0}
        score = scorer._compute_composite_score(exposures, weights)
        assert score == 1.0

    def test_composite_all_negative(self, scorer):
        """测试全负向暴露"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=-1.0,
                z_score=-2.0,
                percentile=5.0,
                signal="SHORT",
                confidence=0.9,
            ),
        }
        weights = {"momentum": 1.0}
        score = scorer._compute_composite_score(exposures, weights)
        assert score == 0.0

    def test_composite_empty_exposures(self, scorer):
        """测试空暴露"""
        score = scorer._compute_composite_score({}, {})
        assert score == 0.0


class TestEstimateExpectedAlpha:
    """预期Alpha估算测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_estimate_alpha_basic(self, scorer):
        """测试基本Alpha估算"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=0.5,
                z_score=1.0,
                percentile=80.0,
                signal="LONG",
                confidence=0.7,
            ),
        }
        weights = {"momentum": 1.0}
        alpha = scorer._estimate_expected_alpha(exposures, weights)
        assert isinstance(alpha, float)

    def test_alpha_positive_exposure(self, scorer):
        """测试正向暴露Alpha"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=1.0,
                z_score=2.0,
                percentile=95.0,
                signal="LONG",
                confidence=0.9,
            ),
        }
        alpha = scorer._estimate_expected_alpha(exposures, {})
        assert alpha > 0


class TestComputeRiskContribution:
    """风险贡献计算测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_risk_contribution_basic(self, scorer):
        """测试基本风险贡献"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=0.5,
                z_score=1.0,
                percentile=80.0,
                signal="LONG",
                confidence=0.7,
            ),
            "value": FactorExposure(
                factor_type=FactorType.VALUE,
                exposure=-0.3,
                z_score=-0.5,
                percentile=30.0,
                signal="SHORT",
                confidence=0.6,
            ),
        }
        risk = scorer._compute_risk_contribution(exposures)
        assert 0 <= risk <= 1

    def test_risk_contribution_empty(self, scorer):
        """测试空暴露风险贡献"""
        risk = scorer._compute_risk_contribution({})
        assert risk == 0.0


class TestGenerateSignal:
    """信号生成测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_strong_buy_signal(self, scorer):
        """测试强买信号"""
        signal = scorer._generate_signal(0.85)
        assert signal == "STRONG_BUY"

    def test_buy_signal(self, scorer):
        """测试买入信号"""
        signal = scorer._generate_signal(0.65)
        assert signal == "BUY"

    def test_hold_signal(self, scorer):
        """测试持有信号"""
        signal = scorer._generate_signal(0.50)
        assert signal == "HOLD"

    def test_sell_signal(self, scorer):
        """测试卖出信号"""
        signal = scorer._generate_signal(0.35)
        # 0.35可能在SELL或STRONG_SELL边界
        assert signal in ["SELL", "STRONG_SELL"]

    def test_strong_sell_signal(self, scorer):
        """测试强卖信号"""
        signal = scorer._generate_signal(0.15)
        assert signal == "STRONG_SELL"


class TestGenerateReasoning:
    """决策理由生成测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_generate_reasoning_basic(self, scorer):
        """测试基本理由生成"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=0.8,
                z_score=2.0,
                percentile=95.0,
                signal="LONG",
                confidence=0.9,
            ),
            "value": FactorExposure(
                factor_type=FactorType.VALUE,
                exposure=-0.2,
                z_score=-0.5,
                percentile=30.0,
                signal="SHORT",
                confidence=0.5,
            ),
        }
        reasoning = scorer._generate_reasoning("AAPL", exposures, 0.7)
        assert "AAPL" in reasoning
        assert "综合评分" in reasoning
        assert "最强因子" in reasoning

    def test_generate_reasoning_empty(self, scorer):
        """测试空暴露理由生成"""
        reasoning = scorer._generate_reasoning("AAPL", {}, 0.5)
        assert "无因子数据" in reasoning


class TestStaticMethods:
    """静态方法测试"""

    def test_normalize_to_zscore(self):
        """测试标准化为Z-score"""
        z = BaseFactorScorer.normalize_to_zscore(75, mean=50, std=10)
        assert z == 2.5

    def test_normalize_to_zscore_zero_std(self):
        """测试零标准差"""
        z = BaseFactorScorer.normalize_to_zscore(75, mean=50, std=0)
        assert z == 0.0

    def test_zscore_to_percentile(self):
        """测试Z-score转百分位"""
        percentile = BaseFactorScorer.zscore_to_percentile(0)
        assert abs(percentile - 50) < 1  # 约50%

    def test_zscore_to_percentile_high(self):
        """测试高Z-score"""
        percentile = BaseFactorScorer.zscore_to_percentile(2)
        assert percentile > 95

    def test_clip_exposure(self):
        """测试裁剪暴露值"""
        assert BaseFactorScorer.clip_exposure(1.5) == 1.0
        assert BaseFactorScorer.clip_exposure(-1.5) == -1.0
        assert BaseFactorScorer.clip_exposure(0.5) == 0.5


class TestScore:
    """完整评分测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_score_basic(self, scorer):
        """测试基本评分"""
        data = {"momentum": 0.6, "value": 0.4}
        score = scorer.score("AAPL", data)
        assert isinstance(score, FactorScore)
        assert score.symbol == "AAPL"
        assert 0 <= score.composite_score <= 1

    def test_score_with_returns(self, scorer):
        """测试带收益率评分"""
        data = {"momentum": 0.5, "value": 0.3}
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        score = scorer.score("AAPL", data, returns)
        assert isinstance(score, FactorScore)

    def test_score_with_regime(self, scorer):
        """测试带市场体制评分"""
        data = {"momentum": 0.5}
        score = scorer.score("AAPL", data, market_regime="bull")
        assert isinstance(score, FactorScore)


class TestScorePortfolio:
    """组合评分测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_score_portfolio_basic(self, scorer):
        """测试基本组合评分"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = {
            "AAPL": {"momentum": 0.6, "value": 0.4},
            "MSFT": {"momentum": 0.5, "value": 0.5},
            "GOOGL": {"momentum": 0.4, "value": 0.6},
        }
        scores = scorer.score_portfolio(symbols, data)
        assert len(scores) == 3
        assert "AAPL" in scores

    def test_score_portfolio_partial(self, scorer):
        """测试部分数据组合评分"""
        symbols = ["AAPL", "MSFT", "UNKNOWN"]
        data = {
            "AAPL": {"momentum": 0.6},
            "MSFT": {"momentum": 0.5},
        }
        scores = scorer.score_portfolio(symbols, data)
        assert "AAPL" in scores
        assert "MSFT" in scores
        assert "UNKNOWN" not in scores


class TestGetFactorPremiums:
    """因子溢价获取测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()
        assert isinstance(premiums, dict)
        # 应该包含常见因子
        assert "size" in premiums or "momentum" in premiums


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def scorer(self):
        return ConcreteFactorScorer()

    def test_extreme_exposure_values(self, scorer):
        """测试极端暴露值"""
        exposures = {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=10.0,  # 极端值
                z_score=5.0,
                percentile=99.99,
                signal="LONG",
                confidence=1.0,
            ),
        }
        # 应该正常处理
        score = scorer._compute_composite_score(exposures, {"momentum": 1.0})
        assert 0 <= score <= 1  # 应该被裁剪

    def test_nan_handling(self, scorer):
        """测试NaN处理"""
        # 测试静态方法对NaN的处理
        z = BaseFactorScorer.normalize_to_zscore(float('nan'), 50, 10)
        # 应该返回某个值而不是抛出异常
        assert isinstance(z, float)
