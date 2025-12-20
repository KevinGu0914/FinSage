"""
Deep tests for StockFactorScorer
股票因子评分器深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.factors.stock_factors import StockFactorScorer, MLEnhancedStockScorer
from finsage.factors.base_factor import FactorType, FactorExposure, FactorScore


class TestStockFactorScorerInit:
    """StockFactorScorer初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        scorer = StockFactorScorer()
        assert scorer.asset_class == "stocks"
        assert FactorType.VALUE in scorer.supported_factors
        assert FactorType.MOMENTUM in scorer.supported_factors

    def test_default_weights(self):
        """测试默认权重"""
        scorer = StockFactorScorer()
        weights = scorer._default_weights()
        assert "value" in weights
        assert "profitability" in weights
        assert weights["profitability"] == 0.25  # 盈利因子权重最高
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_supported_factors(self):
        """测试支持的因子"""
        scorer = StockFactorScorer()
        factors = scorer.supported_factors
        assert FactorType.MARKET in factors
        assert FactorType.SIZE in factors
        assert FactorType.VALUE in factors
        assert FactorType.PROFITABILITY in factors
        assert FactorType.INVESTMENT in factors
        assert FactorType.MOMENTUM in factors


class TestComputeMarketExposure:
    """市场因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_high_beta(self, scorer):
        """测试高Beta"""
        data = {"beta": 1.5}
        exposure = scorer._compute_market_exposure(data, None)
        assert exposure.exposure > 0
        assert exposure.signal == "HIGH_BETA"

    def test_low_beta(self, scorer):
        """测试低Beta"""
        data = {"beta": 0.5}
        exposure = scorer._compute_market_exposure(data, None)
        assert exposure.exposure < 0
        assert exposure.signal == "LOW_BETA"

    def test_neutral_beta(self, scorer):
        """测试中性Beta"""
        data = {"beta": 1.0}
        exposure = scorer._compute_market_exposure(data, None)
        assert abs(exposure.exposure) < 0.1
        assert exposure.signal == "NEUTRAL"


class TestComputeSizeExposure:
    """规模因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_small_cap(self, scorer):
        """测试小盘股"""
        data = {"market_cap": 1e9}  # 10亿
        exposure = scorer._compute_size_exposure(data)
        assert exposure.exposure > 0  # 小盘=正SMB暴露
        assert exposure.signal == "SMALL_CAP"

    def test_large_cap(self, scorer):
        """测试大盘股"""
        data = {"market_cap": 500e9}  # 5000亿
        exposure = scorer._compute_size_exposure(data)
        assert exposure.exposure < 0  # 大盘=负SMB暴露
        assert exposure.signal == "LARGE_CAP"

    def test_mid_cap(self, scorer):
        """测试中盘股"""
        data = {"market_cap": 10e9}  # 100亿
        exposure = scorer._compute_size_exposure(data)
        assert exposure.signal == "NEUTRAL"


class TestComputeValueExposure:
    """价值因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_value_stock(self, scorer):
        """测试价值股"""
        data = {"book_to_market": 0.9, "pe_ratio": 10}
        exposure = scorer._compute_value_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "VALUE"

    def test_growth_stock(self, scorer):
        """测试成长股"""
        data = {"book_to_market": 0.2, "pe_ratio": 40}
        exposure = scorer._compute_value_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "GROWTH"

    def test_blend_stock(self, scorer):
        """测试混合股"""
        data = {"book_to_market": 0.5, "pe_ratio": 20}
        exposure = scorer._compute_value_exposure(data)
        assert exposure.signal == "BLEND"

    def test_negative_pe(self, scorer):
        """测试负PE(亏损公司)"""
        data = {"book_to_market": 0.5, "pe_ratio": -10}
        exposure = scorer._compute_value_exposure(data)
        assert isinstance(exposure, FactorExposure)


class TestComputeProfitabilityExposure:
    """盈利因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_high_quality(self, scorer):
        """测试高质量公司"""
        data = {"roe": 0.25, "operating_margin": 0.25, "gross_margin": 0.50}
        exposure = scorer._compute_profitability_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "HIGH_QUALITY"

    def test_low_quality(self, scorer):
        """测试低质量公司"""
        data = {"roe": 0.02, "operating_margin": 0.03, "gross_margin": 0.25}
        exposure = scorer._compute_profitability_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "LOW_QUALITY"

    def test_neutral_quality(self, scorer):
        """测试中性质量"""
        data = {"roe": 0.15, "operating_margin": 0.15, "gross_margin": 0.40}
        exposure = scorer._compute_profitability_exposure(data)
        assert exposure.signal == "NEUTRAL"


class TestComputeInvestmentExposure:
    """投资因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_conservative_investment(self, scorer):
        """测试保守投资"""
        data = {"asset_growth": 0.02, "capex_to_revenue": 0.03}
        exposure = scorer._compute_investment_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "CONSERVATIVE"

    def test_aggressive_investment(self, scorer):
        """测试激进投资"""
        data = {"asset_growth": 0.30, "capex_to_revenue": 0.10}
        exposure = scorer._compute_investment_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "AGGRESSIVE"


class TestComputeMomentumExposure:
    """动量因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_strong_momentum(self, scorer):
        """测试强动量"""
        data = {"price_change_12m": 0.30, "price_change_1m": 0.02}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure > 0
        assert exposure.signal == "STRONG_MOMENTUM"

    def test_reversal_signal(self, scorer):
        """测试反转信号"""
        data = {"price_change_12m": -0.15, "price_change_1m": -0.02}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure < 0
        assert exposure.signal == "REVERSAL"

    def test_with_returns_series(self, scorer):
        """测试带收益率序列"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 260))
        exposure = scorer._compute_momentum_exposure({}, returns)
        assert isinstance(exposure, FactorExposure)


class TestComputeFactorExposures:
    """因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_compute_all_exposures(self, scorer):
        """测试计算所有暴露"""
        data = {
            "beta": 1.1,
            "market_cap": 100e9,
            "book_to_market": 0.5,
            "pe_ratio": 20,
            "roe": 0.15,
            "operating_margin": 0.15,
            "gross_margin": 0.40,
            "asset_growth": 0.10,
            "price_change_12m": 0.10,
        }
        exposures = scorer._compute_factor_exposures("AAPL", data)
        assert "market" in exposures
        assert "size" in exposures
        assert "value" in exposures
        assert "profitability" in exposures
        assert "investment" in exposures
        assert "momentum" in exposures


class TestGetFactorPremiums:
    """因子溢价获取测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()
        assert premiums["market"] == 0.07
        assert premiums["size"] == 0.02
        assert premiums["value"] == 0.03
        assert premiums["momentum"] == 0.06


class TestGetFactorSummary:
    """因子摘要测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_get_factor_summary(self, scorer):
        """测试获取因子摘要"""
        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={
                "market": FactorExposure(
                    factor_type=FactorType.MARKET,
                    exposure=0.1,
                    z_score=0.2,
                    percentile=55,
                    signal="NEUTRAL",
                    confidence=0.8,
                ),
                "value": FactorExposure(
                    factor_type=FactorType.VALUE,
                    exposure=0.3,
                    z_score=0.6,
                    percentile=70,
                    signal="VALUE",
                    confidence=0.75,
                ),
            },
            composite_score=0.65,
            expected_alpha=0.04,
            risk_contribution=0.25,
            signal="BUY",
            reasoning="看涨",
        )
        summary = scorer.get_factor_summary(score)
        assert "AAPL" in summary
        assert "五因子" in summary
        assert "综合评分" in summary


class TestScore:
    """完整评分测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_score_basic(self, scorer):
        """测试基本评分"""
        data = {
            "beta": 1.1,
            "market_cap": 2500e9,
            "book_to_market": 0.4,
            "pe_ratio": 30,
            "roe": 0.25,
            "operating_margin": 0.30,
            "asset_growth": 0.08,
            "price_change_12m": 0.15,
        }
        score = scorer.score("AAPL", data)
        assert isinstance(score, FactorScore)
        assert score.asset_class == "stocks"
        assert 0 <= score.composite_score <= 1

    def test_score_with_returns(self, scorer):
        """测试带收益率评分"""
        data = {"beta": 1.0, "market_cap": 100e9}
        returns = pd.Series(np.random.normal(0.001, 0.02, 260))
        score = scorer.score("AAPL", data, returns)
        assert isinstance(score, FactorScore)

    def test_score_with_regime(self, scorer):
        """测试带市场体制评分"""
        data = {"beta": 1.2, "market_cap": 50e9}
        score = scorer.score("AAPL", data, market_regime="bull")
        assert isinstance(score, FactorScore)


class TestMLEnhancedStockScorer:
    """ML增强股票评分器测试"""

    def test_init_with_ml(self):
        """测试带ML初始化"""
        scorer = MLEnhancedStockScorer(use_ml=True)
        assert scorer.use_ml is True

    def test_init_without_ml(self):
        """测试不带ML初始化"""
        scorer = MLEnhancedStockScorer(use_ml=False)
        assert scorer.use_ml is False

    def test_score_with_ml(self):
        """测试ML增强评分"""
        scorer = MLEnhancedStockScorer(use_ml=True)
        data = {
            "beta": 1.1,
            "market_cap": 100e9,
            "book_to_market": 0.5,
            "roe": 0.15,
            "volatility": 0.20,
            "turnover": 0.10,
        }
        score = scorer.score("AAPL", data)
        assert isinstance(score, FactorScore)

    def test_compute_ml_alpha(self):
        """测试ML Alpha计算"""
        scorer = MLEnhancedStockScorer(use_ml=True)
        exposures = {
            "market": FactorExposure(
                factor_type=FactorType.MARKET,
                exposure=0.1,
                z_score=0.2,
                percentile=55,
                signal="NEUTRAL",
                confidence=0.8,
            ),
        }
        data = {"volatility": 0.2, "turnover": 0.1}
        alpha = scorer._compute_ml_alpha(data, exposures)
        assert isinstance(alpha, float)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_zero_market_cap(self, scorer):
        """测试零市值"""
        data = {"market_cap": 0}
        exposure = scorer._compute_size_exposure(data)
        assert isinstance(exposure, FactorExposure)

    def test_negative_roe(self, scorer):
        """测试负ROE"""
        data = {"roe": -0.10, "operating_margin": -0.05, "gross_margin": 0.20}
        exposure = scorer._compute_profitability_exposure(data)
        assert exposure.exposure < 0

    def test_extreme_momentum(self, scorer):
        """测试极端动量"""
        data = {"price_change_12m": 2.0, "price_change_1m": 0.3}  # 200%涨幅
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure <= 1.0  # 应该被裁剪

    def test_missing_data(self, scorer):
        """测试缺失数据"""
        data = {}  # 空数据
        exposures = scorer._compute_factor_exposures("AAPL", data)
        assert len(exposures) == 6  # 应该返回所有因子的默认值


class TestScorePortfolio:
    """组合评分测试"""

    @pytest.fixture
    def scorer(self):
        return StockFactorScorer()

    def test_score_portfolio(self, scorer):
        """测试组合评分"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = {
            "AAPL": {"beta": 1.1, "market_cap": 2500e9, "roe": 0.25},
            "MSFT": {"beta": 0.9, "market_cap": 2000e9, "roe": 0.30},
            "GOOGL": {"beta": 1.0, "market_cap": 1500e9, "roe": 0.20},
        }
        scores = scorer.score_portfolio(symbols, data)
        assert len(scores) == 3
        assert all(isinstance(s, FactorScore) for s in scores.values())


class TestFactorWeights:
    """因子权重测试"""

    def test_custom_weights(self):
        """测试自定义权重"""
        config = {
            "factor_weights": {
                "market": 0.05,
                "size": 0.10,
                "value": 0.30,
                "profitability": 0.30,
                "investment": 0.10,
                "momentum": 0.15,
            }
        }
        scorer = StockFactorScorer(config=config)
        assert scorer.factor_weights["value"] == 0.30

    def test_regime_adjustment(self):
        """测试市场体制权重调整"""
        scorer = StockFactorScorer()
        bull_weights = scorer._adjust_weights_for_regime("bull")
        bear_weights = scorer._adjust_weights_for_regime("bear")
        # 牛市应增加动量，熊市应减少动量
        assert bull_weights.get("momentum", 0) >= bear_weights.get("momentum", 0)
