#!/usr/bin/env python
"""
Factor Scorers Deep Testing - 因子评分器深度测试
Coverage: base_factor.py, stock_factors.py, bond_factors.py,
         reits_factors.py, crypto_factors.py, commodity_factors.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# Test 1: FactorType Enum
# ============================================================

class TestFactorType:
    """测试 FactorType 枚举"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.base_factor import FactorType
        assert FactorType is not None

    def test_stock_factors(self):
        """测试股票因子"""
        from finsage.factors.base_factor import FactorType

        assert FactorType.MARKET.value == "market"
        assert FactorType.SIZE.value == "size"
        assert FactorType.VALUE.value == "value"
        assert FactorType.PROFITABILITY.value == "profitability"
        assert FactorType.INVESTMENT.value == "investment"
        assert FactorType.MOMENTUM.value == "momentum"

    def test_bond_factors(self):
        """测试债券因子"""
        from finsage.factors.base_factor import FactorType

        assert FactorType.CARRY.value == "carry"
        assert FactorType.LOW_RISK.value == "low_risk"
        assert FactorType.CREDIT.value == "credit"
        assert FactorType.DURATION.value == "duration"


# ============================================================
# Test 2: FactorExposure
# ============================================================

class TestFactorExposure:
    """测试 FactorExposure 数据类"""

    def test_creation(self):
        """测试创建"""
        from finsage.factors.base_factor import FactorExposure, FactorType

        exposure = FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=0.5,
            z_score=1.2,
            percentile=88.5,
            signal="LONG",
            confidence=0.85
        )

        assert exposure.factor_type == FactorType.VALUE
        assert exposure.exposure == 0.5
        assert exposure.z_score == 1.2
        assert exposure.signal == "LONG"

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.factors.base_factor import FactorExposure, FactorType

        exposure = FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=0.75,
            z_score=2.0,
            percentile=97.7,
            signal="LONG",
            confidence=0.9
        )

        d = exposure.to_dict()

        assert d["factor"] == "momentum"
        assert d["exposure"] == 0.75
        assert d["z_score"] == 2.0
        assert d["signal"] == "LONG"


# ============================================================
# Test 3: FactorScore
# ============================================================

class TestFactorScore:
    """测试 FactorScore 数据类"""

    def test_creation(self):
        """测试创建"""
        from finsage.factors.base_factor import FactorScore, FactorExposure, FactorType

        exposures = {
            "value": FactorExposure(
                FactorType.VALUE, 0.5, 1.0, 84, "LONG", 0.8
            )
        }

        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp="2024-01-01T00:00:00",
            factor_exposures=exposures,
            composite_score=0.72,
            expected_alpha=0.05,
            risk_contribution=0.3,
            signal="BUY",
            reasoning="Good value exposure"
        )

        assert score.symbol == "AAPL"
        assert score.composite_score == 0.72
        assert score.signal == "BUY"

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.factors.base_factor import FactorScore, FactorExposure, FactorType

        exposures = {
            "value": FactorExposure(
                FactorType.VALUE, 0.5, 1.0, 84, "LONG", 0.8
            )
        }

        score = FactorScore(
            symbol="MSFT",
            asset_class="stocks",
            timestamp="2024-01-01",
            factor_exposures=exposures,
            composite_score=0.65,
            expected_alpha=0.03,
            risk_contribution=0.25,
            signal="HOLD",
            reasoning="Moderate factors"
        )

        d = score.to_dict()

        assert "symbol" in d
        assert "factor_exposures" in d
        assert "composite_score" in d
        assert d["signal"] == "HOLD"

    def test_get_exposure(self):
        """测试获取特定因子暴露"""
        from finsage.factors.base_factor import FactorScore, FactorExposure, FactorType

        value_exp = FactorExposure(FactorType.VALUE, 0.5, 1.0, 84, "LONG", 0.8)
        mom_exp = FactorExposure(FactorType.MOMENTUM, 0.3, 0.5, 70, "NEUTRAL", 0.7)

        exposures = {
            "value": value_exp,
            "momentum": mom_exp,
        }

        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp="2024-01-01",
            factor_exposures=exposures,
            composite_score=0.6,
            expected_alpha=0.02,
            risk_contribution=0.2,
            signal="HOLD",
            reasoning=""
        )

        assert score.get_exposure(FactorType.VALUE) == value_exp
        assert score.get_exposure(FactorType.MOMENTUM) == mom_exp
        assert score.get_exposure(FactorType.SIZE) is None


# ============================================================
# Test 4: StockFactorScorer
# ============================================================

class TestStockFactorScorer:
    """测试股票因子评分器"""

    @pytest.fixture
    def stock_scorer(self):
        """创建股票评分器"""
        from finsage.factors.stock_factors import StockFactorScorer
        return StockFactorScorer()

    @pytest.fixture
    def stock_data(self):
        """创建测试股票数据"""
        return {
            "market_cap": 3e12,  # 3 trillion
            "beta": 1.2,
            "book_to_market": 0.15,  # Growth stock
            "roe": 0.35,
            "operating_margin": 0.30,
            "asset_growth": 0.10,
            "price_change_12m": 0.25,
        }

    def test_import(self):
        """测试导入"""
        from finsage.factors.stock_factors import StockFactorScorer
        assert StockFactorScorer is not None

    def test_initialization_default(self, stock_scorer):
        """测试默认初始化"""
        assert stock_scorer.asset_class == "stocks"
        assert stock_scorer.lookback_period == 252

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        from finsage.factors.stock_factors import StockFactorScorer

        config = {
            "lookback_period": 126,
            "factor_weights": {"value": 0.5, "momentum": 0.5},
        }
        scorer = StockFactorScorer(config)

        assert scorer.lookback_period == 126
        assert scorer.factor_weights["value"] == 0.5

    def test_supported_factors(self, stock_scorer):
        """测试支持的因子"""
        from finsage.factors.base_factor import FactorType

        factors = stock_scorer.supported_factors

        assert FactorType.MARKET in factors
        assert FactorType.SIZE in factors
        assert FactorType.VALUE in factors
        assert FactorType.PROFITABILITY in factors
        assert FactorType.INVESTMENT in factors
        assert FactorType.MOMENTUM in factors

    def test_score_basic(self, stock_scorer, stock_data):
        """测试基本评分"""
        score = stock_scorer.score("AAPL", stock_data)

        assert score.symbol == "AAPL"
        assert score.asset_class == "stocks"
        assert 0 <= score.composite_score <= 1
        assert score.signal in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]

    def test_score_with_returns(self, stock_scorer, stock_data):
        """测试带收益率的评分"""
        returns = pd.Series(np.random.randn(252) * 0.02)

        score = stock_scorer.score("AAPL", stock_data, returns)

        assert score.composite_score is not None

    def test_score_with_market_regime(self, stock_scorer, stock_data):
        """测试不同市场状态下的评分"""
        score_bull = stock_scorer.score("AAPL", stock_data, market_regime="bull")
        score_bear = stock_scorer.score("AAPL", stock_data, market_regime="bear")
        score_neutral = stock_scorer.score("AAPL", stock_data, market_regime="neutral")

        # 不同市场状态应该产生不同评分
        assert score_bull.composite_score is not None
        assert score_bear.composite_score is not None
        assert score_neutral.composite_score is not None

    def test_score_portfolio(self, stock_scorer):
        """测试组合评分"""
        data = {
            "AAPL": {"market_cap": 3e12, "beta": 1.2, "roe": 0.35},
            "MSFT": {"market_cap": 2.8e12, "beta": 1.1, "roe": 0.40},
            "GOOGL": {"market_cap": 2e12, "beta": 1.0, "roe": 0.25},
        }

        scores = stock_scorer.score_portfolio(
            ["AAPL", "MSFT", "GOOGL"],
            data
        )

        assert "AAPL" in scores
        assert "MSFT" in scores
        assert "GOOGL" in scores

    def test_factor_exposures_all_present(self, stock_scorer, stock_data):
        """测试所有因子暴露都存在"""
        score = stock_scorer.score("AAPL", stock_data)

        expected_factors = ["market", "size", "value", "profitability", "investment", "momentum"]
        for factor in expected_factors:
            assert factor in score.factor_exposures

    def test_value_factor_growth_stock(self, stock_scorer):
        """测试成长股的价值因子 (低B/M)"""
        growth_data = {
            "market_cap": 1e12,
            "book_to_market": 0.05,  # 低B/M = 成长股
            "roe": 0.30,
        }

        score = stock_scorer.score("GROWTH", growth_data)
        value_exposure = score.factor_exposures["value"]

        # 成长股应该有负的价值因子暴露
        assert value_exposure.exposure < 0

    def test_value_factor_value_stock(self, stock_scorer):
        """测试价值股的价值因子 (高B/M)"""
        value_data = {
            "market_cap": 100e9,
            "book_to_market": 0.8,  # 高B/M = 价值股
            "roe": 0.15,
        }

        score = stock_scorer.score("VALUE", value_data)
        value_exposure = score.factor_exposures["value"]

        # 价值股应该有正的价值因子暴露
        assert value_exposure.exposure > 0

    def test_size_factor_small_cap(self, stock_scorer):
        """测试小盘股的规模因子"""
        small_data = {
            "market_cap": 500e6,  # 5亿 = 小盘
            "roe": 0.15,
        }

        score = stock_scorer.score("SMALL", small_data)
        size_exposure = score.factor_exposures["size"]

        # 小盘股应该有正的规模因子暴露 (SMB做多小盘)
        assert size_exposure.exposure > 0

    def test_size_factor_mega_cap(self, stock_scorer):
        """测试大盘股的规模因子"""
        large_data = {
            "market_cap": 500e9,  # 5000亿 = 大盘
            "roe": 0.20,
        }

        score = stock_scorer.score("LARGE", large_data)
        size_exposure = score.factor_exposures["size"]

        # 大盘股应该有负的规模因子暴露
        assert size_exposure.exposure < 0

    def test_momentum_factor_positive(self, stock_scorer):
        """测试正动量"""
        momentum_data = {
            "market_cap": 100e9,
            "price_change_12m": 0.50,  # 涨50%
        }

        score = stock_scorer.score("MOM", momentum_data)
        mom_exposure = score.factor_exposures["momentum"]

        # 正动量应该有正暴露
        assert mom_exposure.exposure > 0

    def test_momentum_factor_negative(self, stock_scorer):
        """测试负动量"""
        momentum_data = {
            "market_cap": 100e9,
            "price_change_12m": -0.30,  # 跌30%
        }

        score = stock_scorer.score("MOM", momentum_data)
        mom_exposure = score.factor_exposures["momentum"]

        # 负动量应该有负暴露
        assert mom_exposure.exposure < 0


# ============================================================
# Test 5: BondFactorScorer
# ============================================================

class TestBondFactorScorer:
    """测试债券因子评分器"""

    @pytest.fixture
    def bond_scorer(self):
        """创建债券评分器"""
        from finsage.factors.bond_factors import BondFactorScorer
        return BondFactorScorer()

    @pytest.fixture
    def bond_data(self):
        """创建测试债券数据"""
        return {
            "yield": 0.05,          # 5% 收益率
            "duration": 7.0,        # 7年久期
            "credit_rating": "AA",  # 信用评级
            "spread": 0.015,        # 1.5% 利差
            "volatility": 0.08,     # 8% 波动率
        }

    def test_import(self):
        """测试导入"""
        from finsage.factors.bond_factors import BondFactorScorer
        assert BondFactorScorer is not None

    def test_initialization(self, bond_scorer):
        """测试初始化"""
        assert bond_scorer.asset_class == "bonds"

    def test_score_basic(self, bond_scorer, bond_data):
        """测试基本评分"""
        score = bond_scorer.score("TLT", bond_data)

        assert score.symbol == "TLT"
        assert score.asset_class == "bonds"
        assert 0 <= score.composite_score <= 1

    def test_supported_factors(self, bond_scorer):
        """测试支持的因子"""
        from finsage.factors.base_factor import FactorType

        factors = bond_scorer.supported_factors

        assert FactorType.CARRY in factors
        assert FactorType.LOW_RISK in factors
        assert FactorType.CREDIT in factors
        assert FactorType.DURATION in factors


# ============================================================
# Test 6: REITsFactorScorer
# ============================================================

class TestREITsFactorScorer:
    """测试REITs因子评分器"""

    @pytest.fixture
    def reits_scorer(self):
        """创建REITs评分器"""
        from finsage.factors.reits_factors import REITsFactorScorer
        return REITsFactorScorer()

    @pytest.fixture
    def reits_data(self):
        """创建测试REITs数据"""
        return {
            "nav_discount": -0.10,  # NAV折价10%
            "dividend_yield": 0.05, # 5%股息率
            "ffo_growth": 0.08,     # FFO增长8%
            "occupancy_rate": 0.95, # 入住率95%
            "debt_ratio": 0.40,     # 负债率40%
        }

    def test_import(self):
        """测试导入"""
        from finsage.factors.reits_factors import REITsFactorScorer
        assert REITsFactorScorer is not None

    def test_initialization(self, reits_scorer):
        """测试初始化"""
        assert reits_scorer.asset_class == "reits"

    def test_score_basic(self, reits_scorer, reits_data):
        """测试基本评分"""
        score = reits_scorer.score("VNQ", reits_data)

        assert score.symbol == "VNQ"
        assert score.asset_class == "reits"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 7: CryptoFactorScorer
# ============================================================

class TestCryptoFactorScorer:
    """测试加密货币因子评分器"""

    @pytest.fixture
    def crypto_scorer(self):
        """创建加密货币评分器"""
        from finsage.factors.crypto_factors import CryptoFactorScorer
        return CryptoFactorScorer()

    @pytest.fixture
    def crypto_data(self):
        """创建测试加密货币数据"""
        return {
            "market_cap": 800e9,           # 8000亿市值
            "active_addresses": 1e6,       # 活跃地址数
            "transaction_volume": 10e9,    # 交易量
            "hash_rate": 400e18,           # 哈希率
            "price_change_30d": 0.15,      # 30天涨幅
            "volatility": 0.60,            # 60%波动率
        }

    def test_import(self):
        """测试导入"""
        from finsage.factors.crypto_factors import CryptoFactorScorer
        assert CryptoFactorScorer is not None

    def test_initialization(self, crypto_scorer):
        """测试初始化"""
        assert crypto_scorer.asset_class == "crypto"

    def test_score_basic(self, crypto_scorer, crypto_data):
        """测试基本评分"""
        score = crypto_scorer.score("BTC", crypto_data)

        assert score.symbol == "BTC"
        assert score.asset_class == "crypto"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 8: CommodityFactorScorer
# ============================================================

class TestCommodityFactorScorer:
    """测试商品因子评分器"""

    @pytest.fixture
    def commodity_scorer(self):
        """创建商品评分器"""
        from finsage.factors.commodity_factors import CommodityFactorScorer
        return CommodityFactorScorer()

    @pytest.fixture
    def commodity_data(self):
        """创建测试商品数据"""
        return {
            "spot_price": 1900,
            "front_month_price": 1905,
            "next_month_price": 1920,
            "inventory_level": 0.3,     # 库存水平 (正常化)
            "storage_cost": 0.02,       # 存储成本
            "convenience_yield": 0.01,  # 便利收益
        }

    def test_import(self):
        """测试导入"""
        from finsage.factors.commodity_factors import CommodityFactorScorer
        assert CommodityFactorScorer is not None

    def test_initialization(self, commodity_scorer):
        """测试初始化"""
        assert commodity_scorer.asset_class == "commodities"

    def test_score_basic(self, commodity_scorer, commodity_data):
        """测试基本评分"""
        score = commodity_scorer.score("GLD", commodity_data)

        assert score.symbol == "GLD"
        assert score.asset_class == "commodities"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 9: BaseFactorScorer Methods
# ============================================================

class TestBaseFactorScorerMethods:
    """测试基类方法"""

    @pytest.fixture
    def scorer(self):
        """使用StockFactorScorer测试基类方法"""
        from finsage.factors.stock_factors import StockFactorScorer
        return StockFactorScorer()

    def test_normalize_to_zscore(self, scorer):
        """测试Z-score标准化"""
        z = scorer.normalize_to_zscore(value=15, mean=10, std=2.5)
        assert z == 2.0  # (15-10)/2.5 = 2

    def test_normalize_to_zscore_zero_std(self, scorer):
        """测试零标准差"""
        z = scorer.normalize_to_zscore(value=15, mean=10, std=0)
        assert z == 0.0

    def test_zscore_to_percentile(self, scorer):
        """测试Z-score转百分位"""
        pct = scorer.zscore_to_percentile(0)
        assert abs(pct - 50) < 0.1  # Z=0 对应50%

        pct = scorer.zscore_to_percentile(2)
        assert pct > 95  # Z=2 大约对应97.7%

    def test_clip_exposure(self, scorer):
        """测试暴露裁剪"""
        assert scorer.clip_exposure(1.5) == 1.0
        assert scorer.clip_exposure(-1.5) == -1.0
        assert scorer.clip_exposure(0.5) == 0.5

    def test_adjust_weights_for_regime_bull(self, scorer):
        """测试牛市权重调整"""
        weights = scorer._adjust_weights_for_regime("bull")

        # 牛市应该增加动量权重
        if "momentum" in scorer.factor_weights:
            assert weights["momentum"] >= scorer.factor_weights["momentum"]

    def test_adjust_weights_for_regime_bear(self, scorer):
        """测试熊市权重调整"""
        weights = scorer._adjust_weights_for_regime("bear")

        # 熊市应该增加价值权重
        if "value" in scorer.factor_weights:
            assert weights["value"] >= scorer.factor_weights["value"]

    def test_generate_signal_strong_buy(self, scorer):
        """测试强买入信号"""
        signal = scorer._generate_signal(0.85)
        assert signal == "STRONG_BUY"

    def test_generate_signal_buy(self, scorer):
        """测试买入信号"""
        signal = scorer._generate_signal(0.65)
        assert signal == "BUY"

    def test_generate_signal_hold(self, scorer):
        """测试持有信号"""
        signal = scorer._generate_signal(0.50)
        assert signal == "HOLD"

    def test_generate_signal_sell(self, scorer):
        """测试卖出信号"""
        signal = scorer._generate_signal(0.42)  # 在sell阈值(0.4)和hold_lower阈值(0.45)之间
        assert signal == "SELL"

    def test_generate_signal_strong_sell(self, scorer):
        """测试强卖出信号"""
        signal = scorer._generate_signal(0.15)
        assert signal == "STRONG_SELL"

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()

        assert "value" in premiums
        assert "momentum" in premiums
        assert premiums["momentum"] > 0


# ============================================================
# Test 10: Edge Cases
# ============================================================

class TestFactorScorerEdgeCases:
    """测试边界情况"""

    def test_score_missing_data(self):
        """测试缺失数据"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        data = {}  # 空数据

        # 应该能处理空数据
        score = scorer.score("TEST", data)
        assert score.composite_score is not None

    def test_score_invalid_values(self):
        """测试无效值"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        data = {
            "market_cap": -1000,  # 无效负值
            "beta": float('nan'),
            "roe": float('inf'),
        }

        # 不应该崩溃
        score = scorer.score("TEST", data)
        assert score is not None

    def test_score_portfolio_with_missing_symbol(self):
        """测试组合评分中缺失符号"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        data = {
            "AAPL": {"market_cap": 3e12, "roe": 0.35},
            # MSFT缺失
        }

        scores = scorer.score_portfolio(
            ["AAPL", "MSFT"],  # 请求MSFT但数据中没有
            data
        )

        assert "AAPL" in scores
        assert "MSFT" not in scores


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Factor Scorers Deep Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
