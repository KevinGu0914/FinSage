#!/usr/bin/env python
"""
Factors Module Tests - 因子模块测试
覆盖: base_factor, stock_factors, bond_factors, reits_factors,
      crypto_factors, commodity_factors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_stock_data():
    """生成示例股票数据"""
    return {
        'market_cap': 2500000000000,  # 2.5 trillion
        'book_to_market': 0.35,
        'pe_ratio': 25.5,
        'roe': 0.35,
        'roa': 0.15,
        'gross_margin': 0.42,
        'capex_growth': 0.05,
        'asset_growth': 0.08,
        'revenue_growth': 0.12,
        'current_price': 180.50,
        'price_52w_high': 200.0,
        'price_52w_low': 140.0,
        'volume_avg_30d': 75000000,
        'beta': 1.15,
    }


@pytest.fixture
def sample_returns():
    """生成示例收益率序列"""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)),
        index=dates,
        name='AAPL'
    )
    return returns


@pytest.fixture
def sample_bond_data():
    """生成示例债券数据"""
    return {
        'yield': 0.045,
        'duration': 7.5,
        'credit_rating': 'AA',
        'spread': 0.015,
        'maturity_years': 10,
        'coupon_rate': 0.04,
        'price': 98.5,
        'ytm': 0.048,
    }


@pytest.fixture
def sample_reits_data():
    """生成示例REITs数据"""
    return {
        'nav': 45.0,
        'price': 42.0,
        'dividend_yield': 0.045,
        'ffo_per_share': 3.2,
        'occupancy_rate': 0.94,
        'debt_ratio': 0.45,
        'sector': 'residential',
        'market_cap': 15000000000,
    }


@pytest.fixture
def sample_crypto_data():
    """生成示例加密货币数据"""
    return {
        'market_cap': 500000000000,
        'volume_24h': 20000000000,
        'active_addresses': 1000000,
        'hash_rate': 350000000,
        'transaction_volume': 15000000000,
        'nvt_ratio': 25,
        'mvrv': 1.8,
        'exchange_balance_pct': 0.12,
        'price': 42000,
        'volatility_30d': 0.55,
    }


@pytest.fixture
def sample_commodity_data():
    """生成示例商品数据"""
    return {
        'spot_price': 85.0,
        'futures_1m': 84.5,
        'futures_3m': 84.0,
        'futures_12m': 83.0,
        'inventory_level': 'normal',
        'production_growth': 0.02,
        'demand_growth': 0.03,
        'volatility_30d': 0.25,
    }


# ============================================================
# Test 1: Factor Type Enum
# ============================================================

class TestFactorType:
    """测试因子类型枚举"""

    def test_factor_types_exist(self):
        """测试所有因子类型存在"""
        from finsage.factors.base_factor import FactorType

        # 股票因子
        assert FactorType.MARKET.value == "market"
        assert FactorType.SIZE.value == "size"
        assert FactorType.VALUE.value == "value"
        assert FactorType.PROFITABILITY.value == "profitability"
        assert FactorType.INVESTMENT.value == "investment"
        assert FactorType.MOMENTUM.value == "momentum"

        # 债券因子
        assert FactorType.CARRY.value == "carry"
        assert FactorType.LOW_RISK.value == "low_risk"
        assert FactorType.CREDIT.value == "credit"
        assert FactorType.DURATION.value == "duration"

        # 商品因子
        assert FactorType.TERM_STRUCTURE.value == "term_structure"
        assert FactorType.BASIS.value == "basis"

        # REITs因子
        assert FactorType.NAV.value == "nav"
        assert FactorType.IDIOSYNCRATIC.value == "idiosyncratic"

        # 加密货币因子
        assert FactorType.NETWORK.value == "network"
        assert FactorType.ADOPTION.value == "adoption"
        assert FactorType.CRASH_RISK.value == "crash_risk"


# ============================================================
# Test 2: Factor Exposure Dataclass
# ============================================================

class TestFactorExposure:
    """测试因子暴露数据类"""

    def test_factor_exposure_creation(self):
        """测试因子暴露创建"""
        from finsage.factors.base_factor import FactorExposure, FactorType

        exposure = FactorExposure(
            factor_type=FactorType.VALUE,
            exposure=0.75,
            z_score=1.5,
            percentile=93.3,
            signal="LONG",
            confidence=0.85
        )

        assert exposure.factor_type == FactorType.VALUE
        assert exposure.exposure == 0.75
        assert exposure.signal == "LONG"

    def test_factor_exposure_to_dict(self):
        """测试转换为字典"""
        from finsage.factors.base_factor import FactorExposure, FactorType

        exposure = FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=0.5,
            z_score=1.0,
            percentile=84.0,
            signal="LONG",
            confidence=0.75
        )

        d = exposure.to_dict()

        assert d["factor"] == "momentum"
        assert d["exposure"] == 0.5
        assert d["signal"] == "LONG"
        assert "confidence" in d


# ============================================================
# Test 3: Factor Score Dataclass
# ============================================================

class TestFactorScore:
    """测试因子评分数据类"""

    def test_factor_score_creation(self):
        """测试因子评分创建"""
        from finsage.factors.base_factor import FactorScore, FactorExposure, FactorType

        exposures = {
            "value": FactorExposure(FactorType.VALUE, 0.5, 1.0, 84.0, "LONG", 0.8),
            "momentum": FactorExposure(FactorType.MOMENTUM, 0.3, 0.6, 72.0, "LONG", 0.7),
        }

        score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp="2024-01-15T10:00:00",
            factor_exposures=exposures,
            composite_score=0.72,
            expected_alpha=0.08,
            risk_contribution=0.35,
            signal="BUY",
            reasoning="Strong value and momentum signals"
        )

        assert score.symbol == "AAPL"
        assert score.composite_score == 0.72
        assert score.signal == "BUY"

    def test_factor_score_to_dict(self):
        """测试转换为字典"""
        from finsage.factors.base_factor import FactorScore, FactorExposure, FactorType

        exposures = {
            "value": FactorExposure(FactorType.VALUE, 0.5, 1.0, 84.0, "LONG", 0.8),
        }

        score = FactorScore(
            symbol="GOOGL",
            asset_class="stocks",
            timestamp="2024-01-15",
            factor_exposures=exposures,
            composite_score=0.65,
            expected_alpha=0.05,
            risk_contribution=0.30,
            signal="BUY",
            reasoning="Good value"
        )

        d = score.to_dict()

        assert d["symbol"] == "GOOGL"
        assert "factor_exposures" in d
        assert d["signal"] == "BUY"

    def test_get_exposure(self):
        """测试获取特定因子暴露"""
        from finsage.factors.base_factor import FactorScore, FactorExposure, FactorType

        value_exposure = FactorExposure(FactorType.VALUE, 0.5, 1.0, 84.0, "LONG", 0.8)
        exposures = {"value": value_exposure}

        score = FactorScore(
            symbol="MSFT",
            asset_class="stocks",
            timestamp="2024-01-15",
            factor_exposures=exposures,
            composite_score=0.60,
            expected_alpha=0.04,
            risk_contribution=0.25,
            signal="BUY",
            reasoning="Test"
        )

        retrieved = score.get_exposure(FactorType.VALUE)
        assert retrieved == value_exposure

        missing = score.get_exposure(FactorType.MOMENTUM)
        assert missing is None


# ============================================================
# Test 4: Base Factor Scorer
# ============================================================

class TestBaseFactorScorer:
    """测试因子评分器基类"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.base_factor import BaseFactorScorer
        assert BaseFactorScorer is not None

    def test_abstract_methods(self):
        """测试抽象方法"""
        from finsage.factors.base_factor import BaseFactorScorer

        # 应该无法直接实例化抽象类
        with pytest.raises(TypeError):
            BaseFactorScorer()

    def test_normalize_to_zscore(self):
        """测试Z-score标准化"""
        from finsage.factors.base_factor import BaseFactorScorer

        z = BaseFactorScorer.normalize_to_zscore(75, 50, 10)
        assert z == 2.5

        z = BaseFactorScorer.normalize_to_zscore(50, 50, 10)
        assert z == 0.0

        # 标准差为0
        z = BaseFactorScorer.normalize_to_zscore(50, 50, 0)
        assert z == 0.0

    def test_clip_exposure(self):
        """测试暴露值裁剪"""
        from finsage.factors.base_factor import BaseFactorScorer

        assert BaseFactorScorer.clip_exposure(0.5) == 0.5
        assert BaseFactorScorer.clip_exposure(1.5) == 1.0
        assert BaseFactorScorer.clip_exposure(-1.5) == -1.0

    def test_zscore_to_percentile(self):
        """测试Z-score转百分位"""
        from finsage.factors.base_factor import BaseFactorScorer

        pct = BaseFactorScorer.zscore_to_percentile(0.0)
        assert abs(pct - 50.0) < 0.1

        pct = BaseFactorScorer.zscore_to_percentile(2.0)
        assert pct > 97


# ============================================================
# Test 5: Stock Factor Scorer
# ============================================================

class TestStockFactorScorer:
    """测试股票因子评分器"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.stock_factors import StockFactorScorer
        assert StockFactorScorer is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        assert scorer.asset_class == "stocks"
        assert len(scorer.supported_factors) >= 5

    def test_default_weights(self):
        """测试默认因子权重"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        weights = scorer._default_weights()

        assert "market" in weights
        assert "size" in weights
        assert "value" in weights
        assert "profitability" in weights
        assert "momentum" in weights

        # 权重应该归一化
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_score(self, sample_stock_data, sample_returns):
        """测试评分计算"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        score = scorer.score("AAPL", sample_stock_data, sample_returns)

        assert score.symbol == "AAPL"
        assert score.asset_class == "stocks"
        assert 0 <= score.composite_score <= 1
        assert score.signal in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]

    def test_score_with_market_regime(self, sample_stock_data, sample_returns):
        """测试带市场状态的评分"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()

        bull_score = scorer.score("AAPL", sample_stock_data, sample_returns, market_regime="bull")
        bear_score = scorer.score("AAPL", sample_stock_data, sample_returns, market_regime="bear")

        assert bull_score is not None
        assert bear_score is not None

    def test_score_portfolio(self, sample_stock_data, sample_returns):
        """测试组合批量评分"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        data = {"AAPL": sample_stock_data, "GOOGL": sample_stock_data}
        returns_df = pd.DataFrame({
            "AAPL": sample_returns,
            "GOOGL": sample_returns * 1.1
        })

        scores = scorer.score_portfolio(["AAPL", "GOOGL"], data, returns_df)

        assert "AAPL" in scores
        assert "GOOGL" in scores


# ============================================================
# Test 6: Bond Factor Scorer
# ============================================================

class TestBondFactorScorer:
    """测试债券因子评分器"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.bond_factors import BondFactorScorer
        assert BondFactorScorer is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.factors.bond_factors import BondFactorScorer

        scorer = BondFactorScorer()
        assert scorer.asset_class == "bonds"

    def test_supported_factors(self):
        """测试支持的因子"""
        from finsage.factors.bond_factors import BondFactorScorer
        from finsage.factors.base_factor import FactorType

        scorer = BondFactorScorer()
        factors = scorer.supported_factors

        assert FactorType.CARRY in factors
        assert FactorType.DURATION in factors

    def test_score(self, sample_bond_data, sample_returns):
        """测试评分计算"""
        from finsage.factors.bond_factors import BondFactorScorer

        scorer = BondFactorScorer()
        score = scorer.score("TLT", sample_bond_data, sample_returns)

        assert score.symbol == "TLT"
        assert score.asset_class == "bonds"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 7: REITs Factor Scorer
# ============================================================

class TestREITsFactorScorer:
    """测试REITs因子评分器"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.reits_factors import REITsFactorScorer
        assert REITsFactorScorer is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.factors.reits_factors import REITsFactorScorer

        scorer = REITsFactorScorer()
        assert scorer.asset_class == "reits"

    def test_supported_factors(self):
        """测试支持的因子"""
        from finsage.factors.reits_factors import REITsFactorScorer
        from finsage.factors.base_factor import FactorType

        scorer = REITsFactorScorer()
        factors = scorer.supported_factors

        assert FactorType.NAV in factors

    def test_score(self, sample_reits_data, sample_returns):
        """测试评分计算"""
        from finsage.factors.reits_factors import REITsFactorScorer

        scorer = REITsFactorScorer()
        score = scorer.score("VNQ", sample_reits_data, sample_returns)

        assert score.symbol == "VNQ"
        assert score.asset_class == "reits"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 8: Crypto Factor Scorer
# ============================================================

class TestCryptoFactorScorer:
    """测试加密货币因子评分器"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.crypto_factors import CryptoFactorScorer
        assert CryptoFactorScorer is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.factors.crypto_factors import CryptoFactorScorer

        scorer = CryptoFactorScorer()
        assert scorer.asset_class == "crypto"

    def test_supported_factors(self):
        """测试支持的因子"""
        from finsage.factors.crypto_factors import CryptoFactorScorer
        from finsage.factors.base_factor import FactorType

        scorer = CryptoFactorScorer()
        factors = scorer.supported_factors

        assert FactorType.NETWORK in factors
        assert FactorType.ADOPTION in factors
        assert FactorType.CRASH_RISK in factors

    def test_score(self, sample_crypto_data, sample_returns):
        """测试评分计算"""
        from finsage.factors.crypto_factors import CryptoFactorScorer

        scorer = CryptoFactorScorer()
        score = scorer.score("BTC", sample_crypto_data, sample_returns)

        assert score.symbol == "BTC"
        assert score.asset_class == "crypto"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 9: Commodity Factor Scorer
# ============================================================

class TestCommodityFactorScorer:
    """测试商品因子评分器"""

    def test_import(self):
        """测试导入"""
        from finsage.factors.commodity_factors import CommodityFactorScorer
        assert CommodityFactorScorer is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.factors.commodity_factors import CommodityFactorScorer

        scorer = CommodityFactorScorer()
        assert scorer.asset_class == "commodities"

    def test_supported_factors(self):
        """测试支持的因子"""
        from finsage.factors.commodity_factors import CommodityFactorScorer
        from finsage.factors.base_factor import FactorType

        scorer = CommodityFactorScorer()
        factors = scorer.supported_factors

        assert FactorType.TERM_STRUCTURE in factors
        assert FactorType.BASIS in factors
        assert FactorType.MOMENTUM in factors

    def test_score(self, sample_commodity_data, sample_returns):
        """测试评分计算"""
        from finsage.factors.commodity_factors import CommodityFactorScorer

        scorer = CommodityFactorScorer()
        score = scorer.score("USO", sample_commodity_data, sample_returns)

        assert score.symbol == "USO"
        assert score.asset_class == "commodities"
        assert 0 <= score.composite_score <= 1


# ============================================================
# Test 10: Signal Generation
# ============================================================

class TestSignalGeneration:
    """测试信号生成逻辑"""

    def test_signal_strong_buy(self):
        """测试强买信号"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        signal = scorer._generate_signal(0.85)
        assert signal == "STRONG_BUY"

    def test_signal_buy(self):
        """测试买入信号"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        signal = scorer._generate_signal(0.65)
        assert signal == "BUY"

    def test_signal_hold(self):
        """测试持有信号"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        signal = scorer._generate_signal(0.50)
        assert signal == "HOLD"

    def test_signal_sell(self):
        """测试卖出信号"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        # sell threshold is 0.4, so 0.42 should be SELL (between 0.4 and 0.45)
        signal = scorer._generate_signal(0.42)
        assert signal == "SELL"

    def test_signal_strong_sell(self):
        """测试强卖信号"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        signal = scorer._generate_signal(0.15)
        assert signal == "STRONG_SELL"


# ============================================================
# Test 11: Weight Adjustment for Market Regime
# ============================================================

class TestWeightAdjustment:
    """测试市场状态权重调整"""

    def test_bull_market_weights(self):
        """测试牛市权重调整"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        scorer.factor_weights = {"momentum": 0.2, "value": 0.3, "size": 0.5}

        adjusted = scorer._adjust_weights_for_regime("bull")

        # 牛市增加动量权重，减少价值权重
        # 注意归一化后比例变化
        assert "momentum" in adjusted
        assert "value" in adjusted

    def test_bear_market_weights(self):
        """测试熊市权重调整"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        scorer.factor_weights = {"momentum": 0.3, "value": 0.3, "low_risk": 0.4}

        adjusted = scorer._adjust_weights_for_regime("bear")

        # 熊市增加低风险和价值权重，减少动量
        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_neutral_market_weights(self):
        """测试中性市场权重"""
        from finsage.factors.stock_factors import StockFactorScorer

        scorer = StockFactorScorer()
        original = scorer.factor_weights.copy()

        adjusted = scorer._adjust_weights_for_regime("neutral")

        # 中性市场不调整
        assert abs(sum(adjusted.values()) - 1.0) < 0.01


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Factors Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
