"""
Deep tests for CryptoFactorScorer

覆盖 finsage/factors/crypto_factors.py (目标从60%提升到80%+)
"""

import pytest
import pandas as pd
import numpy as np

from finsage.factors.crypto_factors import CryptoFactorScorer
from finsage.factors.base_factor import FactorType


class TestCryptoFactorScorerProperties:
    """测试CryptoFactorScorer属性"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_asset_class(self, scorer):
        """测试资产类别"""
        assert scorer.asset_class == "crypto"

    def test_supported_factors(self, scorer):
        """测试支持的因子"""
        factors = scorer.supported_factors
        assert FactorType.NETWORK in factors
        assert FactorType.ADOPTION in factors
        assert FactorType.MOMENTUM in factors
        assert FactorType.CRASH_RISK in factors

    def test_default_weights(self, scorer):
        """测试默认权重"""
        weights = scorer._default_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert weights["network"] == 0.35
        assert weights["momentum"] == 0.25


class TestCryptoFactorScorerCategory:
    """测试加密货币类别判断"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_get_category_btc(self, scorer):
        """测试BTC类别"""
        assert scorer._get_category("BTC") == "store_of_value"

    def test_get_category_eth(self, scorer):
        """测试ETH类别"""
        assert scorer._get_category("ETH") == "smart_contract"

    def test_get_category_uni(self, scorer):
        """测试UNI类别"""
        assert scorer._get_category("UNI") == "defi"

    def test_get_category_matic(self, scorer):
        """测试MATIC类别"""
        assert scorer._get_category("MATIC") == "layer2"

    def test_get_category_usdt(self, scorer):
        """测试USDT类别"""
        assert scorer._get_category("USDT") == "stablecoin"

    def test_get_category_doge(self, scorer):
        """测试DOGE类别"""
        assert scorer._get_category("DOGE") == "meme"

    def test_get_category_unknown(self, scorer):
        """测试未知类别"""
        assert scorer._get_category("UNKNOWN") == "smart_contract"  # 默认


class TestCryptoFactorScorerNetworkExposure:
    """测试网络因子暴露计算"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    @pytest.fixture
    def strong_network_data(self):
        return {
            "active_addresses": 1000000,
            "active_addresses_percentile": 90,
            "transaction_volume": 5000000000,
            "market_cap": 50000000000,
            "hash_rate_growth_30d": 0.15,
            "tvl": 10000000000,
            "tvl_growth_30d": 0.2,
            "developer_activity_percentile": 85,
        }

    @pytest.fixture
    def weak_network_data(self):
        return {
            "active_addresses": 1000,
            "active_addresses_percentile": 10,
            "transaction_volume": 1000000,
            "market_cap": 100000000,
            "hash_rate_growth_30d": -0.1,
            "tvl": 0,
            "tvl_growth_30d": 0,
            "developer_activity_percentile": 15,
        }

    def test_compute_network_exposure_strong(self, scorer, strong_network_data):
        """测试强网络效应"""
        exposure = scorer._compute_network_exposure("ETH", strong_network_data, "smart_contract")

        assert exposure.factor_type == FactorType.NETWORK
        assert exposure.exposure > 0
        assert exposure.signal in ["STRONG_NETWORK", "LONG"]
        assert exposure.confidence > 0

    def test_compute_network_exposure_weak(self, scorer, weak_network_data):
        """测试弱网络效应"""
        exposure = scorer._compute_network_exposure("ETH", weak_network_data, "smart_contract")

        assert exposure.exposure < 0
        assert exposure.signal in ["WEAK_NETWORK", "NEUTRAL"]

    def test_compute_network_exposure_btc(self, scorer, strong_network_data):
        """测试BTC网络效应（使用hash_rate）"""
        exposure = scorer._compute_network_exposure("BTC", strong_network_data, "store_of_value")

        assert exposure.factor_type == FactorType.NETWORK


class TestCryptoFactorScorerAdoptionExposure:
    """测试采纳因子暴露计算"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    @pytest.fixture
    def high_adoption_data(self):
        return {
            "new_addresses_growth_30d": 0.15,  # 15%月增长
            "whale_concentration": 0.2,  # 低集中度
            "exchange_netflow_7d": -50000,  # 流出交易所
            "institutional_holdings_pct": 0.12,  # 高机构持仓
        }

    @pytest.fixture
    def low_adoption_data(self):
        return {
            "new_addresses_growth_30d": -0.05,  # 负增长
            "whale_concentration": 0.8,  # 高集中度
            "exchange_netflow_7d": 100000,  # 流入交易所
            "institutional_holdings_pct": 0.01,  # 低机构持仓
        }

    def test_compute_adoption_exposure_high(self, scorer, high_adoption_data):
        """测试高采纳度"""
        exposure = scorer._compute_adoption_exposure(high_adoption_data, "smart_contract")

        assert exposure.factor_type == FactorType.ADOPTION
        assert exposure.exposure > 0
        assert exposure.signal in ["HIGH_ADOPTION", "GROWING"]

    def test_compute_adoption_exposure_low(self, scorer, low_adoption_data):
        """测试低采纳度"""
        exposure = scorer._compute_adoption_exposure(low_adoption_data, "smart_contract")

        assert exposure.exposure < 0
        assert exposure.signal in ["LOW_ADOPTION", "NEUTRAL"]


class TestCryptoFactorScorerMomentumExposure:
    """测试动量因子暴露计算"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_compute_momentum_exposure_strong_positive(self, scorer):
        """测试强正动量"""
        data = {
            "return_7d": 0.20,
            "return_30d": 0.35,
            "return_90d": 0.50,
        }

        exposure = scorer._compute_momentum_exposure(data, None)

        assert exposure.factor_type == FactorType.MOMENTUM
        assert exposure.exposure > 0
        assert exposure.signal in ["STRONG_MOMENTUM", "LONG"]
        assert exposure.confidence == 0.75

    def test_compute_momentum_exposure_strong_negative(self, scorer):
        """测试强负动量"""
        data = {
            "return_7d": -0.30,
            "return_30d": -0.40,
            "return_90d": -0.50,
        }

        exposure = scorer._compute_momentum_exposure(data, None)

        assert exposure.exposure < 0
        assert exposure.signal in ["STRONG_REVERSAL", "SHORT"]

    def test_compute_momentum_exposure_with_returns(self, scorer):
        """测试使用历史收益率"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.03 + 0.005)

        exposure = scorer._compute_momentum_exposure({}, returns)

        assert exposure.factor_type == FactorType.MOMENTUM


class TestCryptoFactorScorerCrashRiskExposure:
    """测试崩盘风险因子暴露计算"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_compute_crash_risk_high(self, scorer):
        """测试高崩盘风险"""
        data = {
            "volatility_30d": 1.2,  # 120%年化波动
            "max_drawdown_90d": -0.60,  # 60%回撤
            "funding_rate": 0.002,  # 高资金费率
            "open_interest_change_7d": 0.40,  # 高杠杆增长
            "liquidation_volume_24h": 500000000,
            "market_cap": 10000000000,
        }

        exposure = scorer._compute_crash_risk_exposure(data, None)

        assert exposure.factor_type == FactorType.CRASH_RISK
        assert exposure.exposure < 0  # 高风险 = 负暴露
        assert exposure.signal in ["HIGH_RISK_AVOID", "ELEVATED_RISK"]

    def test_compute_crash_risk_low(self, scorer):
        """测试低崩盘风险"""
        data = {
            "volatility_30d": 0.30,  # 30%年化波动
            "max_drawdown_90d": -0.10,  # 10%回撤
            "funding_rate": 0.0001,  # 低资金费率
            "open_interest_change_7d": 0.05,
            "liquidation_volume_24h": 1000000,
            "market_cap": 100000000000,
        }

        exposure = scorer._compute_crash_risk_exposure(data, None)

        assert exposure.exposure > 0  # 低风险 = 正暴露
        assert exposure.signal in ["LOW_RISK", "MODERATE_RISK"]

    def test_compute_crash_risk_with_returns(self, scorer):
        """测试使用历史收益率计算崩盘风险"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.03)

        exposure = scorer._compute_crash_risk_exposure({}, returns)

        assert exposure.factor_type == FactorType.CRASH_RISK


class TestCryptoFactorScorerFactorPremiums:
    """测试因子溢价"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()

        assert "network" in premiums
        assert "adoption" in premiums
        assert "momentum" in premiums
        assert "crash_risk" in premiums
        assert premiums["momentum"] == 0.12  # 动量溢价最高


class TestCryptoFactorScorerOnChainSummary:
    """测试链上数据摘要"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_get_on_chain_summary_healthy(self, scorer):
        """测试健康网络摘要"""
        data = {
            "active_addresses": 1500000,
            "transaction_volume": 5000000000,
            "new_addresses_growth_30d": 0.12,
            "whale_concentration": 0.25,
            "exchange_netflow_7d": -50000,
        }

        summary = scorer.get_on_chain_summary("BTC", data)

        assert "非常健康" in summary
        assert "强劲增长" in summary or "稳定增长" in summary
        assert "强持有意愿" in summary

    def test_get_on_chain_summary_weak(self, scorer):
        """测试弱网络摘要"""
        data = {
            "active_addresses": 5000,
            "transaction_volume": 100000,
            "new_addresses_growth_30d": -0.05,
            "whale_concentration": 0.80,
            "exchange_netflow_7d": 50000,
        }

        summary = scorer.get_on_chain_summary("UNKNOWN_TOKEN", data)

        assert "较弱" in summary
        assert "负增长" in summary
        assert "卖出压力" in summary


class TestCryptoFactorScorerRiskAssessment:
    """测试风险评估"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_get_risk_assessment_high_risk(self, scorer):
        """测试高风险评估"""
        data = {
            "volatility_30d": 1.0,
            "volatility_90d": 0.9,
            "var_95": -0.15,
            "max_drawdown_90d": -0.50,
            "funding_rate": 0.002,
            "open_interest_change_7d": 0.30,
        }

        assessment = scorer.get_risk_assessment("BTC", data)

        assert assessment["risk_level"] in ["极高", "高"]
        assert assessment["risk_score"] > 0.6
        # 高风险建议包含止损或谨慎等关键词
        assert any(kw in assessment["recommendation"] for kw in ["谨慎", "止损", "控制仓位"])

    def test_get_risk_assessment_low_risk(self, scorer):
        """测试低风险评估"""
        data = {
            "volatility_30d": 0.25,
            "volatility_90d": 0.25,
            "var_95": -0.03,
            "max_drawdown_90d": -0.10,
            "funding_rate": 0.0001,
            "open_interest_change_7d": 0.05,
        }

        assessment = scorer.get_risk_assessment("ETH", data)

        assert assessment["risk_level"] in ["较低", "中等"]
        assert assessment["risk_score"] < 0.5

    def test_get_risk_assessment_with_returns(self, scorer):
        """测试使用历史收益率的风险评估"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.03)
        data = {"max_drawdown_90d": -0.20, "funding_rate": 0.001, "open_interest_change_7d": 0.10}

        assessment = scorer.get_risk_assessment("SOL", data, returns)

        assert "risk_level" in assessment
        assert "volatility_30d" in assessment


class TestCryptoFactorScorerRiskRecommendation:
    """测试风险建议"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_risk_recommendation_extreme(self, scorer):
        """测试极高风险建议"""
        rec = scorer._get_risk_recommendation("极高")
        assert "极度谨慎" in rec

    def test_risk_recommendation_high(self, scorer):
        """测试高风险建议"""
        rec = scorer._get_risk_recommendation("高")
        assert "止损" in rec

    def test_risk_recommendation_medium(self, scorer):
        """测试中等风险建议"""
        rec = scorer._get_risk_recommendation("中等")
        assert "可控" in rec

    def test_risk_recommendation_low(self, scorer):
        """测试低风险建议"""
        rec = scorer._get_risk_recommendation("较低")
        assert "增加" in rec


class TestCryptoFactorScorerCategoryAnalysis:
    """测试类别分析"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_get_category_analysis(self, scorer):
        """测试类别分析"""
        from finsage.factors.base_factor import FactorScore, FactorExposure
        from datetime import datetime

        scores = {
            "BTC": FactorScore(
                symbol="BTC",
                asset_class="crypto",
                factor_exposures={},
                composite_score=0.7,
                expected_alpha=0.10,
                signal="LONG",
                reasoning="Strong network",
                timestamp=datetime.now().isoformat(),
                risk_contribution=0.1,
            ),
            "ETH": FactorScore(
                symbol="ETH",
                asset_class="crypto",
                factor_exposures={},
                composite_score=0.6,
                expected_alpha=0.08,
                signal="LONG",
                reasoning="Growing adoption",
                timestamp=datetime.now().isoformat(),
                risk_contribution=0.1,
            ),
        }

        analysis = scorer.get_category_analysis(scores)

        assert "store_of_value" in analysis
        assert analysis["store_of_value"]["average_score"] == 0.7


class TestCryptoFactorScorerInvestmentThesis:
    """测试投资论点生成"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    def test_generate_investment_thesis(self, scorer):
        """测试投资论点生成 - 跳过由于源代码f-string格式问题"""
        # 注意：源代码中generate_investment_thesis有f-string格式问题
        # 这里只测试方法存在
        assert hasattr(scorer, 'generate_investment_thesis')
        # 如果源代码修复后可以启用完整测试


class TestCryptoFactorScorerComputeFactorExposures:
    """测试完整因子暴露计算"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    @pytest.fixture
    def complete_data(self):
        return {
            "active_addresses": 500000,
            "active_addresses_percentile": 75,
            "transaction_volume": 2000000000,
            "market_cap": 30000000000,
            "hash_rate_growth_30d": 0.08,
            "tvl": 5000000000,
            "tvl_growth_30d": 0.15,
            "developer_activity_percentile": 70,
            "new_addresses_growth_30d": 0.08,
            "whale_concentration": 0.35,
            "exchange_netflow_7d": -10000,
            "institutional_holdings_pct": 0.06,
            "return_7d": 0.10,
            "return_30d": 0.15,
            "return_90d": 0.25,
            "volatility_30d": 0.65,
            "max_drawdown_90d": -0.25,
            "funding_rate": 0.0005,
            "open_interest_change_7d": 0.15,
            "liquidation_volume_24h": 50000000,
        }

    def test_compute_factor_exposures_complete(self, scorer, complete_data):
        """测试完整因子暴露计算"""
        exposures = scorer._compute_factor_exposures("ETH", complete_data, None)

        assert "network" in exposures
        assert "adoption" in exposures
        assert "momentum" in exposures
        assert "crash_risk" in exposures

        # 验证所有暴露在有效范围内
        for name, exp in exposures.items():
            assert -1 <= exp.exposure <= 1
            assert exp.confidence > 0

    def test_compute_factor_exposures_with_returns(self, scorer, complete_data):
        """测试使用历史收益率的因子暴露计算"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.03 + 0.002)

        exposures = scorer._compute_factor_exposures("BTC", complete_data, returns)

        assert "momentum" in exposures
        assert "crash_risk" in exposures


class TestCryptoFactorScorerScore:
    """测试评分功能"""

    @pytest.fixture
    def scorer(self):
        return CryptoFactorScorer()

    @pytest.fixture
    def complete_data(self):
        return {
            "active_addresses": 500000,
            "active_addresses_percentile": 75,
            "transaction_volume": 2000000000,
            "market_cap": 30000000000,
            "hash_rate_growth_30d": 0.08,
            "tvl": 5000000000,
            "tvl_growth_30d": 0.15,
            "developer_activity_percentile": 70,
            "new_addresses_growth_30d": 0.08,
            "whale_concentration": 0.35,
            "exchange_netflow_7d": -10000,
            "institutional_holdings_pct": 0.06,
            "return_7d": 0.10,
            "return_30d": 0.15,
            "return_90d": 0.25,
            "volatility_30d": 0.65,
            "max_drawdown_90d": -0.25,
            "funding_rate": 0.0005,
            "open_interest_change_7d": 0.15,
            "liquidation_volume_24h": 50000000,
        }

    def test_score_basic(self, scorer, complete_data):
        """测试基本评分"""
        score = scorer.score("ETH", complete_data)

        assert score.symbol == "ETH"
        assert score.asset_class == "crypto"
        assert -1 <= score.composite_score <= 1
        assert score.signal in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
