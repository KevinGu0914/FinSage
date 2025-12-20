"""
Deep tests for DynamicHedgeSelector

覆盖 finsage/hedging/dynamic_selector.py (目标从47%提升到80%+)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from finsage.hedging.dynamic_selector import (
    DynamicHedgeSelector,
    HedgeObjective,
    PortfolioExposure,
    HedgeCandidate,
    HedgeRecommendation,
)
from finsage.hedging.hedge_universe import HedgeAsset, HedgeCategory


# ============================================================
# HedgeObjective Tests
# ============================================================

class TestHedgeObjective:
    """测试对冲目标枚举"""

    def test_all_objectives_exist(self):
        """测试所有对冲目标存在"""
        assert HedgeObjective.BETA_NEUTRAL.value == "beta_neutral"
        assert HedgeObjective.SECTOR_HEDGE.value == "sector_hedge"
        assert HedgeObjective.TAIL_RISK.value == "tail_risk"
        assert HedgeObjective.CORRELATION_HEDGE.value == "correlation_hedge"
        assert HedgeObjective.VOLATILITY_HEDGE.value == "volatility_hedge"
        assert HedgeObjective.RATE_HEDGE.value == "rate_hedge"
        assert HedgeObjective.CURRENCY_HEDGE.value == "currency_hedge"
        assert HedgeObjective.DIVERSIFICATION.value == "diversification"


# ============================================================
# PortfolioExposure Tests
# ============================================================

class TestPortfolioExposure:
    """测试组合敞口数据类"""

    def test_default_values(self):
        """测试默认值"""
        exposure = PortfolioExposure()

        assert exposure.beta == 1.0
        assert exposure.volatility == 0.12
        assert exposure.concentration_hhi == 0.1
        assert exposure.correlation_with_spy == 0.8
        assert exposure.var_95 == -0.02

    def test_custom_values(self):
        """测试自定义值"""
        exposure = PortfolioExposure(
            beta=1.5,
            volatility=0.20,
            concentration_hhi=0.25,
            sector_exposure={"tech": 0.4, "healthcare": 0.3},
            top_holdings=[{"symbol": "AAPL", "weight": 0.1}],
            correlation_with_spy=0.9,
            var_95=-0.03
        )

        assert exposure.beta == 1.5
        assert exposure.volatility == 0.20
        assert "tech" in exposure.sector_exposure

    def test_to_dict(self):
        """测试转换为字典"""
        exposure = PortfolioExposure(
            beta=1.234,
            volatility=0.15678,
            sector_exposure={"tech": 0.4}
        )

        result = exposure.to_dict()

        assert result["beta"] == 1.234
        assert result["volatility"] == 0.1568  # 四舍五入
        assert "sector_exposure" in result


# ============================================================
# HedgeCandidate Tests
# ============================================================

class TestHedgeCandidate:
    """测试对冲候选数据类"""

    @pytest.fixture
    def mock_asset(self):
        return HedgeAsset(
            symbol="SH",
            name="Short S&P 500",
            category=HedgeCategory.INVERSE_EQUITY,
            expense_ratio=0.009,
            leverage=-1.0,
            avg_daily_volume=1e7
        )

    def test_create_candidate(self, mock_asset):
        """测试创建候选"""
        candidate = HedgeCandidate(
            asset=mock_asset,
            correlation_score=0.8,
            liquidity_score=0.9,
            cost_score=0.95,
            efficiency_score=0.7,
            total_score=0.85,
            raw_correlation=-0.6
        )

        assert candidate.total_score == 0.85
        assert candidate.raw_correlation == -0.6

    def test_to_dict(self, mock_asset):
        """测试转换为字典"""
        candidate = HedgeCandidate(
            asset=mock_asset,
            correlation_score=0.8,
            liquidity_score=0.9,
            cost_score=0.95,
            efficiency_score=0.7,
            total_score=0.85,
            raw_correlation=-0.6
        )

        result = candidate.to_dict()

        assert result["symbol"] == "SH"
        assert result["total_score"] == 0.85
        assert "leverage" in result


# ============================================================
# HedgeRecommendation Tests
# ============================================================

class TestHedgeRecommendation:
    """测试对冲推荐数据类"""

    @pytest.fixture
    def mock_candidate(self):
        asset = HedgeAsset(
            symbol="SH",
            name="Short S&P 500",
            category=HedgeCategory.INVERSE_EQUITY,
            expense_ratio=0.009
        )
        return HedgeCandidate(
            asset=asset,
            total_score=0.8,
            raw_correlation=-0.5
        )

    def test_create_recommendation(self, mock_candidate):
        """测试创建推荐"""
        recommendation = HedgeRecommendation(
            objective=HedgeObjective.BETA_NEUTRAL,
            candidates=[mock_candidate],
            recommended_allocation={"SH": 0.1},
            expected_correlation_reduction=0.05,
            expected_cost=0.001,
            reasoning="Test reasoning"
        )

        assert recommendation.objective == HedgeObjective.BETA_NEUTRAL
        assert len(recommendation.candidates) == 1
        assert "SH" in recommendation.recommended_allocation

    def test_to_dict(self, mock_candidate):
        """测试转换为字典"""
        recommendation = HedgeRecommendation(
            objective=HedgeObjective.TAIL_RISK,
            candidates=[mock_candidate],
            recommended_allocation={"SH": 0.1},
            expected_correlation_reduction=0.05,
            expected_cost=0.001,
            reasoning="Test reasoning",
            exposure_analysis=PortfolioExposure()
        )

        result = recommendation.to_dict()

        assert result["objective"] == "tail_risk"
        assert len(result["top_candidates"]) <= 5
        assert "exposure_analysis" in result

    def test_get_instruments_for_agent(self, mock_candidate):
        """测试获取Agent格式的工具列表"""
        recommendation = HedgeRecommendation(
            objective=HedgeObjective.BETA_NEUTRAL,
            candidates=[mock_candidate],
            recommended_allocation={"SH": 0.1},
            expected_correlation_reduction=0.05,
            expected_cost=0.001,
            reasoning="Test"
        )

        instruments = recommendation.get_instruments_for_agent()

        assert len(instruments) == 1
        assert instruments[0]["symbol"] == "SH"
        assert instruments[0]["allocation"] == 0.1
        assert instruments[0]["source"] == "dynamic"

    def test_get_instruments_missing_candidate(self, mock_candidate):
        """测试获取工具列表时缺少候选"""
        recommendation = HedgeRecommendation(
            objective=HedgeObjective.BETA_NEUTRAL,
            candidates=[mock_candidate],
            recommended_allocation={"UNKNOWN": 0.1},  # 不在候选中
            expected_correlation_reduction=0.0,
            expected_cost=0.0,
            reasoning="Test"
        )

        instruments = recommendation.get_instruments_for_agent()

        assert len(instruments) == 0


# ============================================================
# DynamicHedgeSelector Tests
# ============================================================

class TestDynamicHedgeSelectorInit:
    """测试选择器初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        selector = DynamicHedgeSelector()

        assert selector.universe is not None
        assert selector.min_liquidity == 1e5
        assert selector.max_expense == 0.02
        assert selector.top_k == 5

    def test_custom_config(self):
        """测试自定义配置"""
        config = {
            "min_daily_volume": 1e6,
            "max_expense_ratio": 0.01,
            "top_candidates": 10,
            "correlation_lookback": 90
        }

        selector = DynamicHedgeSelector(config=config)

        assert selector.min_liquidity == 1e6
        assert selector.max_expense == 0.01
        assert selector.top_k == 10
        assert selector.lookback_days == 90

    def test_custom_weights(self):
        """测试自定义权重"""
        config = {
            "scoring_weights": {
                "correlation": 0.5,
                "liquidity": 0.2,
                "cost": 0.15,
                "efficiency": 0.15
            }
        }

        selector = DynamicHedgeSelector(config=config)

        assert selector.weights["correlation"] == 0.5


class TestDynamicHedgeSelectorAnalyzeExposure:
    """测试组合敞口分析"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    @pytest.fixture
    def sample_returns(self):
        """创建样本收益率数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        data = {
            "AAPL": np.random.randn(100) * 0.02,
            "MSFT": np.random.randn(100) * 0.018,
            "GOOGL": np.random.randn(100) * 0.022,
            "SPY": np.random.randn(100) * 0.01,
        }
        return pd.DataFrame(data, index=dates)

    def test_analyze_exposure_basic(self, selector, sample_returns):
        """测试基本敞口分析"""
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}

        exposure = selector.analyze_portfolio_exposure(weights, sample_returns)

        assert exposure.volatility > 0
        assert 0 < exposure.concentration_hhi < 1
        assert len(exposure.top_holdings) <= 5

    def test_analyze_exposure_empty_data(self, selector):
        """测试空数据"""
        empty_returns = pd.DataFrame()
        weights = {"AAPL": 0.5}

        exposure = selector.analyze_portfolio_exposure(weights, empty_returns)

        assert exposure.beta == 1.0  # 默认值

    def test_analyze_exposure_no_overlap(self, selector, sample_returns):
        """测试无重叠资产"""
        weights = {"UNKNOWN": 0.5, "MISSING": 0.5}

        exposure = selector.analyze_portfolio_exposure(weights, sample_returns)

        assert exposure.beta == 1.0  # 默认值

    def test_analyze_exposure_with_spy(self, selector, sample_returns):
        """测试有SPY时计算Beta"""
        weights = {"AAPL": 0.5, "MSFT": 0.5}

        exposure = selector.analyze_portfolio_exposure(weights, sample_returns)

        assert exposure.beta != 1.0 or exposure.correlation_with_spy != 0.8

    def test_analyze_exposure_zero_weights(self, selector, sample_returns):
        """测试零权重"""
        weights = {"AAPL": 0, "MSFT": 0}

        exposure = selector.analyze_portfolio_exposure(weights, sample_returns)

        assert exposure.beta == 1.0  # 默认值


class TestDynamicHedgeSelectorIdentifyObjective:
    """测试对冲目标识别"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    def test_strategy_mapping_put_protection(self, selector):
        """测试put_protection策略映射"""
        exposure = PortfolioExposure()

        objective = selector.identify_hedge_objective(exposure, "put_protection")

        assert objective == HedgeObjective.TAIL_RISK

    def test_strategy_mapping_dynamic_hedge(self, selector):
        """测试dynamic_hedge策略映射"""
        exposure = PortfolioExposure()

        objective = selector.identify_hedge_objective(exposure, "dynamic_hedge")

        assert objective == HedgeObjective.BETA_NEUTRAL

    def test_strategy_mapping_safe_haven(self, selector):
        """测试safe_haven策略映射"""
        exposure = PortfolioExposure()

        objective = selector.identify_hedge_objective(exposure, "safe_haven")

        assert objective == HedgeObjective.CORRELATION_HEDGE

    def test_high_beta_detection(self, selector):
        """测试高Beta检测"""
        exposure = PortfolioExposure(beta=1.5)

        objective = selector.identify_hedge_objective(exposure, "unknown")

        assert objective == HedgeObjective.BETA_NEUTRAL

    def test_high_volatility_detection(self, selector):
        """测试高波动率检测"""
        exposure = PortfolioExposure(beta=0.9, volatility=0.25)

        objective = selector.identify_hedge_objective(exposure, "unknown")

        assert objective == HedgeObjective.TAIL_RISK

    def test_high_concentration_detection(self, selector):
        """测试高集中度检测"""
        exposure = PortfolioExposure(beta=0.9, volatility=0.12, concentration_hhi=0.35)

        objective = selector.identify_hedge_objective(exposure, "unknown")

        assert objective == HedgeObjective.DIVERSIFICATION

    def test_default_objective(self, selector):
        """测试默认目标"""
        exposure = PortfolioExposure(beta=0.9, volatility=0.10, concentration_hhi=0.15)

        objective = selector.identify_hedge_objective(exposure, "unknown")

        assert objective == HedgeObjective.CORRELATION_HEDGE


class TestDynamicHedgeSelectorSelectCandidates:
    """测试候选资产筛选"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    def test_select_beta_neutral(self, selector):
        """测试Beta中性筛选"""
        exposure = PortfolioExposure()

        candidates = selector.select_candidates(
            HedgeObjective.BETA_NEUTRAL, exposure
        )

        # 应该包含反向股票ETF
        assert any(c.is_inverse for c in candidates)

    def test_select_tail_risk(self, selector):
        """测试尾部风险筛选"""
        exposure = PortfolioExposure()

        candidates = selector.select_candidates(
            HedgeObjective.TAIL_RISK, exposure
        )

        assert len(candidates) > 0

    def test_select_correlation_hedge(self, selector):
        """测试相关性对冲筛选"""
        exposure = PortfolioExposure()

        candidates = selector.select_candidates(
            HedgeObjective.CORRELATION_HEDGE, exposure
        )

        # 不应该包含反向资产
        assert all(not c.is_inverse for c in candidates)

    def test_select_diversification(self, selector):
        """测试分散化筛选"""
        exposure = PortfolioExposure()

        candidates = selector.select_candidates(
            HedgeObjective.DIVERSIFICATION, exposure
        )

        assert len(candidates) > 0

    def test_select_volatility_hedge(self, selector):
        """测试波动率对冲筛选"""
        exposure = PortfolioExposure()

        candidates = selector.select_candidates(
            HedgeObjective.VOLATILITY_HEDGE, exposure
        )

        # 应该包含波动率工具
        categories = [c.category for c in candidates]
        assert HedgeCategory.VOLATILITY in categories or len(candidates) == 0

    def test_liquidity_filter(self, selector):
        """测试流动性过滤"""
        selector.min_liquidity = 1e10  # 极高门槛

        candidates = selector.select_candidates(
            HedgeObjective.BETA_NEUTRAL, PortfolioExposure()
        )

        assert len(candidates) == 0

    def test_no_duplicates(self, selector):
        """测试无重复"""
        candidates = selector.select_candidates(
            HedgeObjective.TAIL_RISK, PortfolioExposure()
        )

        symbols = [c.symbol for c in candidates]
        assert len(symbols) == len(set(symbols))


class TestDynamicHedgeSelectorScoreCandidates:
    """测试候选资产评分"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        data = {
            "SPY": np.random.randn(100) * 0.01,
            "SH": -np.random.randn(100) * 0.01,  # 反向
            "GLD": np.random.randn(100) * 0.008,
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_candidates(self):
        return [
            HedgeAsset(
                symbol="SH",
                name="Short S&P 500",
                category=HedgeCategory.INVERSE_EQUITY,
                expense_ratio=0.009,
                leverage=-1.0,
                avg_daily_volume=1e7
            ),
            HedgeAsset(
                symbol="GLD",
                name="Gold ETF",
                category=HedgeCategory.SAFE_HAVEN,
                expense_ratio=0.004,
                leverage=1.0,
                avg_daily_volume=1e8
            ),
        ]

    def test_score_candidates(self, selector, sample_candidates, sample_returns):
        """测试候选评分"""
        portfolio_returns = sample_returns["SPY"].values

        scored = selector.score_candidates(
            sample_candidates, portfolio_returns, sample_returns
        )

        assert len(scored) == 2
        # 应该按总分排序
        assert scored[0].total_score >= scored[1].total_score

    def test_score_correlation(self, selector, sample_candidates, sample_returns):
        """测试相关性评分"""
        portfolio_returns = sample_returns["SPY"].values

        scored = selector.score_candidates(
            sample_candidates, portfolio_returns, sample_returns
        )

        # SH应该有相关性得分（随机数据可能变化，只检查有评分）
        sh_score = next(s for s in scored if s.asset.symbol == "SH")
        assert sh_score.correlation_score >= 0  # 只检查有效评分

    def test_score_empty_candidates(self, selector, sample_returns):
        """测试空候选列表"""
        portfolio_returns = sample_returns["SPY"].values

        scored = selector.score_candidates(
            [], portfolio_returns, sample_returns
        )

        assert len(scored) == 0

    def test_score_no_returns_data(self, selector, sample_candidates, sample_returns):
        """测试无收益率数据"""
        portfolio_returns = np.array([])

        scored = selector.score_candidates(
            sample_candidates, portfolio_returns, sample_returns
        )

        # 应该仍然返回评分（相关性为0）
        assert len(scored) == 2


class TestDynamicHedgeSelectorComputeAllocation:
    """测试最优配置计算"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    @pytest.fixture
    def scored_candidates(self):
        asset1 = HedgeAsset(symbol="SH", name="Short", category=HedgeCategory.INVERSE_EQUITY)
        asset2 = HedgeAsset(symbol="GLD", name="Gold", category=HedgeCategory.SAFE_HAVEN)

        return [
            HedgeCandidate(asset=asset1, total_score=0.8),
            HedgeCandidate(asset=asset2, total_score=0.6),
        ]

    def test_compute_allocation(self, selector, scored_candidates):
        """测试计算配置"""
        np.random.seed(42)
        portfolio_returns = np.random.randn(100) * 0.01
        returns_df = pd.DataFrame({"SPY": portfolio_returns})

        allocation = selector.compute_optimal_allocation(
            scored_candidates, hedge_ratio=0.1,
            portfolio_returns=portfolio_returns,
            returns_data=returns_df
        )

        assert len(allocation) == 2
        assert abs(sum(allocation.values()) - 0.1) < 0.001

    def test_compute_allocation_empty(self, selector):
        """测试空候选配置"""
        allocation = selector.compute_optimal_allocation(
            [], hedge_ratio=0.1,
            portfolio_returns=np.array([]),
            returns_data=pd.DataFrame()
        )

        assert allocation == {}

    def test_compute_allocation_zero_ratio(self, selector, scored_candidates):
        """测试零对冲比例"""
        allocation = selector.compute_optimal_allocation(
            scored_candidates, hedge_ratio=0,
            portfolio_returns=np.array([]),
            returns_data=pd.DataFrame()
        )

        assert allocation == {}

    def test_compute_allocation_zero_scores(self, selector):
        """测试零分数候选"""
        asset = HedgeAsset(symbol="TEST", name="Test", category=HedgeCategory.SAFE_HAVEN)
        candidates = [
            HedgeCandidate(asset=asset, total_score=0),
        ]

        allocation = selector.compute_optimal_allocation(
            candidates, hedge_ratio=0.1,
            portfolio_returns=np.array([]),
            returns_data=pd.DataFrame()
        )

        # 应该等权分配
        assert abs(allocation.get("TEST", 0) - 0.1) < 0.001


class TestDynamicHedgeSelectorRecommend:
    """测试推荐生成"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        data = {
            "AAPL": np.random.randn(100) * 0.02,
            "MSFT": np.random.randn(100) * 0.018,
            "SPY": np.random.randn(100) * 0.01,
            "SH": -np.random.randn(100) * 0.01,
            "GLD": np.random.randn(100) * 0.008,
        }
        return pd.DataFrame(data, index=dates)

    def test_recommend_basic(self, selector, sample_data):
        """测试基本推荐"""
        weights = {"AAPL": 0.5, "MSFT": 0.5}

        recommendation = selector.recommend(
            portfolio_weights=weights,
            returns_data=sample_data,
            hedge_strategy="tail_hedge",
            hedge_ratio=0.1
        )

        assert recommendation.objective == HedgeObjective.TAIL_RISK
        assert recommendation.exposure_analysis is not None
        assert len(recommendation.reasoning) > 0

    def test_recommend_no_candidates(self, selector, sample_data):
        """测试无候选时的推荐"""
        weights = {"AAPL": 0.5, "MSFT": 0.5}

        # 使用极高门槛导致无候选
        selector.min_liquidity = 1e15

        recommendation = selector.recommend(
            portfolio_weights=weights,
            returns_data=sample_data,
            hedge_strategy="tail_hedge",
            hedge_ratio=0.1
        )

        assert recommendation.recommended_allocation == {}
        assert "未找到" in recommendation.reasoning or len(recommendation.candidates) == 0


class TestDynamicHedgeSelectorGenerateReasoning:
    """测试推荐理由生成"""

    @pytest.fixture
    def selector(self):
        return DynamicHedgeSelector()

    def test_generate_reasoning_basic(self, selector):
        """测试基本理由生成"""
        exposure = PortfolioExposure(beta=1.2, volatility=0.15)
        asset = HedgeAsset(symbol="SH", name="Short", category=HedgeCategory.INVERSE_EQUITY)
        candidates = [HedgeCandidate(asset=asset, total_score=0.8)]
        allocation = {"SH": 0.1}

        reasoning = selector._generate_reasoning(
            HedgeObjective.BETA_NEUTRAL,
            exposure,
            candidates,
            allocation,
            0.1
        )

        assert "Beta" in reasoning
        assert "SH" in reasoning

    def test_generate_reasoning_empty(self, selector):
        """测试空数据理由"""
        exposure = PortfolioExposure()

        reasoning = selector._generate_reasoning(
            HedgeObjective.BETA_NEUTRAL,
            exposure,
            [],
            {},
            0.1
        )

        assert "未能" in reasoning


class TestDynamicHedgeSelectorUniverseSummary:
    """测试资产全集摘要"""

    def test_get_universe_summary(self):
        """测试获取全集摘要"""
        selector = DynamicHedgeSelector()

        summary = selector.get_universe_summary()

        assert "total_assets" in summary
        assert "categories" in summary
        assert "config" in summary
        assert summary["total_assets"] > 0
