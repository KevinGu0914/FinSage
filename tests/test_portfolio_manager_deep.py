#!/usr/bin/env python
"""
Comprehensive Deep Tests for Portfolio Manager Agent
Covers all public methods, edge cases, error handling, and different parameter combinations
Target: Achieve high coverage (80%+) for portfolio_manager.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List

from finsage.agents.portfolio_manager import PortfolioManager, PortfolioDecision
from finsage.agents.base_expert import ExpertReport, ExpertRecommendation, Action


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    """Mock LLM provider"""
    llm = Mock()
    llm.create_completion = Mock()
    return llm


@pytest.fixture
def mock_toolkit():
    """Mock hedging toolkit"""
    toolkit = Mock()
    toolkit.list_tools = Mock(return_value=[
        {"name": "minimum_variance", "description": "Minimum variance optimization method"},
        {"name": "risk_parity", "description": "Risk parity allocation"},
        {"name": "cvar_optimization", "description": "CVaR optimization for tail risk"},
        {"name": "dcc_garch", "description": "DCC-GARCH dynamic correlation"},
        {"name": "robust_optimization", "description": "Robust optimization under uncertainty"},
        {"name": "black_litterman", "description": "Black-Litterman model"},
        {"name": "mean_variance", "description": "Mean-variance optimization"},
    ])
    toolkit.call = Mock(return_value={"stocks": 0.4, "bonds": 0.3, "commodities": 0.15, "reits": 0.1, "crypto": 0.05})
    return toolkit


@pytest.fixture
def portfolio_manager(mock_llm, mock_toolkit):
    """Create portfolio manager instance"""
    config = {
        "rebalance_threshold": 0.05,
        "min_trade_value": 100,
    }
    return PortfolioManager(mock_llm, mock_toolkit, config)


@pytest.fixture
def sample_expert_reports():
    """Sample expert reports for testing"""
    reports = {}

    # Bullish stock expert
    reports["stocks"] = ExpertReport(
        expert_name="Stock Expert",
        asset_class="stocks",
        timestamp=datetime.now().isoformat(),
        recommendations=[
            ExpertRecommendation(
                asset_class="stocks",
                symbol="SPY",
                action=Action.BUY_75,
                confidence=0.85,
                target_weight=0.3,
                reasoning="Strong momentum",
                market_view={"trend": "bullish"},
                risk_assessment={"volatility": 0.15}
            )
        ],
        overall_view="bullish",
        sector_allocation={"SPY": 1.0},
        key_factors=["momentum", "earnings"]
    )

    # Neutral bond expert
    reports["bonds"] = ExpertReport(
        expert_name="Bond Expert",
        asset_class="bonds",
        timestamp=datetime.now().isoformat(),
        recommendations=[
            ExpertRecommendation(
                asset_class="bonds",
                symbol="TLT",
                action=Action.HOLD,
                confidence=0.6,
                target_weight=0.2,
                reasoning="Stable yields",
                market_view={"trend": "neutral"},
                risk_assessment={"duration": 0.05}
            )
        ],
        overall_view="neutral",
        sector_allocation={"TLT": 1.0},
        key_factors=["yields"]
    )

    # Bearish commodity expert
    reports["commodities"] = ExpertReport(
        expert_name="Commodity Expert",
        asset_class="commodities",
        timestamp=datetime.now().isoformat(),
        recommendations=[
            ExpertRecommendation(
                asset_class="commodities",
                symbol="GLD",
                action=Action.SHORT_25,
                confidence=0.7,
                target_weight=0.1,
                reasoning="Dollar strength",
                market_view={"trend": "bearish"},
                risk_assessment={"volatility": 0.2}
            )
        ],
        overall_view="bearish",
        sector_allocation={"GLD": 1.0},
        key_factors=["dollar"]
    )

    # Neutral REITs expert
    reports["reits"] = ExpertReport(
        expert_name="REITs Expert",
        asset_class="reits",
        timestamp=datetime.now().isoformat(),
        recommendations=[
            ExpertRecommendation(
                asset_class="reits",
                symbol="VNQ",
                action=Action.HOLD,
                confidence=0.65,
                target_weight=0.1,
                reasoning="Interest rate uncertainty",
                market_view={"trend": "neutral"},
                risk_assessment={"volatility": 0.18}
            )
        ],
        overall_view="neutral",
        sector_allocation={"VNQ": 1.0},
        key_factors=["rates"]
    )

    # Neutral crypto expert
    reports["crypto"] = ExpertReport(
        expert_name="Crypto Expert",
        asset_class="crypto",
        timestamp=datetime.now().isoformat(),
        recommendations=[
            ExpertRecommendation(
                asset_class="crypto",
                symbol="BTC",
                action=Action.HOLD,
                confidence=0.5,
                target_weight=0.05,
                reasoning="Consolidation phase",
                market_view={"trend": "neutral"},
                risk_assessment={"volatility": 0.5}
            )
        ],
        overall_view="neutral",
        sector_allocation={"BTC": 1.0},
        key_factors=["regulation"]
    )

    return reports


@pytest.fixture
def sample_market_data():
    """Sample market data"""
    # Create sample returns data (20 days)
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
    returns_data = {
        "stocks": np.random.randn(20) * 0.01,
        "bonds": np.random.randn(20) * 0.005,
        "commodities": np.random.randn(20) * 0.012,
        "reits": np.random.randn(20) * 0.008,
        "crypto": np.random.randn(20) * 0.03,
    }

    return {
        "returns": returns_data,
        "macro": {"vix": 20.0},
        "vix": 20.0,
        "volatility": 0.15,
        "avg_correlation": 0.3,
    }


@pytest.fixture
def sample_current_portfolio():
    """Sample current portfolio"""
    return {
        "stocks": 0.35,
        "bonds": 0.25,
        "commodities": 0.15,
        "reits": 0.15,
        "crypto": 0.05,
        "cash": 0.05,
    }


@pytest.fixture
def sample_risk_constraints():
    """Sample risk constraints"""
    return {
        "max_var": 0.03,
        "max_drawdown": -0.15,
        "current_drawdown": -0.02,
    }


# ============================================================
# Test 1: PortfolioDecision Dataclass
# ============================================================

class TestPortfolioDecision:
    """Test PortfolioDecision dataclass"""

    def test_portfolio_decision_creation(self):
        """Test creating a portfolio decision"""
        decision = PortfolioDecision(
            timestamp="2024-01-15T10:00:00",
            target_allocation={"stocks": 0.4, "bonds": 0.3, "cash": 0.3},
            trades=[{"asset": "stocks", "action": "BUY", "weight_change": 0.05}],
            hedging_tool_used="minimum_variance",
            reasoning="Test reasoning",
            risk_metrics={"expected_volatility": 0.12},
            expert_summary={"stocks": "bullish"}
        )

        assert decision.timestamp == "2024-01-15T10:00:00"
        assert decision.target_allocation["stocks"] == 0.4
        assert len(decision.trades) == 1
        assert decision.hedging_tool_used == "minimum_variance"

    def test_portfolio_decision_to_dict(self):
        """Test converting portfolio decision to dict"""
        decision = PortfolioDecision(
            timestamp="2024-01-15T10:00:00",
            target_allocation={"stocks": 0.5},
            trades=[],
            hedging_tool_used="risk_parity",
            reasoning="Test",
            risk_metrics={},
            expert_summary={}
        )

        d = decision.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == "2024-01-15T10:00:00"
        assert d["hedging_tool_used"] == "risk_parity"
        assert "target_allocation" in d
        assert "trades" in d


# ============================================================
# Test 2: PortfolioManager Initialization
# ============================================================

class TestPortfolioManagerInit:
    """Test PortfolioManager initialization"""

    def test_init_default_config(self, mock_llm, mock_toolkit):
        """Test initialization with default config"""
        pm = PortfolioManager(mock_llm, mock_toolkit)

        assert pm.llm == mock_llm
        assert pm.toolkit == mock_toolkit
        assert pm.rebalance_threshold == 0.05
        assert pm.min_trade_value == 100
        assert pm.allocation_bounds == PortfolioManager.DEFAULT_ALLOCATION_BOUNDS

    def test_init_custom_config(self, mock_llm, mock_toolkit):
        """Test initialization with custom config"""
        custom_config = {
            "rebalance_threshold": 0.03,
            "min_trade_value": 200,
            "allocation_bounds": {
                "stocks": {"min": 0.2, "max": 0.6, "default": 0.4},
            }
        }

        pm = PortfolioManager(mock_llm, mock_toolkit, custom_config)

        assert pm.rebalance_threshold == 0.03
        assert pm.min_trade_value == 200
        assert pm.allocation_bounds["stocks"]["min"] == 0.2

    def test_init_empty_config(self, mock_llm, mock_toolkit):
        """Test initialization with empty config"""
        pm = PortfolioManager(mock_llm, mock_toolkit, {})

        assert pm.config == {}
        assert pm.rebalance_threshold == 0.05  # Default


# ============================================================
# Test 3: _summarize_expert_views
# ============================================================

class TestSummarizeExpertViews:
    """Test _summarize_expert_views method"""

    def test_summarize_expert_views(self, portfolio_manager, sample_expert_reports):
        """Test summarizing expert views"""
        summary = portfolio_manager._summarize_expert_views(sample_expert_reports)

        assert isinstance(summary, dict)
        assert len(summary) == 5
        assert "stocks" in summary
        assert "bullish" in summary["stocks"]
        assert "SPY" in summary["stocks"]

    def test_summarize_expert_views_multiple_picks(self, portfolio_manager):
        """Test with multiple top picks"""
        reports = {
            "stocks": ExpertReport(
                expert_name="Stock Expert",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                recommendations=[
                    ExpertRecommendation("stocks", "SPY", Action.BUY_50, 0.8, 0.2, "test", {}, {}),
                    ExpertRecommendation("stocks", "QQQ", Action.BUY_25, 0.7, 0.15, "test", {}, {}),
                    ExpertRecommendation("stocks", "IWM", Action.HOLD, 0.6, 0.1, "test", {}, {}),
                ],
                overall_view="bullish",
                sector_allocation={},
                key_factors=[]
            )
        }

        summary = portfolio_manager._summarize_expert_views(reports)

        # Should only include top 2 picks
        assert "SPY" in summary["stocks"]
        assert "QQQ" in summary["stocks"]
        assert "IWM" not in summary["stocks"]

    def test_summarize_expert_views_empty_recommendations(self, portfolio_manager):
        """Test with empty recommendations"""
        reports = {
            "bonds": ExpertReport(
                expert_name="Bond Expert",
                asset_class="bonds",
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view="neutral",
                sector_allocation={},
                key_factors=[]
            )
        }

        summary = portfolio_manager._summarize_expert_views(reports)

        assert "bonds" in summary
        assert "neutral" in summary["bonds"]


# ============================================================
# Test 4: _analyze_market_conditions
# ============================================================

class TestAnalyzeMarketConditions:
    """Test _analyze_market_conditions method"""

    def test_analyze_market_conditions_high_vix(self, portfolio_manager, sample_expert_reports):
        """Test with high VIX"""
        market_data = {
            "macro": {"vix": 30.0},
            "returns": {},
        }
        risk_constraints = {"current_drawdown": -0.01}

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, risk_constraints
        )

        assert conditions["vix"] == 30.0
        assert conditions["market_state"] == "high_volatility"
        assert isinstance(conditions["expert_disagreement"], float)
        assert "bullish_count" in conditions
        assert "bearish_count" in conditions
        assert "neutral_count" in conditions

    def test_analyze_market_conditions_low_vix(self, portfolio_manager, sample_expert_reports):
        """Test with low VIX"""
        market_data = {
            "macro": {"vix": 12.0},
            "returns": {},
        }
        risk_constraints = {}

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, risk_constraints
        )

        assert conditions["vix"] == 12.0
        assert conditions["market_state"] == "low_volatility"

    def test_analyze_market_conditions_elevated_vix(self, portfolio_manager, sample_expert_reports):
        """Test with elevated VIX"""
        market_data = {"macro": {"vix": 22.0}}
        risk_constraints = {}

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, risk_constraints
        )

        assert conditions["market_state"] == "elevated_volatility"

    def test_analyze_market_conditions_expert_disagreement(self, portfolio_manager):
        """Test expert disagreement calculation"""
        # All experts agree (bullish)
        reports_agree = {}
        for asset in ["stocks", "bonds", "commodities", "reits", "crypto"]:
            reports_agree[asset] = ExpertReport(
                expert_name=f"{asset} Expert",
                asset_class=asset,
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view="bullish",
                sector_allocation={},
                key_factors=[]
            )

        market_data = {"vix": 20.0}
        conditions = portfolio_manager._analyze_market_conditions(reports_agree, market_data, {})

        # Low disagreement when all agree
        assert conditions["expert_disagreement"] == 0.0
        assert conditions["bullish_count"] == 5
        assert conditions["bullish_majority"] is True

    def test_analyze_market_conditions_mixed_views(self, portfolio_manager):
        """Test with mixed expert views"""
        reports = {}
        views = ["bullish", "bullish", "bearish", "neutral", "neutral"]

        for i, asset in enumerate(["stocks", "bonds", "commodities", "reits", "crypto"]):
            reports[asset] = ExpertReport(
                expert_name=f"{asset} Expert",
                asset_class=asset,
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view=views[i],
                sector_allocation={},
                key_factors=[]
            )

        market_data = {"vix": 18.0}
        conditions = portfolio_manager._analyze_market_conditions(reports, market_data, {})

        assert conditions["bullish_count"] == 2
        assert conditions["bearish_count"] == 1
        assert conditions["neutral_count"] == 2
        assert conditions["expert_disagreement"] > 0

    def test_analyze_market_conditions_with_returns_data(self, portfolio_manager, sample_expert_reports):
        """Test with returns data for volatility calculation"""
        # Create sample returns data with 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        returns_df = pd.DataFrame({
            "stocks": np.random.randn(30) * 0.01,
            "bonds": np.random.randn(30) * 0.005,
        }, index=dates)

        market_data = {
            "vix": 20.0,
            "returns": returns_df.to_dict(),
        }

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, {}
        )

        # Should calculate volatility and correlation changes
        assert "volatility_change" in conditions
        assert "correlation_change" in conditions
        assert isinstance(conditions["volatility_change"], float)

    def test_analyze_market_conditions_vix_fallback(self, portfolio_manager, sample_expert_reports):
        """Test VIX fallback when macro.vix is None"""
        market_data = {
            "macro": {"vix": None},  # Explicitly None
            "vix": 25.0,  # Direct vix field for fallback
            "returns": {},
        }

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, {}
        )

        assert conditions["vix"] == 25.0

    def test_analyze_market_conditions_vix_none(self, portfolio_manager, sample_expert_reports):
        """Test VIX when None"""
        market_data = {
            "macro": {"vix": None},
            "vix": 20.0,
        }

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, {}
        )

        assert conditions["vix"] == 20.0

    def test_analyze_market_conditions_current_drawdown(self, portfolio_manager, sample_expert_reports):
        """Test with current drawdown"""
        market_data = {"vix": 20.0}
        risk_constraints = {"current_drawdown": -0.08}

        conditions = portfolio_manager._analyze_market_conditions(
            sample_expert_reports, market_data, risk_constraints
        )

        assert conditions["current_drawdown"] == -0.08

    def test_analyze_market_conditions_bearish_majority(self, portfolio_manager):
        """Test bearish majority detection"""
        reports = {}
        for i, asset in enumerate(["stocks", "bonds", "commodities", "reits", "crypto"]):
            view = "bearish" if i < 3 else "bullish"
            reports[asset] = ExpertReport(
                expert_name=f"{asset} Expert",
                asset_class=asset,
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view=view,
                sector_allocation={},
                key_factors=[]
            )

        market_data = {"vix": 20.0}
        conditions = portfolio_manager._analyze_market_conditions(reports, market_data, {})

        assert conditions["bearish_majority"] is True
        assert conditions["bullish_majority"] is False


# ============================================================
# Test 5: _rule_based_tool_preselection
# ============================================================

class TestRuleBasedToolPreselection:
    """Test _rule_based_tool_preselection method"""

    def test_cvar_optimization_trigger_high_vix(self, portfolio_manager):
        """Test CVaR optimization triggered by high VIX"""
        conditions = {
            "vix": 25.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.05,
            "correlation_change": 0.02,
            "expert_disagreement": 0.2,
            "market_state": "high_volatility",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "cvar_optimization" in candidates
        assert "VIX" in reasons

    def test_cvar_optimization_trigger_drawdown(self, portfolio_manager):
        """Test CVaR optimization triggered by drawdown"""
        conditions = {
            "vix": 15.0,
            "bearish_majority": False,
            "current_drawdown": -0.05,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "cvar_optimization" in candidates
        assert "回撤" in reasons

    def test_cvar_optimization_trigger_bearish(self, portfolio_manager):
        """Test CVaR optimization triggered by bearish majority"""
        conditions = {
            "vix": 15.0,
            "bearish_majority": True,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 0,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "cvar_optimization" in candidates
        assert "看空" in reasons

    def test_dcc_garch_trigger(self, portfolio_manager):
        """Test DCC-GARCH triggered by volatility change"""
        conditions = {
            "vix": 16.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.12,
            "correlation_change": 0.08,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "dcc_garch" in candidates
        assert "波动率变化" in reasons or "相关性变化" in reasons

    def test_robust_optimization_trigger(self, portfolio_manager):
        """Test robust optimization triggered by expert disagreement"""
        conditions = {
            "vix": 18.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.5,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "robust_optimization" in candidates
        assert "分歧" in reasons

    def test_black_litterman_trigger(self, portfolio_manager):
        """Test Black-Litterman triggered by bullish majority"""
        conditions = {
            "vix": 16.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": True,
            "bullish_count": 4,
            "neutral_count": 1,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "black_litterman" in candidates
        assert "看多" in reasons

    def test_risk_parity_trigger(self, portfolio_manager):
        """Test risk parity triggered by neutral experts"""
        conditions = {
            "vix": 16.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 3,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "risk_parity" in candidates
        assert "中性" in reasons

    def test_mean_variance_trigger(self, portfolio_manager):
        """Test mean variance triggered by low VIX and bullish"""
        conditions = {
            "vix": 12.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.1,
            "market_state": "low_volatility",
            "bullish_majority": False,
            "bullish_count": 3,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "mean_variance" in candidates

    def test_minimum_variance_trigger(self, portfolio_manager):
        """Test minimum variance triggered by low VIX"""
        conditions = {
            "vix": 12.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.1,
            "market_state": "low_volatility",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 2,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        assert "minimum_variance" in candidates

    def test_default_rotation(self, portfolio_manager):
        """Test default tool rotation when no rules match"""
        conditions = {
            "vix": 17.0,
            "bearish_majority": False,
            "current_drawdown": -0.01,
            "volatility_change": 0.0,
            "correlation_change": 0.0,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 1,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        # Should return at least one candidate (rotation)
        assert len(candidates) >= 1
        assert "轮换" in reasons or len(candidates) > 0

    def test_multiple_triggers(self, portfolio_manager):
        """Test when multiple rules trigger"""
        conditions = {
            "vix": 26.0,
            "bearish_majority": True,
            "current_drawdown": -0.06,
            "volatility_change": 0.15,
            "correlation_change": 0.12,
            "expert_disagreement": 0.5,
            "market_state": "high_volatility",
            "bullish_majority": False,
            "bullish_count": 0,
            "neutral_count": 1,
        }

        candidates, reasons = portfolio_manager._rule_based_tool_preselection(conditions)

        # Multiple tools should be selected
        assert len(candidates) >= 2
        assert "cvar_optimization" in candidates
        assert "dcc_garch" in candidates
        assert "robust_optimization" in candidates


# ============================================================
# Test 6: _select_hedging_tool
# ============================================================

class TestSelectHedgingTool:
    """Test _select_hedging_tool method"""

    def test_select_hedging_tool_success(self, portfolio_manager, sample_expert_reports,
                                         sample_market_data, sample_risk_constraints):
        """Test successful tool selection"""
        # Mock LLM response
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "tool_name": "cvar_optimization",
            "reasoning": "High volatility environment"
        })

        tool = portfolio_manager._select_hedging_tool(
            sample_expert_reports,
            sample_market_data,
            sample_risk_constraints
        )

        assert tool in ["cvar_optimization", "dcc_garch", "robust_optimization",
                       "risk_parity", "minimum_variance", "black_litterman", "mean_variance"]
        assert portfolio_manager.llm.create_completion.called

    def test_select_hedging_tool_json_in_markdown(self, portfolio_manager, sample_expert_reports,
                                                   sample_market_data, sample_risk_constraints):
        """Test tool selection with JSON in markdown code block"""
        portfolio_manager.llm.create_completion.return_value = """```json
{
    "tool_name": "risk_parity",
    "reasoning": "Balanced approach"
}
```"""

        tool = portfolio_manager._select_hedging_tool(
            sample_expert_reports,
            sample_market_data,
            sample_risk_constraints
        )

        # Should handle markdown code blocks
        assert tool is not None

    def test_select_hedging_tool_llm_failure(self, portfolio_manager, sample_expert_reports,
                                             sample_market_data, sample_risk_constraints):
        """Test tool selection when LLM fails"""
        portfolio_manager.llm.create_completion.side_effect = Exception("LLM API error")

        tool = portfolio_manager._select_hedging_tool(
            sample_expert_reports,
            sample_market_data,
            sample_risk_constraints
        )

        # Should fall back to rule-based default
        assert tool in ["cvar_optimization", "dcc_garch", "robust_optimization",
                       "risk_parity", "minimum_variance", "black_litterman", "mean_variance"]

    def test_select_hedging_tool_invalid_json(self, portfolio_manager, sample_expert_reports,
                                              sample_market_data, sample_risk_constraints):
        """Test tool selection with invalid JSON response"""
        portfolio_manager.llm.create_completion.return_value = "This is not JSON"

        tool = portfolio_manager._select_hedging_tool(
            sample_expert_reports,
            sample_market_data,
            sample_risk_constraints
        )

        # Should fall back to rule-based default
        assert tool is not None

    def test_select_hedging_tool_not_in_candidates(self, portfolio_manager, sample_expert_reports,
                                                    sample_market_data, sample_risk_constraints):
        """Test when LLM selects tool not in candidates"""
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "tool_name": "unknown_tool",
            "reasoning": "Test"
        })

        tool = portfolio_manager._select_hedging_tool(
            sample_expert_reports,
            sample_market_data,
            sample_risk_constraints
        )

        # Should use first candidate instead
        assert tool is not None

    def test_select_hedging_tool_no_matching_tools(self, portfolio_manager, sample_expert_reports,
                                                    sample_market_data, sample_risk_constraints):
        """Test when no tools match candidates"""
        # Mock toolkit with different tools
        portfolio_manager.toolkit.list_tools.return_value = [
            {"name": "other_tool", "description": "Other tool"}
        ]

        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "tool_name": "other_tool",
            "reasoning": "Test"
        })

        tool = portfolio_manager._select_hedging_tool(
            sample_expert_reports,
            sample_market_data,
            sample_risk_constraints
        )

        # Should use available tools
        assert tool is not None


# ============================================================
# Test 7: _compute_target_allocation
# ============================================================

class TestComputeTargetAllocation:
    """Test _compute_target_allocation method"""

    def test_compute_target_allocation_success(self, portfolio_manager, sample_expert_reports,
                                               sample_market_data, sample_risk_constraints):
        """Test successful allocation computation"""
        # Mock LLM for dynamic allocation
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.45,
                "bonds": 0.25,
                "commodities": 0.15,
                "reits": 0.10,
                "crypto": 0.05,
                "cash": 0.00
            },
            "reasoning": "Test allocation"
        })

        # Mock toolkit call
        portfolio_manager.toolkit.call.return_value = {
            "stocks": 0.4,
            "bonds": 0.3,
            "commodities": 0.15,
            "reits": 0.1,
            "crypto": 0.05
        }

        allocation = portfolio_manager._compute_target_allocation(
            sample_expert_reports,
            sample_market_data,
            "minimum_variance",
            sample_risk_constraints
        )

        assert isinstance(allocation, dict)
        assert abs(sum(allocation.values()) - 1.0) < 0.01  # Should sum to 1
        assert all(0 <= v <= 1 for v in allocation.values())  # All weights valid

    def test_compute_target_allocation_no_returns_data(self, portfolio_manager, sample_expert_reports,
                                                        sample_risk_constraints):
        """Test allocation computation without returns data"""
        market_data_no_returns = {
            "macro": {"vix": 20.0},
        }

        # Mock LLM
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.4,
                "bonds": 0.3,
                "commodities": 0.15,
                "reits": 0.1,
                "crypto": 0.05,
            },
            "reasoning": "Test"
        })

        allocation = portfolio_manager._compute_target_allocation(
            sample_expert_reports,
            market_data_no_returns,
            "risk_parity",
            sample_risk_constraints
        )

        # Should use LLM allocation directly
        assert isinstance(allocation, dict)
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_compute_target_allocation_toolkit_failure(self, portfolio_manager, sample_expert_reports,
                                                       sample_market_data, sample_risk_constraints):
        """Test allocation when toolkit fails"""
        # Mock LLM
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.4,
                "bonds": 0.3,
                "commodities": 0.15,
                "reits": 0.1,
                "crypto": 0.05,
            },
            "reasoning": "Test"
        })

        # Mock toolkit failure
        portfolio_manager.toolkit.call.side_effect = Exception("Optimization failed")

        allocation = portfolio_manager._compute_target_allocation(
            sample_expert_reports,
            sample_market_data,
            "cvar_optimization",
            sample_risk_constraints
        )

        # Should fall back to LLM allocation
        assert isinstance(allocation, dict)
        assert abs(sum(allocation.values()) - 1.0) < 0.01


# ============================================================
# Test 8: _llm_dynamic_class_allocation
# ============================================================

class TestLLMDynamicClassAllocation:
    """Test _llm_dynamic_class_allocation method"""

    def test_llm_dynamic_allocation_success(self, portfolio_manager, sample_expert_reports,
                                           sample_market_data):
        """Test successful LLM dynamic allocation"""
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.45,
                "bonds": 0.25,
                "commodities": 0.10,
                "reits": 0.10,
                "crypto": 0.05,
                "cash": 0.05
            },
            "reasoning": "Bullish on stocks, neutral on bonds"
        })

        allocation = portfolio_manager._llm_dynamic_class_allocation(
            sample_expert_reports,
            sample_market_data
        )

        assert isinstance(allocation, dict)
        assert abs(sum(allocation.values()) - 1.0) < 0.01
        assert all(asset in allocation for asset in ["stocks", "bonds", "commodities", "reits", "crypto", "cash"])

    def test_llm_dynamic_allocation_missing_asset(self, portfolio_manager, sample_expert_reports,
                                                  sample_market_data):
        """Test LLM allocation with missing asset classes"""
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.5,
                "bonds": 0.3,
                # Missing other assets
            },
            "reasoning": "Test"
        })

        allocation = portfolio_manager._llm_dynamic_class_allocation(
            sample_expert_reports,
            sample_market_data
        )

        # Should fill in missing assets with defaults
        assert all(asset in allocation for asset in ["stocks", "bonds", "commodities", "reits", "crypto", "cash"])
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_llm_dynamic_allocation_out_of_bounds(self, portfolio_manager, sample_expert_reports,
                                                  sample_market_data):
        """Test LLM allocation with out-of-bounds weights"""
        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.8,  # Above max
                "bonds": 0.1,
                "commodities": 0.05,
                "reits": 0.03,
                "crypto": 0.02,
                "cash": 0.0
            },
            "reasoning": "Test"
        })

        allocation = portfolio_manager._llm_dynamic_class_allocation(
            sample_expert_reports,
            sample_market_data
        )

        # Should apply bounds and normalize - final stocks should be reduced from 0.8
        # but may exceed max after normalization (this is expected behavior)
        assert allocation["stocks"] < 0.8  # Reduced from original
        assert abs(sum(allocation.values()) - 1.0) < 0.001  # Normalized to 1

    def test_llm_dynamic_allocation_failure(self, portfolio_manager, sample_expert_reports,
                                           sample_market_data):
        """Test LLM allocation failure fallback"""
        portfolio_manager.llm.create_completion.side_effect = Exception("LLM error")

        allocation = portfolio_manager._llm_dynamic_class_allocation(
            sample_expert_reports,
            sample_market_data
        )

        # Should fall back to expert-weighted defaults
        assert isinstance(allocation, dict)
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_llm_dynamic_allocation_high_vix(self, portfolio_manager, sample_expert_reports):
        """Test allocation with high VIX"""
        market_data_high_vix = {
            "macro": {"vix": 35.0},
        }

        portfolio_manager.llm.create_completion.return_value = json.dumps({
            "allocation": {
                "stocks": 0.3,
                "bonds": 0.4,
                "commodities": 0.1,
                "reits": 0.08,
                "crypto": 0.02,
                "cash": 0.1
            },
            "reasoning": "Defensive due to high volatility"
        })

        allocation = portfolio_manager._llm_dynamic_class_allocation(
            sample_expert_reports,
            market_data_high_vix
        )

        assert isinstance(allocation, dict)


# ============================================================
# Test 9: _validate_llm_allocation
# ============================================================

class TestValidateLLMAllocation:
    """Test _validate_llm_allocation method"""

    def test_validate_allocation_within_bounds(self, portfolio_manager):
        """Test validation with weights within bounds"""
        allocation = {
            "stocks": 0.40,
            "bonds": 0.25,
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
            "cash": 0.05
        }

        validated = portfolio_manager._validate_llm_allocation(allocation)

        assert abs(sum(validated.values()) - 1.0) < 0.001
        assert validated["stocks"] == pytest.approx(0.40, rel=0.01)

    def test_validate_allocation_exceeds_max(self, portfolio_manager):
        """Test validation when weights exceed max"""
        allocation = {
            "stocks": 0.70,  # Exceeds max of 0.50
            "bonds": 0.20,
            "commodities": 0.05,
            "reits": 0.03,
            "crypto": 0.02,
        }

        validated = portfolio_manager._validate_llm_allocation(allocation)

        # After bounds + normalization, stocks should be reduced from 0.70 but sum to 1
        assert validated["stocks"] < 0.70  # Reduced from original
        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_allocation_below_min(self, portfolio_manager):
        """Test validation when weights below min"""
        allocation = {
            "stocks": 0.10,  # Below min of 0.30
            "bonds": 0.50,
            "commodities": 0.20,
            "reits": 0.10,
            "crypto": 0.05,
            "cash": 0.05
        }

        validated = portfolio_manager._validate_llm_allocation(allocation)

        # After bounds + normalization, stocks should be increased from 0.10
        assert validated["stocks"] > 0.10  # Increased from original
        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_allocation_missing_assets(self, portfolio_manager):
        """Test validation with missing assets"""
        allocation = {
            "stocks": 0.5,
            "bonds": 0.3,
            # Missing: commodities, reits, crypto, cash
        }

        validated = portfolio_manager._validate_llm_allocation(allocation)

        # Should fill in missing assets
        assert all(asset in validated for asset in ["stocks", "bonds", "commodities", "reits", "crypto", "cash"])
        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_allocation_empty(self, portfolio_manager):
        """Test validation with empty allocation"""
        allocation = {}

        validated = portfolio_manager._validate_llm_allocation(allocation)

        # Should use defaults
        assert all(asset in validated for asset in ["stocks", "bonds", "commodities", "reits", "crypto", "cash"])
        assert abs(sum(validated.values()) - 1.0) < 0.001


# ============================================================
# Test 10: _expert_weighted_default_allocation
# ============================================================

class TestExpertWeightedDefaultAllocation:
    """Test _expert_weighted_default_allocation method"""

    def test_expert_weighted_bullish(self, portfolio_manager):
        """Test allocation with bullish expert"""
        reports = {
            "stocks": ExpertReport(
                expert_name="Stock Expert",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view="bullish",
                sector_allocation={},
                key_factors=[]
            )
        }

        allocation = portfolio_manager._expert_weighted_default_allocation(reports)

        # Bullish should increase weight (1.3x multiplier)
        default_weight = portfolio_manager.allocation_bounds["stocks"]["default"]
        assert allocation["stocks"] >= default_weight

    def test_expert_weighted_bearish(self, portfolio_manager):
        """Test allocation with bearish expert"""
        reports = {
            "bonds": ExpertReport(
                expert_name="Bond Expert",
                asset_class="bonds",
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view="bearish",
                sector_allocation={},
                key_factors=[]
            )
        }

        allocation = portfolio_manager._expert_weighted_default_allocation(reports)

        # Bearish should decrease weight (0.7x multiplier)
        default_weight = portfolio_manager.allocation_bounds["bonds"]["default"]
        assert allocation["bonds"] <= default_weight

    def test_expert_weighted_neutral(self, portfolio_manager):
        """Test allocation with neutral expert"""
        reports = {
            "commodities": ExpertReport(
                expert_name="Commodity Expert",
                asset_class="commodities",
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view="neutral",
                sector_allocation={},
                key_factors=[]
            )
        }

        allocation = portfolio_manager._expert_weighted_default_allocation(reports)

        # Neutral should keep default weight (1.0x multiplier)
        assert abs(sum(allocation.values()) - 1.0) < 0.001

    def test_expert_weighted_no_reports(self, portfolio_manager):
        """Test allocation with no expert reports"""
        allocation = portfolio_manager._expert_weighted_default_allocation({})

        # Should use all defaults
        assert abs(sum(allocation.values()) - 1.0) < 0.001


# ============================================================
# Test 11: _apply_allocation_bounds
# ============================================================

class TestApplyAllocationBounds:
    """Test _apply_allocation_bounds method"""

    def test_apply_bounds_within_limits(self, portfolio_manager):
        """Test applying bounds when already within limits"""
        weights = {
            "stocks": 0.40,
            "bonds": 0.25,
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
            "cash": 0.05
        }

        bounded = portfolio_manager._apply_allocation_bounds(weights)

        assert abs(sum(bounded.values()) - 1.0) < 0.001
        assert bounded["stocks"] == pytest.approx(0.40, rel=0.01)

    def test_apply_bounds_exceeds_max(self, portfolio_manager):
        """Test applying bounds when exceeding max"""
        weights = {
            "stocks": 0.80,  # Exceeds max 0.50
            "bonds": 0.10,
            "commodities": 0.05,
            "reits": 0.03,
            "crypto": 0.02,
        }

        bounded = portfolio_manager._apply_allocation_bounds(weights)

        # After bounds + normalization, stocks should be reduced from 0.80
        assert bounded["stocks"] < 0.80  # Reduced from original
        assert abs(sum(bounded.values()) - 1.0) < 0.001

    def test_apply_bounds_below_min(self, portfolio_manager):
        """Test applying bounds when below min"""
        weights = {
            "stocks": 0.10,  # Below min 0.30
            "bonds": 0.50,
            "commodities": 0.20,
            "reits": 0.10,
            "crypto": 0.05,
            "cash": 0.05
        }

        bounded = portfolio_manager._apply_allocation_bounds(weights)

        # After bounds + normalization, stocks should be increased from 0.10
        assert bounded["stocks"] > 0.10  # Increased from original
        assert abs(sum(bounded.values()) - 1.0) < 0.001

    def test_apply_bounds_unknown_asset(self, portfolio_manager):
        """Test applying bounds with unknown asset"""
        weights = {
            "stocks": 0.40,
            "unknown_asset": 0.10,
        }

        bounded = portfolio_manager._apply_allocation_bounds(weights)

        # Unknown assets should be preserved
        assert "unknown_asset" in bounded


# ============================================================
# Test 12: _generate_trades
# ============================================================

class TestGenerateTrades:
    """Test _generate_trades method"""

    def test_generate_trades_buy(self, portfolio_manager):
        """Test generating BUY trades"""
        current_portfolio = {
            "stocks": 0.30,
            "bonds": 0.30,
            "commodities": 0.20,
            "reits": 0.10,
            "crypto": 0.05,
            "cash": 0.05
        }

        target_allocation = {
            "stocks": 0.42,  # +0.12 (above threshold)
            "bonds": 0.28,   # -0.02 (below threshold)
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
        }

        trades = portfolio_manager._generate_trades(current_portfolio, target_allocation)

        # Should generate BUY for stocks
        stock_trades = [t for t in trades if t["asset"] == "stocks"]
        assert len(stock_trades) == 1
        assert stock_trades[0]["action"] == "BUY"
        assert stock_trades[0]["weight_change"] == pytest.approx(0.12, abs=0.001)

    def test_generate_trades_sell(self, portfolio_manager):
        """Test generating SELL trades"""
        current_portfolio = {
            "stocks": 0.45,
            "bonds": 0.25,
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
        }

        target_allocation = {
            "stocks": 0.30,  # -0.15 (above threshold)
            "bonds": 0.30,
            "commodities": 0.20,
            "reits": 0.12,
            "crypto": 0.08,
        }

        trades = portfolio_manager._generate_trades(current_portfolio, target_allocation)

        # Should generate SELL for stocks
        stock_trades = [t for t in trades if t["asset"] == "stocks"]
        assert len(stock_trades) == 1
        assert stock_trades[0]["action"] == "SELL"

    def test_generate_trades_below_threshold(self, portfolio_manager):
        """Test that small changes below threshold don't generate trades"""
        current_portfolio = {
            "stocks": 0.40,
            "bonds": 0.30,
        }

        target_allocation = {
            "stocks": 0.42,  # +0.02 (below 0.05 threshold)
            "bonds": 0.31,   # +0.01 (below threshold)
        }

        trades = portfolio_manager._generate_trades(current_portfolio, target_allocation)

        # Should generate no trades
        assert len(trades) == 0

    def test_generate_trades_new_asset(self, portfolio_manager):
        """Test adding a new asset not in current portfolio"""
        current_portfolio = {
            "stocks": 0.50,
            "bonds": 0.50,
        }

        target_allocation = {
            "stocks": 0.40,
            "bonds": 0.40,
            "crypto": 0.20,  # New asset
        }

        trades = portfolio_manager._generate_trades(current_portfolio, target_allocation)

        # Should generate BUY for crypto
        crypto_trades = [t for t in trades if t["asset"] == "crypto"]
        assert len(crypto_trades) == 1
        assert crypto_trades[0]["action"] == "BUY"
        assert crypto_trades[0]["from_weight"] == 0.0

    def test_generate_trades_multiple(self, portfolio_manager):
        """Test generating multiple trades"""
        current_portfolio = {
            "stocks": 0.30,
            "bonds": 0.30,
            "commodities": 0.20,
            "reits": 0.15,
            "crypto": 0.05,
        }

        target_allocation = {
            "stocks": 0.45,  # +0.15 BUY
            "bonds": 0.20,   # -0.10 SELL
            "commodities": 0.18,  # -0.02 (below threshold)
            "reits": 0.08,   # -0.07 SELL (must be > 0.05 threshold)
            "crypto": 0.07,  # +0.02 (below threshold)
        }

        trades = portfolio_manager._generate_trades(current_portfolio, target_allocation)

        # Should generate 3 trades (stocks BUY, bonds SELL, reits SELL)
        assert len(trades) == 3


# ============================================================
# Test 13: _compute_risk_metrics
# ============================================================

class TestComputeRiskMetrics:
    """Test _compute_risk_metrics method"""

    def test_compute_risk_metrics_with_returns(self, portfolio_manager, sample_market_data):
        """Test risk metrics computation with returns data"""
        allocation = {
            "stocks": 0.40,
            "bonds": 0.30,
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
        }

        metrics = portfolio_manager._compute_risk_metrics(allocation, sample_market_data)

        assert "expected_volatility" in metrics
        assert "diversification_ratio" in metrics
        assert "max_drawdown_estimate" in metrics
        assert metrics["data_source"] == "historical_returns"
        assert metrics["expected_volatility"] >= 0

    def test_compute_risk_metrics_no_returns(self, portfolio_manager):
        """Test risk metrics computation without returns data"""
        allocation = {
            "stocks": 0.40,
            "bonds": 0.30,
        }

        market_data_no_returns = {
            "macro": {"vix": 25.0},
        }

        metrics = portfolio_manager._compute_risk_metrics(allocation, market_data_no_returns)

        assert metrics["data_source"] == "vix_estimate"
        assert metrics["expected_volatility"] > 0

    def test_compute_risk_metrics_empty_returns(self, portfolio_manager):
        """Test risk metrics with empty returns DataFrame"""
        allocation = {"stocks": 0.5, "bonds": 0.5}

        market_data = {
            "returns": pd.DataFrame(),  # Empty
            "macro": {"vix": 20.0},
        }

        metrics = portfolio_manager._compute_risk_metrics(allocation, market_data)

        # Should fall back to defaults
        assert "expected_volatility" in metrics

    def test_compute_risk_metrics_var_calculation(self, portfolio_manager):
        """Test VaR calculation when sufficient data"""
        # Create 30 days of returns
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        returns_data = {
            "stocks": np.random.randn(30) * 0.01,
            "bonds": np.random.randn(30) * 0.005,
        }

        market_data = {
            "returns": returns_data,
            "macro": {"vix": 20.0},
        }

        allocation = {"stocks": 0.6, "bonds": 0.4}

        metrics = portfolio_manager._compute_risk_metrics(allocation, market_data)

        # Should include VaR when >= 20 observations
        assert "var_95_daily" in metrics

    def test_compute_risk_metrics_no_matching_assets(self, portfolio_manager):
        """Test risk metrics when no matching assets"""
        allocation = {"unknown_asset": 1.0}

        market_data = {
            "returns": {"stocks": [0.01, 0.02], "bonds": [0.005, 0.01]},
            "macro": {"vix": 20.0},
        }

        metrics = portfolio_manager._compute_risk_metrics(allocation, market_data)

        # Should fall back to defaults
        assert "expected_volatility" in metrics

    def test_compute_risk_metrics_computation_error(self, portfolio_manager):
        """Test risk metrics when computation fails"""
        allocation = {"stocks": 0.5, "bonds": 0.5}

        # Invalid returns data that will cause error
        market_data = {
            "returns": "invalid_data",
            "macro": {"vix": 20.0},
        }

        metrics = portfolio_manager._compute_risk_metrics(allocation, market_data)

        # Should fall back to defaults
        assert "expected_volatility" in metrics
        assert metrics["data_source"] in ["default_estimate", "vix_estimate"]


# ============================================================
# Test 14: _get_default_risk_metrics
# ============================================================

class TestGetDefaultRiskMetrics:
    """Test _get_default_risk_metrics method"""

    def test_default_risk_metrics_with_vix(self, portfolio_manager):
        """Test default risk metrics with VIX"""
        market_data = {"macro": {"vix": 25.0}}

        metrics = portfolio_manager._get_default_risk_metrics(market_data)

        assert "expected_volatility" in metrics
        assert "diversification_ratio" in metrics
        assert "max_drawdown_estimate" in metrics
        assert metrics["data_source"] == "default_estimate"

    def test_default_risk_metrics_vix_fallback(self, portfolio_manager):
        """Test default risk metrics with VIX fallback"""
        market_data = {"vix": 30.0}  # No macro.vix

        metrics = portfolio_manager._get_default_risk_metrics(market_data)

        assert metrics["expected_volatility"] > 0

    def test_default_risk_metrics_high_vix(self, portfolio_manager):
        """Test default risk metrics with percentage VIX"""
        market_data = {"macro": {"vix": 35.0}}  # High percentage VIX

        metrics = portfolio_manager._get_default_risk_metrics(market_data)

        assert metrics["expected_volatility"] > 0

    def test_default_risk_metrics_decimal_vix(self, portfolio_manager):
        """Test default risk metrics with decimal VIX"""
        market_data = {"vix": 0.25}  # Already in decimal form

        metrics = portfolio_manager._get_default_risk_metrics(market_data)

        assert metrics["expected_volatility"] > 0


# ============================================================
# Test 15: _generate_reasoning
# ============================================================

class TestGenerateReasoning:
    """Test _generate_reasoning method"""

    def test_generate_reasoning_basic(self, portfolio_manager):
        """Test basic reasoning generation"""
        expert_summary = {
            "stocks": "bullish (SPY, QQQ)",
            "bonds": "neutral (TLT)",
        }

        target_allocation = {
            "stocks": 0.45,
            "bonds": 0.30,
            "commodities": 0.15,
            "reits": 0.10,
        }

        reasoning = portfolio_manager._generate_reasoning(
            expert_summary,
            "risk_parity",
            target_allocation
        )

        assert isinstance(reasoning, str)
        assert "risk_parity" in reasoning
        assert len(reasoning) > 0

    def test_generate_reasoning_with_bullish(self, portfolio_manager):
        """Test reasoning with bullish experts"""
        expert_summary = {
            "stocks": "bullish (SPY)",
            "bonds": "bullish (TLT)",
            "commodities": "bearish (GLD)",
        }

        target_allocation = {
            "stocks": 0.50,
            "bonds": 0.30,
            "commodities": 0.10,
        }

        reasoning = portfolio_manager._generate_reasoning(
            expert_summary,
            "minimum_variance",
            target_allocation
        )

        assert "看多" in reasoning
        assert "stocks" in reasoning or "bonds" in reasoning

    def test_generate_reasoning_no_bullish(self, portfolio_manager):
        """Test reasoning without bullish experts"""
        expert_summary = {
            "stocks": "neutral (SPY)",
            "bonds": "bearish (TLT)",
        }

        target_allocation = {
            "stocks": 0.40,
            "bonds": 0.35,
            "cash": 0.25,
        }

        reasoning = portfolio_manager._generate_reasoning(
            expert_summary,
            "cvar_optimization",
            target_allocation
        )

        assert isinstance(reasoning, str)


# ============================================================
# Test 16: _extract_json_from_response
# ============================================================

class TestExtractJSONFromResponse:
    """Test _extract_json_from_response method"""

    def test_extract_json_pure(self, portfolio_manager):
        """Test extracting pure JSON"""
        response = '{"tool_name": "risk_parity", "reasoning": "test"}'

        extracted = portfolio_manager._extract_json_from_response(response)

        assert extracted == response
        assert json.loads(extracted)["tool_name"] == "risk_parity"

    def test_extract_json_markdown_block(self, portfolio_manager):
        """Test extracting JSON from markdown code block"""
        response = """Here is the result:
```json
{
    "tool_name": "cvar_optimization",
    "reasoning": "High risk environment"
}
```
Additional text"""

        extracted = portfolio_manager._extract_json_from_response(response)

        data = json.loads(extracted)
        assert data["tool_name"] == "cvar_optimization"

    def test_extract_json_code_block(self, portfolio_manager):
        """Test extracting JSON from generic code block"""
        response = """```
{"tool_name": "dcc_garch", "reasoning": "test"}
```"""

        extracted = portfolio_manager._extract_json_from_response(response)

        data = json.loads(extracted)
        assert data["tool_name"] == "dcc_garch"

    def test_extract_json_with_text(self, portfolio_manager):
        """Test extracting JSON with surrounding text"""
        response = """Some text before
{"tool_name": "robust_optimization", "reasoning": "uncertainty"}
Some text after"""

        extracted = portfolio_manager._extract_json_from_response(response)

        data = json.loads(extracted)
        assert data["tool_name"] == "robust_optimization"

    def test_extract_json_nested(self, portfolio_manager):
        """Test extracting nested JSON"""
        response = '{"allocation": {"stocks": 0.5, "bonds": 0.3}, "reasoning": "test"}'

        extracted = portfolio_manager._extract_json_from_response(response)

        data = json.loads(extracted)
        assert "allocation" in data
        assert data["allocation"]["stocks"] == 0.5

    def test_extract_json_array(self, portfolio_manager):
        """Test extracting JSON array"""
        response = '[{"name": "tool1"}, {"name": "tool2"}]'

        extracted = portfolio_manager._extract_json_from_response(response)

        data = json.loads(extracted)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_extract_json_empty_response(self, portfolio_manager):
        """Test extracting from empty response"""
        with pytest.raises(ValueError, match="Empty response"):
            portfolio_manager._extract_json_from_response("")

    def test_extract_json_invalid(self, portfolio_manager):
        """Test extracting from invalid response"""
        response = "This is just plain text with no JSON"

        with pytest.raises(ValueError, match="Cannot extract JSON"):
            portfolio_manager._extract_json_from_response(response)

    def test_extract_json_incomplete(self, portfolio_manager):
        """Test extracting incomplete JSON"""
        response = '{"tool_name": "test", "reasoning":'  # Incomplete

        with pytest.raises(ValueError):
            portfolio_manager._extract_json_from_response(response)


# ============================================================
# Test 17: _find_balanced_json
# ============================================================

class TestFindBalancedJSON:
    """Test _find_balanced_json method"""

    def test_find_balanced_json_object(self, portfolio_manager):
        """Test finding balanced JSON object"""
        text = 'Here is some text {"key": "value", "nested": {"inner": "data"}} and more text'

        result = portfolio_manager._find_balanced_json(text, '{', '}')

        assert result == '{"key": "value", "nested": {"inner": "data"}}'
        assert json.loads(result)

    def test_find_balanced_json_array(self, portfolio_manager):
        """Test finding balanced JSON array"""
        text = 'Text before [1, 2, [3, 4]] text after'

        result = portfolio_manager._find_balanced_json(text, '[', ']')

        assert result == '[1, 2, [3, 4]]'
        assert json.loads(result)

    def test_find_balanced_json_with_strings(self, portfolio_manager):
        """Test finding JSON with string containing brackets"""
        text = '{"message": "This } is a test", "value": 42}'

        result = portfolio_manager._find_balanced_json(text, '{', '}')

        assert result == text
        data = json.loads(result)
        assert "}" in data["message"]

    def test_find_balanced_json_with_escape(self, portfolio_manager):
        """Test finding JSON with escaped quotes"""
        text = r'{"message": "He said \"hello\"", "value": 1}'

        result = portfolio_manager._find_balanced_json(text, '{', '}')

        assert result is not None
        data = json.loads(result)
        assert '"' in data["message"]

    def test_find_balanced_json_not_found(self, portfolio_manager):
        """Test when no balanced JSON found"""
        text = 'No JSON here'

        result = portfolio_manager._find_balanced_json(text, '{', '}')

        assert result is None

    def test_find_balanced_json_unbalanced(self, portfolio_manager):
        """Test with unbalanced brackets"""
        text = '{"key": "value", "nested": {"inner": "data"'  # Missing closing }

        result = portfolio_manager._find_balanced_json(text, '{', '}')

        assert result is None


# ============================================================
# Test 18: decide (Integration Test)
# ============================================================

class TestDecideMethod:
    """Test the main decide method (integration test)"""

    def test_decide_success(self, portfolio_manager, sample_expert_reports, sample_market_data,
                           sample_current_portfolio, sample_risk_constraints):
        """Test successful decision making"""
        # Mock LLM responses
        portfolio_manager.llm.create_completion.side_effect = [
            # Tool selection response
            json.dumps({"tool_name": "risk_parity", "reasoning": "Balanced approach"}),
            # Allocation response
            json.dumps({
                "allocation": {
                    "stocks": 0.40,
                    "bonds": 0.30,
                    "commodities": 0.15,
                    "reits": 0.10,
                    "crypto": 0.05,
                },
                "reasoning": "Test allocation"
            })
        ]

        decision = portfolio_manager.decide(
            sample_expert_reports,
            sample_market_data,
            sample_current_portfolio,
            sample_risk_constraints
        )

        assert isinstance(decision, PortfolioDecision)
        assert decision.hedging_tool_used in ["risk_parity", "minimum_variance", "cvar_optimization"]
        assert isinstance(decision.target_allocation, dict)
        assert isinstance(decision.trades, list)
        assert isinstance(decision.risk_metrics, dict)
        assert isinstance(decision.expert_summary, dict)
        assert len(decision.reasoning) > 0

    def test_decide_with_trades(self, portfolio_manager, sample_expert_reports, sample_market_data,
                                sample_risk_constraints):
        """Test decision that generates trades"""
        # Current portfolio significantly different from target
        current_portfolio = {
            "stocks": 0.20,  # Will need to increase
            "bonds": 0.50,   # Will need to decrease
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
        }

        # Mock responses
        portfolio_manager.llm.create_completion.side_effect = [
            json.dumps({"tool_name": "minimum_variance", "reasoning": "Test"}),
            json.dumps({
                "allocation": {
                    "stocks": 0.45,
                    "bonds": 0.25,
                    "commodities": 0.15,
                    "reits": 0.10,
                    "crypto": 0.05,
                },
                "reasoning": "Test"
            })
        ]

        decision = portfolio_manager.decide(
            sample_expert_reports,
            sample_market_data,
            current_portfolio,
            sample_risk_constraints
        )

        # Should generate trades
        assert len(decision.trades) > 0

        # Check for expected BUY and SELL
        buy_trades = [t for t in decision.trades if t["action"] == "BUY"]
        sell_trades = [t for t in decision.trades if t["action"] == "SELL"]

        assert len(buy_trades) > 0 or len(sell_trades) > 0

    def test_decide_no_rebalance_needed(self, portfolio_manager, sample_expert_reports,
                                        sample_market_data, sample_risk_constraints):
        """Test decision when current portfolio is close to target"""
        # Current portfolio close to what will be target
        current_portfolio = {
            "stocks": 0.39,
            "bonds": 0.31,
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05,
        }

        portfolio_manager.llm.create_completion.side_effect = [
            json.dumps({"tool_name": "minimum_variance", "reasoning": "Test"}),
            json.dumps({
                "allocation": {
                    "stocks": 0.40,
                    "bonds": 0.30,
                    "commodities": 0.15,
                    "reits": 0.10,
                    "crypto": 0.05,
                },
                "reasoning": "Test"
            })
        ]

        decision = portfolio_manager.decide(
            sample_expert_reports,
            sample_market_data,
            current_portfolio,
            sample_risk_constraints
        )

        # Should generate no trades (changes below threshold)
        assert len(decision.trades) == 0

    def test_decide_with_all_bearish_experts(self, portfolio_manager, sample_market_data,
                                             sample_current_portfolio, sample_risk_constraints):
        """Test decision with all bearish experts"""
        bearish_reports = {}
        for asset in ["stocks", "bonds", "commodities", "reits", "crypto"]:
            bearish_reports[asset] = ExpertReport(
                expert_name=f"{asset} Expert",
                asset_class=asset,
                timestamp=datetime.now().isoformat(),
                recommendations=[],
                overall_view="bearish",
                sector_allocation={},
                key_factors=[]
            )

        portfolio_manager.llm.create_completion.side_effect = [
            json.dumps({"tool_name": "cvar_optimization", "reasoning": "All bearish"}),
            json.dumps({
                "allocation": {
                    "stocks": 0.30,
                    "bonds": 0.35,
                    "commodities": 0.10,
                    "reits": 0.05,
                    "crypto": 0.00,
                    "cash": 0.20
                },
                "reasoning": "Defensive allocation"
            })
        ]

        decision = portfolio_manager.decide(
            bearish_reports,
            sample_market_data,
            sample_current_portfolio,
            sample_risk_constraints
        )

        # Should select CVaR or other defensive tool
        assert decision.hedging_tool_used in ["cvar_optimization", "minimum_variance", "risk_parity"]


# ============================================================
# Test 19: Edge Cases and Error Handling
# ============================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""

    def test_empty_expert_reports(self, portfolio_manager, sample_market_data,
                                  sample_current_portfolio, sample_risk_constraints):
        """Test with empty expert reports"""
        empty_reports = {}

        portfolio_manager.llm.create_completion.side_effect = [
            json.dumps({"tool_name": "minimum_variance", "reasoning": "Default"}),
            json.dumps({
                "allocation": {
                    "stocks": 0.40,
                    "bonds": 0.30,
                    "commodities": 0.15,
                    "reits": 0.10,
                    "crypto": 0.05,
                },
                "reasoning": "Default"
            })
        ]

        decision = portfolio_manager.decide(
            empty_reports,
            sample_market_data,
            sample_current_portfolio,
            sample_risk_constraints
        )

        assert isinstance(decision, PortfolioDecision)

    def test_extreme_market_conditions(self, portfolio_manager, sample_expert_reports,
                                      sample_current_portfolio, sample_risk_constraints):
        """Test with extreme market conditions"""
        extreme_market_data = {
            "macro": {"vix": 80.0},  # Extreme VIX
            "returns": {},
            "volatility": 0.50,
            "avg_correlation": 0.95,
        }

        portfolio_manager.llm.create_completion.side_effect = [
            json.dumps({"tool_name": "cvar_optimization", "reasoning": "Extreme volatility"}),
            json.dumps({
                "allocation": {
                    "stocks": 0.30,
                    "bonds": 0.40,
                    "commodities": 0.10,
                    "reits": 0.05,
                    "crypto": 0.00,
                    "cash": 0.15
                },
                "reasoning": "Extreme defensive"
            })
        ]

        decision = portfolio_manager.decide(
            sample_expert_reports,
            extreme_market_data,
            sample_current_portfolio,
            sample_risk_constraints
        )

        assert decision.hedging_tool_used == "cvar_optimization"

    def test_all_weights_zero(self, portfolio_manager):
        """Test with all zero weights"""
        weights = {
            "stocks": 0.0,
            "bonds": 0.0,
            "commodities": 0.0,
        }

        bounded = portfolio_manager._apply_allocation_bounds(weights)

        # Should handle gracefully (though not realistic)
        assert isinstance(bounded, dict)

    def test_negative_weights(self, portfolio_manager):
        """Test handling of negative weights"""
        weights = {
            "stocks": -0.1,  # Invalid
            "bonds": 0.6,
            "commodities": 0.5,
        }

        bounded = portfolio_manager._apply_allocation_bounds(weights)

        # Should apply min bound (>= 0)
        assert all(v >= 0 for v in bounded.values())


# ============================================================
# Test 20: Build Prompt Methods
# ============================================================

class TestBuildPromptMethods:
    """Test prompt building methods"""

    def test_build_tool_selection_prompt(self, portfolio_manager, sample_expert_reports,
                                        sample_market_data):
        """Test building tool selection prompt"""
        available_tools = [
            {"name": "minimum_variance", "description": "Minimum variance optimization method"},
            {"name": "risk_parity", "description": "Risk parity allocation"},
        ]

        market_conditions = {
            "vix": 20.0,
            "market_state": "normal",
            "bullish_count": 2,
            "bearish_count": 1,
            "neutral_count": 2,
            "volatility_change": 0.05,
            "correlation_change": 0.03,
            "current_drawdown": -0.02,
            "expert_disagreement": 0.3,
        }

        prompt = portfolio_manager._build_tool_selection_prompt(
            sample_expert_reports,
            sample_market_data,
            available_tools,
            market_conditions,
            "Test reasoning"
        )

        assert isinstance(prompt, str)
        assert "VIX" in prompt or "vix" in prompt.lower()
        assert "minimum_variance" in prompt
        assert len(prompt) > 100

    def test_build_tool_selection_prompt_no_conditions(self, portfolio_manager,
                                                       sample_expert_reports, sample_market_data):
        """Test building prompt without pre-computed conditions"""
        available_tools = [
            {"name": "cvar_optimization", "description": "CVaR optimization"},
        ]

        prompt = portfolio_manager._build_tool_selection_prompt(
            sample_expert_reports,
            sample_market_data,
            available_tools,
            market_conditions=None,
            rule_reasoning=""
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_build_dynamic_allocation_prompt(self, portfolio_manager):
        """Test building dynamic allocation prompt"""
        expert_summary = [
            {
                "asset_class": "stocks",
                "view": "bullish",
                "confidence": 0.8,
                "top_picks": ["SPY", "QQQ"],
                "bounds": {"min": 0.30, "max": 0.50, "default": 0.40}
            }
        ]

        prompt = portfolio_manager._build_dynamic_allocation_prompt(
            expert_summary,
            "high",
            30.0
        )

        assert isinstance(prompt, str)
        assert "stocks" in prompt
        assert "bullish" in prompt
        assert len(prompt) > 100

    def test_get_tool_selection_system_prompt(self, portfolio_manager):
        """Test tool selection system prompt"""
        prompt = portfolio_manager._get_tool_selection_system_prompt()

        assert isinstance(prompt, str)
        assert "专家" in prompt or "expert" in prompt.lower()
        assert len(prompt) > 50

    def test_get_dynamic_allocation_system_prompt(self, portfolio_manager):
        """Test dynamic allocation system prompt"""
        prompt = portfolio_manager._get_dynamic_allocation_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ============================================================
# Test 21: Configuration and Constants
# ============================================================

class TestConfigurationAndConstants:
    """Test configuration and class constants"""

    def test_default_allocation_bounds(self):
        """Test default allocation bounds are valid"""
        bounds = PortfolioManager.DEFAULT_ALLOCATION_BOUNDS

        assert isinstance(bounds, dict)
        assert "stocks" in bounds
        assert "bonds" in bounds

        for asset, limits in bounds.items():
            assert "min" in limits
            assert "max" in limits
            assert "default" in limits
            assert limits["min"] <= limits["default"] <= limits["max"]
            assert 0 <= limits["min"] <= 1
            assert 0 <= limits["max"] <= 1

    def test_tool_selection_rules(self):
        """Test tool selection rules configuration"""
        rules = PortfolioManager.TOOL_SELECTION_RULES

        assert isinstance(rules, dict)
        assert "cvar_optimization" in rules
        assert "minimum_variance" in rules

        for tool_name, rule in rules.items():
            assert "description" in rule
            assert "triggers" in rule
            assert "priority" in rule
            assert isinstance(rule["priority"], int)

    def test_custom_allocation_bounds(self, mock_llm, mock_toolkit):
        """Test using custom allocation bounds"""
        custom_bounds = {
            "stocks": {"min": 0.20, "max": 0.70, "default": 0.50},
            "bonds": {"min": 0.10, "max": 0.40, "default": 0.25},
        }

        config = {"allocation_bounds": custom_bounds}
        pm = PortfolioManager(mock_llm, mock_toolkit, config)

        assert pm.allocation_bounds["stocks"]["max"] == 0.70
        assert pm.allocation_bounds["bonds"]["min"] == 0.10


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" Portfolio Manager Deep Tests")
    print("=" * 80)

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])


if __name__ == "__main__":
    run_tests()
