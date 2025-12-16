#!/usr/bin/env python
"""
Deep Orchestrator Tests - 深度协调器测试
Comprehensive test coverage for FinSageOrchestrator

Coverage focus:
- All public and private methods
- Edge cases and error handling
- Different parameter combinations
- Checkpoint save/load/resume functionality
- Date generation with holidays
- Risk adjustments and emergency rebalancing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from typing import Dict, Any
import json
import pickle
import tempfile
import shutil


# ============================================================
# Test 1: CheckpointState Deep Tests
# ============================================================

class TestCheckpointStateDeep:
    """深度测试 CheckpointState"""

    def test_checkpoint_state_complete_fields(self):
        """测试完整字段的CheckpointState"""
        from finsage.core.orchestrator import CheckpointState

        state = CheckpointState(
            start_date="2024-01-01",
            end_date="2024-12-31",
            rebalance_frequency="weekly",
            current_date_index=10,
            total_dates=52,
            results={"decisions": [], "risk_assessments": []},
            portfolio_state={"cash": 100000, "positions": {}},
            timestamp="2024-03-15T10:30:00",
            risk_controller_state={"peak_value": 1.1, "max_drawdown_history": 0.05}
        )

        assert state.start_date == "2024-01-01"
        assert state.end_date == "2024-12-31"
        assert state.rebalance_frequency == "weekly"
        assert state.current_date_index == 10
        assert state.total_dates == 52
        assert "decisions" in state.results
        assert state.portfolio_state["cash"] == 100000
        assert state.risk_controller_state["peak_value"] == 1.1

    def test_checkpoint_state_optional_risk_controller(self):
        """测试可选的risk_controller_state"""
        from finsage.core.orchestrator import CheckpointState

        state = CheckpointState(
            start_date="2024-01-01",
            end_date="2024-12-31",
            rebalance_frequency="monthly",
            current_date_index=0,
            total_dates=12,
            results={},
            portfolio_state={},
            timestamp="2024-01-01T00:00:00"
        )

        assert state.risk_controller_state is None

    def test_checkpoint_state_to_dict_complete(self):
        """测试完整的to_dict转换"""
        from finsage.core.orchestrator import CheckpointState

        state = CheckpointState(
            start_date="2024-01-01",
            end_date="2024-06-30",
            rebalance_frequency="daily",
            current_date_index=50,
            total_dates=126,
            results={"test": "data"},
            portfolio_state={"value": 1000000},
            timestamp="2024-03-01T12:00:00",
            risk_controller_state={"peak_value": 1.0}
        )

        d = state.to_dict()

        assert isinstance(d, dict)
        assert d["start_date"] == "2024-01-01"
        assert d["current_date_index"] == 50
        assert d["risk_controller_state"]["peak_value"] == 1.0


# ============================================================
# Test 2: FinSageOrchestrator Initialization Tests
# ============================================================

class TestOrchestratorInitialization:
    """测试协调器初始化"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_init_with_default_config(self, mock_llm, mock_loader, mock_market,
                                      mock_hedging, mock_screener, mock_env,
                                      mock_stock, mock_bond, mock_commodity,
                                      mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试使用默认配置初始化"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            assert orchestrator.config is not None
            assert orchestrator.checkpoint_dir == tmpdir
            assert os.path.exists(tmpdir)
            mock_llm.assert_called_once()
            mock_loader.assert_called_once()

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_init_with_custom_config(self, mock_llm, mock_loader, mock_market,
                                     mock_hedging, mock_screener, mock_env,
                                     mock_stock, mock_bond, mock_commodity,
                                     mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试使用自定义配置初始化"""
        from finsage.core.orchestrator import FinSageOrchestrator
        from finsage.config import FinSageConfig

        config = FinSageConfig()
        config.trading.initial_capital = 2_000_000.0

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(config=config, checkpoint_dir=tmpdir)

            assert orchestrator.config.trading.initial_capital == 2_000_000.0

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_init_creates_checkpoint_directory(self, mock_llm, mock_loader, mock_market,
                                               mock_hedging, mock_screener, mock_env,
                                               mock_stock, mock_bond, mock_commodity,
                                               mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试初始化创建checkpoint目录"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "checkpoints", "test")

            orchestrator = FinSageOrchestrator(checkpoint_dir=checkpoint_dir)

            assert os.path.exists(checkpoint_dir)

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_init_components_creates_all_experts(self, mock_llm, mock_loader, mock_market,
                                                 mock_hedging, mock_screener, mock_env,
                                                 mock_stock, mock_bond, mock_commodity,
                                                 mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试_init_components创建所有专家"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            assert "stocks" in orchestrator.experts
            assert "bonds" in orchestrator.experts
            assert "commodities" in orchestrator.experts
            assert "reits" in orchestrator.experts
            assert "crypto" in orchestrator.experts
            assert len(orchestrator.experts) == 5


# ============================================================
# Test 3: Date Generation Tests
# ============================================================

class TestDateGeneration:
    """测试日期生成功能"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_generate_daily_dates(self, mock_llm, mock_loader, mock_market,
                                   mock_hedging, mock_screener, mock_env,
                                   mock_stock, mock_bond, mock_commodity,
                                   mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试生成每日日期"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            dates = orchestrator._generate_rebalance_dates(
                "2024-01-02", "2024-01-05", "daily"
            )

            # Should skip weekends
            assert len(dates) > 0
            # Check no weekends (2024-01-02 is Tuesday, 01-05 is Friday)
            for date_str in dates:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                assert dt.weekday() < 5  # Monday=0, Friday=4

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_generate_weekly_dates(self, mock_llm, mock_loader, mock_market,
                                    mock_hedging, mock_screener, mock_env,
                                    mock_stock, mock_bond, mock_commodity,
                                    mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试生成每周日期"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            dates = orchestrator._generate_rebalance_dates(
                "2024-01-01", "2024-01-31", "weekly"
            )

            assert len(dates) > 0
            assert len(dates) <= 5  # At most 5 weeks in January

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_generate_monthly_dates(self, mock_llm, mock_loader, mock_market,
                                     mock_hedging, mock_screener, mock_env,
                                     mock_stock, mock_bond, mock_commodity,
                                     mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试生成每月日期"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            dates = orchestrator._generate_rebalance_dates(
                "2024-01-01", "2024-12-31", "monthly"
            )

            assert len(dates) > 0
            assert len(dates) <= 13  # At most ~12-13 monthly intervals

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_generate_dates_skips_new_year(self, mock_llm, mock_loader, mock_market,
                                           mock_hedging, mock_screener, mock_env,
                                           mock_stock, mock_bond, mock_commodity,
                                           mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试跳过新年假日"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            dates = orchestrator._generate_rebalance_dates(
                "2023-12-29", "2024-01-05", "daily"
            )

            # Should not include 2024-01-01 (New Year's Day)
            assert "2024-01-01" not in dates

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_generate_dates_skips_christmas(self, mock_llm, mock_loader, mock_market,
                                            mock_hedging, mock_screener, mock_env,
                                            mock_stock, mock_bond, mock_commodity,
                                            mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试跳过圣诞节假日"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            dates = orchestrator._generate_rebalance_dates(
                "2024-12-20", "2024-12-31", "daily"
            )

            # Should not include 2024-12-25 (Christmas)
            assert "2024-12-25" not in dates


# ============================================================
# Test 4: Holiday Calculation Tests
# ============================================================

class TestHolidayCalculation:
    """测试假日计算功能"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_get_us_market_holidays_2024(self, mock_llm, mock_loader, mock_market,
                                         mock_hedging, mock_screener, mock_env,
                                         mock_stock, mock_bond, mock_commodity,
                                         mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试获取2024年美国市场假日"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            holidays = orchestrator._get_us_market_holidays(2024, 2024)

            # Check major holidays
            assert any("New Year" in name for name in holidays.values())
            assert any("Independence Day" in name for name in holidays.values())
            assert any("Christmas" in name for name in holidays.values())
            assert any("Thanksgiving" in name for name in holidays.values())

            # Should have multiple holidays
            assert len(holidays) >= 8

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_get_nth_weekday_mlk_day(self, mock_llm, mock_loader, mock_market,
                                     mock_hedging, mock_screener, mock_env,
                                     mock_stock, mock_bond, mock_commodity,
                                     mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试计算第n个工作日(MLK Day - 1月第3个周一)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            # MLK Day 2024: January 15 (3rd Monday)
            mlk_day = orchestrator._get_nth_weekday(2024, 1, 0, 3)  # 0=Monday, 3rd

            assert mlk_day.month == 1
            assert mlk_day.weekday() == 0  # Monday
            assert mlk_day.day == 15

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_get_last_weekday_memorial_day(self, mock_llm, mock_loader, mock_market,
                                           mock_hedging, mock_screener, mock_env,
                                           mock_stock, mock_bond, mock_commodity,
                                           mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试计算最后一个工作日(Memorial Day - 5月最后一个周一)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            # Memorial Day 2024: May 27 (last Monday)
            memorial_day = orchestrator._get_last_weekday(2024, 5, 0)  # 0=Monday

            assert memorial_day.month == 5
            assert memorial_day.weekday() == 0  # Monday
            assert memorial_day.day == 27

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_get_last_weekday_december(self, mock_llm, mock_loader, mock_market,
                                       mock_hedging, mock_screener, mock_env,
                                       mock_stock, mock_bond, mock_commodity,
                                       mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试12月最后一个周一(边界情况)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            last_monday = orchestrator._get_last_weekday(2024, 12, 0)

            assert last_monday.month == 12
            assert last_monday.weekday() == 0


# ============================================================
# Test 5: Checkpoint Save/Load/Resume Tests
# ============================================================

class TestCheckpointOperations:
    """测试断点操作"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_get_checkpoint_path(self, mock_llm, mock_loader, mock_market,
                                 mock_hedging, mock_screener, mock_env,
                                 mock_stock, mock_bond, mock_commodity,
                                 mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试获取checkpoint路径"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            path = orchestrator._get_checkpoint_path()

            assert path == os.path.join(tmpdir, "finsage_checkpoint.pkl")

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_save_checkpoint(self, mock_llm, mock_loader, mock_market,
                            mock_hedging, mock_screener, mock_env,
                            mock_stock, mock_bond, mock_commodity,
                            mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试保存checkpoint"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock env.get_state()
        mock_env_instance = MagicMock()
        mock_env_instance.get_state.return_value = {"portfolio_value": 1000000}
        mock_env.return_value = mock_env_instance

        # Mock risk controller state
        mock_risk_instance = MagicMock()
        mock_risk_instance.peak_value = 1.1
        mock_risk_instance.max_drawdown_history = 0.05
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = {"decisions": [], "risk_assessments": []}

            orchestrator._save_checkpoint(
                start_date="2024-01-01",
                end_date="2024-12-31",
                rebalance_frequency="weekly",
                current_date_index=10,
                total_dates=52,
                results=results
            )

            checkpoint_path = orchestrator._get_checkpoint_path()
            assert os.path.exists(checkpoint_path)

            # Check JSON progress file also created
            json_path = checkpoint_path.replace(".pkl", "_progress.json")
            assert os.path.exists(json_path)

            with open(json_path, 'r') as f:
                progress = json.load(f)
                assert progress["current_date_index"] == 10
                assert progress["total_dates"] == 52

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_load_checkpoint_not_exists(self, mock_llm, mock_loader, mock_market,
                                        mock_hedging, mock_screener, mock_env,
                                        mock_stock, mock_bond, mock_commodity,
                                        mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试加载不存在的checkpoint"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            checkpoint = orchestrator._load_checkpoint()

            assert checkpoint is None

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_save_and_load_checkpoint(self, mock_llm, mock_loader, mock_market,
                                      mock_hedging, mock_screener, mock_env,
                                      mock_stock, mock_bond, mock_commodity,
                                      mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试保存和加载checkpoint"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock env.get_state()
        mock_env_instance = MagicMock()
        mock_env_instance.get_state.return_value = {"portfolio_value": 1000000}
        mock_env.return_value = mock_env_instance

        # Mock risk controller state
        mock_risk_instance = MagicMock()
        mock_risk_instance.peak_value = 1.15
        mock_risk_instance.max_drawdown_history = 0.08
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = {"decisions": [{"date": "2024-01-01"}], "risk_assessments": []}

            orchestrator._save_checkpoint(
                start_date="2024-01-01",
                end_date="2024-12-31",
                rebalance_frequency="monthly",
                current_date_index=5,
                total_dates=12,
                results=results
            )

            # Load checkpoint
            loaded = orchestrator._load_checkpoint()

            assert loaded is not None
            assert loaded.start_date == "2024-01-01"
            assert loaded.current_date_index == 5
            assert loaded.total_dates == 12
            assert loaded.rebalance_frequency == "monthly"
            assert len(loaded.results["decisions"]) == 1

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_clear_checkpoint(self, mock_llm, mock_loader, mock_market,
                             mock_hedging, mock_screener, mock_env,
                             mock_stock, mock_bond, mock_commodity,
                             mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试清理checkpoint"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock env.get_state()
        mock_env_instance = MagicMock()
        mock_env_instance.get_state.return_value = {"portfolio_value": 1000000}
        mock_env.return_value = mock_env_instance

        # Mock risk controller
        mock_risk_instance = MagicMock()
        mock_risk_instance.peak_value = 1.0
        mock_risk_instance.max_drawdown_history = 0.0
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            # Save checkpoint
            orchestrator._save_checkpoint(
                start_date="2024-01-01",
                end_date="2024-12-31",
                rebalance_frequency="weekly",
                current_date_index=10,
                total_dates=52,
                results={}
            )

            checkpoint_path = orchestrator._get_checkpoint_path()
            assert os.path.exists(checkpoint_path)

            # Clear checkpoint
            orchestrator._clear_checkpoint()

            assert not os.path.exists(checkpoint_path)


# ============================================================
# Test 6: Data Filtering and Extraction Tests
# ============================================================

class TestDataProcessing:
    """测试数据处理功能"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_filter_market_data(self, mock_llm, mock_loader, mock_market,
                                mock_hedging, mock_screener, mock_env,
                                mock_stock, mock_bond, mock_commodity,
                                mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试过滤市场数据"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock stock expert symbols
        mock_stock_instance = MagicMock()
        mock_stock_instance.symbols = ["AAPL", "MSFT"]
        mock_stock.return_value = mock_stock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            market_data = {
                "AAPL": {"close": 150.0},
                "MSFT": {"close": 300.0},
                "TLT": {"close": 100.0},
                "GLD": {"close": 180.0}
            }

            filtered = orchestrator._filter_market_data(market_data, "stocks")

            assert "AAPL" in filtered
            assert "MSFT" in filtered
            assert "TLT" not in filtered
            assert "GLD" not in filtered

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_extract_technical_indicators(self, mock_llm, mock_loader, mock_market,
                                          mock_hedging, mock_screener, mock_env,
                                          mock_stock, mock_bond, mock_commodity,
                                          mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试提取技术指标"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            market_data = {
                "AAPL": {
                    "close": 150.0,
                    "rsi": 65.0,
                    "macd": 2.5,
                    "macd_signal": 2.0,
                    "ma_20": 148.0,
                    "ma_50": 145.0,
                    "bb_upper": 155.0,
                    "bb_lower": 145.0,
                    "trend": "uptrend"
                }
            }

            indicators = orchestrator._extract_technical_indicators(
                market_data, ["AAPL"]
            )

            assert "AAPL" in indicators
            assert indicators["AAPL"]["rsi"] == 65.0
            assert indicators["AAPL"]["macd"] == 2.5
            assert indicators["AAPL"]["price"] == 150.0
            assert indicators["AAPL"]["trend"] == "uptrend"

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_extract_technical_indicators_with_defaults(self, mock_llm, mock_loader, mock_market,
                                                        mock_hedging, mock_screener, mock_env,
                                                        mock_stock, mock_bond, mock_commodity,
                                                        mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试提取技术指标(缺失字段使用默认值)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            market_data = {
                "AAPL": {
                    "close": 150.0,
                    # Missing most indicators
                }
            }

            indicators = orchestrator._extract_technical_indicators(
                market_data, ["AAPL"]
            )

            assert "AAPL" in indicators
            # Should use defaults
            assert indicators["AAPL"]["rsi"] == 50
            assert indicators["AAPL"]["macd"] == 0
            assert indicators["AAPL"]["price"] == 150.0


# ============================================================
# Test 7: Risk Adjustments Tests
# ============================================================

class TestRiskAdjustments:
    """测试风险调整功能"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_apply_risk_adjustments_reduce(self, mock_llm, mock_loader, mock_market,
                                           mock_hedging, mock_screener, mock_env,
                                           mock_stock, mock_bond, mock_commodity,
                                           mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试应用风险调整(减少持仓)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            allocation = {"AAPL": 0.3, "MSFT": 0.3, "TLT": 0.4}
            recommendations = {
                "reduce_aapl": {"asset": "AAPL", "to": 0.15}
            }

            adjusted = orchestrator._apply_risk_adjustments(allocation, recommendations)

            # AAPL should be reduced
            assert adjusted["AAPL"] < 0.3
            # Should be normalized
            assert abs(sum(adjusted.values()) - 1.0) < 1e-6

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_apply_risk_adjustments_no_recommendations(self, mock_llm, mock_loader, mock_market,
                                                       mock_hedging, mock_screener, mock_env,
                                                       mock_stock, mock_bond, mock_commodity,
                                                       mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试应用风险调整(无建议)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            allocation = {"AAPL": 0.5, "TLT": 0.5}
            recommendations = {}

            adjusted = orchestrator._apply_risk_adjustments(allocation, recommendations)

            # Should remain same (normalized)
            assert abs(adjusted["AAPL"] - 0.5) < 1e-6
            assert abs(adjusted["TLT"] - 0.5) < 1e-6

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_apply_risk_adjustments_empty_allocation(self, mock_llm, mock_loader, mock_market,
                                                     mock_hedging, mock_screener, mock_env,
                                                     mock_stock, mock_bond, mock_commodity,
                                                     mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试应用风险调整(空配置)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            allocation = {}
            recommendations = {"reduce_aapl": {"asset": "AAPL", "to": 0.1}}

            adjusted = orchestrator._apply_risk_adjustments(allocation, recommendations)

            # Should handle empty allocation
            assert adjusted == {}


# ============================================================
# Test 8: Dynamic Universe Refresh Tests
# ============================================================

class TestDynamicUniverseRefresh:
    """测试动态资产池刷新"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_refresh_dynamic_universe_force(self, mock_llm, mock_loader, mock_market,
                                            mock_hedging, mock_screener, mock_env,
                                            mock_stock, mock_bond, mock_commodity,
                                            mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试强制刷新动态资产池"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock screener
        mock_screener_instance = MagicMock()
        mock_screener_instance.get_dynamic_universe.return_value = ["AAPL", "MSFT", "GOOGL"]
        mock_screener.return_value = mock_screener_instance

        # Mock experts
        mock_stock_instance = MagicMock()
        mock_stock.return_value = mock_stock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            result = orchestrator._refresh_dynamic_universe("2024-01-15", force=True)

            # Should refresh all asset classes
            assert len(result) > 0
            assert mock_screener_instance.get_dynamic_universe.called

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_refresh_dynamic_universe_no_refresh_needed(self, mock_llm, mock_loader, mock_market,
                                                        mock_hedging, mock_screener, mock_env,
                                                        mock_stock, mock_bond, mock_commodity,
                                                        mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试不需要刷新(时间未到)"""
        from finsage.core.orchestrator import FinSageOrchestrator

        mock_screener_instance = MagicMock()
        mock_screener.return_value = mock_screener_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            # Set last refresh to today
            orchestrator._last_universe_refresh = datetime.strptime("2024-01-15", "%Y-%m-%d")
            orchestrator._refresh_interval_days = 7  # Weekly

            # Try to refresh same day
            result = orchestrator._refresh_dynamic_universe("2024-01-15", force=False)

            # Should return empty (no refresh)
            assert result == {}
            assert not mock_screener_instance.get_dynamic_universe.called

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_refresh_dynamic_universe_interval_passed(self, mock_llm, mock_loader, mock_market,
                                                      mock_hedging, mock_screener, mock_env,
                                                      mock_stock, mock_bond, mock_commodity,
                                                      mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试刷新间隔已过"""
        from finsage.core.orchestrator import FinSageOrchestrator

        mock_screener_instance = MagicMock()
        mock_screener_instance.get_dynamic_universe.return_value = ["AAPL", "MSFT"]
        mock_screener.return_value = mock_screener_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            # Set last refresh to 8 days ago
            orchestrator._last_universe_refresh = datetime.strptime("2024-01-01", "%Y-%m-%d")
            orchestrator._refresh_interval_days = 7  # Weekly

            # Try to refresh after 8 days
            result = orchestrator._refresh_dynamic_universe("2024-01-09", force=False)

            # Should refresh
            assert len(result) > 0
            assert mock_screener_instance.get_dynamic_universe.called


# ============================================================
# Test 9: Expert Report Collection Tests
# ============================================================

class TestExpertReportCollection:
    """测试专家报告收集"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_collect_expert_reports_success(self, mock_llm, mock_loader, mock_market,
                                            mock_hedging, mock_screener, mock_env,
                                            mock_stock, mock_bond, mock_commodity,
                                            mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试成功收集专家报告"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock expert instances
        mock_stock_instance = MagicMock()
        mock_stock_instance.symbols = ["AAPL"]
        mock_stock_instance.analyze.return_value = {"recommendation": "buy"}
        mock_stock.return_value = mock_stock_instance

        mock_bond_instance = MagicMock()
        mock_bond_instance.symbols = ["TLT"]
        mock_bond_instance.analyze.return_value = {"recommendation": "hold"}
        mock_bond.return_value = mock_bond_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            market_data = {
                "AAPL": {"close": 150.0, "rsi": 65},
                "TLT": {"close": 100.0, "rsi": 50},
                "news": []
            }

            reports = orchestrator._collect_expert_reports(market_data, "2024-01-15")

            assert "stocks" in reports
            assert "bonds" in reports

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_collect_expert_reports_expert_fails(self, mock_llm, mock_loader, mock_market,
                                                  mock_hedging, mock_screener, mock_env,
                                                  mock_stock, mock_bond, mock_commodity,
                                                  mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试专家分析失败的情况"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock stock expert to raise exception
        mock_stock_instance = MagicMock()
        mock_stock_instance.symbols = ["AAPL"]
        mock_stock_instance.analyze.side_effect = Exception("API Error")
        mock_stock.return_value = mock_stock_instance

        # Mock bond expert to work normally
        mock_bond_instance = MagicMock()
        mock_bond_instance.symbols = ["TLT"]
        mock_bond_instance.analyze.return_value = {"recommendation": "hold"}
        mock_bond.return_value = mock_bond_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            market_data = {
                "AAPL": {"close": 150.0},
                "TLT": {"close": 100.0},
                "news": []
            }

            reports = orchestrator._collect_expert_reports(market_data, "2024-01-15")

            # Stock expert should fail but bonds should work
            assert "stocks" not in reports
            assert "bonds" in reports


# ============================================================
# Test 10: Save Results Tests
# ============================================================

class TestSaveResults:
    """测试保存结果"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_save_results(self, mock_llm, mock_loader, mock_market,
                         mock_hedging, mock_screener, mock_env,
                         mock_stock, mock_bond, mock_commodity,
                         mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试保存结果到文件"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "decisions": [],
                "final_metrics": {"sharpe": 1.5}
            }

            output_path = os.path.join(tmpdir, "results", "test_results.json")

            orchestrator.save_results(results, output_path)

            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                loaded = json.load(f)
                assert loaded["start_date"] == "2024-01-01"
                assert loaded["final_metrics"]["sharpe"] == 1.5

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_save_results_creates_directory(self, mock_llm, mock_loader, mock_market,
                                            mock_hedging, mock_screener, mock_env,
                                            mock_stock, mock_bond, mock_commodity,
                                            mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试保存结果自动创建目录"""
        from finsage.core.orchestrator import FinSageOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = {"test": "data"}
            output_path = os.path.join(tmpdir, "deep", "nested", "results.json")

            orchestrator.save_results(results, output_path)

            assert os.path.exists(output_path)


# ============================================================
# Test 11: Run Method Tests (Integration-style with Mocks)
# ============================================================

class TestRunMethod:
    """测试run方法"""

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_run_simple_scenario(self, mock_llm, mock_loader, mock_market,
                                 mock_hedging, mock_screener, mock_env,
                                 mock_stock, mock_bond, mock_commodity,
                                 mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试简单运行场景"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock all components
        mock_market_instance = MagicMock()
        mock_market_instance.get_market_snapshot.return_value = {
            "AAPL": {"close": 150.0, "rsi": 50},
            "news": []
        }
        mock_market.return_value = mock_market_instance

        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = None
        mock_portfolio = MagicMock()
        mock_portfolio.weights = {"AAPL": 1.0}
        mock_portfolio.portfolio_value = 1000000
        mock_portfolio.to_dict.return_value = {"value": 1000000}
        mock_env_instance.portfolio = mock_portfolio
        mock_env_instance.step.return_value = (
            mock_portfolio,  # portfolio_state
            0.01,  # reward
            False,  # done
            {"trades": []}  # info
        )
        mock_env_instance.get_state.return_value = {"value": 1000000}
        mock_env_instance.get_metrics.return_value = {"sharpe": 1.5}
        mock_env.return_value = mock_env_instance

        # Mock experts
        for expert_mock in [mock_stock, mock_bond, mock_commodity, mock_reits, mock_crypto]:
            expert_instance = MagicMock()
            expert_instance.symbols = ["AAPL"]
            expert_instance.analyze.return_value = {"recommendation": "hold"}
            expert_mock.return_value = expert_instance

        # Mock portfolio manager
        mock_pm_instance = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target_allocation = {"AAPL": 1.0}
        mock_decision.to_dict.return_value = {"allocation": {"AAPL": 1.0}}
        mock_pm_instance.decide.return_value = mock_decision
        mock_pm.return_value = mock_pm_instance

        # Mock risk controller
        mock_risk_instance = MagicMock()
        mock_risk_instance.get_constraints.return_value = {}
        mock_assessment = MagicMock()
        mock_assessment.veto = False
        mock_assessment.emergency_rebalance = False
        mock_assessment.intraday_alerts = []
        mock_assessment.to_dict.return_value = {"status": "ok"}
        mock_risk_instance.assess_with_intraday.return_value = mock_assessment
        mock_risk_instance.peak_value = 1.0
        mock_risk_instance.max_drawdown_history = 0.0
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = orchestrator.run(
                start_date="2024-01-02",
                end_date="2024-01-03",
                rebalance_frequency="daily",
                resume=False
            )

            assert "decisions" in results
            assert "risk_assessments" in results
            assert "final_metrics" in results
            assert results["start_date"] == "2024-01-02"
            assert results["end_date"] == "2024-01-03"

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_run_with_risk_veto(self, mock_llm, mock_loader, mock_market,
                                mock_hedging, mock_screener, mock_env,
                                mock_stock, mock_bond, mock_commodity,
                                mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试风险否决场景"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock market data
        mock_market_instance = MagicMock()
        mock_market_instance.get_market_snapshot.return_value = {
            "AAPL": {"close": 150.0, "rsi": 50},
            "news": []
        }
        mock_market.return_value = mock_market_instance

        # Mock env
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = None
        mock_portfolio = MagicMock()
        mock_portfolio.weights = {"AAPL": 0.5, "TLT": 0.5}
        mock_portfolio.portfolio_value = 1000000
        mock_portfolio.to_dict.return_value = {"value": 1000000}
        mock_env_instance.portfolio = mock_portfolio
        mock_env_instance.step.return_value = (mock_portfolio, 0.01, False, {"trades": []})
        mock_env_instance.get_state.return_value = {"value": 1000000}
        mock_env_instance.get_metrics.return_value = {"sharpe": 1.5}
        mock_env.return_value = mock_env_instance

        # Mock experts
        for expert_mock in [mock_stock, mock_bond, mock_commodity, mock_reits, mock_crypto]:
            expert_instance = MagicMock()
            expert_instance.symbols = ["AAPL"]
            expert_instance.analyze.return_value = {"recommendation": "hold"}
            expert_mock.return_value = expert_instance

        # Mock portfolio manager
        mock_pm_instance = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target_allocation = {"AAPL": 0.9, "TLT": 0.1}  # Risky allocation
        mock_decision.to_dict.return_value = {"allocation": {"AAPL": 0.9, "TLT": 0.1}}
        mock_pm_instance.decide.return_value = mock_decision
        mock_pm.return_value = mock_pm_instance

        # Mock risk controller - VETO
        mock_risk_instance = MagicMock()
        mock_risk_instance.get_constraints.return_value = {}
        mock_assessment = MagicMock()
        mock_assessment.veto = True  # VETO!
        mock_assessment.emergency_rebalance = False
        mock_assessment.recommendations = {"reduce_aapl": {"asset": "AAPL", "to": 0.5}}
        mock_assessment.intraday_alerts = []
        mock_assessment.to_dict.return_value = {"status": "veto"}
        mock_risk_instance.assess_with_intraday.return_value = mock_assessment
        mock_risk_instance.peak_value = 1.0
        mock_risk_instance.max_drawdown_history = 0.0
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = orchestrator.run(
                start_date="2024-01-02",
                end_date="2024-01-03",
                rebalance_frequency="daily",
                resume=False
            )

            assert "decisions" in results
            # Should have completed despite veto
            assert len(results["decisions"]) > 0

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_run_with_emergency_rebalance(self, mock_llm, mock_loader, mock_market,
                                          mock_hedging, mock_screener, mock_env,
                                          mock_stock, mock_bond, mock_commodity,
                                          mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试紧急再平衡场景"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock market data
        mock_market_instance = MagicMock()
        mock_market_instance.get_market_snapshot.return_value = {
            "AAPL": {"close": 150.0, "rsi": 50},
            "news": []
        }
        mock_market.return_value = mock_market_instance

        # Mock env
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = None
        mock_portfolio = MagicMock()
        mock_portfolio.weights = {"AAPL": 1.0}
        mock_portfolio.portfolio_value = 1000000
        mock_portfolio.to_dict.return_value = {"value": 1000000}
        mock_env_instance.portfolio = mock_portfolio
        mock_env_instance.step.return_value = (mock_portfolio, 0.01, False, {"trades": []})
        mock_env_instance.get_state.return_value = {"value": 1000000}
        mock_env_instance.get_metrics.return_value = {"sharpe": 1.5}
        mock_env.return_value = mock_env_instance

        # Mock experts
        for expert_mock in [mock_stock, mock_bond, mock_commodity, mock_reits, mock_crypto]:
            expert_instance = MagicMock()
            expert_instance.symbols = ["AAPL"]
            expert_instance.analyze.return_value = {"recommendation": "hold"}
            expert_mock.return_value = expert_instance

        # Mock portfolio manager
        mock_pm_instance = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target_allocation = {"AAPL": 1.0}
        mock_decision.to_dict.return_value = {"allocation": {"AAPL": 1.0}}
        mock_pm_instance.decide.return_value = mock_decision
        mock_pm.return_value = mock_pm_instance

        # Mock risk controller - EMERGENCY
        mock_risk_instance = MagicMock()
        mock_risk_instance.get_constraints.return_value = {}
        mock_assessment = MagicMock()
        mock_assessment.veto = False
        mock_assessment.emergency_rebalance = True  # EMERGENCY!
        mock_assessment.defensive_allocation = {"TLT": 0.7, "cash": 0.3}
        mock_assessment.intraday_alerts = []
        mock_assessment.to_dict.return_value = {"status": "emergency"}
        mock_risk_instance.assess_with_intraday.return_value = mock_assessment
        mock_risk_instance.peak_value = 1.0
        mock_risk_instance.max_drawdown_history = 0.0
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            results = orchestrator.run(
                start_date="2024-01-02",
                end_date="2024-01-03",
                rebalance_frequency="daily",
                resume=False
            )

            assert "decisions" in results
            # Emergency rebalance should trigger defensive allocation
            assert len(results["decisions"]) > 0

    @patch('finsage.core.orchestrator.RiskController')
    @patch('finsage.core.orchestrator.PortfolioManager')
    @patch('finsage.core.orchestrator.CryptoExpert')
    @patch('finsage.core.orchestrator.REITsExpert')
    @patch('finsage.core.orchestrator.CommodityExpert')
    @patch('finsage.core.orchestrator.BondExpert')
    @patch('finsage.core.orchestrator.StockExpert')
    @patch('finsage.core.orchestrator.MultiAssetTradingEnv')
    @patch('finsage.core.orchestrator.DynamicStockScreener')
    @patch('finsage.core.orchestrator.HedgingToolkit')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.LLMProvider')
    def test_run_with_data_error_continues(self, mock_llm, mock_loader, mock_market,
                                           mock_hedging, mock_screener, mock_env,
                                           mock_stock, mock_bond, mock_commodity,
                                           mock_reits, mock_crypto, mock_pm, mock_risk):
        """测试数据错误时继续运行"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # Mock market data - raise error on first call, succeed on second
        mock_market_instance = MagicMock()
        call_count = [0]

        def get_snapshot_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyError("Data not found")
            return {"AAPL": {"close": 150.0, "rsi": 50}, "news": []}

        mock_market_instance.get_market_snapshot.side_effect = get_snapshot_side_effect
        mock_market.return_value = mock_market_instance

        # Mock env
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = None
        mock_portfolio = MagicMock()
        mock_portfolio.weights = {"AAPL": 1.0}
        mock_portfolio.portfolio_value = 1000000
        mock_portfolio.to_dict.return_value = {"value": 1000000}
        mock_env_instance.portfolio = mock_portfolio
        mock_env_instance.step.return_value = (mock_portfolio, 0.01, False, {"trades": []})
        mock_env_instance.get_state.return_value = {"value": 1000000}
        mock_env_instance.get_metrics.return_value = {"sharpe": 1.5}
        mock_env.return_value = mock_env_instance

        # Mock experts
        for expert_mock in [mock_stock, mock_bond, mock_commodity, mock_reits, mock_crypto]:
            expert_instance = MagicMock()
            expert_instance.symbols = ["AAPL"]
            expert_instance.analyze.return_value = {"recommendation": "hold"}
            expert_mock.return_value = expert_instance

        # Mock portfolio manager
        mock_pm_instance = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target_allocation = {"AAPL": 1.0}
        mock_decision.to_dict.return_value = {"allocation": {"AAPL": 1.0}}
        mock_pm_instance.decide.return_value = mock_decision
        mock_pm.return_value = mock_pm_instance

        # Mock risk controller
        mock_risk_instance = MagicMock()
        mock_risk_instance.get_constraints.return_value = {}
        mock_assessment = MagicMock()
        mock_assessment.veto = False
        mock_assessment.emergency_rebalance = False
        mock_assessment.intraday_alerts = []
        mock_assessment.to_dict.return_value = {"status": "ok"}
        mock_risk_instance.assess_with_intraday.return_value = mock_assessment
        mock_risk_instance.peak_value = 1.0
        mock_risk_instance.max_drawdown_history = 0.0
        mock_risk.return_value = mock_risk_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FinSageOrchestrator(checkpoint_dir=tmpdir)

            # Should handle error and continue
            results = orchestrator.run(
                start_date="2024-01-02",
                end_date="2024-01-04",
                rebalance_frequency="daily",
                resume=False
            )

            assert "decisions" in results
            # Should have at least one successful decision (from second/third date)
            # First date failed, but should continue


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Deep Orchestrator Tests - Comprehensive Coverage")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short", "-x"])


if __name__ == "__main__":
    run_tests()
