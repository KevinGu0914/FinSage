#!/usr/bin/env python
"""
Multi-Asset Trading Environment Tests
======================================
覆盖: finsage/environment/multi_asset_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_market_data():
    """生成示例市场数据"""
    return {
        "AAPL": {"close": 175.0, "price": 175.0, "volume": 50000000},
        "MSFT": {"close": 380.0, "price": 380.0, "volume": 30000000},
        "SPY": {"close": 470.0, "price": 470.0, "volume": 80000000},
        "TLT": {"close": 95.0, "price": 95.0, "volume": 20000000},
        "GLD": {"close": 185.0, "price": 185.0, "volume": 10000000},
        "VNQ": {"close": 85.0, "price": 85.0, "volume": 5000000},
    }


@pytest.fixture
def sample_asset_universe():
    """生成示例资产池"""
    return {
        "stocks": ["AAPL", "MSFT", "SPY"],
        "bonds": ["TLT"],
        "commodities": ["GLD"],
        "reits": ["VNQ"],
    }


@pytest.fixture
def sample_target_allocation():
    """生成示例目标配置"""
    return {
        "stocks": 0.4,
        "bonds": 0.3,
        "commodities": 0.2,
        "reits": 0.1,
    }


# ============================================================
# Test 1: Module Imports
# ============================================================

class TestModuleImports:
    """测试模块导入"""

    def test_import_multi_asset_trading_env(self):
        """测试导入MultiAssetTradingEnv"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv
        assert MultiAssetTradingEnv is not None

    def test_import_env_config(self):
        """测试导入EnvConfig"""
        from finsage.environment.multi_asset_env import EnvConfig
        assert EnvConfig is not None


# ============================================================
# Test 2: EnvConfig Dataclass
# ============================================================

class TestEnvConfig:
    """测试EnvConfig配置类"""

    def test_env_config_default_values(self):
        """测试EnvConfig默认值"""
        from finsage.environment.multi_asset_env import EnvConfig

        config = EnvConfig()
        assert config.initial_capital == 1_000_000.0
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.min_trade_value == 100.0
        assert config.max_single_asset == 0.15
        assert config.max_asset_class == 0.50
        assert config.rebalance_threshold == 0.02

    def test_env_config_custom_values(self):
        """测试EnvConfig自定义值"""
        from finsage.environment.multi_asset_env import EnvConfig

        config = EnvConfig(
            initial_capital=2_000_000.0,
            transaction_cost=0.002,
            slippage=0.001,
        )
        assert config.initial_capital == 2_000_000.0
        assert config.transaction_cost == 0.002
        assert config.slippage == 0.001


# ============================================================
# Test 3: Environment Initialization
# ============================================================

class TestMultiAssetEnvInit:
    """测试环境初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        assert env is not None
        assert env.config is not None
        assert env.portfolio is not None

    def test_init_with_config(self):
        """测试带配置初始化"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv, EnvConfig

        config = EnvConfig(initial_capital=2_000_000.0)
        env = MultiAssetTradingEnv(config=config)

        assert env.config.initial_capital == 2_000_000.0

    def test_init_with_asset_universe(self, sample_asset_universe):
        """测试带资产池初始化"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv(asset_universe=sample_asset_universe)
        assert env.asset_universe == sample_asset_universe

    def test_init_default_universe(self):
        """测试默认资产池"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        assert env.asset_universe is not None
        assert len(env.asset_universe) > 0


# ============================================================
# Test 4: Environment Reset
# ============================================================

class TestEnvReset:
    """测试环境重置"""

    def test_reset_basic(self):
        """测试基本重置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        portfolio = env.reset()

        assert portfolio is not None
        assert env.current_step == 0
        assert env.done is False

    def test_reset_with_initial_capital(self):
        """测试带初始资金重置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        portfolio = env.reset(initial_capital=500_000.0)

        assert portfolio.initial_capital == 500_000.0
        assert portfolio.cash == 500_000.0

    def test_reset_with_start_date(self):
        """测试带开始日期重置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        portfolio = env.reset(start_date="2024-01-01")

        assert env.current_date == "2024-01-01"


# ============================================================
# Test 5: Step Function
# ============================================================

class TestEnvStep:
    """测试step函数"""

    def test_step_basic(self, sample_market_data, sample_target_allocation):
        """测试基本step"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        portfolio, reward, done, info = env.step(
            target_allocation=sample_target_allocation,
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        assert portfolio is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_step_count(self, sample_market_data, sample_target_allocation):
        """测试step增加步数"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        initial_step = env.current_step
        env.step(
            target_allocation=sample_target_allocation,
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        assert env.current_step == initial_step + 1

    def test_step_returns_info(self, sample_market_data, sample_target_allocation):
        """测试step返回info"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        _, _, _, info = env.step(
            target_allocation=sample_target_allocation,
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        assert "step" in info
        assert "timestamp" in info
        assert "trades" in info
        assert "portfolio_value" in info

    def test_step_sequence(self, sample_market_data, sample_target_allocation):
        """测试连续多步"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        for i in range(5):
            portfolio, reward, done, info = env.step(
                target_allocation=sample_target_allocation,
                market_data=sample_market_data,
                timestamp=f"2024-01-{i+2:02d}"
            )

            if done:
                break

        assert env.current_step >= 1


# ============================================================
# Test 6: Price Extraction
# ============================================================

class TestPriceExtraction:
    """测试价格提取"""

    def test_extract_prices_from_dict(self, sample_market_data):
        """测试从字典提取价格"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        prices = env._extract_prices(sample_market_data)

        assert "AAPL" in prices
        assert prices["AAPL"] == 175.0

    def test_extract_prices_skips_non_price_keys(self, sample_market_data):
        """测试跳过非价格键"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        # 添加应该跳过的键
        market_data_with_extra = {
            **sample_market_data,
            "news": ["some news"],
            "macro": {"gdp": 2.5},
        }

        prices = env._extract_prices(market_data_with_extra)

        assert "news" not in prices
        assert "macro" not in prices


# ============================================================
# Test 7: Rebalance
# ============================================================

class TestRebalance:
    """测试再平衡"""

    def test_rebalance_basic(self, sample_market_data, sample_target_allocation):
        """测试基本再平衡"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        prices = env._extract_prices(sample_market_data)
        trades = env._rebalance(
            target_allocation=sample_target_allocation,
            prices=prices,
            timestamp="2024-01-02"
        )

        assert isinstance(trades, list)

    def test_rebalance_with_symbol_allocation(self, sample_market_data):
        """测试股票代码级别配置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        symbol_allocation = {
            "AAPL": 0.2,
            "MSFT": 0.2,
            "SPY": 0.2,
            "TLT": 0.2,
            "GLD": 0.1,
            "VNQ": 0.1,
        }

        prices = env._extract_prices(sample_market_data)
        trades = env._rebalance(
            target_allocation=symbol_allocation,
            prices=prices,
            timestamp="2024-01-02"
        )

        assert isinstance(trades, list)


# ============================================================
# Test 8: Reward Calculation
# ============================================================

class TestRewardCalculation:
    """测试奖励计算"""

    def test_calculate_reward(self):
        """测试计算奖励"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        reward = env._calculate_reward()

        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)


# ============================================================
# Test 9: Default Universe
# ============================================================

class TestDefaultUniverse:
    """测试默认资产池"""

    def test_default_universe_structure(self):
        """测试默认资产池结构"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        universe = env._default_universe()

        assert isinstance(universe, dict)
        # 应该包含多个资产类别
        assert len(universe) > 0


# ============================================================
# Test 10: Portfolio State Integration
# ============================================================

class TestPortfolioStateIntegration:
    """测试投资组合状态集成"""

    def test_portfolio_value_tracking(self, sample_market_data, sample_target_allocation):
        """测试投资组合价值跟踪"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        initial_value = env.portfolio.portfolio_value

        env.step(
            target_allocation=sample_target_allocation,
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        # 价值应该已更新
        assert env.portfolio.portfolio_value is not None

    def test_cash_management(self):
        """测试现金管理"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        portfolio = env.reset()

        assert portfolio.cash > 0


# ============================================================
# Test 11: Expert Reports Integration
# ============================================================

class TestExpertReportsIntegration:
    """测试专家报告集成"""

    def test_step_with_expert_reports(self, sample_market_data, sample_target_allocation):
        """测试带专家报告的step"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        expert_reports = {
            "Stock_Expert": {
                "recommendations": [
                    {"symbol": "AAPL", "weight": 0.5},
                    {"symbol": "MSFT", "weight": 0.5},
                ]
            }
        }

        portfolio, reward, done, info = env.step(
            target_allocation=sample_target_allocation,
            market_data=sample_market_data,
            timestamp="2024-01-02",
            expert_reports=expert_reports
        )

        assert portfolio is not None


# ============================================================
# Test 12: Edge Cases
# ============================================================

class TestEdgeCases:
    """测试边缘情况"""

    def test_empty_market_data(self, sample_target_allocation):
        """测试空市场数据"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        portfolio, reward, done, info = env.step(
            target_allocation=sample_target_allocation,
            market_data={},
            timestamp="2024-01-02"
        )

        assert portfolio is not None

    def test_empty_allocation(self, sample_market_data):
        """测试空配置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        portfolio, reward, done, info = env.step(
            target_allocation={},
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        assert portfolio is not None

    def test_zero_allocation(self, sample_market_data):
        """测试零配置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        zero_allocation = {
            "stocks": 0.0,
            "bonds": 0.0,
            "commodities": 0.0,
            "reits": 0.0,
        }

        portfolio, reward, done, info = env.step(
            target_allocation=zero_allocation,
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        assert portfolio is not None


# ============================================================
# Test 13: Expand Class Allocation
# ============================================================

class TestExpandClassAllocation:
    """测试资产类别配置展开"""

    def test_expand_class_allocation_basic(self, sample_market_data, sample_asset_universe):
        """测试基本配置展开"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv(asset_universe=sample_asset_universe)
        env.reset()

        class_allocation = {"stocks": 0.6, "bonds": 0.4}
        prices = env._extract_prices(sample_market_data)

        symbol_allocation = env._expand_class_allocation(
            class_allocation=class_allocation,
            prices=prices
        )

        assert isinstance(symbol_allocation, dict)
        # 股票应该被展开到具体标的
        total_weight = sum(symbol_allocation.values())
        assert abs(total_weight - 1.0) < 0.1 or total_weight > 0

    def test_expand_class_allocation_with_cash(self, sample_market_data, sample_asset_universe):
        """测试带现金的配置展开"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv(asset_universe=sample_asset_universe)
        env.reset()

        class_allocation = {"stocks": 0.5, "cash": 0.5}
        prices = env._extract_prices(sample_market_data)

        symbol_allocation = env._expand_class_allocation(
            class_allocation=class_allocation,
            prices=prices
        )

        assert "cash" in symbol_allocation
        assert symbol_allocation["cash"] == 0.5


# ============================================================
# Test 14: Get Asset Class
# ============================================================

class TestGetAssetClass:
    """测试获取资产类别"""

    def test_get_asset_class_stock(self, sample_asset_universe):
        """测试获取股票资产类别"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv(asset_universe=sample_asset_universe)

        asset_class = env._get_asset_class("AAPL")
        assert asset_class == "stocks"

    def test_get_asset_class_bond(self, sample_asset_universe):
        """测试获取债券资产类别"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv(asset_universe=sample_asset_universe)

        asset_class = env._get_asset_class("TLT")
        assert asset_class == "bonds"

    def test_get_asset_class_unknown(self, sample_asset_universe):
        """测试获取未知资产类别"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv(asset_universe=sample_asset_universe)

        asset_class = env._get_asset_class("UNKNOWN_SYMBOL")
        assert asset_class == "other"  # 函数返回 "other" 作为未知类别


# ============================================================
# Test 15: Calculate Reward Variations
# ============================================================

class TestCalculateRewardVariations:
    """测试奖励计算变体"""

    def test_calculate_reward_after_step(self, sample_market_data, sample_target_allocation):
        """测试step后的奖励计算"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        # 执行多个step
        for i in range(3):
            _, reward, _, _ = env.step(
                target_allocation=sample_target_allocation,
                market_data=sample_market_data,
                timestamp=f"2024-01-{i+2:02d}"
            )

            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)


# ============================================================
# Test 16: Short Position Tests
# ============================================================

class TestShortPositions:
    """测试做空仓位"""

    def test_step_with_negative_weight(self, sample_market_data):
        """测试带负权重的step（做空）"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        # 包含负权重（做空）的配置
        allocation_with_short = {
            "AAPL": 0.3,
            "MSFT": -0.1,  # 做空
            "SPY": 0.4,
        }

        portfolio, reward, done, info = env.step(
            target_allocation=allocation_with_short,
            market_data=sample_market_data,
            timestamp="2024-01-02"
        )

        assert portfolio is not None


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Multi-Asset Trading Environment Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
