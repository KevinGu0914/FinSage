#!/usr/bin/env python
"""
Config Module Tests - 配置模块测试
覆盖: config.py (LLMConfig, TradingConfig, RiskConfig, AssetConfig, FMPConfig, FinSageConfig)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: LLM Config
# ============================================================

class TestLLMConfig:
    """测试LLM配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import LLMConfig
        assert LLMConfig is not None

    def test_default_values(self):
        """测试默认值"""
        from finsage.config import LLMConfig

        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.timeout == 60

    def test_custom_values(self):
        """测试自定义值"""
        from finsage.config import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.5,
            max_tokens=4000
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 4000

    def test_api_key_from_env(self):
        """测试从环境变量获取API key"""
        from finsage.config import LLMConfig

        # 使用模拟环境变量
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig()
            # 注意: api_key 可能从环境变量获取


# ============================================================
# Test 2: Trading Config
# ============================================================

class TestTradingConfig:
    """测试交易配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import TradingConfig
        assert TradingConfig is not None

    def test_default_values(self):
        """测试默认值"""
        from finsage.config import TradingConfig

        config = TradingConfig()
        assert config.initial_capital == 1_000_000.0
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.min_trade_value == 100.0
        assert config.rebalance_frequency == "daily"
        assert config.rebalance_threshold == 0.02

    def test_custom_values(self):
        """测试自定义值"""
        from finsage.config import TradingConfig

        config = TradingConfig(
            initial_capital=500_000,
            transaction_cost=0.002,
            rebalance_frequency="weekly"
        )

        assert config.initial_capital == 500_000
        assert config.transaction_cost == 0.002
        assert config.rebalance_frequency == "weekly"


# ============================================================
# Test 3: Risk Config
# ============================================================

class TestRiskConfig:
    """测试风控配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import RiskConfig
        assert RiskConfig is not None

    def test_default_values(self):
        """测试默认值"""
        from finsage.config import RiskConfig

        config = RiskConfig()

        # 硬性约束
        assert config.max_single_asset == 0.15
        assert config.max_asset_class == 0.50
        assert config.max_drawdown_trigger == 0.15
        assert config.max_portfolio_var_95 == 0.03

        # 软性约束
        assert config.target_volatility == 0.12
        assert config.max_correlation_cluster == 0.60
        assert config.min_diversification_ratio == 1.2

    def test_custom_constraints(self):
        """测试自定义约束"""
        from finsage.config import RiskConfig

        config = RiskConfig(
            max_single_asset=0.20,
            max_drawdown_trigger=0.10,
            target_volatility=0.08
        )

        assert config.max_single_asset == 0.20
        assert config.max_drawdown_trigger == 0.10
        assert config.target_volatility == 0.08


# ============================================================
# Test 4: Asset Config
# ============================================================

class TestAssetConfig:
    """测试资产配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import AssetConfig
        assert AssetConfig is not None

    def test_allocation_bounds(self):
        """测试配置边界"""
        from finsage.config import AssetConfig

        config = AssetConfig()
        bounds = config.allocation_bounds

        assert "stocks" in bounds
        assert "bonds" in bounds
        assert "commodities" in bounds
        assert "reits" in bounds
        assert "crypto" in bounds
        assert "cash" in bounds

        # 检查stocks边界
        assert bounds["stocks"]["min"] == 0.30
        assert bounds["stocks"]["max"] == 0.50
        assert bounds["stocks"]["default"] == 0.40

    def test_default_universe(self):
        """测试默认资产池"""
        from finsage.config import AssetConfig

        config = AssetConfig()
        universe = config.default_universe

        assert "stocks" in universe
        assert "bonds" in universe
        assert "commodities" in universe
        assert "reits" in universe
        assert "crypto" in universe

        # 检查stocks资产
        assert "SPY" in universe["stocks"]
        assert "QQQ" in universe["stocks"]

        # 检查bonds资产
        assert "TLT" in universe["bonds"]

    def test_universe_has_expected_assets(self):
        """测试资产池包含预期资产"""
        from finsage.config import AssetConfig

        config = AssetConfig()
        universe = config.default_universe

        # 股票应该包含主要ETF和股票
        stocks = universe["stocks"]
        assert len(stocks) >= 5

        # 债券应该包含多种类型
        bonds = universe["bonds"]
        assert len(bonds) >= 3


# ============================================================
# Test 5: FMP Config
# ============================================================

class TestFMPConfig:
    """测试FMP API配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import FMPConfig
        assert FMPConfig is not None

    def test_default_values(self):
        """测试默认值"""
        from finsage.config import FMPConfig

        config = FMPConfig()
        assert config.tier == "ultra"
        assert "financialmodelingprep.com" in config.base_url

    def test_endpoints(self):
        """测试API端点"""
        from finsage.config import FMPConfig

        config = FMPConfig()
        endpoints = config.endpoints

        # 检查必要的端点
        assert "quote" in endpoints
        assert "historical_price" in endpoints
        assert "income_statement" in endpoints
        assert "key_metrics" in endpoints
        assert "stock_news" in endpoints

    def test_screener_defaults(self):
        """测试筛选器默认值"""
        from finsage.config import FMPConfig

        config = FMPConfig()
        defaults = config.screener_defaults

        assert "exchange" in defaults
        assert "NYSE" in defaults["exchange"]
        assert defaults["is_actively_trading"] == True


# ============================================================
# Test 6: Data Config
# ============================================================

class TestDataConfig:
    """测试数据配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import DataConfig
        assert DataConfig is not None

    def test_default_values(self):
        """测试默认值"""
        from finsage.config import DataConfig

        config = DataConfig()
        assert config.data_source == "fmp"
        assert config.cache_dir is not None


# ============================================================
# Test 7: FinSage Config (Main Config)
# ============================================================

class TestFinSageConfig:
    """测试FinSage主配置"""

    def test_import(self):
        """测试导入"""
        from finsage.config import FinSageConfig
        assert FinSageConfig is not None

    def test_default_initialization(self):
        """测试默认初始化"""
        from finsage.config import FinSageConfig

        config = FinSageConfig()

        assert config.llm is not None
        assert config.trading is not None
        assert config.risk is not None
        assert config.assets is not None
        assert config.data is not None

    def test_sub_configs(self):
        """测试子配置"""
        from finsage.config import FinSageConfig, LLMConfig, TradingConfig, RiskConfig

        config = FinSageConfig()

        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.trading, TradingConfig)
        assert isinstance(config.risk, RiskConfig)

    def test_config_values_accessible(self):
        """测试配置值可访问"""
        from finsage.config import FinSageConfig

        config = FinSageConfig()

        # 访问嵌套配置
        assert config.llm.model == "gpt-4o-mini"
        assert config.trading.initial_capital == 1_000_000.0
        assert config.risk.max_single_asset == 0.15


# ============================================================
# Test 8: Config Validation
# ============================================================

class TestConfigValidation:
    """测试配置验证"""

    def test_allocation_bounds_sum(self):
        """测试配置边界合理性"""
        from finsage.config import AssetConfig

        config = AssetConfig()
        bounds = config.allocation_bounds

        # 所有最小值之和应该小于等于1
        min_sum = sum(b["min"] for b in bounds.values())
        assert min_sum <= 1.0, f"Minimum allocations sum to {min_sum}, should be <= 1.0"

        # 所有默认值之和应该等于1
        default_sum = sum(b["default"] for b in bounds.values())
        assert abs(default_sum - 1.0) < 0.01, f"Default allocations sum to {default_sum}, should be ~1.0"

    def test_risk_constraints_reasonable(self):
        """测试风险约束合理性"""
        from finsage.config import RiskConfig

        config = RiskConfig()

        # 单资产上限应该小于资产类别上限
        assert config.max_single_asset <= config.max_asset_class

        # 波动率目标应该在合理范围内
        assert 0.05 <= config.target_volatility <= 0.30

    def test_trading_config_reasonable(self):
        """测试交易配置合理性"""
        from finsage.config import TradingConfig

        config = TradingConfig()

        # 交易成本应该在合理范围内
        assert 0 <= config.transaction_cost <= 0.01

        # 滑点应该在合理范围内
        assert 0 <= config.slippage <= 0.01

        # 再平衡阈值应该在合理范围内
        assert 0.01 <= config.rebalance_threshold <= 0.10


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Config Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
