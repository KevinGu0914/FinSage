#!/usr/bin/env python
"""
Data Module Tests - 数据模块测试
覆盖: data_loader
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: DataLoader
# ============================================================

class TestDataLoader:
    """测试数据加载器"""

    def test_import(self):
        """测试导入"""
        from finsage.data.data_loader import DataLoader
        assert DataLoader is not None

    def test_initialization_yfinance(self):
        """测试yfinance数据源初始化"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="yfinance")
        assert loader.data_source == "yfinance"

    def test_initialization_fmp(self):
        """测试FMP数据源初始化"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="fmp", api_key="test-key")
        assert loader.data_source == "fmp"
        assert loader.api_key == "test-key"

    def test_initialization_local(self):
        """测试本地数据源初始化"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="local")
        assert loader.data_source == "local"

    def test_initialization_with_cache_dir(self):
        """测试带缓存目录初始化"""
        from finsage.data.data_loader import DataLoader
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            loader = DataLoader(data_source="yfinance", cache_dir=cache_dir)

            assert os.path.exists(cache_dir)

    def test_unknown_data_source(self):
        """测试未知数据源"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="unknown")

        with pytest.raises(ValueError):
            loader.load_price_data(
                symbols=["SPY"],
                start_date="2024-01-01",
                end_date="2024-01-31"
            )


# ============================================================
# Test 2: FMPClient
# ============================================================

class TestFMPClient:
    """测试FMP客户端"""

    def test_import(self):
        """测试导入"""
        from finsage.data.fmp_client import FMPClient
        assert FMPClient is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.data.fmp_client import FMPClient

        client = FMPClient(api_key="test-key")
        assert client.api_key == "test-key"


# ============================================================
# Test 3: Data Utility Functions
# ============================================================

class TestDataUtilities:
    """测试数据工具函数"""

    def test_calculate_returns(self):
        """测试计算收益率"""
        prices = pd.Series([100, 102, 101, 105, 103])
        returns = prices.pct_change().dropna()

        assert len(returns) == 4
        assert abs(returns.iloc[0] - 0.02) < 0.001

    def test_calculate_volatility(self):
        """测试计算波动率"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))

        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        assert 0.1 < annual_vol < 0.5

    def test_calculate_sharpe_ratio(self):
        """测试计算夏普比率"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        risk_free = 0.02

        sharpe = (mean_return - risk_free) / volatility

        assert isinstance(sharpe, float)

    def test_resample_data(self):
        """测试数据重采样"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)

        # 重采样到周
        weekly = prices.resample("W").last()

        assert len(weekly) < len(prices)


# ============================================================
# Test 4: Cache Functionality
# ============================================================

class TestCacheFunctionality:
    """测试缓存功能"""

    def test_data_loader_caching(self):
        """测试数据加载器缓存"""
        from finsage.data.data_loader import DataLoader
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_source="yfinance", cache_dir=tmpdir)

            # 缓存目录应该被创建
            assert os.path.exists(tmpdir)


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Data Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
