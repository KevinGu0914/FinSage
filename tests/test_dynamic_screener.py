#!/usr/bin/env python
"""
Dynamic Asset Screener Tests
=============================
覆盖: finsage/data/dynamic_screener.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_market_data():
    """生成示例市场数据"""
    return {
        "AAPL": {"close": 175.0, "volume": 50000000, "return_1m": 0.05, "return_12m": 0.30},
        "MSFT": {"close": 380.0, "volume": 30000000, "return_1m": 0.03, "return_12m": 0.25},
        "GOOGL": {"close": 140.0, "volume": 25000000, "return_1m": 0.02, "return_12m": 0.20},
        "SPY": {"close": 470.0, "volume": 80000000, "return_1m": 0.02, "return_12m": 0.15},
        "QQQ": {"close": 400.0, "volume": 50000000, "return_1m": 0.03, "return_12m": 0.25},
        "TLT": {"close": 95.0, "volume": 20000000, "return_1m": -0.01, "return_12m": -0.10},
        "GLD": {"close": 185.0, "volume": 10000000, "return_1m": 0.02, "return_12m": 0.12},
        "VNQ": {"close": 85.0, "volume": 5000000, "return_1m": 0.01, "return_12m": 0.05},
    }


@pytest.fixture
def sample_returns_df():
    """生成示例收益率DataFrame"""
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)

    symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "TLT", "GLD", "VNQ"]
    returns = pd.DataFrame(
        np.random.randn(252, len(symbols)) * 0.02,
        index=dates,
        columns=symbols
    )
    return returns


@pytest.fixture
def mock_scorer():
    """Mock factor scorer"""
    scorer = MagicMock()
    score_result = MagicMock()
    score_result.composite_score = 0.75
    score_result.signal = "BUY"
    scorer.score.return_value = score_result
    return scorer


# ============================================================
# Test 1: Module Imports
# ============================================================

class TestModuleImports:
    """测试模块导入"""

    def test_import_dynamic_screener(self):
        """测试导入DynamicAssetScreener"""
        from finsage.data.dynamic_screener import DynamicAssetScreener
        assert DynamicAssetScreener is not None

    def test_import_screened_asset(self):
        """测试导入ScreenedAsset"""
        from finsage.data.dynamic_screener import ScreenedAsset
        assert ScreenedAsset is not None


# ============================================================
# Test 2: Initialization
# ============================================================

class TestScreenerInitialization:
    """测试筛选器初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        assert screener is not None
        assert screener.cache_hours == 168  # 默认 1 周

    def test_init_with_api_key(self):
        """测试带API密钥初始化"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener(api_key="test_api_key")
        assert screener.api_key == "test_api_key"

    def test_init_with_cache_hours(self):
        """测试带缓存时间初始化"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener(cache_hours=24)
        assert screener.cache_hours == 24

    def test_init_with_data_provider(self):
        """测试带数据提供者初始化"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        provider = MagicMock()
        screener = DynamicAssetScreener(data_provider=provider)
        assert screener.data_provider is provider


# ============================================================
# Test 3: Candidate Pools
# ============================================================

class TestCandidatePools:
    """测试候选池"""

    def test_candidate_pools_defined(self):
        """测试候选池定义"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        assert DynamicAssetScreener.CANDIDATE_POOLS is not None
        assert "stocks" in DynamicAssetScreener.CANDIDATE_POOLS
        assert "bonds" in DynamicAssetScreener.CANDIDATE_POOLS
        assert "commodities" in DynamicAssetScreener.CANDIDATE_POOLS
        assert "reits" in DynamicAssetScreener.CANDIDATE_POOLS
        assert "crypto" in DynamicAssetScreener.CANDIDATE_POOLS

    def test_stocks_candidate_pool(self):
        """测试股票候选池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        stocks = DynamicAssetScreener.CANDIDATE_POOLS["stocks"]
        assert "tech" in stocks
        assert "financial" in stocks
        assert "AAPL" in stocks["tech"]
        assert "MSFT" in stocks["tech"]

    def test_bonds_candidate_pool(self):
        """测试债券候选池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        bonds = DynamicAssetScreener.CANDIDATE_POOLS["bonds"]
        assert "treasuries" in bonds
        assert "TLT" in bonds["treasuries"]

    def test_commodities_candidate_pool(self):
        """测试商品候选池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        commodities = DynamicAssetScreener.CANDIDATE_POOLS["commodities"]
        assert "precious_metals" in commodities
        assert "GLD" in commodities["precious_metals"]

    def test_reits_candidate_pool(self):
        """测试REITs候选池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        reits = DynamicAssetScreener.CANDIDATE_POOLS["reits"]
        assert "diversified" in reits
        assert "VNQ" in reits["diversified"]

    def test_crypto_candidate_pool(self):
        """测试加密货币候选池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        crypto = DynamicAssetScreener.CANDIDATE_POOLS["crypto"]
        assert "major" in crypto
        assert "BTC-USD" in crypto["major"]


# ============================================================
# Test 4: Screening Configs
# ============================================================

class TestScreeningConfigs:
    """测试筛选配置"""

    def test_screening_configs_defined(self):
        """测试筛选配置定义"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        assert DynamicAssetScreener.SCREENING_CONFIGS is not None
        assert "stocks" in DynamicAssetScreener.SCREENING_CONFIGS
        assert "bonds" in DynamicAssetScreener.SCREENING_CONFIGS

    def test_stocks_screening_config(self):
        """测试股票筛选配置"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        config = DynamicAssetScreener.SCREENING_CONFIGS["stocks"]
        assert "max_assets" in config
        assert "min_score" in config
        assert "core_etfs" in config
        assert config["max_assets"] > 0
        assert 0 <= config["min_score"] <= 1

    def test_core_etfs_in_configs(self):
        """测试核心ETF配置"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        stocks_config = DynamicAssetScreener.SCREENING_CONFIGS["stocks"]
        assert "SPY" in stocks_config["core_etfs"] or "QQQ" in stocks_config["core_etfs"]

        bonds_config = DynamicAssetScreener.SCREENING_CONFIGS["bonds"]
        assert "TLT" in bonds_config["core_etfs"] or "AGG" in bonds_config["core_etfs"]


# ============================================================
# Test 5: Get Candidate Symbols
# ============================================================

class TestGetCandidateSymbols:
    """测试获取候选符号"""

    def test_get_stock_candidates(self):
        """测试获取股票候选"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        candidates = screener._get_candidate_symbols("stocks")

        assert len(candidates) > 0
        assert "AAPL" in candidates
        assert "MSFT" in candidates

    def test_get_bond_candidates(self):
        """测试获取债券候选"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        candidates = screener._get_candidate_symbols("bonds")

        assert len(candidates) > 0
        assert "TLT" in candidates

    def test_get_candidates_no_duplicates(self):
        """测试获取候选没有重复"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        candidates = screener._get_candidate_symbols("stocks")

        assert len(candidates) == len(set(candidates))


# ============================================================
# Test 6: Prepare Factor Data
# ============================================================

class TestPrepareFactorData:
    """测试准备因子数据"""

    def test_prepare_factor_data_with_market_data(self, sample_market_data):
        """测试从市场数据准备因子数据"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        data = screener._prepare_factor_data(
            symbol="AAPL",
            asset_class="stocks",
            market_data=sample_market_data
        )

        assert data is not None
        assert "price" in data
        assert data["price"] == 175.0

    def test_prepare_factor_data_default_values(self):
        """测试准备因子数据默认值"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        data = screener._prepare_factor_data(
            symbol="AAPL",
            asset_class="stocks",
            market_data=None
        )

        assert data is not None
        assert "price" in data
        assert "market_cap" in data

    def test_prepare_factor_data_bonds(self):
        """测试准备债券因子数据"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        data = screener._prepare_factor_data(
            symbol="TLT",
            asset_class="bonds",
            market_data=None
        )

        assert data is not None


# ============================================================
# Test 7: Cache Management
# ============================================================

class TestCacheManagement:
    """测试缓存管理"""

    def test_cache_initialization(self):
        """测试缓存初始化"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        assert screener._cache == {}
        assert screener._cache_time == {}

    def test_is_cache_valid(self):
        """测试缓存有效性检查"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener(cache_hours=1)

        # 空缓存应该无效
        assert not screener._is_cache_valid("test_key")

    def test_update_cache(self):
        """测试更新缓存"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        test_symbols = ["AAPL", "MSFT"]

        screener._update_cache("test_key", test_symbols)

        assert "test_key" in screener._cache
        assert screener._cache["test_key"] == test_symbols
        assert "test_key" in screener._cache_time


# ============================================================
# Test 8: Get Fallback Symbols
# ============================================================

class TestGetFallbackSymbols:
    """测试获取后备符号"""

    def test_get_fallback_symbols_stocks(self):
        """测试获取股票后备符号"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        fallback = screener._get_fallback_symbols("stocks")

        assert len(fallback) > 0
        # 核心 ETF 应该在后备中
        assert "SPY" in fallback or "QQQ" in fallback

    def test_get_fallback_symbols_bonds(self):
        """测试获取债券后备符号"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        fallback = screener._get_fallback_symbols("bonds")

        assert len(fallback) > 0

    def test_get_fallback_symbols_commodities(self):
        """测试获取商品后备符号"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        fallback = screener._get_fallback_symbols("commodities")

        assert len(fallback) > 0

    def test_get_fallback_symbols_unknown_class(self):
        """测试获取未知类别后备符号"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        fallback = screener._get_fallback_symbols("unknown")

        # 应该返回空列表或默认列表
        assert isinstance(fallback, list)


# ============================================================
# Test 9: Get Dynamic Universe
# ============================================================

class TestGetDynamicUniverse:
    """测试获取动态资产池"""

    def test_get_dynamic_universe_stocks(self, sample_market_data):
        """测试获取股票动态资产池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()

        # 使用 fallback (没有 scorer)
        universe = screener.get_dynamic_universe(
            asset_class="stocks",
            market_data=sample_market_data
        )

        assert isinstance(universe, list)
        assert len(universe) > 0

    def test_get_dynamic_universe_with_cache(self, sample_market_data):
        """测试带缓存获取动态资产池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()

        # 首次获取
        universe1 = screener.get_dynamic_universe(
            asset_class="stocks",
            market_data=sample_market_data
        )

        # 第二次应该从缓存获取
        universe2 = screener.get_dynamic_universe(
            asset_class="stocks",
            market_data=sample_market_data
        )

        assert universe1 == universe2

    def test_get_dynamic_universe_force_refresh(self, sample_market_data):
        """测试强制刷新动态资产池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()

        # 首次获取
        universe1 = screener.get_dynamic_universe(
            asset_class="stocks",
            market_data=sample_market_data
        )

        # 强制刷新
        universe2 = screener.get_dynamic_universe(
            asset_class="stocks",
            market_data=sample_market_data,
            force_refresh=True
        )

        assert isinstance(universe2, list)

    def test_get_dynamic_universe_with_regime(self, sample_market_data):
        """测试带市场体制获取动态资产池"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()

        universe = screener.get_dynamic_universe(
            asset_class="stocks",
            market_data=sample_market_data,
            market_regime="bull"
        )

        assert isinstance(universe, list)


# ============================================================
# Test 10: Screen By Factors (with mock)
# ============================================================

class TestScreenByFactors:
    """测试因子筛选"""

    def test_screen_by_factors_no_scorer(self, sample_market_data):
        """测试无评分器时的因子筛选"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()

        # 没有实际的 scorer，应该返回 fallback
        result = screener._screen_by_factors(
            asset_class="stocks",
            market_data=sample_market_data
        )

        assert isinstance(result, list)

    def test_screen_by_factors_with_mocked_scorer(self, sample_market_data, mock_scorer):
        """测试带模拟评分器的因子筛选"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        screener._scorers["stocks"] = mock_scorer

        # 调用 _screen_by_factors
        result = screener._screen_by_factors(
            asset_class="stocks",
            market_data=sample_market_data
        )

        assert isinstance(result, list)


# ============================================================
# Test 11: ScreenedAsset Dataclass
# ============================================================

class TestScreenedAssetDataclass:
    """测试ScreenedAsset数据类"""

    def test_create_screened_asset(self):
        """测试创建ScreenedAsset"""
        from finsage.data.dynamic_screener import ScreenedAsset

        asset = ScreenedAsset(
            symbol="AAPL",
            asset_class="stocks",
            factor_score=0.85,
            signal="STRONG_BUY",
            top_factors=[("momentum", 0.9), ("value", 0.8)],
            expected_alpha=0.05
        )

        assert asset.symbol == "AAPL"
        assert asset.asset_class == "stocks"
        assert asset.factor_score == 0.85
        assert asset.signal == "STRONG_BUY"
        assert len(asset.top_factors) == 2
        assert asset.expected_alpha == 0.05


# ============================================================
# Test 12: Get Scorer
# ============================================================

class TestGetScorer:
    """测试获取评分器"""

    def test_get_scorer_lazy_init(self):
        """测试评分器延迟初始化"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()

        # 初始为空
        assert screener._scorers == {}

    def test_get_scorer_unknown_asset_class(self):
        """测试获取未知资产类别评分器"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        scorer = screener._get_scorer("unknown_asset")

        assert scorer is None


# ============================================================
# Test 13: Additional Asset Class Data Preparation
# ============================================================

class TestAdditionalFactorData:
    """测试其他资产类别的因子数据准备"""

    def test_prepare_factor_data_commodities(self):
        """测试准备商品因子数据"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        data = screener._prepare_factor_data(
            symbol="GLD",
            asset_class="commodities",
            market_data=None
        )

        assert data is not None
        assert "price" in data
        assert "front_price" in data
        assert "back_price" in data
        assert "storage_cost" in data

    def test_prepare_factor_data_reits(self):
        """测试准备REITs因子数据"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        data = screener._prepare_factor_data(
            symbol="VNQ",
            asset_class="reits",
            market_data=None
        )

        assert data is not None
        assert "nav" in data
        assert "dividend_yield" in data
        assert "sector" in data

    def test_prepare_factor_data_crypto(self):
        """测试准备加密货币因子数据"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        data = screener._prepare_factor_data(
            symbol="BTC-USD",
            asset_class="crypto",
            market_data=None
        )

        assert data is not None
        assert "market_cap" in data
        assert "active_addresses" in data
        assert "volatility_30d" in data


# ============================================================
# Test 14: REITs Sector Inference
# ============================================================

class TestREITsSectorInference:
    """测试REITs行业推断"""

    def test_infer_reit_sector_data_center(self):
        """测试数据中心REITs"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        sector = screener._infer_reit_sector("DLR")
        assert sector == "data_center"

    def test_infer_reit_sector_logistics(self):
        """测试物流REITs"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        sector = screener._infer_reit_sector("PLD")
        assert sector == "logistics"

    def test_infer_reit_sector_tower(self):
        """测试通信塔REITs"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        sector = screener._infer_reit_sector("AMT")
        assert sector == "tower"

    def test_infer_reit_sector_unknown(self):
        """测试未知REITs"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        sector = screener._infer_reit_sector("UNKNOWN")
        assert sector == "diversified"


# ============================================================
# Test 15: Cache Validation
# ============================================================

class TestCacheValidation:
    """测试缓存验证"""

    def test_is_cache_valid_no_key(self):
        """测试缓存键不存在"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        assert not screener._is_cache_valid("nonexistent_key")

    def test_is_cache_valid_no_time(self):
        """测试缓存时间不存在"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        screener._cache["test_key"] = ["data"]
        # 没有设置时间

        assert not screener._is_cache_valid("test_key")

    def test_is_cache_valid_fresh(self):
        """测试新鲜缓存"""
        from finsage.data.dynamic_screener import DynamicAssetScreener
        from datetime import datetime

        screener = DynamicAssetScreener()
        screener._cache["test_key"] = ["data"]
        screener._cache_time["test_key"] = datetime.now()

        assert screener._is_cache_valid("test_key")


# ============================================================
# Test 16: Get Fallback Symbols for All Asset Classes
# ============================================================

class TestGetFallbackSymbolsAll:
    """测试所有资产类别的后备符号"""

    def test_get_fallback_symbols_reits(self):
        """测试获取REITs后备符号"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        fallback = screener._get_fallback_symbols("reits")

        assert len(fallback) > 0
        assert "VNQ" in fallback

    def test_get_fallback_symbols_crypto(self):
        """测试获取加密货币后备符号"""
        from finsage.data.dynamic_screener import DynamicAssetScreener

        screener = DynamicAssetScreener()
        fallback = screener._get_fallback_symbols("crypto")

        assert len(fallback) > 0
        assert "BTC-USD" in fallback


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Dynamic Asset Screener Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
