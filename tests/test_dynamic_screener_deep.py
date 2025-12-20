"""
Deep tests for DynamicAssetScreener

覆盖 finsage/data/dynamic_screener.py (目标从23%提升到80%+)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from finsage.data.dynamic_screener import (
    DynamicAssetScreener,
    ScreenedAsset,
    DynamicStockScreener,
)


# ============================================================
# ScreenedAsset Tests
# ============================================================

class TestScreenedAsset:
    """测试ScreenedAsset数据类"""

    def test_create_screened_asset(self):
        """测试创建ScreenedAsset"""
        asset = ScreenedAsset(
            symbol="AAPL",
            asset_class="stocks",
            factor_score=0.75,
            signal="BUY",
            top_factors=[("momentum", 0.8), ("value", 0.6)],
            expected_alpha=0.05
        )

        assert asset.symbol == "AAPL"
        assert asset.asset_class == "stocks"
        assert asset.factor_score == 0.75
        assert asset.signal == "BUY"
        assert len(asset.top_factors) == 2
        assert asset.expected_alpha == 0.05

    def test_screened_asset_with_strong_signals(self):
        """测试强信号资产"""
        asset = ScreenedAsset(
            symbol="NVDA",
            asset_class="stocks",
            factor_score=0.95,
            signal="STRONG_BUY",
            top_factors=[("momentum", 0.95), ("profitability", 0.9), ("size", 0.85)],
            expected_alpha=0.12
        )

        assert asset.signal == "STRONG_BUY"
        assert asset.factor_score > 0.9
        assert len(asset.top_factors) == 3

    def test_screened_asset_crypto(self):
        """测试加密货币资产"""
        asset = ScreenedAsset(
            symbol="BTC-USD",
            asset_class="crypto",
            factor_score=0.65,
            signal="HOLD",
            top_factors=[("network", 0.7), ("adoption", 0.6)],
            expected_alpha=0.03
        )

        assert asset.asset_class == "crypto"
        assert "BTC" in asset.symbol


# ============================================================
# DynamicAssetScreener Tests
# ============================================================

class TestDynamicAssetScreenerInit:
    """测试DynamicAssetScreener初始化"""

    def test_init_without_api_key(self):
        """测试不提供API密钥的初始化"""
        with patch.dict('os.environ', {}, clear=True):
            screener = DynamicAssetScreener(api_key=None)
            assert screener.api_key is None
            assert screener._cache == {}
            assert screener._scorers == {}

    def test_init_with_api_key(self):
        """测试提供API密钥的初始化"""
        screener = DynamicAssetScreener(api_key="test_key_123")
        assert screener.api_key == "test_key_123"

    def test_init_with_custom_cache_hours(self):
        """测试自定义缓存时间"""
        screener = DynamicAssetScreener(cache_hours=24)
        assert screener.cache_hours == 24

    def test_init_with_data_provider(self):
        """测试提供数据提供者"""
        mock_provider = Mock()
        screener = DynamicAssetScreener(data_provider=mock_provider)
        assert screener.data_provider == mock_provider

    def test_init_with_env_api_key(self):
        """测试从环境变量获取API密钥"""
        with patch.dict('os.environ', {'FMP_API_KEY': 'env_test_key'}):
            screener = DynamicAssetScreener()
            assert screener.api_key == 'env_test_key'


class TestDynamicAssetScreenerGetScorer:
    """测试因子评分器获取"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener(api_key="test_key")

    def test_get_scorer_stocks(self, screener):
        """测试获取股票评分器"""
        scorer = screener._get_scorer("stocks")
        assert scorer is not None
        assert screener._scorers.get("stocks") is not None

    def test_get_scorer_bonds(self, screener):
        """测试获取债券评分器"""
        scorer = screener._get_scorer("bonds")
        assert scorer is not None

    def test_get_scorer_commodities(self, screener):
        """测试获取商品评分器"""
        scorer = screener._get_scorer("commodities")
        assert scorer is not None

    def test_get_scorer_reits(self, screener):
        """测试获取REITs评分器"""
        scorer = screener._get_scorer("reits")
        assert scorer is not None

    def test_get_scorer_crypto(self, screener):
        """测试获取加密货币评分器"""
        scorer = screener._get_scorer("crypto")
        assert scorer is not None

    def test_get_scorer_unknown_asset_class(self, screener):
        """测试获取未知资产类别评分器"""
        scorer = screener._get_scorer("unknown_asset")
        assert scorer is None

    def test_get_scorer_caching(self, screener):
        """测试评分器缓存"""
        scorer1 = screener._get_scorer("stocks")
        scorer2 = screener._get_scorer("stocks")
        assert scorer1 is scorer2  # 应该是同一个对象


class TestDynamicAssetScreenerCandidatePools:
    """测试候选池获取"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_get_candidate_symbols_stocks(self, screener):
        """测试获取股票候选池"""
        symbols = screener._get_candidate_symbols("stocks")
        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "SPY" in symbols

    def test_get_candidate_symbols_bonds(self, screener):
        """测试获取债券候选池"""
        symbols = screener._get_candidate_symbols("bonds")
        assert len(symbols) > 0
        assert "TLT" in symbols
        assert "AGG" in symbols

    def test_get_candidate_symbols_commodities(self, screener):
        """测试获取商品候选池"""
        symbols = screener._get_candidate_symbols("commodities")
        assert len(symbols) > 0
        assert "GLD" in symbols
        assert "USO" in symbols

    def test_get_candidate_symbols_reits(self, screener):
        """测试获取REITs候选池"""
        symbols = screener._get_candidate_symbols("reits")
        assert len(symbols) > 0
        assert "VNQ" in symbols

    def test_get_candidate_symbols_crypto(self, screener):
        """测试获取加密货币候选池"""
        symbols = screener._get_candidate_symbols("crypto")
        assert len(symbols) > 0
        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_get_candidate_symbols_no_duplicates(self, screener):
        """测试候选池无重复"""
        symbols = screener._get_candidate_symbols("stocks")
        assert len(symbols) == len(set(symbols))

    def test_get_candidate_symbols_unknown(self, screener):
        """测试获取未知资产类别的候选池"""
        symbols = screener._get_candidate_symbols("unknown")
        assert symbols == []


class TestDynamicAssetScreenerPrepareFactorData:
    """测试因子数据准备"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_prepare_factor_data_stocks_default(self, screener):
        """测试股票默认数据准备"""
        data = screener._prepare_factor_data("AAPL", "stocks")

        assert "price" in data
        assert "market_cap" in data
        assert "pe_ratio" in data
        assert "book_to_market" in data
        assert "beta" in data
        assert data["price"] == 100  # 默认值

    def test_prepare_factor_data_stocks_with_market_data(self, screener):
        """测试股票使用市场数据"""
        market_data = {
            "AAPL": {
                "close": 150.0,
                "volume": 1000000,
                "return_1m": 0.05,
                "return_12m": 0.15
            }
        }

        data = screener._prepare_factor_data("AAPL", "stocks", market_data)

        assert data["price"] == 150.0
        assert data["volume"] == 1000000
        assert data["price_change_1m"] == 0.05
        assert data["price_change_12m"] == 0.15

    def test_prepare_factor_data_bonds(self, screener):
        """测试债券数据准备"""
        data = screener._prepare_factor_data("TLT", "bonds")

        assert "yield_to_maturity" in data
        assert "duration" in data
        assert "credit_spread" in data
        assert "yield_curve_slope" in data

    def test_prepare_factor_data_commodities(self, screener):
        """测试商品数据准备"""
        data = screener._prepare_factor_data("GLD", "commodities")

        assert "front_price" in data
        assert "back_price" in data
        assert "spot_price" in data
        assert "storage_cost" in data

    def test_prepare_factor_data_reits(self, screener):
        """测试REITs数据准备"""
        data = screener._prepare_factor_data("VNQ", "reits")

        assert "nav" in data
        assert "nav_premium" in data
        assert "dividend_yield" in data
        assert "cap_rate" in data
        assert "sector" in data

    def test_prepare_factor_data_crypto(self, screener):
        """测试加密货币数据准备"""
        data = screener._prepare_factor_data("BTC-USD", "crypto")

        assert "market_cap" in data
        assert "active_addresses" in data
        assert "volatility_30d" in data
        assert "max_drawdown_90d" in data
        assert "funding_rate" in data

    @patch("requests.get")
    def test_prepare_factor_data_with_api(self, mock_get, screener):
        """测试使用API获取基本面数据"""
        screener.api_key = "test_key"
        mock_response = Mock()
        mock_response.json.return_value = [{
            "mktCap": 3e12,
            "beta": 1.2,
            "peRatio": 25
        }]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        data = screener._prepare_factor_data("AAPL", "stocks")

        assert data["market_cap"] == 3e12
        assert data["beta"] == 1.2
        assert data["pe_ratio"] == 25


class TestDynamicAssetScreenerInferReitSector:
    """测试REITs行业推断"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_infer_data_center(self, screener):
        """测试数据中心REITs"""
        assert screener._infer_reit_sector("DLR") == "data_center"
        assert screener._infer_reit_sector("EQIX") == "data_center"

    def test_infer_logistics(self, screener):
        """测试物流REITs"""
        assert screener._infer_reit_sector("PLD") == "logistics"
        assert screener._infer_reit_sector("STAG") == "logistics"

    def test_infer_residential(self, screener):
        """测试住宅REITs"""
        assert screener._infer_reit_sector("EQR") == "residential"
        assert screener._infer_reit_sector("AVB") == "residential"

    def test_infer_healthcare(self, screener):
        """测试医疗REITs"""
        assert screener._infer_reit_sector("WELL") == "healthcare"
        assert screener._infer_reit_sector("VTR") == "healthcare"

    def test_infer_retail(self, screener):
        """测试零售REITs"""
        assert screener._infer_reit_sector("SPG") == "retail"
        assert screener._infer_reit_sector("O") == "retail"

    def test_infer_office(self, screener):
        """测试办公REITs"""
        assert screener._infer_reit_sector("BXP") == "office"
        assert screener._infer_reit_sector("ARE") == "office"

    def test_infer_tower(self, screener):
        """测试通信塔REITs"""
        assert screener._infer_reit_sector("AMT") == "tower"
        assert screener._infer_reit_sector("CCI") == "tower"

    def test_infer_diversified(self, screener):
        """测试综合REITs"""
        assert screener._infer_reit_sector("VNQ") == "diversified"

    def test_infer_unknown(self, screener):
        """测试未知REITs"""
        assert screener._infer_reit_sector("UNKNOWN_REIT") == "diversified"


class TestDynamicAssetScreenerFallback:
    """测试回退符号列表"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_get_fallback_symbols_stocks(self, screener):
        """测试股票回退列表"""
        symbols = screener._get_fallback_symbols("stocks")
        assert len(symbols) > 0
        assert "SPY" in symbols
        assert "QQQ" in symbols

    def test_get_fallback_symbols_bonds(self, screener):
        """测试债券回退列表"""
        symbols = screener._get_fallback_symbols("bonds")
        assert "TLT" in symbols
        assert "AGG" in symbols

    def test_get_fallback_symbols_commodities(self, screener):
        """测试商品回退列表"""
        symbols = screener._get_fallback_symbols("commodities")
        assert "GLD" in symbols

    def test_get_fallback_symbols_reits(self, screener):
        """测试REITs回退列表"""
        symbols = screener._get_fallback_symbols("reits")
        assert "VNQ" in symbols

    def test_get_fallback_symbols_crypto(self, screener):
        """测试加密货币回退列表"""
        symbols = screener._get_fallback_symbols("crypto")
        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_get_fallback_symbols_unknown(self, screener):
        """测试未知资产类别回退列表"""
        symbols = screener._get_fallback_symbols("unknown")
        assert symbols == []


class TestDynamicAssetScreenerCache:
    """测试缓存功能"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener(cache_hours=1)

    def test_cache_not_valid_empty(self, screener):
        """测试空缓存无效"""
        assert screener._is_cache_valid("test_key") is False

    def test_update_and_check_cache(self, screener):
        """测试更新和检查缓存"""
        screener._update_cache("test_key", ["AAPL", "MSFT"])

        assert screener._is_cache_valid("test_key") is True
        assert screener._cache["test_key"] == ["AAPL", "MSFT"]

    def test_cache_expiration(self, screener):
        """测试缓存过期"""
        screener._update_cache("test_key", ["AAPL"])

        # 手动设置过期时间
        screener._cache_time["test_key"] = datetime.now() - timedelta(hours=2)

        assert screener._is_cache_valid("test_key") is False

    def test_cache_missing_time(self, screener):
        """测试缺少时间戳的缓存"""
        screener._cache["test_key"] = ["AAPL"]
        # 不设置 _cache_time

        assert screener._is_cache_valid("test_key") is False


class TestDynamicAssetScreenerGetDynamicUniverse:
    """测试获取动态资产池"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_get_dynamic_universe_stocks(self, screener):
        """测试获取股票动态池"""
        symbols = screener.get_dynamic_universe("stocks")

        assert len(symbols) > 0
        # 应该包含核心ETF
        assert "SPY" in symbols or "QQQ" in symbols

    def test_get_dynamic_universe_bonds(self, screener):
        """测试获取债券动态池"""
        symbols = screener.get_dynamic_universe("bonds")

        assert len(symbols) > 0

    def test_get_dynamic_universe_with_cache(self, screener):
        """测试使用缓存"""
        # 第一次调用
        symbols1 = screener.get_dynamic_universe("stocks")

        # 第二次调用应该使用缓存
        symbols2 = screener.get_dynamic_universe("stocks")

        assert symbols1 == symbols2

    def test_get_dynamic_universe_force_refresh(self, screener):
        """测试强制刷新"""
        # 先填充缓存
        screener._update_cache("stocks_current_default", ["OLD_SYMBOL"])

        # 强制刷新
        symbols = screener.get_dynamic_universe("stocks", force_refresh=True)

        assert "OLD_SYMBOL" not in symbols

    def test_get_dynamic_universe_with_date(self, screener):
        """测试指定日期"""
        symbols = screener.get_dynamic_universe("stocks", date="2024-01-15")

        assert len(symbols) > 0

    def test_get_dynamic_universe_with_market_regime(self, screener):
        """测试指定市场状态"""
        symbols = screener.get_dynamic_universe("stocks", market_regime="bull")

        assert len(symbols) > 0

    def test_get_dynamic_universe_with_returns_df(self, screener):
        """测试提供收益率数据"""
        returns_df = pd.DataFrame({
            "AAPL": np.random.randn(100) * 0.02,
            "MSFT": np.random.randn(100) * 0.02,
        })

        symbols = screener.get_dynamic_universe(
            "stocks",
            returns_df=returns_df,
            force_refresh=True
        )

        assert len(symbols) > 0


class TestDynamicAssetScreenerScreenByFactors:
    """测试因子筛选"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_screen_by_factors_stocks(self, screener):
        """测试股票因子筛选"""
        symbols = screener._screen_by_factors("stocks")

        assert len(symbols) > 0
        # 核心ETF应该被包含
        has_core = "SPY" in symbols or "QQQ" in symbols
        assert has_core

    def test_screen_by_factors_with_market_data(self, screener):
        """测试使用市场数据进行因子筛选"""
        market_data = {
            "AAPL": {"close": 150, "return_1m": 0.05},
            "MSFT": {"close": 350, "return_1m": 0.03},
        }

        symbols = screener._screen_by_factors("stocks", market_data=market_data)

        assert len(symbols) > 0

    def test_screen_by_factors_no_scorer(self, screener):
        """测试无评分器时回退"""
        with patch.object(screener, '_get_scorer', return_value=None):
            symbols = screener._screen_by_factors("stocks")

            # 应该返回回退列表
            assert len(symbols) > 0


class TestDynamicAssetScreenerSectorRotation:
    """测试行业轮动"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_sector_rotation_risk_on(self, screener):
        """测试风险偏好市场"""
        picks = screener.get_sector_rotation_picks(market_regime="risk_on")

        assert "stocks" in picks
        assert "commodities" in picks
        assert "crypto" in picks
        assert "bonds" not in picks

    def test_sector_rotation_risk_off(self, screener):
        """测试风险厌恶市场"""
        picks = screener.get_sector_rotation_picks(market_regime="risk_off")

        assert "bonds" in picks
        assert "commodities" in picks  # 黄金作为避险
        assert "crypto" not in picks

    def test_sector_rotation_neutral(self, screener):
        """测试中性市场"""
        picks = screener.get_sector_rotation_picks(market_regime="neutral")

        assert "stocks" in picks
        assert "bonds" in picks
        assert "commodities" in picks

    def test_sector_rotation_max_picks(self, screener):
        """测试每类最多10个"""
        picks = screener.get_sector_rotation_picks(market_regime="neutral")

        for asset_class, symbols in picks.items():
            assert len(symbols) <= 10


class TestDynamicAssetScreenerRefreshAll:
    """测试刷新所有池"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_refresh_all(self, screener):
        """测试刷新所有资产类别"""
        result = screener.refresh_all()

        assert "stocks" in result
        assert "bonds" in result
        assert "commodities" in result
        assert "reits" in result
        assert "crypto" in result

    def test_refresh_all_with_market_data(self, screener):
        """测试使用市场数据刷新"""
        market_data = {"AAPL": {"close": 150}}
        returns_df = pd.DataFrame({"AAPL": np.random.randn(50)})

        result = screener.refresh_all(
            market_data=market_data,
            returns_df=returns_df
        )

        assert len(result) == 5  # 5个资产类别


class TestDynamicAssetScreenerFactorRankings:
    """测试因子排名"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_get_factor_rankings_stocks(self, screener):
        """测试股票因子排名"""
        rankings = screener.get_factor_rankings("stocks", top_n=10)

        assert len(rankings) <= 10
        if rankings:
            assert all(isinstance(r, ScreenedAsset) for r in rankings)
            assert all(r.asset_class == "stocks" for r in rankings)
            # 应该按评分降序排列
            scores = [r.factor_score for r in rankings]
            assert scores == sorted(scores, reverse=True)

    def test_get_factor_rankings_no_scorer(self, screener):
        """测试无评分器返回空列表"""
        with patch.object(screener, '_get_scorer', return_value=None):
            rankings = screener.get_factor_rankings("stocks")
            assert rankings == []

    def test_get_factor_rankings_with_returns(self, screener):
        """测试使用收益率数据"""
        returns_df = pd.DataFrame({
            "AAPL": np.random.randn(100) * 0.02,
            "MSFT": np.random.randn(100) * 0.02,
        })

        rankings = screener.get_factor_rankings(
            "stocks",
            returns_df=returns_df,
            top_n=5
        )

        assert len(rankings) <= 5


class TestDynamicAssetScreenerFetchFundamental:
    """测试获取基本面数据"""

    @pytest.fixture
    def screener(self):
        screener = DynamicAssetScreener(api_key="test_key")
        return screener

    @patch("requests.get")
    def test_fetch_fundamental_data_success(self, mock_get, screener):
        """测试成功获取基本面数据"""
        mock_response = Mock()
        mock_response.json.return_value = [{
            "mktCap": 2.5e12,
            "beta": 1.1,
            "peRatio": 28
        }]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        data = screener._fetch_fundamental_data("AAPL")

        assert data["market_cap"] == 2.5e12
        assert data["beta"] == 1.1
        assert data["pe_ratio"] == 28

    @patch("requests.get")
    def test_fetch_fundamental_data_empty_response(self, mock_get, screener):
        """测试空响应"""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        data = screener._fetch_fundamental_data("AAPL")

        assert data == {}

    @patch("requests.get")
    def test_fetch_fundamental_data_error(self, mock_get, screener):
        """测试API错误"""
        mock_get.side_effect = Exception("API Error")

        data = screener._fetch_fundamental_data("AAPL")

        assert data == {}

    @patch("requests.get")
    def test_fetch_fundamental_data_dict_response(self, mock_get, screener):
        """测试字典格式响应"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "mktCap": 1e12,
            "beta": 0.9,
            "peRatio": 15
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        data = screener._fetch_fundamental_data("AAPL")

        assert data["market_cap"] == 1e12

    @patch("requests.get")
    def test_fetch_fundamental_data_missing_pe(self, mock_get, screener):
        """测试缺少PE的情况"""
        mock_response = Mock()
        mock_response.json.return_value = [{
            "mktCap": 1e12,
            "beta": 1.0,
            "peRatio": None
        }]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        data = screener._fetch_fundamental_data("AAPL")

        assert data["pe_ratio"] == 20  # 默认值


class TestDynamicStockScreenerAlias:
    """测试向后兼容别名"""

    def test_alias_exists(self):
        """测试别名存在"""
        assert DynamicStockScreener is DynamicAssetScreener

    def test_alias_can_instantiate(self):
        """测试别名可以实例化"""
        screener = DynamicStockScreener()
        assert isinstance(screener, DynamicAssetScreener)


class TestDynamicAssetScreenerEdgeCases:
    """测试边界情况"""

    @pytest.fixture
    def screener(self):
        return DynamicAssetScreener()

    def test_prepare_factor_data_with_price_key(self, screener):
        """测试使用price键代替close"""
        market_data = {
            "AAPL": {"price": 200.0}
        }

        data = screener._prepare_factor_data("AAPL", "stocks", market_data)

        assert data["price"] == 200.0

    def test_screen_by_factors_exception_handling(self, screener):
        """测试筛选异常处理"""
        # 创建一个会抛出异常的mock scorer
        mock_scorer = Mock()
        mock_scorer.score.side_effect = Exception("Score error")

        with patch.object(screener, '_get_scorer', return_value=mock_scorer):
            # 应该返回回退列表而不是崩溃
            symbols = screener._screen_by_factors("stocks")
            assert len(symbols) > 0

    def test_get_dynamic_universe_fallback_on_error(self, screener):
        """测试获取动态池时发生错误后回退"""
        with patch.object(screener, '_screen_by_factors', side_effect=Exception("Error")):
            symbols = screener.get_dynamic_universe("stocks", force_refresh=True)

            # 应该返回回退列表
            assert len(symbols) > 0
