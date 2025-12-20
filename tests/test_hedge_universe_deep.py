"""
Deep tests for Hedge Asset Universe
对冲资产全集深度测试
"""

import pytest

from finsage.hedging.hedge_universe import (
    HedgeCategory,
    HedgeAsset,
    HedgeAssetUniverse
)


class TestHedgeCategory:
    """HedgeCategory枚举测试"""

    def test_all_categories_defined(self):
        """测试所有类别已定义"""
        expected_categories = [
            "inverse_equity", "inverse_sector", "sector_etf",
            "volatility", "safe_haven", "fixed_income",
            "commodity", "international", "currency"
        ]

        for cat in expected_categories:
            assert hasattr(HedgeCategory, cat.upper())

    def test_category_values(self):
        """测试类别值"""
        assert HedgeCategory.INVERSE_EQUITY.value == "inverse_equity"
        assert HedgeCategory.VOLATILITY.value == "volatility"
        assert HedgeCategory.SAFE_HAVEN.value == "safe_haven"


class TestHedgeAsset:
    """HedgeAsset数据类测试"""

    def test_basic_asset_creation(self):
        """测试基本资产创建"""
        asset = HedgeAsset(
            symbol="TEST",
            name="Test Asset",
            category=HedgeCategory.SAFE_HAVEN
        )

        assert asset.symbol == "TEST"
        assert asset.name == "Test Asset"
        assert asset.category == HedgeCategory.SAFE_HAVEN

    def test_default_values(self):
        """测试默认值"""
        asset = HedgeAsset(
            symbol="TEST",
            name="Test Asset",
            category=HedgeCategory.SAFE_HAVEN
        )

        assert asset.leverage == 1.0
        assert asset.expense_ratio == 0.001
        assert asset.avg_daily_volume == 1e6
        assert asset.typical_spread == 0.001
        assert asset.underlying is None
        assert asset.sector is None
        assert asset.tags == []

    def test_custom_values(self):
        """测试自定义值"""
        asset = HedgeAsset(
            symbol="SH",
            name="ProShares Short S&P500",
            category=HedgeCategory.INVERSE_EQUITY,
            leverage=-1.0,
            expense_ratio=0.0089,
            avg_daily_volume=3e6,
            underlying="SPY",
            tags=["sp500", "short"]
        )

        assert asset.leverage == -1.0
        assert asset.expense_ratio == 0.0089
        assert asset.underlying == "SPY"
        assert "sp500" in asset.tags

    def test_is_inverse_property(self):
        """测试反向属性"""
        inverse_asset = HedgeAsset(
            symbol="SH", name="Short", category=HedgeCategory.INVERSE_EQUITY,
            leverage=-1.0
        )
        long_asset = HedgeAsset(
            symbol="SPY", name="Long", category=HedgeCategory.SECTOR_ETF,
            leverage=1.0
        )

        assert inverse_asset.is_inverse == True
        assert long_asset.is_inverse == False

    def test_is_leveraged_property(self):
        """测试杠杆属性"""
        leveraged_asset = HedgeAsset(
            symbol="SDS", name="2x Short", category=HedgeCategory.INVERSE_EQUITY,
            leverage=-2.0
        )
        non_leveraged = HedgeAsset(
            symbol="SH", name="1x Short", category=HedgeCategory.INVERSE_EQUITY,
            leverage=-1.0
        )

        assert leveraged_asset.is_leveraged == True
        assert non_leveraged.is_leveraged == False

    def test_total_cost_estimate(self):
        """测试总成本估算"""
        asset = HedgeAsset(
            symbol="TEST", name="Test", category=HedgeCategory.SAFE_HAVEN,
            expense_ratio=0.004,
            typical_spread=0.001
        )

        expected_cost = 0.004 + 0.001 * 252
        assert abs(asset.total_cost_estimate - expected_cost) < 0.001

    def test_to_dict(self):
        """测试转换为字典"""
        asset = HedgeAsset(
            symbol="GLD", name="Gold ETF", category=HedgeCategory.SAFE_HAVEN,
            leverage=1.0, expense_ratio=0.004, tags=["gold"]
        )

        d = asset.to_dict()

        assert d["symbol"] == "GLD"
        assert d["name"] == "Gold ETF"
        assert d["category"] == "safe_haven"
        assert d["leverage"] == 1.0
        assert d["is_inverse"] == False
        assert "gold" in d["tags"]


class TestHedgeAssetUniverseInit:
    """HedgeAssetUniverse初始化测试"""

    def test_init_creates_assets(self):
        """测试初始化创建资产"""
        universe = HedgeAssetUniverse()

        assert len(universe._assets) > 0

    def test_init_has_many_assets(self):
        """测试初始化有足够多的资产"""
        universe = HedgeAssetUniverse()

        # 应该有70+资产
        assert len(universe._assets) >= 50


class TestGetMethods:
    """获取方法测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_existing_asset(self, universe):
        """测试获取已存在资产"""
        asset = universe.get("SH")

        assert asset is not None
        assert asset.symbol == "SH"

    def test_get_nonexistent_asset(self, universe):
        """测试获取不存在资产"""
        asset = universe.get("NONEXISTENT")

        assert asset is None

    def test_get_all(self, universe):
        """测试获取所有资产"""
        all_assets = universe.get_all()

        assert len(all_assets) > 0
        assert isinstance(all_assets, dict)

    def test_get_all_symbols(self, universe):
        """测试获取所有代码"""
        symbols = universe.get_all_symbols()

        assert len(symbols) > 0
        assert isinstance(symbols, list)
        assert "SH" in symbols


class TestGetByCategory:
    """按类别获取测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_inverse_equity(self, universe):
        """测试获取反向股票ETF"""
        assets = universe.get_by_category(HedgeCategory.INVERSE_EQUITY)

        assert len(assets) > 0
        for asset in assets:
            assert asset.category == HedgeCategory.INVERSE_EQUITY

    def test_get_volatility(self, universe):
        """测试获取波动率工具"""
        assets = universe.get_by_category(HedgeCategory.VOLATILITY)

        assert len(assets) > 0
        for asset in assets:
            assert asset.category == HedgeCategory.VOLATILITY

    def test_get_safe_haven(self, universe):
        """测试获取避险资产"""
        assets = universe.get_by_category(HedgeCategory.SAFE_HAVEN)

        assert len(assets) > 0
        for asset in assets:
            assert asset.category == HedgeCategory.SAFE_HAVEN

    def test_get_fixed_income(self, universe):
        """测试获取固定收益"""
        assets = universe.get_by_category(HedgeCategory.FIXED_INCOME)

        assert len(assets) > 0
        for asset in assets:
            assert asset.category == HedgeCategory.FIXED_INCOME


class TestGetBySector:
    """按行业获取测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_technology_sector(self, universe):
        """测试获取科技行业"""
        assets = universe.get_by_sector("technology")

        assert len(assets) > 0
        for asset in assets:
            assert asset.sector == "technology"

    def test_get_financials_sector(self, universe):
        """测试获取金融行业"""
        assets = universe.get_by_sector("financials")

        assert len(assets) > 0
        for asset in assets:
            assert asset.sector == "financials"

    def test_get_nonexistent_sector(self, universe):
        """测试获取不存在行业"""
        assets = universe.get_by_sector("nonexistent_sector")

        assert assets == []


class TestGetByTag:
    """按标签获取测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_gold_tag(self, universe):
        """测试获取黄金标签"""
        assets = universe.get_by_tag("gold")

        assert len(assets) > 0
        for asset in assets:
            assert "gold" in asset.tags

    def test_get_short_tag(self, universe):
        """测试获取做空标签"""
        assets = universe.get_by_tag("short")

        assert len(assets) > 0
        for asset in assets:
            assert "short" in asset.tags

    def test_get_leveraged_tag(self, universe):
        """测试获取杠杆标签"""
        assets = universe.get_by_tag("leveraged")

        assert len(assets) > 0
        for asset in assets:
            assert "leveraged" in asset.tags

    def test_get_nonexistent_tag(self, universe):
        """测试获取不存在标签"""
        assets = universe.get_by_tag("nonexistent_tag")

        assert assets == []


class TestGetInverseAssets:
    """获取反向资产测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_inverse_assets(self, universe):
        """测试获取所有反向资产"""
        inverse_assets = universe.get_inverse_assets()

        assert len(inverse_assets) > 0
        for asset in inverse_assets:
            assert asset.is_inverse == True
            assert asset.leverage < 0


class TestGetLeveragedAssets:
    """获取杠杆资产测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_leveraged_assets(self, universe):
        """测试获取所有杠杆资产"""
        leveraged_assets = universe.get_leveraged_assets()

        assert len(leveraged_assets) > 0
        for asset in leveraged_assets:
            assert asset.is_leveraged == True
            assert abs(asset.leverage) > 1


class TestGetInverseFor:
    """获取特定标的反向ETF测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_get_inverse_for_spy(self, universe):
        """测试获取SPY的反向ETF"""
        inverse_spy = universe.get_inverse_for("SPY")

        assert len(inverse_spy) > 0
        for asset in inverse_spy:
            assert asset.underlying == "SPY"
            assert asset.is_inverse == True

    def test_get_inverse_for_qqq(self, universe):
        """测试获取QQQ的反向ETF"""
        inverse_qqq = universe.get_inverse_for("QQQ")

        assert len(inverse_qqq) > 0
        for asset in inverse_qqq:
            assert asset.underlying == "QQQ"
            assert asset.is_inverse == True

    def test_get_inverse_for_nonexistent(self, universe):
        """测试获取不存在标的的反向ETF"""
        inverse = universe.get_inverse_for("NONEXISTENT")

        assert inverse == []


class TestFilter:
    """多条件过滤测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_filter_by_category(self, universe):
        """测试按类别过滤"""
        result = universe.filter(category=HedgeCategory.SAFE_HAVEN)

        assert len(result) > 0
        for asset in result:
            assert asset.category == HedgeCategory.SAFE_HAVEN

    def test_filter_by_sector(self, universe):
        """测试按行业过滤"""
        result = universe.filter(sector="technology")

        assert len(result) > 0
        for asset in result:
            assert asset.sector == "technology"

    def test_filter_by_tags(self, universe):
        """测试按标签过滤"""
        result = universe.filter(tags=["gold", "silver"])

        assert len(result) > 0
        for asset in result:
            assert any(t in asset.tags for t in ["gold", "silver"])

    def test_filter_by_min_volume(self, universe):
        """测试按最小成交量过滤"""
        min_vol = 1e7
        result = universe.filter(min_volume=min_vol)

        for asset in result:
            assert asset.avg_daily_volume >= min_vol

    def test_filter_by_max_expense(self, universe):
        """测试按最大费率过滤"""
        max_exp = 0.005
        result = universe.filter(max_expense=max_exp)

        for asset in result:
            assert asset.expense_ratio <= max_exp

    def test_filter_inverse_only(self, universe):
        """测试仅反向资产"""
        result = universe.filter(inverse_only=True)

        assert len(result) > 0
        for asset in result:
            assert asset.is_inverse == True

    def test_filter_leveraged_only(self, universe):
        """测试仅杠杆资产"""
        result = universe.filter(leveraged_only=True)

        assert len(result) > 0
        for asset in result:
            assert asset.is_leveraged == True

    def test_filter_multiple_conditions(self, universe):
        """测试多条件过滤"""
        result = universe.filter(
            category=HedgeCategory.INVERSE_EQUITY,
            inverse_only=True,
            leveraged_only=True
        )

        for asset in result:
            assert asset.category == HedgeCategory.INVERSE_EQUITY
            assert asset.is_inverse == True
            assert asset.is_leveraged == True


class TestSummary:
    """资产统计摘要测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_summary_structure(self, universe):
        """测试摘要结构"""
        summary = universe.summary()

        assert "total" in summary
        assert "inverse" in summary
        assert "leveraged" in summary

    def test_summary_totals(self, universe):
        """测试摘要总数"""
        summary = universe.summary()

        assert summary["total"] > 0
        assert summary["inverse"] > 0
        assert summary["leveraged"] > 0

    def test_summary_categories(self, universe):
        """测试摘要类别计数"""
        summary = universe.summary()

        for category in HedgeCategory:
            assert category.value in summary


class TestSpecificAssets:
    """特定资产测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_sh_exists(self, universe):
        """测试SH存在"""
        asset = universe.get("SH")

        assert asset is not None
        assert asset.name == "ProShares Short S&P500"
        assert asset.leverage == -1.0

    def test_sqqq_exists(self, universe):
        """测试SQQQ存在"""
        asset = universe.get("SQQQ")

        assert asset is not None
        assert asset.leverage == -3.0
        assert "nasdaq" in asset.tags

    def test_gld_exists(self, universe):
        """测试GLD存在"""
        asset = universe.get("GLD")

        assert asset is not None
        assert asset.category == HedgeCategory.SAFE_HAVEN
        assert "gold" in asset.tags

    def test_tlt_exists(self, universe):
        """测试TLT存在"""
        asset = universe.get("TLT")

        assert asset is not None
        assert asset.category == HedgeCategory.FIXED_INCOME
        assert "treasury" in asset.tags

    def test_vxx_exists(self, universe):
        """测试VXX存在"""
        asset = universe.get("VXX")

        assert asset is not None
        assert asset.category == HedgeCategory.VOLATILITY
        assert "vix" in asset.tags


class TestRegistrationMethods:
    """注册方法测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_inverse_equity_registered(self, universe):
        """测试反向股票ETF已注册"""
        assets = universe.get_by_category(HedgeCategory.INVERSE_EQUITY)
        assert len(assets) >= 10

    def test_inverse_sector_registered(self, universe):
        """测试反向行业ETF已注册"""
        assets = universe.get_by_category(HedgeCategory.INVERSE_SECTOR)
        assert len(assets) >= 5

    def test_sector_etf_registered(self, universe):
        """测试行业ETF已注册"""
        assets = universe.get_by_category(HedgeCategory.SECTOR_ETF)
        assert len(assets) >= 5

    def test_volatility_registered(self, universe):
        """测试波动率工具已注册"""
        assets = universe.get_by_category(HedgeCategory.VOLATILITY)
        assert len(assets) >= 3

    def test_safe_haven_registered(self, universe):
        """测试避险资产已注册"""
        assets = universe.get_by_category(HedgeCategory.SAFE_HAVEN)
        assert len(assets) >= 3

    def test_fixed_income_registered(self, universe):
        """测试固定收益已注册"""
        assets = universe.get_by_category(HedgeCategory.FIXED_INCOME)
        assert len(assets) >= 5

    def test_commodity_registered(self, universe):
        """测试商品已注册"""
        assets = universe.get_by_category(HedgeCategory.COMMODITY)
        assert len(assets) >= 3

    def test_international_registered(self, universe):
        """测试国际市场已注册"""
        assets = universe.get_by_category(HedgeCategory.INTERNATIONAL)
        assert len(assets) >= 5

    def test_currency_registered(self, universe):
        """测试货币已注册"""
        assets = universe.get_by_category(HedgeCategory.CURRENCY)
        assert len(assets) >= 3


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def universe(self):
        return HedgeAssetUniverse()

    def test_full_workflow(self, universe):
        """测试完整工作流"""
        # 获取所有资产
        all_assets = universe.get_all()
        assert len(all_assets) > 0

        # 按类别筛选
        inverse_equity = universe.get_by_category(HedgeCategory.INVERSE_EQUITY)
        assert len(inverse_equity) > 0

        # 按标签筛选
        gold_assets = universe.get_by_tag("gold")
        assert len(gold_assets) > 0

        # 获取特定标的反向ETF
        spy_inverse = universe.get_inverse_for("SPY")
        assert len(spy_inverse) > 0

        # 多条件过滤
        filtered = universe.filter(
            category=HedgeCategory.INVERSE_EQUITY,
            min_volume=1e6,
            leveraged_only=True
        )
        assert all(a.is_leveraged for a in filtered)

        # 获取摘要
        summary = universe.summary()
        assert summary["total"] > 0

    def test_consistency(self, universe):
        """测试一致性"""
        # 所有反向资产应该有负杠杆
        inverse = universe.get_inverse_assets()
        for asset in inverse:
            assert asset.leverage < 0

        # 所有杠杆资产应该有>1的绝对杠杆
        leveraged = universe.get_leveraged_assets()
        for asset in leveraged:
            assert abs(asset.leverage) > 1

        # 类别计数应该匹配
        summary = universe.summary()
        total_by_category = sum(
            len(universe.get_by_category(cat))
            for cat in HedgeCategory
        )
        assert total_by_category == summary["total"]
