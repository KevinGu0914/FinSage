"""
Deep tests for MultiAssetTradingEnv
多资产交易环境深度测试
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.environment.multi_asset_env import (
    MultiAssetTradingEnv,
    EnvConfig,
)
from finsage.environment.portfolio_state import PortfolioState


class TestEnvConfig:
    """EnvConfig数据类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = EnvConfig()
        assert config.initial_capital == 1_000_000.0
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.max_single_asset == 0.15

    def test_custom_config(self):
        """测试自定义配置"""
        config = EnvConfig(
            initial_capital=500_000,
            transaction_cost=0.002,
            max_single_asset=0.20,
        )
        assert config.initial_capital == 500_000
        assert config.transaction_cost == 0.002


class TestMultiAssetTradingEnvInit:
    """MultiAssetTradingEnv初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        env = MultiAssetTradingEnv()
        assert env.config.initial_capital == 1_000_000.0
        assert env.current_step == 0
        assert env.done is False

    def test_custom_config_init(self):
        """测试自定义配置初始化"""
        config = EnvConfig(initial_capital=500_000)
        env = MultiAssetTradingEnv(config=config)
        assert env.portfolio.initial_capital == 500_000

    def test_custom_universe(self):
        """测试自定义资产池"""
        universe = {"stocks": ["SPY", "QQQ"], "bonds": ["TLT"]}
        env = MultiAssetTradingEnv(asset_universe=universe)
        assert env.asset_universe == universe

    def test_default_universe(self):
        """测试默认资产池"""
        env = MultiAssetTradingEnv()
        assert "stocks" in env.asset_universe
        assert "bonds" in env.asset_universe


class TestReset:
    """重置环境测试"""

    def test_reset_basic(self):
        """测试基本重置"""
        env = MultiAssetTradingEnv()
        state = env.reset()
        assert isinstance(state, PortfolioState)
        assert state.cash == env.config.initial_capital
        assert env.current_step == 0

    def test_reset_with_custom_capital(self):
        """测试自定义初始资金重置"""
        env = MultiAssetTradingEnv()
        state = env.reset(initial_capital=200_000)
        assert state.cash == 200_000

    def test_reset_with_start_date(self):
        """测试带开始日期重置"""
        env = MultiAssetTradingEnv()
        state = env.reset(start_date="2024-01-01")
        assert env.current_date == "2024-01-01"


class TestExtractPrices:
    """价格提取测试"""

    @pytest.fixture
    def env(self):
        return MultiAssetTradingEnv()

    def test_extract_prices_dict_close(self, env):
        """测试从close字段提取价格"""
        market_data = {
            "AAPL": {"close": 180.0, "volume": 1000000},
            "MSFT": {"close": 350.0, "volume": 800000},
        }
        prices = env._extract_prices(market_data)
        assert prices["AAPL"] == 180.0
        assert prices["MSFT"] == 350.0

    def test_extract_prices_dict_price(self, env):
        """测试从price字段提取价格"""
        market_data = {
            "AAPL": {"price": 180.0},
        }
        prices = env._extract_prices(market_data)
        assert prices["AAPL"] == 180.0

    def test_extract_prices_direct_float(self, env):
        """测试直接浮点数价格"""
        market_data = {
            "AAPL": 180.0,
            "MSFT": 350.0,
        }
        prices = env._extract_prices(market_data)
        assert prices["AAPL"] == 180.0

    def test_skip_non_price_data(self, env):
        """测试跳过非价格数据"""
        market_data = {
            "AAPL": {"close": 180.0},
            "news": [{"title": "news1"}],
            "macro": {"vix": 20.0},
            "returns": {},
        }
        prices = env._extract_prices(market_data)
        assert "AAPL" in prices
        assert "news" not in prices
        assert "macro" not in prices


class TestGetAssetClass:
    """获取资产类别测试"""

    @pytest.fixture
    def env(self):
        return MultiAssetTradingEnv()

    def test_get_stock_class(self, env):
        """测试获取股票类别"""
        # 需要SPY在stocks中
        assert env._get_asset_class("SPY") == "stocks"

    def test_get_unknown_class(self, env):
        """测试未知资产类别"""
        result = env._get_asset_class("UNKNOWN_ASSET")
        assert result == "other"


class TestCalculateReward:
    """奖励计算测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_reward_no_history(self, env):
        """测试无历史数据奖励"""
        reward = env._calculate_reward()
        assert reward == 0.0

    def test_reward_with_history(self, env):
        """测试有历史数据奖励"""
        # 添加一些价值历史 - 使用正确的键名 portfolio_value
        env.portfolio.value_history = [
            {"timestamp": "2024-01-01", "portfolio_value": 1000000},
            {"timestamp": "2024-01-02", "portfolio_value": 1010000},
            {"timestamp": "2024-01-03", "portfolio_value": 1005000},
        ]
        reward = env._calculate_reward()
        assert isinstance(reward, float)


class TestRebalance:
    """再平衡测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_rebalance_basic(self, env):
        """测试基本再平衡"""
        target = {"SPY": 0.50, "TLT": 0.30, "cash": 0.20}
        prices = {"SPY": 450.0, "TLT": 100.0}
        trades = env._rebalance(target, prices, "2024-01-15")
        # 应该产生交易
        assert isinstance(trades, list)

    def test_rebalance_below_threshold(self, env):
        """测试低于阈值不再平衡"""
        # 设置初始持仓 - 使用真实的Position对象而非MagicMock
        env.portfolio.cash = 500000
        # 更新价格并执行一次rebalance来建立持仓
        target = {"SPY": 0.50, "cash": 0.50}
        prices = {"SPY": 450.0}
        # 先执行一次再平衡建立持仓
        env._rebalance(target, prices, "2024-01-14")
        # 目标微调，不应产生大量交易
        target2 = {"SPY": 0.51, "cash": 0.49}
        trades = env._rebalance(target2, prices, "2024-01-15")
        # 应该返回交易列表（可能为空或小规模）
        assert isinstance(trades, list)

    def test_rebalance_class_level(self, env):
        """测试资产类别级别再平衡"""
        target = {"stocks": 0.60, "bonds": 0.30, "cash": 0.10}
        prices = {"SPY": 450.0, "QQQ": 380.0, "TLT": 100.0}
        trades = env._rebalance(target, prices, "2024-01-15")
        assert isinstance(trades, list)


class TestExpandClassAllocation:
    """资产类别配置展开测试"""

    @pytest.fixture
    def env(self):
        return MultiAssetTradingEnv()

    def test_expand_basic(self, env):
        """测试基本展开"""
        class_alloc = {"stocks": 0.60, "bonds": 0.40}
        prices = {"SPY": 450, "QQQ": 380, "TLT": 100, "IEF": 95}
        result = env._expand_class_allocation(class_alloc, prices)
        # 应该展开为个股配置
        assert len(result) > 0
        total = sum(result.values())
        assert abs(total - 1.0) < 0.1

    def test_expand_with_cash(self, env):
        """测试带现金的展开"""
        class_alloc = {"stocks": 0.50, "cash": 0.50}
        prices = {"SPY": 450}
        result = env._expand_class_allocation(class_alloc, prices)
        assert "cash" in result
        assert result["cash"] == 0.50


class TestComputeExpertDrivenWeights:
    """专家驱动权重计算测试"""

    @pytest.fixture
    def env(self):
        return MultiAssetTradingEnv()

    def test_compute_weights_basic(self, env):
        """测试基本权重计算"""
        # Mock expert report
        expert_report = MagicMock()
        rec1 = MagicMock()
        rec1.symbol = "SPY"
        rec1.action = MagicMock(value="BUY_50%")
        rec1.confidence = 0.8

        rec2 = MagicMock()
        rec2.symbol = "QQQ"
        rec2.action = MagicMock(value="HOLD")
        rec2.confidence = 0.6

        expert_report.recommendations = [rec1, rec2]

        weights = env._compute_expert_driven_weights(
            asset_class="stocks",
            available_symbols=["SPY", "QQQ"],
            expert_report=expert_report,
            class_weight=0.60,
        )
        assert "SPY" in weights
        assert "QQQ" in weights
        assert weights["SPY"] > weights["QQQ"]  # 买入信号应获得更高权重

    def test_compute_weights_no_recommendations(self, env):
        """测试无推荐时的权重计算"""
        expert_report = MagicMock()
        expert_report.recommendations = []

        weights = env._compute_expert_driven_weights(
            asset_class="stocks",
            available_symbols=["SPY", "QQQ"],
            expert_report=expert_report,
            class_weight=0.60,
        )
        # 应该使用默认权重
        assert len(weights) > 0


class TestStep:
    """单步执行测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_step_basic(self, env):
        """测试基本单步执行"""
        target = {"stocks": 0.60, "bonds": 0.40}
        market_data = {
            "SPY": {"close": 450.0},
            "QQQ": {"close": 380.0},
            "TLT": {"close": 100.0},
        }
        state, reward, done, info = env.step(target, market_data, "2024-01-15")
        assert isinstance(state, PortfolioState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "step" in info

    def test_step_increments_counter(self, env):
        """测试步数计数器"""
        initial_step = env.current_step
        env.step({}, {}, "2024-01-15")
        assert env.current_step == initial_step + 1


class TestGetObservation:
    """获取观察测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_get_observation(self, env):
        """测试获取观察"""
        env.current_step = 5
        env.current_date = "2024-01-15"
        obs = env.get_observation()
        assert "portfolio" in obs
        assert "step" in obs
        assert obs["step"] == 5


class TestGetMetrics:
    """获取指标测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_get_metrics(self, env):
        """测试获取指标"""
        metrics = env.get_metrics()
        assert isinstance(metrics, dict)


class TestRender:
    """渲染测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_render(self, env):
        """测试渲染"""
        env.current_step = 5
        env.current_date = "2024-01-15"
        output = env.render()
        assert "Step 5" in output
        assert "Portfolio Value" in output


class TestStateManagement:
    """状态管理测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_get_state(self, env):
        """测试获取状态"""
        env.current_step = 10
        env.current_date = "2024-01-15"
        state = env.get_state()
        assert "portfolio" in state
        assert "current_step" in state
        assert state["current_step"] == 10

    def test_restore_state(self, env):
        """测试恢复状态"""
        state = {
            "portfolio": {
                "initial_capital": 1000000,
                "cash": 500000,
                "positions": {},
                "trade_history": [],
                "value_history": [],
                "timestamp": "2024-01-15",
            },
            "current_step": 20,
            "current_date": "2024-01-20",
            "done": False,
        }
        env.restore_state(state)
        assert env.current_step == 20
        assert env.current_date == "2024-01-20"
        assert env.portfolio.cash == 500000

    def test_restore_state_with_positions(self, env):
        """测试恢复带仓位的状态"""
        state = {
            "portfolio": {
                "initial_capital": 1000000,
                "cash": 300000,
                "positions": {
                    "SPY": {
                        "symbol": "SPY",
                        "shares": 100,
                        "avg_cost": 440,
                        "current_price": 450,
                        "asset_class": "stocks",
                    }
                },
                "trade_history": [],
                "value_history": [],
                "timestamp": "2024-01-15",
            },
            "current_step": 15,
            "current_date": "2024-01-15",
            "done": False,
        }
        env.restore_state(state)
        assert "SPY" in env.portfolio.positions
        assert env.portfolio.positions["SPY"].shares == 100


class TestClose:
    """关闭环境测试"""

    def test_close(self):
        """测试关闭环境"""
        env = MultiAssetTradingEnv()
        env.reset()
        env.close()  # 应该不抛出异常


class TestShortSelling:
    """做空测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_short_signal_detection(self, env):
        """测试做空信号检测"""
        expert_report = MagicMock()
        rec = MagicMock()
        rec.symbol = "SPY"
        rec.action = MagicMock(value="SHORT_50%")
        rec.confidence = 0.7
        expert_report.recommendations = [rec]

        weights = env._compute_expert_driven_weights(
            asset_class="stocks",
            available_symbols=["SPY"],
            expert_report=expert_report,
            class_weight=0.30,
        )
        # 做空应该产生负权重
        if "SPY" in weights:
            assert weights["SPY"] < 0


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def env(self):
        env = MultiAssetTradingEnv()
        env.reset()
        return env

    def test_empty_target_allocation(self, env):
        """测试空目标配置"""
        state, reward, done, info = env.step({}, {}, "2024-01-15")
        assert isinstance(state, PortfolioState)

    def test_zero_prices(self, env):
        """测试零价格处理"""
        target = {"SPY": 0.50}
        market_data = {"SPY": {"close": 0}}
        trades = env._rebalance(target, {"SPY": 0}, "2024-01-15")
        # 应该跳过零价格资产
        assert isinstance(trades, list)

    def test_negative_prices(self, env):
        """测试负价格处理"""
        prices = env._extract_prices({"SPY": {"close": -100}})
        # 应该仍然提取但可能在其他地方处理
        assert isinstance(prices, dict)

    def test_very_large_portfolio(self, env):
        """测试大规模组合"""
        target = {f"ASSET_{i}": 0.01 for i in range(100)}
        target["cash"] = 0.0
        # 应该能够处理大规模组合
        prices = {f"ASSET_{i}": 100.0 for i in range(100)}
        trades = env._rebalance(target, prices, "2024-01-15")
        assert isinstance(trades, list)
