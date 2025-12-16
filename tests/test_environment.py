#!/usr/bin/env python
"""
Environment Module Tests - 环境模块测试
覆盖: portfolio_state, multi_asset_env
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: Position Dataclass
# ============================================================

class TestPosition:
    """测试持仓数据类"""

    def test_import(self):
        """测试导入"""
        from finsage.environment.portfolio_state import Position
        assert Position is not None

    def test_long_position(self):
        """测试多头仓位"""
        from finsage.environment.portfolio_state import Position

        pos = Position(
            symbol="AAPL",
            shares=100,
            avg_cost=150.0,
            current_price=170.0,
            asset_class="stocks"
        )

        assert not pos.is_short
        assert pos.market_value == 17000.0  # 100 * 170
        assert pos.unrealized_pnl == 2000.0  # 100 * (170 - 150)
        assert abs(pos.unrealized_pnl_pct - 0.1333) < 0.001

    def test_short_position(self):
        """测试空头仓位"""
        from finsage.environment.portfolio_state import Position

        pos = Position(
            symbol="SPY",
            shares=-100,  # 空头
            avg_cost=450.0,  # 卖出价
            current_price=430.0,  # 当前价
            asset_class="stocks"
        )

        assert pos.is_short
        assert pos.market_value == -43000.0  # -100 * 430
        # 空头盈利: 卖出价 - 当前价
        assert pos.unrealized_pnl == 2000.0  # 100 * (450 - 430)

    def test_short_position_loss(self):
        """测试空头亏损"""
        from finsage.environment.portfolio_state import Position

        pos = Position(
            symbol="QQQ",
            shares=-50,
            avg_cost=380.0,  # 卖出价
            current_price=400.0,  # 价格上涨，空头亏损
            asset_class="stocks"
        )

        assert pos.is_short
        # 空头亏损: 卖出价 - 当前价 = 380 - 400 = -20 * 50 = -1000
        assert pos.unrealized_pnl == -1000.0

    def test_margin_requirement(self):
        """测试保证金要求"""
        from finsage.environment.portfolio_state import Position

        # 多头无保证金要求
        long_pos = Position("AAPL", 100, 150, 170, "stocks")
        assert long_pos.margin_requirement == 0.0

        # 空头需要保证金
        short_pos = Position("SPY", -100, 450, 450, "stocks")
        assert short_pos.margin_requirement == 22500.0  # |45000| * 0.5


# ============================================================
# Test 2: Portfolio State
# ============================================================

class TestPortfolioState:
    """测试组合状态"""

    def test_import(self):
        """测试导入"""
        from finsage.environment.portfolio_state import PortfolioState
        assert PortfolioState is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(
            initial_capital=1_000_000,
            cash=1_000_000
        )

        assert portfolio.initial_capital == 1_000_000
        assert portfolio.cash == 1_000_000
        assert portfolio.portfolio_value == 1_000_000
        assert portfolio.total_return == 0.0

    def test_long_market_value(self):
        """测试多头市值计算"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions = {
            "AAPL": Position("AAPL", 100, 150, 160, "stocks"),
            "GOOGL": Position("GOOGL", 50, 100, 110, "stocks"),
        }

        assert portfolio.long_market_value == 21500  # 100*160 + 50*110

    def test_short_market_value(self):
        """测试空头市值计算"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions = {
            "SPY": Position("SPY", -100, 450, 440, "stocks"),
        }

        assert portfolio.short_market_value == -44000  # -100 * 440

    def test_portfolio_value(self):
        """测试组合总价值"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions = {
            "AAPL": Position("AAPL", 100, 150, 160, "stocks"),  # 多头 16000
        }

        # 组合价值 = 现金 + 市值
        assert portfolio.portfolio_value == 66000  # 50000 + 16000

    def test_weights(self):
        """测试权重计算"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions = {
            "AAPL": Position("AAPL", 100, 150, 150, "stocks"),  # 15000
        }

        weights = portfolio.weights
        # 总价值 = 50000 + 15000 = 65000
        assert abs(weights["cash"] - 50000/65000) < 0.001
        assert abs(weights["AAPL"] - 15000/65000) < 0.001

    def test_class_weights(self):
        """测试资产类别权重"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=30000)
        portfolio.positions = {
            "AAPL": Position("AAPL", 100, 200, 200, "stocks"),  # 20000
            "GOOGL": Position("GOOGL", 100, 100, 100, "stocks"),  # 10000
            "TLT": Position("TLT", 100, 100, 100, "bonds"),  # 10000
        }

        class_weights = portfolio.class_weights
        # 总价值 = 30000 + 20000 + 10000 + 10000 = 70000
        assert abs(class_weights["stocks"] - 30000/70000) < 0.001
        assert abs(class_weights["bonds"] - 10000/70000) < 0.001
        assert abs(class_weights["cash"] - 30000/70000) < 0.001

    def test_execute_trade_buy(self):
        """测试买入交易"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(initial_capital=100000, cash=100000)
        trade = portfolio.execute_trade(
            symbol="AAPL",
            shares=100,
            price=150.0,
            asset_class="stocks"
        )

        assert trade["action"] == "BUY"
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].shares == 100
        assert portfolio.cash < 100000  # 扣除了购买成本和手续费

    def test_execute_trade_sell(self):
        """测试卖出交易"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions["AAPL"] = Position("AAPL", 100, 140, 150, "stocks")

        trade = portfolio.execute_trade(
            symbol="AAPL",
            shares=-50,  # 卖出50股
            price=160.0,
            asset_class="stocks"
        )

        assert trade["action"] == "SELL"
        assert portfolio.positions["AAPL"].shares == 50
        assert trade["realized_pnl"] > 0  # 有盈利

    def test_execute_trade_short(self):
        """测试做空交易"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(initial_capital=100000, cash=100000)
        trade = portfolio.execute_trade(
            symbol="SPY",
            shares=-100,  # 做空
            price=450.0,
            asset_class="stocks",
            is_short=True
        )

        assert trade["action"] == "SHORT"
        assert "SPY" in portfolio.positions
        assert portfolio.positions["SPY"].shares < 0
        assert portfolio.positions["SPY"].is_short

    def test_execute_trade_cover_short(self):
        """测试平仓空头"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions["SPY"] = Position("SPY", -100, 450, 430, "stocks")

        trade = portfolio.execute_trade(
            symbol="SPY",
            shares=100,  # 买入平仓
            price=420.0,  # 价格下跌，空头盈利
            asset_class="stocks"
        )

        assert trade["action"] == "COVER_SHORT"
        assert "SPY" not in portfolio.positions  # 仓位已平
        assert trade["realized_pnl"] > 0  # 有盈利

    def test_update_prices(self):
        """测试价格更新"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions["AAPL"] = Position("AAPL", 100, 150, 150, "stocks")

        portfolio.update_prices({"AAPL": 160})
        assert portfolio.positions["AAPL"].current_price == 160

    def test_record_value(self):
        """测试价值记录"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(initial_capital=100000, cash=100000)
        portfolio.record_value("2024-01-15", {})

        assert len(portfolio.value_history) == 1
        assert portfolio.value_history[0]["portfolio_value"] == 100000

    def test_get_returns(self):
        """测试收益率计算"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(initial_capital=100000, cash=100000)
        portfolio.value_history = [
            {"portfolio_value": 100000},
            {"portfolio_value": 101000},
            {"portfolio_value": 102010},
        ]

        returns = portfolio.get_returns()
        assert len(returns) == 2
        assert abs(returns[0] - 0.01) < 0.0001

    def test_get_metrics(self):
        """测试组合指标计算"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(initial_capital=100000, cash=105000)
        portfolio.value_history = [
            {"portfolio_value": 100000 + i * 200 + np.random.randn() * 100}
            for i in range(50)
        ]

        metrics = portfolio.get_metrics()

        assert "cumulative_return" in metrics
        assert "annual_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_apply_short_borrowing_cost(self):
        """测试借股成本"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions["SPY"] = Position("SPY", -100, 450, 450, "stocks")

        initial_cash = portfolio.cash
        cost = portfolio.apply_short_borrowing_cost(days=365)

        # 年化成本 = |45000| * 0.02 = 900
        assert abs(cost - 900) < 1
        assert portfolio.cash < initial_cash

    def test_gross_and_net_exposure(self):
        """测试总敞口和净敞口"""
        from finsage.environment.portfolio_state import PortfolioState, Position

        portfolio = PortfolioState(initial_capital=100000, cash=50000)
        portfolio.positions = {
            "AAPL": Position("AAPL", 100, 150, 150, "stocks"),  # 多头 15000
            "SPY": Position("SPY", -50, 400, 400, "stocks"),  # 空头 -20000
        }

        assert portfolio.long_market_value == 15000
        assert portfolio.short_market_value == -20000
        assert portfolio.gross_exposure == 35000  # |15000| + |20000|
        assert portfolio.net_exposure == -5000  # 15000 - 20000

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.environment.portfolio_state import PortfolioState

        portfolio = PortfolioState(initial_capital=100000, cash=100000)
        d = portfolio.to_dict()

        assert "cash" in d
        assert "portfolio_value" in d
        assert "positions" in d
        assert "weights" in d


# ============================================================
# Test 3: Environment Config
# ============================================================

class TestEnvConfig:
    """测试环境配置"""

    def test_import(self):
        """测试导入"""
        from finsage.environment.multi_asset_env import EnvConfig
        assert EnvConfig is not None

    def test_default_config(self):
        """测试默认配置"""
        from finsage.environment.multi_asset_env import EnvConfig

        config = EnvConfig()
        assert config.initial_capital == 1_000_000
        assert config.transaction_cost == 0.001
        assert config.max_single_asset == 0.15

    def test_custom_config(self):
        """测试自定义配置"""
        from finsage.environment.multi_asset_env import EnvConfig

        config = EnvConfig(
            initial_capital=500_000,
            transaction_cost=0.002,
            max_single_asset=0.20
        )

        assert config.initial_capital == 500_000
        assert config.transaction_cost == 0.002


# ============================================================
# Test 4: Multi Asset Trading Environment
# ============================================================

class TestMultiAssetTradingEnv:
    """测试多资产交易环境"""

    def test_import(self):
        """测试导入"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv
        assert MultiAssetTradingEnv is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        assert env.portfolio is not None
        assert env.asset_universe is not None
        assert not env.done

    def test_custom_initialization(self):
        """测试自定义初始化"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv, EnvConfig

        config = EnvConfig(initial_capital=500_000)
        universe = {"stocks": ["SPY", "QQQ"]}
        env = MultiAssetTradingEnv(config=config, asset_universe=universe)

        assert env.portfolio.initial_capital == 500_000
        assert "stocks" in env.asset_universe

    def test_reset(self):
        """测试环境重置"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        portfolio = env.reset(initial_capital=200_000)

        assert portfolio.initial_capital == 200_000
        assert portfolio.cash == 200_000
        assert env.current_step == 0
        assert not env.done

    def test_get_observation(self):
        """测试获取观察"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.reset()

        obs = env.get_observation()

        assert "portfolio" in obs
        assert "market_data" in obs

    def test_asset_universe_structure(self):
        """测试资产池结构"""
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv

        env = MultiAssetTradingEnv()
        env.asset_universe = {
            "stocks": ["SPY", "QQQ"],
            "bonds": ["TLT"],
        }

        # 验证资产池结构正确
        assert "stocks" in env.asset_universe
        assert "bonds" in env.asset_universe
        assert "SPY" in env.asset_universe["stocks"]
        assert "TLT" in env.asset_universe["bonds"]


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Environment Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
