#!/usr/bin/env python
"""
Portfolio State Deep Testing - 组合状态深度测试
Coverage: portfolio_state.py (Position, PortfolioState)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime

from finsage.environment.portfolio_state import Position, PortfolioState


# ============================================================
# Test 1: Position Class
# ============================================================

class TestPosition:
    """测试 Position 类"""

    def test_long_position_creation(self):
        """测试创建多头仓位"""
        pos = Position(
            symbol="AAPL",
            shares=100,
            avg_cost=150.0,
            current_price=160.0,
            asset_class="stocks"
        )

        assert pos.symbol == "AAPL"
        assert pos.shares == 100
        assert pos.avg_cost == 150.0
        assert pos.current_price == 160.0
        assert pos.asset_class == "stocks"

    def test_short_position_creation(self):
        """测试创建空头仓位"""
        pos = Position(
            symbol="TSLA",
            shares=-50,  # 负数表示空头
            avg_cost=200.0,
            current_price=180.0,
            asset_class="stocks"
        )

        assert pos.symbol == "TSLA"
        assert pos.shares == -50
        assert pos.is_short

    def test_is_short_property(self):
        """测试 is_short 属性"""
        long_pos = Position("AAPL", 100, 150, 160, "stocks")
        short_pos = Position("TSLA", -50, 200, 180, "stocks")

        assert not long_pos.is_short
        assert short_pos.is_short

    def test_market_value_long(self):
        """测试多头市值计算"""
        pos = Position("AAPL", 100, 150, 160, "stocks")

        assert pos.market_value == 100 * 160  # 16000

    def test_market_value_short(self):
        """测试空头市值计算 (负值)"""
        pos = Position("TSLA", -50, 200, 180, "stocks")

        # 空头市值 = -50 * 180 = -9000 (负债)
        assert pos.market_value == -50 * 180
        assert pos.market_value < 0

    def test_unrealized_pnl_long_profit(self):
        """测试多头盈利"""
        pos = Position("AAPL", 100, 150, 170, "stocks")

        # 盈利 = 100 * (170 - 150) = 2000
        assert pos.unrealized_pnl == 100 * (170 - 150)

    def test_unrealized_pnl_long_loss(self):
        """测试多头亏损"""
        pos = Position("AAPL", 100, 150, 130, "stocks")

        # 亏损 = 100 * (130 - 150) = -2000
        assert pos.unrealized_pnl == 100 * (130 - 150)
        assert pos.unrealized_pnl < 0

    def test_unrealized_pnl_short_profit(self):
        """测试空头盈利 (价格下跌)"""
        pos = Position("TSLA", -50, 200, 180, "stocks")

        # 空头盈利 = |shares| * (avg_cost - current) = 50 * (200 - 180) = 1000
        assert pos.unrealized_pnl == 50 * (200 - 180)
        assert pos.unrealized_pnl > 0

    def test_unrealized_pnl_short_loss(self):
        """测试空头亏损 (价格上涨)"""
        pos = Position("TSLA", -50, 200, 220, "stocks")

        # 空头亏损 = |shares| * (avg_cost - current) = 50 * (200 - 220) = -1000
        assert pos.unrealized_pnl == 50 * (200 - 220)
        assert pos.unrealized_pnl < 0

    def test_unrealized_pnl_pct_long(self):
        """测试多头未实现盈亏百分比"""
        pos = Position("AAPL", 100, 150, 165, "stocks")

        # 盈利百分比 = (165 - 150) / 150 = 10%
        assert abs(pos.unrealized_pnl_pct - 0.10) < 0.001

    def test_unrealized_pnl_pct_short(self):
        """测试空头未实现盈亏百分比"""
        pos = Position("TSLA", -50, 200, 180, "stocks")

        # 空头盈利百分比 = (200 - 180) / 200 = 10%
        assert abs(pos.unrealized_pnl_pct - 0.10) < 0.001

    def test_unrealized_pnl_pct_zero_cost(self):
        """测试零成本时的未实现盈亏百分比"""
        pos = Position("TEST", 100, 0, 100, "stocks")

        assert pos.unrealized_pnl_pct == 0.0

    def test_margin_requirement_long(self):
        """测试多头无保证金要求"""
        pos = Position("AAPL", 100, 150, 160, "stocks")

        assert pos.margin_requirement == 0.0

    def test_margin_requirement_short(self):
        """测试空头保证金要求"""
        pos = Position("TSLA", -50, 200, 180, "stocks")

        # 保证金 = |市值| * 0.5 = 9000 * 0.5 = 4500
        expected = abs(-50 * 180) * 0.5
        assert pos.margin_requirement == expected


# ============================================================
# Test 2: PortfolioState Initialization
# ============================================================

class TestPortfolioStateInit:
    """测试 PortfolioState 初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        state = PortfolioState()

        assert state.initial_capital == 1_000_000.0
        assert state.cash == 1_000_000.0
        assert len(state.positions) == 0
        assert len(state.trade_history) == 0
        assert len(state.value_history) == 0

    def test_custom_initialization(self):
        """测试自定义初始化"""
        state = PortfolioState(
            initial_capital=500_000.0,
            cash=500_000.0,
        )

        assert state.initial_capital == 500_000.0
        assert state.cash == 500_000.0

    def test_timestamp_auto_set(self):
        """测试时间戳自动设置"""
        state = PortfolioState()

        assert state.timestamp != ""
        # 应该是有效的ISO格式时间
        datetime.fromisoformat(state.timestamp)

    def test_short_config_defaults(self):
        """测试做空配置默认值"""
        state = PortfolioState()

        assert state.short_borrow_rate == 0.02
        assert state.short_margin_ratio == 0.5


# ============================================================
# Test 3: PortfolioState Properties
# ============================================================

class TestPortfolioStateProperties:
    """测试 PortfolioState 属性"""

    @pytest.fixture
    def state_with_positions(self):
        """创建带有仓位的状态"""
        state = PortfolioState(
            initial_capital=100_000.0,
            cash=20_000.0,
        )
        state.positions["AAPL"] = Position("AAPL", 100, 150, 160, "stocks")  # 多头
        state.positions["TSLA"] = Position("TSLA", -50, 200, 180, "stocks")  # 空头
        state.positions["BND"] = Position("BND", 200, 100, 105, "bonds")    # 多头
        return state

    def test_long_market_value(self, state_with_positions):
        """测试多头市值"""
        state = state_with_positions

        # AAPL: 100 * 160 = 16000
        # BND: 200 * 105 = 21000
        expected = 16000 + 21000
        assert state.long_market_value == expected

    def test_short_market_value(self, state_with_positions):
        """测试空头市值 (负值)"""
        state = state_with_positions

        # TSLA: -50 * 180 = -9000
        assert state.short_market_value == -9000

    def test_total_market_value(self, state_with_positions):
        """测试净市值"""
        state = state_with_positions

        # 16000 + 21000 + (-9000) = 28000
        expected = 16000 + 21000 - 9000
        assert state.total_market_value == expected

    def test_gross_exposure(self, state_with_positions):
        """测试总敞口"""
        state = state_with_positions

        # |多头| + |空头| = 37000 + 9000 = 46000
        expected = 37000 + 9000
        assert state.gross_exposure == expected

    def test_net_exposure(self, state_with_positions):
        """测试净敞口"""
        state = state_with_positions

        # 多头 - |空头| = 37000 - 9000 = 28000
        expected = 37000 - 9000
        assert state.net_exposure == expected

    def test_short_margin_required(self, state_with_positions):
        """测试空头所需保证金"""
        state = state_with_positions

        # |空头市值| * 0.5 = 9000 * 0.5 = 4500
        assert state.short_margin_required == 4500

    def test_portfolio_value(self, state_with_positions):
        """测试组合总价值"""
        state = state_with_positions

        # 现金 + 净市值 = 20000 + 28000 = 48000
        expected = 20000 + 28000
        assert state.portfolio_value == expected

    def test_total_return(self, state_with_positions):
        """测试总收益率"""
        state = state_with_positions

        # (48000 - 100000) / 100000 = -52%
        expected = (48000 - 100000) / 100000
        assert abs(state.total_return - expected) < 0.001

    def test_weights(self, state_with_positions):
        """测试资产权重"""
        state = state_with_positions
        weights = state.weights

        total = state.portfolio_value  # 48000

        assert "AAPL" in weights
        assert "TSLA" in weights
        assert "BND" in weights
        assert "cash" in weights

        # AAPL: 16000 / 48000 = 0.333
        assert abs(weights["AAPL"] - 16000 / 48000) < 0.001

        # TSLA空头: -9000 / 48000 = -0.1875
        assert abs(weights["TSLA"] - (-9000 / 48000)) < 0.001

    def test_weights_zero_value(self):
        """测试零价值时的权重"""
        state = PortfolioState(cash=0)
        weights = state.weights

        assert weights == {}

    def test_class_weights(self, state_with_positions):
        """测试资产类别权重"""
        state = state_with_positions
        class_weights = state.class_weights

        total = state.portfolio_value  # 48000

        # stocks: 16000 - 9000 = 7000 / 48000 = 0.1458
        assert "stocks" in class_weights
        # bonds: 21000 / 48000 = 0.4375
        assert "bonds" in class_weights
        assert "cash" in class_weights


# ============================================================
# Test 4: PortfolioState Trading
# ============================================================

class TestPortfolioStateTrading:
    """测试交易功能"""

    def test_buy_long_new_position(self):
        """测试买入新多头仓位"""
        state = PortfolioState(cash=100_000.0)

        trade = state.execute_trade(
            symbol="AAPL",
            shares=100,
            price=150.0,
            asset_class="stocks"
        )

        assert trade["action"] == "BUY"
        assert "AAPL" in state.positions
        assert state.positions["AAPL"].shares == 100
        # 现金减少: 100 * 150 + 0.1% 手续费
        expected_cost = 100 * 150 * 1.001
        assert abs(state.cash - (100_000 - expected_cost)) < 1

    def test_buy_long_add_position(self):
        """测试加仓多头"""
        state = PortfolioState(cash=100_000.0)
        state.positions["AAPL"] = Position("AAPL", 100, 150, 160, "stocks")

        trade = state.execute_trade(
            symbol="AAPL",
            shares=50,
            price=165.0,
            asset_class="stocks"
        )

        assert trade["action"] == "BUY"
        assert state.positions["AAPL"].shares == 150
        # 平均成本: (100 * 150 + 50 * 165) / 150 = 155
        expected_avg = (100 * 150 + 50 * 165) / 150
        assert abs(state.positions["AAPL"].avg_cost - expected_avg) < 0.01

    def test_sell_long_partial(self):
        """测试部分卖出多头"""
        state = PortfolioState(cash=10_000.0)
        state.positions["AAPL"] = Position("AAPL", 100, 150, 170, "stocks")

        trade = state.execute_trade(
            symbol="AAPL",
            shares=-50,  # 卖出50股
            price=170.0,
            asset_class="stocks"
        )

        assert trade["action"] == "SELL"
        assert state.positions["AAPL"].shares == 50
        # 已实现盈亏: 50 * (170 - 150) - 手续费
        assert trade["realized_pnl"] > 0

    def test_sell_long_all(self):
        """测试全部卖出多头"""
        state = PortfolioState(cash=10_000.0)
        state.positions["AAPL"] = Position("AAPL", 100, 150, 170, "stocks")

        trade = state.execute_trade(
            symbol="AAPL",
            shares=-100,
            price=170.0,
            asset_class="stocks"
        )

        assert trade["action"] == "SELL"
        assert "AAPL" not in state.positions

    def test_open_short_position(self):
        """测试开空仓"""
        state = PortfolioState(cash=50_000.0)

        trade = state.execute_trade(
            symbol="TSLA",
            shares=-50,
            price=200.0,
            asset_class="stocks",
            is_short=True
        )

        assert trade["action"] == "SHORT"
        assert "TSLA" in state.positions
        assert state.positions["TSLA"].shares == -50
        assert state.positions["TSLA"].is_short

    def test_add_short_position(self):
        """测试加空仓"""
        state = PortfolioState(cash=50_000.0)
        state.positions["TSLA"] = Position("TSLA", -50, 200, 195, "stocks")

        trade = state.execute_trade(
            symbol="TSLA",
            shares=-20,
            price=195.0,
            asset_class="stocks"
        )

        assert trade["action"] == "ADD_SHORT"
        assert state.positions["TSLA"].shares == -70

    def test_cover_short_partial(self):
        """测试部分平空仓"""
        state = PortfolioState(cash=20_000.0)
        state.positions["TSLA"] = Position("TSLA", -50, 200, 180, "stocks")

        trade = state.execute_trade(
            symbol="TSLA",
            shares=20,  # 买入20股平仓
            price=180.0,
            asset_class="stocks"
        )

        assert trade["action"] == "COVER_SHORT"
        assert state.positions["TSLA"].shares == -30
        # 盈利: 20 * (200 - 180) - 手续费
        assert trade["realized_pnl"] > 0

    def test_cover_short_all(self):
        """测试全部平空仓"""
        state = PortfolioState(cash=20_000.0)
        state.positions["TSLA"] = Position("TSLA", -50, 200, 180, "stocks")

        trade = state.execute_trade(
            symbol="TSLA",
            shares=50,
            price=180.0,
            asset_class="stocks"
        )

        assert trade["action"] == "COVER_SHORT"
        assert "TSLA" not in state.positions

    def test_trade_exceeds_cash(self):
        """测试交易超过现金"""
        state = PortfolioState(cash=1_000.0)

        trade = state.execute_trade(
            symbol="AAPL",
            shares=100,  # 想买 15000+ 美元
            price=150.0,
            asset_class="stocks"
        )

        # 应该只买到现金能支持的数量
        assert state.positions["AAPL"].shares < 100
        assert state.cash >= 0

    def test_trade_history_recorded(self):
        """测试交易历史记录"""
        state = PortfolioState(cash=100_000.0)

        state.execute_trade("AAPL", 100, 150.0, "stocks")
        state.execute_trade("MSFT", 50, 300.0, "stocks")

        assert len(state.trade_history) == 2
        assert state.trade_history[0]["symbol"] == "AAPL"
        assert state.trade_history[1]["symbol"] == "MSFT"


# ============================================================
# Test 5: PortfolioState Price Updates
# ============================================================

class TestPortfolioStatePriceUpdates:
    """测试价格更新"""

    def test_update_prices(self):
        """测试更新价格"""
        state = PortfolioState()
        state.positions["AAPL"] = Position("AAPL", 100, 150, 150, "stocks")
        state.positions["MSFT"] = Position("MSFT", 50, 300, 300, "stocks")

        state.update_prices({"AAPL": 160, "MSFT": 310})

        assert state.positions["AAPL"].current_price == 160
        assert state.positions["MSFT"].current_price == 310

    def test_update_prices_nonexistent_symbol(self):
        """测试更新不存在的符号"""
        state = PortfolioState()
        state.positions["AAPL"] = Position("AAPL", 100, 150, 150, "stocks")

        # 不应该崩溃
        state.update_prices({"AAPL": 160, "INVALID": 100})

        assert state.positions["AAPL"].current_price == 160


# ============================================================
# Test 6: Short Borrowing Cost
# ============================================================

class TestShortBorrowingCost:
    """测试空头借股成本"""

    def test_apply_short_borrowing_cost(self):
        """测试应用借股成本"""
        state = PortfolioState(cash=50_000.0)
        state.positions["TSLA"] = Position("TSLA", -100, 200, 200, "stocks")

        initial_cash = state.cash
        cost = state.apply_short_borrowing_cost(days=1)

        # 空头市值 = 100 * 200 = 20000
        # 日借股成本 = 20000 * 0.02 / 365 ≈ 1.10
        expected_cost = 20000 * 0.02 / 365
        assert abs(cost - expected_cost) < 0.01
        assert state.cash == initial_cash - cost

    def test_no_cost_without_short(self):
        """测试无空仓时无借股成本"""
        state = PortfolioState(cash=50_000.0)
        state.positions["AAPL"] = Position("AAPL", 100, 150, 160, "stocks")

        initial_cash = state.cash
        cost = state.apply_short_borrowing_cost(days=1)

        assert cost == 0.0
        assert state.cash == initial_cash

    def test_multiple_days_cost(self):
        """测试多天借股成本"""
        state = PortfolioState(cash=50_000.0)
        state.positions["TSLA"] = Position("TSLA", -100, 200, 200, "stocks")

        cost = state.apply_short_borrowing_cost(days=30)

        expected = 20000 * 0.02 / 365 * 30
        assert abs(cost - expected) < 0.1


# ============================================================
# Test 7: Value Recording
# ============================================================

class TestValueRecording:
    """测试价值记录"""

    def test_record_value(self):
        """测试记录组合价值"""
        state = PortfolioState(cash=80_000.0)
        state.positions["AAPL"] = Position("AAPL", 100, 150, 150, "stocks")

        state.record_value("2024-01-01", {"AAPL": 160})

        assert len(state.value_history) == 1
        record = state.value_history[0]
        assert record["timestamp"] == "2024-01-01"
        assert record["portfolio_value"] == state.portfolio_value

    def test_record_value_with_short(self):
        """测试带空头的价值记录"""
        state = PortfolioState(cash=50_000.0)
        state.positions["TSLA"] = Position("TSLA", -50, 200, 200, "stocks")

        state.record_value("2024-01-01", {"TSLA": 195})

        record = state.value_history[0]
        assert record["short_value"] < 0
        assert record["borrow_cost"] > 0


# ============================================================
# Test 8: Returns Calculation
# ============================================================

class TestReturnsCalculation:
    """测试收益计算"""

    def test_get_returns_empty(self):
        """测试空历史收益"""
        state = PortfolioState()
        returns = state.get_returns()

        assert len(returns) == 0

    def test_get_returns_insufficient_data(self):
        """测试数据不足"""
        state = PortfolioState()
        state.value_history = [{"portfolio_value": 100_000}]
        returns = state.get_returns()

        assert len(returns) == 0

    def test_get_returns_calculation(self):
        """测试收益计算"""
        state = PortfolioState()
        state.value_history = [
            {"portfolio_value": 100_000},
            {"portfolio_value": 101_000},
            {"portfolio_value": 100_500},
        ]
        returns = state.get_returns()

        assert len(returns) == 2
        # 第一天: (101000 - 100000) / 100000 = 0.01
        assert abs(returns[0] - 0.01) < 0.001
        # 第二天: (100500 - 101000) / 101000 ≈ -0.00495
        assert abs(returns[1] - (-500 / 101000)) < 0.001


# ============================================================
# Test 9: Metrics Calculation
# ============================================================

class TestMetricsCalculation:
    """测试指标计算"""

    def test_get_metrics_empty(self):
        """测试空历史指标"""
        state = PortfolioState()
        metrics = state.get_metrics()

        assert "cumulative_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_get_metrics_with_data(self):
        """测试有数据的指标"""
        state = PortfolioState(initial_capital=100_000.0, cash=100_000.0)

        # 模拟多天数据 - 需要让portfolio_value反映实际累计收益
        for i in range(10):
            value = 100_000 + i * 1000  # 每天涨1000
            state.value_history.append({
                "portfolio_value": value,
            })

        # 设置最终现金以匹配最后的portfolio_value
        state.cash = 109_000.0  # 初始100000涨到109000
        metrics = state.get_metrics()

        # cumulative_return 基于 (portfolio_value - initial) / initial = (109000-100000)/100000 = 0.09
        assert metrics["cumulative_return"] > 0
        assert metrics["volatility"] >= 0
        assert metrics["n_trades"] == 0

    def test_metrics_sharpe_ratio(self):
        """测试夏普比率计算"""
        state = PortfolioState(initial_capital=100_000.0)

        # 稳定增长
        for i in range(20):
            state.value_history.append({
                "portfolio_value": 100_000 * (1.001 ** i)
            })

        metrics = state.get_metrics()

        # 稳定增长应该有正的夏普比率
        assert metrics["sharpe_ratio"] > 0

    def test_metrics_max_drawdown(self):
        """测试最大回撤"""
        state = PortfolioState(initial_capital=100_000.0)

        # 先涨后跌
        state.value_history = [
            {"portfolio_value": 100_000},
            {"portfolio_value": 110_000},  # 涨10%
            {"portfolio_value": 105_000},  # 回撤5%
            {"portfolio_value": 95_000},   # 继续跌，回撤约13.6%
        ]

        metrics = state.get_metrics()

        # 最大回撤从110000跌到95000 = -13.6%
        expected_dd = (95_000 - 110_000) / 110_000
        assert abs(metrics["max_drawdown"] - expected_dd) < 0.01


# ============================================================
# Test 10: to_dict Conversion
# ============================================================

class TestToDictConversion:
    """测试字典转换"""

    def test_to_dict_basic(self):
        """测试基本转换"""
        state = PortfolioState(initial_capital=100_000.0, cash=80_000.0)
        state.positions["AAPL"] = Position("AAPL", 100, 150, 160, "stocks")

        d = state.to_dict()

        assert "timestamp" in d
        assert "cash" in d
        assert "portfolio_value" in d
        assert "total_return" in d
        assert "positions" in d
        assert "weights" in d
        assert "class_weights" in d

    def test_to_dict_with_short(self):
        """测试带空头的转换"""
        state = PortfolioState(cash=50_000.0)
        state.positions["TSLA"] = Position("TSLA", -50, 200, 180, "stocks")

        d = state.to_dict()

        assert d["short_market_value"] < 0
        assert d["positions"]["TSLA"]["is_short"] == True

    def test_to_dict_empty_portfolio(self):
        """测试空组合转换"""
        state = PortfolioState()
        d = state.to_dict()

        assert d["positions"] == {}
        assert d["portfolio_value"] == state.cash


# ============================================================
# Test 11: Edge Cases
# ============================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_very_small_position(self):
        """测试极小仓位"""
        state = PortfolioState(cash=100_000.0)

        state.execute_trade("AAPL", 0.001, 150.0, "stocks")

        assert "AAPL" in state.positions

    def test_zero_price(self):
        """测试零价格"""
        pos = Position("TEST", 100, 100, 0, "stocks")

        assert pos.market_value == 0
        assert pos.unrealized_pnl == 100 * (0 - 100)

    def test_negative_initial_capital(self):
        """测试负初始资本"""
        state = PortfolioState(initial_capital=-100_000)

        # 应该能创建但总收益率计算有意义
        assert state.initial_capital == -100_000

    def test_concurrent_long_short_same_symbol(self):
        """测试同一符号的多空转换"""
        state = PortfolioState(cash=100_000.0)

        # 先买入多头
        state.execute_trade("AAPL", 100, 150.0, "stocks")
        assert state.positions["AAPL"].shares == 100

        # 卖出多头并开空
        state.execute_trade("AAPL", -150, 160.0, "stocks", is_short=True)

        # 应该剩余空仓或全部平仓
        if "AAPL" in state.positions:
            assert state.positions["AAPL"].shares <= 0


# ============================================================
# Test 12: Complex Scenarios
# ============================================================

class TestComplexScenarios:
    """测试复杂场景"""

    def test_multi_asset_portfolio(self):
        """测试多资产组合"""
        state = PortfolioState(cash=500_000.0)

        # 买入多种资产
        state.execute_trade("AAPL", 100, 150.0, "stocks")
        state.execute_trade("MSFT", 50, 350.0, "stocks")
        state.execute_trade("TLT", 200, 100.0, "bonds")
        state.execute_trade("GLD", 100, 180.0, "commodities")

        assert len(state.positions) == 4
        assert "stocks" in state.class_weights
        assert "bonds" in state.class_weights
        assert "commodities" in state.class_weights

    def test_long_short_portfolio(self):
        """测试多空组合"""
        state = PortfolioState(cash=200_000.0)

        # 做多
        state.execute_trade("AAPL", 100, 150.0, "stocks")
        state.execute_trade("MSFT", 50, 350.0, "stocks")

        # 做空
        state.execute_trade("TSLA", -30, 250.0, "stocks", is_short=True)

        assert state.long_market_value > 0
        assert state.short_market_value < 0
        assert state.gross_exposure > state.net_exposure

    def test_full_trading_cycle(self):
        """测试完整交易周期"""
        state = PortfolioState(cash=100_000.0)

        # 1. 买入
        state.execute_trade("AAPL", 100, 150.0, "stocks")
        assert len(state.trade_history) == 1

        # 2. 记录价值
        state.record_value("2024-01-01", {"AAPL": 155})
        assert len(state.value_history) == 1

        # 3. 加仓
        state.execute_trade("AAPL", 50, 155.0, "stocks")

        # 4. 记录价值
        state.record_value("2024-01-02", {"AAPL": 160})

        # 5. 卖出
        state.execute_trade("AAPL", -150, 160.0, "stocks")
        assert "AAPL" not in state.positions

        # 6. 检查指标
        metrics = state.get_metrics()
        assert metrics["n_trades"] == 3


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Portfolio State Deep Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
