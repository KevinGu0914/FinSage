"""
Portfolio State
组合状态管理
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class Position:
    """
    单一资产持仓

    支持做多和做空:
    - shares > 0: 多头仓位 (买入持有)
    - shares < 0: 空头仓位 (卖空借入)
    """
    symbol: str
    shares: float  # 正数=多头, 负数=空头
    avg_cost: float
    current_price: float
    asset_class: str

    @property
    def is_short(self) -> bool:
        """是否为空头仓位"""
        return self.shares < 0

    @property
    def market_value(self) -> float:
        """
        市值
        - 多头: 正值 (资产)
        - 空头: 负值 (负债) - 需要买回股票还给券商
        """
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """
        未实现盈亏
        - 多头: shares * (current - avg_cost), 价格涨赚钱
        - 空头: |shares| * (avg_cost - current), 价格跌赚钱
        """
        if self.is_short:
            # 空头: 卖出价(avg_cost) - 当前价(current_price) 为盈利
            return abs(self.shares) * (self.avg_cost - self.current_price)
        else:
            # 多头: 当前价 - 买入价 为盈利
            return self.shares * (self.current_price - self.avg_cost)

    @property
    def unrealized_pnl_pct(self) -> float:
        """未实现盈亏百分比"""
        if self.avg_cost > 0:
            if self.is_short:
                return (self.avg_cost - self.current_price) / self.avg_cost
            else:
                return (self.current_price - self.avg_cost) / self.avg_cost
        return 0.0

    @property
    def margin_requirement(self) -> float:
        """
        保证金要求 (仅空头)
        空头仓位需要存入保证金作为担保
        通常要求为空头市值的50%
        """
        if self.is_short:
            return abs(self.market_value) * 0.5  # 50% 保证金
        return 0.0


@dataclass
class PortfolioState:
    """
    组合状态

    跟踪多资产组合的完整状态，包括:
    - 现金余额
    - 各资产持仓 (多头和空头)
    - 历史交易记录
    - 组合价值历史

    做空机制:
    - 空头仓位的 shares < 0
    - 空头市值为负数 (表示负债)
    - 卖空收入暂存到现金账户
    - 需要维持保证金
    """
    initial_capital: float = 1_000_000.0
    cash: float = 1_000_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    value_history: List[Dict] = field(default_factory=list)
    timestamp: str = ""

    # 做空配置
    short_borrow_rate: float = 0.02  # 年化借股费率 2%
    short_margin_ratio: float = 0.5  # 空头保证金比例 50%

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def long_market_value(self) -> float:
        """多头总市值"""
        return sum(pos.market_value for pos in self.positions.values() if not pos.is_short)

    @property
    def short_market_value(self) -> float:
        """空头总市值 (负数)"""
        return sum(pos.market_value for pos in self.positions.values() if pos.is_short)

    @property
    def total_market_value(self) -> float:
        """净市值 = 多头市值 + 空头市值(负数)"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        """总敞口 = |多头| + |空头|"""
        return self.long_market_value + abs(self.short_market_value)

    @property
    def net_exposure(self) -> float:
        """净敞口 = 多头 - |空头|"""
        return self.long_market_value - abs(self.short_market_value)

    @property
    def short_margin_required(self) -> float:
        """空头所需保证金"""
        return abs(self.short_market_value) * self.short_margin_ratio

    @property
    def portfolio_value(self) -> float:
        """
        组合总价值
        = 现金 + 多头市值 + 空头未实现盈亏
        注: 空头市值本身是负数(负债)，但卖空所得已加入现金
        """
        return self.cash + self.total_market_value

    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.portfolio_value - self.initial_capital) / self.initial_capital

    @property
    def weights(self) -> Dict[str, float]:
        """当前资产权重"""
        total = self.portfolio_value
        if total <= 0:
            return {}
        weights = {
            symbol: pos.market_value / total
            for symbol, pos in self.positions.items()
        }
        weights["cash"] = self.cash / total
        return weights

    def get_weights(self) -> Dict[str, float]:
        """获取当前资产权重（方法版本，兼容旧接口）"""
        return self.weights

    @property
    def class_weights(self) -> Dict[str, float]:
        """按资产类别的权重"""
        total = self.portfolio_value
        if total <= 0:
            return {}

        class_values = {}
        for pos in self.positions.values():
            asset_class = pos.asset_class
            class_values[asset_class] = class_values.get(asset_class, 0) + pos.market_value

        class_weights = {c: v / total for c, v in class_values.items()}
        class_weights["cash"] = self.cash / total
        return class_weights

    def update_prices(self, prices: Dict[str, float]):
        """更新当前价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def execute_trade(
        self,
        symbol: str,
        shares: float,
        price: float,
        asset_class: str,
        timestamp: str = "",
        is_short: bool = False
    ) -> Dict[str, Any]:
        """
        执行交易 (支持做多和做空)

        Args:
            symbol: 资产代码
            shares: 股数 (正数买入/平仓，负数卖出/开空)
            price: 成交价格
            asset_class: 资产类别
            timestamp: 交易时间
            is_short: 是否为做空操作 (仅在开新仓位时使用)

        Returns:
            交易记录

        交易类型:
        1. 买入多头 (shares > 0, 无现有仓位或有多头仓位)
        2. 卖出多头 (shares < 0, 有多头仓位)
        3. 开空仓 (is_short=True 或 shares < 0 且无现有仓位)
        4. 平空仓 (shares > 0, 有空头仓位)
        """
        trade_value = abs(shares * price)
        commission = trade_value * 0.001  # 0.1% 交易成本

        # 检查是否有现有仓位
        existing_pos = self.positions.get(symbol)
        has_long = existing_pos and not existing_pos.is_short
        has_short = existing_pos and existing_pos.is_short

        action = ""
        realized_pnl = 0.0

        if shares > 0:
            # 买入或平空
            if has_short:
                # 平空仓 (买入股票还给券商)
                action = "COVER_SHORT"
                pos = existing_pos
                cover_shares = min(shares, abs(pos.shares))

                # 计算平仓盈亏: 卖出价(avg_cost) - 买回价(price)
                realized_pnl = cover_shares * (pos.avg_cost - price) - commission

                # 减少空头仓位 (shares是负数，加上正数来减少)
                pos.shares += cover_shares
                self.cash -= cover_shares * price + commission  # 买回股票的成本

                if abs(pos.shares) <= 0.001:
                    del self.positions[symbol]
                else:
                    pos.current_price = price

                # 如果还有剩余shares，可能是要开多头
                remaining = shares - cover_shares
                if remaining > 0.001:
                    # 开多头仓位
                    self._open_long(symbol, remaining, price, asset_class, commission)
                    action = "COVER_SHORT_AND_BUY"

            else:
                # 买入多头 (开仓或加仓)
                action = "BUY"
                total_cost = shares * price + commission
                if total_cost > self.cash:
                    shares = (self.cash - commission) / price
                    total_cost = shares * price + commission

                self.cash -= total_cost

                if has_long:
                    pos = existing_pos
                    total_shares = pos.shares + shares
                    pos.avg_cost = (pos.shares * pos.avg_cost + shares * price) / total_shares
                    pos.shares = total_shares
                    pos.current_price = price
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=shares,
                        avg_cost=price,
                        current_price=price,
                        asset_class=asset_class
                    )

        else:  # shares < 0
            shares_to_trade = abs(shares)

            if has_long:
                # 卖出多头
                action = "SELL"
                pos = existing_pos
                sell_shares = min(shares_to_trade, pos.shares)

                # 计算已实现盈亏
                realized_pnl = sell_shares * (price - pos.avg_cost) - commission

                proceeds = sell_shares * price - commission
                self.cash += proceeds

                pos.shares -= sell_shares
                if pos.shares <= 0.001:
                    del self.positions[symbol]
                else:
                    pos.current_price = price

                # 如果还有剩余要卖，开空仓
                remaining = shares_to_trade - sell_shares
                if remaining > 0.001 and is_short:
                    self._open_short(symbol, remaining, price, asset_class, commission)
                    action = "SELL_AND_SHORT"

            elif has_short:
                # 加空仓
                action = "ADD_SHORT"
                pos = existing_pos
                short_proceeds = shares_to_trade * price - commission

                # 检查保证金是否足够
                margin_needed = shares_to_trade * price * self.short_margin_ratio
                if margin_needed > self.cash:
                    shares_to_trade = self.cash / (price * self.short_margin_ratio)
                    short_proceeds = shares_to_trade * price - commission

                self.cash += short_proceeds  # 卖空收入
                # 修复：先保存旧仓位数量，再更新
                old_short_shares = abs(pos.shares)  # 加仓前的仓位数量
                pos.shares -= shares_to_trade  # 增加空头 (更负)
                pos.current_price = price
                # 更新平均成本 (使用旧仓位数量进行加权)
                total_short = abs(pos.shares)
                if total_short > 1e-10:  # 避免除零
                    pos.avg_cost = (old_short_shares * pos.avg_cost + shares_to_trade * price) / total_short

            else:
                # 开新空仓
                action = "SHORT"
                self._open_short(symbol, shares_to_trade, price, asset_class, commission)

        # 记录交易
        trade_record = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "symbol": symbol,
            "shares": abs(shares),
            "action": action,
            "price": price,
            "commission": commission,
            "realized_pnl": realized_pnl,
            "portfolio_value": self.portfolio_value,
            "is_short": action in ["SHORT", "ADD_SHORT", "SELL_AND_SHORT"],
        }
        self.trade_history.append(trade_record)

        return trade_record

    def _open_long(self, symbol: str, shares: float, price: float, asset_class: str, commission: float):
        """开多头仓位"""
        total_cost = shares * price + commission
        if total_cost > self.cash:
            shares = (self.cash - commission) / price
            total_cost = shares * price + commission

        self.cash -= total_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            avg_cost=price,
            current_price=price,
            asset_class=asset_class
        )

    def _open_short(self, symbol: str, shares: float, price: float, asset_class: str, commission: float):
        """
        开空头仓位

        做空机制:
        1. 从券商借入股票
        2. 立即卖出，收入进入现金账户
        3. 需要保证金 (通常为卖空价值的50%)
        4. 未来需要买回股票还给券商
        """
        # 检查保证金
        margin_needed = shares * price * self.short_margin_ratio
        if margin_needed > self.cash:
            # 调整做空数量
            shares = self.cash / (price * self.short_margin_ratio)

        short_proceeds = shares * price - commission
        self.cash += short_proceeds  # 卖空收入

        self.positions[symbol] = Position(
            symbol=symbol,
            shares=-shares,  # 负数表示空头
            avg_cost=price,  # 卖出价格
            current_price=price,
            asset_class=asset_class
        )

    def apply_short_borrowing_cost(self, days: int = 1):
        """
        应用空头借股成本

        借股成本按日计算，每日扣除:
        cost = |空头市值| × 年化借股费率 / 365 × 天数

        典型借股费率:
        - 容易借到的股票: 0.5% - 1% 年化
        - 一般难度: 1% - 3% 年化
        - 难借股票: 3% - 10%+ 年化

        本系统默认使用 2% 年化借股费率
        """
        if abs(self.short_market_value) > 0:
            daily_rate = self.short_borrow_rate / 365
            borrow_cost = abs(self.short_market_value) * daily_rate * days
            self.cash -= borrow_cost
            return borrow_cost
        return 0.0

    def record_value(self, timestamp: str, prices: Dict[str, float]):
        """记录组合价值"""
        self.update_prices(prices)

        # 应用每日借股成本 (如果有空头仓位)
        borrow_cost = self.apply_short_borrowing_cost(days=1)

        self.value_history.append({
            "timestamp": timestamp,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "market_value": self.total_market_value,
            "long_value": self.long_market_value,
            "short_value": self.short_market_value,
            "borrow_cost": borrow_cost,
            "weights": self.weights.copy()
        })

    def get_returns(self) -> np.ndarray:
        """计算日收益率序列"""
        if len(self.value_history) < 2:
            return np.array([])

        values = np.array([v["portfolio_value"] for v in self.value_history])
        returns = np.diff(values) / values[:-1]
        return returns

    def get_metrics(self) -> Dict[str, float]:
        """计算组合指标"""
        returns = self.get_returns()
        if len(returns) == 0:
            # 返回有意义的默认值而不是空字典
            return {
                "cumulative_return": self.total_return,
                "annual_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "n_trades": len(self.trade_history),
            }

        # 累计收益
        cumulative_return = self.total_return

        # 年化收益 (修复短期数据计算问题)
        n_days = len(returns)
        if n_days > 1:
            annual_return = (1 + cumulative_return) ** (252.0 / n_days) - 1
        else:
            annual_return = 0.0

        # 波动率
        volatility = np.std(returns) * np.sqrt(252)

        # 夏普比率
        risk_free = 0.02
        sharpe = (annual_return - risk_free) / volatility if volatility > 0 else 0

        # 最大回撤
        values = np.array([v["portfolio_value"] for v in self.value_history])
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - risk_free) / downside_std if downside_std > 0 else 0

        # 胜率
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

        return {
            "cumulative_return": cumulative_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "n_trades": len(self.trade_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "total_return": self.total_return,
            "long_market_value": self.long_market_value,
            "short_market_value": self.short_market_value,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "positions": {
                symbol: {
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "asset_class": pos.asset_class,
                    "is_short": pos.is_short,
                }
                for symbol, pos in self.positions.items()
            },
            "weights": self.weights,
            "class_weights": self.class_weights,
        }
