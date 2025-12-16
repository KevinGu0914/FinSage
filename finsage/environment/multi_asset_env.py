"""
Multi-Asset Trading Environment
多资产交易环境
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import logging

from finsage.environment.portfolio_state import PortfolioState
from finsage.config import AssetConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """环境配置"""
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    min_trade_value: float = 100.0
    max_single_asset: float = 0.15
    max_asset_class: float = 0.50
    rebalance_threshold: float = 0.02  # 降低至2%，更频繁触发再平衡


class MultiAssetTradingEnv:
    """
    多资产交易环境

    支持多种资产类别的交易模拟:
    - 股票 (stocks)
    - 债券 (bonds)
    - 大宗商品 (commodities)
    - REITs
    - 加密货币 (crypto)

    功能:
    - 模拟交易执行
    - 计算收益和风险指标
    - 支持再平衡操作
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        asset_universe: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初始化环境

        Args:
            config: 环境配置
            asset_universe: 资产池 {asset_class: [symbols]}
        """
        self.config = config or EnvConfig()
        self.asset_universe = asset_universe or self._default_universe()

        # 状态
        self.portfolio = PortfolioState(
            initial_capital=self.config.initial_capital,
            cash=self.config.initial_capital
        )
        self.current_step = 0
        self.current_date = None
        self.market_data = {}
        self.done = False

        logger.info("MultiAssetTradingEnv initialized")

    def _default_universe(self) -> Dict[str, List[str]]:
        """默认资产池 - 从 config.py AssetConfig 获取"""
        return AssetConfig().default_universe

    def reset(
        self,
        initial_capital: Optional[float] = None,
        start_date: Optional[str] = None
    ) -> PortfolioState:
        """
        重置环境

        Args:
            initial_capital: 初始资金
            start_date: 开始日期

        Returns:
            初始组合状态
        """
        capital = initial_capital or self.config.initial_capital
        self.portfolio = PortfolioState(
            initial_capital=capital,
            cash=capital
        )
        self.current_step = 0
        self.current_date = start_date
        self.done = False

        logger.info(f"Environment reset with capital={capital}")
        return self.portfolio

    def step(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        timestamp: str,
        expert_reports: Optional[Dict[str, Any]] = None
    ) -> Tuple[PortfolioState, float, bool, Dict]:
        """
        执行一步交易

        Args:
            target_allocation: 目标配置 {symbol: weight}
            market_data: 市场数据 {symbol: {price, ...}}
            timestamp: 时间戳
            expert_reports: 专家报告 (用于个股权重分配)

        Returns:
            Tuple: (新状态, 奖励, 是否结束, 信息)
        """
        self.current_step += 1
        self.current_date = timestamp
        self.market_data = market_data

        # 获取当前价格
        prices = self._extract_prices(market_data)
        self.portfolio.update_prices(prices)

        # 执行再平衡 (传递专家报告用于个股权重分配)
        trades = self._rebalance(target_allocation, prices, timestamp, expert_reports)

        # 记录价值
        self.portfolio.record_value(timestamp, prices)

        # 计算奖励
        reward = self._calculate_reward()

        # 返回信息
        info = {
            "step": self.current_step,
            "timestamp": timestamp,
            "trades": trades,
            "portfolio_value": self.portfolio.portfolio_value,
            "return": self.portfolio.total_return,
        }

        return self.portfolio, reward, self.done, info

    def _extract_prices(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """提取价格数据"""
        prices = {}
        # 需要跳过的非价格数据键
        skip_keys = {"news", "macro", "returns", "covariance", "volatilities", "expected_returns"}

        for symbol, data in market_data.items():
            # 跳过非价格数据
            if symbol in skip_keys:
                continue

            if isinstance(data, dict):
                # 确保是包含价格信息的字典
                price = data.get("close", data.get("price", None))
                if price is not None:
                    try:
                        prices[symbol] = float(price)
                    except (TypeError, ValueError):
                        continue
            elif isinstance(data, (int, float)):
                prices[symbol] = float(data)
            # 跳过列表或其他类型
        return prices

    def _rebalance(
        self,
        target_allocation: Dict[str, float],
        prices: Dict[str, float],
        timestamp: str,
        expert_reports: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        执行再平衡 (支持做空)

        Args:
            target_allocation: 目标配置 (可以是资产类别级别或股票代码级别)
                              负权重表示做空
            prices: 当前价格
            timestamp: 时间戳
            expert_reports: 专家报告 (用于个股权重分配)

        Returns:
            交易记录列表
        """
        trades = []
        portfolio_value = self.portfolio.portfolio_value
        current_weights = self.portfolio.weights

        # 检查 target_allocation 是资产类别级别还是股票代码级别
        asset_classes = set(self.asset_universe.keys())
        is_class_level = any(key in asset_classes for key in target_allocation.keys())

        if is_class_level:
            # 资产类别级别配置，需要转换为股票代码级别
            # 使用专家报告来决定个股权重分配
            symbol_allocation = self._expand_class_allocation(target_allocation, prices, expert_reports)
            logger.info(f"Expanded class allocation to {len(symbol_allocation)} symbols")
        else:
            symbol_allocation = target_allocation

        # 计算需要的交易
        for symbol, target_weight in symbol_allocation.items():
            if symbol == "cash":
                continue

            current_weight = current_weights.get(symbol, 0)

            # 检查当前是否有空头仓位
            current_pos = self.portfolio.positions.get(symbol)
            if current_pos and current_pos.is_short:
                # 空头仓位的权重是负数
                current_weight = -abs(current_weight)

            weight_diff = target_weight - current_weight

            # 检查是否超过再平衡阈值
            if abs(weight_diff) < self.config.rebalance_threshold:
                continue

            if symbol not in prices or prices[symbol] <= 0:
                logger.debug(f"Skipping {symbol}: not in prices or price <= 0")
                continue

            price = prices[symbol]
            target_value = target_weight * portfolio_value
            current_value = current_weight * portfolio_value
            trade_value = target_value - current_value

            # 应用滑点
            if trade_value > 0:
                price = price * (1 + self.config.slippage)
            else:
                price = price * (1 - self.config.slippage)

            shares = trade_value / price

            # 检查最小交易金额
            if abs(trade_value) < self.config.min_trade_value:
                continue

            # 确定资产类别
            asset_class = self._get_asset_class(symbol)

            # 确定是否为做空操作
            is_short = target_weight < 0

            # 执行交易
            trade_record = self.portfolio.execute_trade(
                symbol=symbol,
                shares=shares,
                price=price,
                asset_class=asset_class,
                timestamp=timestamp,
                is_short=is_short
            )
            trades.append(trade_record)

            action_type = trade_record.get('action', 'TRADE')
            logger.info(f"Executed trade: {action_type} {abs(shares):.2f} shares of {symbol} @ ${price:.2f}")

        return trades

    def _expand_class_allocation(
        self,
        class_allocation: Dict[str, float],
        prices: Dict[str, float],
        expert_reports: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        将资产类别级别的配置展开为股票代码级别

        策略: 使用专家报告中的个股推荐来分配权重
        - 如果有专家报告，根据专家的 action 和 confidence 计算权重
        - BUY 信号获得更高权重，SELL 信号获得更低权重
        - 没有专家报告时使用平均分配

        Args:
            class_allocation: 资产类别配置 {"stocks": 0.40, "bonds": 0.25, ...}
            prices: 当前价格
            expert_reports: 专家报告 (可选)

        Returns:
            股票代码级别配置 {"SPY": 0.05, "QQQ": 0.05, ...}
        """
        symbol_allocation = {}

        for asset_class, class_weight in class_allocation.items():
            if asset_class == "cash":
                symbol_allocation["cash"] = class_weight
                continue

            # 获取该类别下的所有资产
            symbols = self.asset_universe.get(asset_class, [])

            # 过滤出有价格数据的资产
            available_symbols = [s for s in symbols if s in prices and prices[s] > 0]

            if not available_symbols:
                logger.warning(f"No available symbols for {asset_class}, skipping")
                continue

            # 检查是否有专家报告
            expert_report = None
            if expert_reports and asset_class in expert_reports:
                expert_report = expert_reports.get(asset_class)

            if expert_report and hasattr(expert_report, 'recommendations'):
                # 使用专家推荐来分配权重
                symbol_weights = self._compute_expert_driven_weights(
                    asset_class=asset_class,
                    available_symbols=available_symbols,
                    expert_report=expert_report,
                    class_weight=class_weight
                )
                symbol_allocation.update(symbol_weights)
                logger.debug(f"{asset_class}: Expert-driven allocation with {len(symbol_weights)} symbols")
            else:
                # 没有专家报告，使用平均分配
                weight_per_symbol = class_weight / len(available_symbols)
                for symbol in available_symbols:
                    symbol_allocation[symbol] = weight_per_symbol

        return symbol_allocation

    def _compute_expert_driven_weights(
        self,
        asset_class: str,
        available_symbols: List[str],
        expert_report: Any,
        class_weight: float
    ) -> Dict[str, float]:
        """
        根据专家报告计算个股权重 (支持做空)

        权重计算逻辑:
        - 从专家推荐中提取每个股票的评分 (action + confidence)
        - BUY_100%: 2.0, BUY_75%: 1.5, BUY_50%: 1.25, BUY_25%: 1.1
        - HOLD: 1.0
        - SELL_25%: 0.75, SELL_50%: 0.5, SELL_75%: 0.25, SELL_100%: 0.0
        - SHORT_25%: -0.25, SHORT_50%: -0.5, SHORT_75%: -0.75, SHORT_100%: -1.0
        - 最终权重 = base_weight * action_multiplier * confidence

        做空逻辑:
        - SHORT信号生成负权重
        - 负权重表示做空该资产
        - 做空规模受保证金限制

        Args:
            asset_class: 资产类别
            available_symbols: 可用股票列表
            expert_report: 专家报告
            class_weight: 该类别的总权重

        Returns:
            个股权重字典 (负权重表示做空)
        """
        # Action 到乘数的映射 (支持做空)
        action_multipliers = {
            # 做多信号
            "BUY_100%": 2.0,
            "BUY_75%": 1.75,
            "BUY_50%": 1.5,
            "BUY_25%": 1.25,
            "HOLD": 1.0,
            # 减仓信号
            "SELL_25%": 0.75,
            "SELL_50%": 0.5,
            "SELL_75%": 0.25,
            "SELL_100%": 0.0,  # 完全平仓
            # 做空信号 (负乘数)
            "SHORT_25%": -0.25,
            "SHORT_50%": -0.5,
            "SHORT_75%": -0.75,
            "SHORT_100%": -1.0,
        }

        # 提取专家推荐中的股票评分
        symbol_scores = {}
        short_symbols = []  # 记录做空的股票

        for rec in expert_report.recommendations:
            symbol = rec.symbol
            if symbol in available_symbols:
                # 获取 action 乘数
                action_str = rec.action.value if hasattr(rec.action, 'value') else str(rec.action)
                multiplier = action_multipliers.get(action_str, 1.0)

                # 结合 confidence
                confidence = rec.confidence if hasattr(rec, 'confidence') else 0.7

                if multiplier < 0:
                    # 做空信号
                    short_symbols.append(symbol)
                    score = multiplier * confidence  # 保持负数
                    logger.info(f"  SHORT signal: {symbol}: action={action_str}, conf={confidence:.2f}, score={score:.2f}")
                else:
                    score = multiplier * confidence

                symbol_scores[symbol] = score
                logger.debug(f"  {symbol}: action={action_str}, conf={confidence:.2f}, score={score:.2f}")

        # 对于没有被专家提及的股票，给予基础评分
        for symbol in available_symbols:
            if symbol not in symbol_scores:
                symbol_scores[symbol] = 0.5  # 默认较低权重

        # 分离多头和空头
        long_scores = {s: v for s, v in symbol_scores.items() if v > 0}
        short_scores = {s: v for s, v in symbol_scores.items() if v < 0}

        symbol_weights = {}

        # 计算多头权重
        total_long_score = sum(long_scores.values())
        if total_long_score > 0:
            for symbol, score in long_scores.items():
                symbol_weights[symbol] = (score / total_long_score) * class_weight

        # 计算空头权重 (使用类别权重的一部分，最多50%)
        if short_scores:
            max_short_weight = class_weight * 0.5  # 最多使用类别权重的50%做空
            total_short_score = abs(sum(short_scores.values()))
            for symbol, score in short_scores.items():
                # 负权重表示做空
                symbol_weights[symbol] = (score / total_short_score) * max_short_weight

        # 如果没有有效评分，使用平均分配
        if not symbol_weights:
            weight_per_symbol = class_weight / len(available_symbols)
            return {s: weight_per_symbol for s in available_symbols}

        return symbol_weights

    def _get_asset_class(self, symbol: str) -> str:
        """获取资产类别"""
        for asset_class, symbols in self.asset_universe.items():
            if symbol in symbols:
                return asset_class
        return "other"

    def _calculate_reward(self) -> float:
        """
        计算奖励

        综合考虑:
        - 收益率
        - 风险调整收益
        - 交易成本惩罚
        """
        returns = self.portfolio.get_returns()
        if len(returns) == 0:
            return 0.0

        # 最近收益
        recent_return = returns[-1] if len(returns) > 0 else 0

        # 风险惩罚 (简化的波动率)
        recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        risk_penalty = 0.5 * recent_vol

        # 交易成本惩罚
        n_recent_trades = len([
            t for t in self.portfolio.trade_history[-5:]
        ])
        trade_penalty = 0.001 * n_recent_trades

        reward = recent_return - risk_penalty - trade_penalty
        return reward

    def get_observation(self) -> Dict[str, Any]:
        """获取当前观察"""
        return {
            "portfolio": self.portfolio.to_dict(),
            "market_data": self.market_data,
            "step": self.current_step,
            "date": self.current_date,
        }

    def get_metrics(self) -> Dict[str, float]:
        """获取组合指标"""
        return self.portfolio.get_metrics()

    def render(self) -> str:
        """渲染当前状态"""
        state = self.portfolio.to_dict()
        lines = [
            f"=== Step {self.current_step} | {self.current_date} ===",
            f"Portfolio Value: ${state['portfolio_value']:,.2f}",
            f"Total Return: {state['total_return']:.2%}",
            f"Cash: ${state['cash']:,.2f} ({state['weights'].get('cash', 0):.1%})",
            "",
            "Positions:",
        ]

        for symbol, pos in state["positions"].items():
            lines.append(
                f"  {symbol}: {pos['shares']:.2f} shares @ ${pos['current_price']:.2f} "
                f"(PnL: ${pos['unrealized_pnl']:,.2f})"
            )

        lines.append("")
        lines.append("Class Weights:")
        for cls, weight in state["class_weights"].items():
            lines.append(f"  {cls}: {weight:.1%}")

        return "\n".join(lines)

    def close(self):
        """关闭环境"""
        logger.info("Environment closed")

    def get_state(self) -> Dict[str, Any]:
        """
        获取环境状态 (用于断点保存)

        Returns:
            可序列化的状态字典
        """
        # 序列化 positions
        positions_state = {}
        for symbol, pos in self.portfolio.positions.items():
            positions_state[symbol] = {
                "symbol": pos.symbol,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "asset_class": pos.asset_class,
            }

        return {
            "portfolio": {
                "initial_capital": self.portfolio.initial_capital,
                "cash": self.portfolio.cash,
                "positions": positions_state,
                "trade_history": self.portfolio.trade_history.copy(),
                "value_history": self.portfolio.value_history.copy(),
                "timestamp": self.portfolio.timestamp,
            },
            "current_step": self.current_step,
            "current_date": self.current_date,
            "done": self.done,
        }

    def restore_state(self, state: Dict[str, Any]):
        """
        恢复环境状态 (用于断点恢复)

        Args:
            state: 之前保存的状态字典
        """
        from finsage.environment.portfolio_state import Position

        portfolio_state = state["portfolio"]

        # 恢复 portfolio
        self.portfolio.initial_capital = portfolio_state["initial_capital"]
        self.portfolio.cash = portfolio_state["cash"]
        self.portfolio.trade_history = portfolio_state["trade_history"]
        self.portfolio.value_history = portfolio_state["value_history"]
        self.portfolio.timestamp = portfolio_state["timestamp"]

        # 恢复 positions
        self.portfolio.positions = {}
        for symbol, pos_data in portfolio_state["positions"].items():
            self.portfolio.positions[symbol] = Position(
                symbol=pos_data["symbol"],
                shares=pos_data["shares"],
                avg_cost=pos_data["avg_cost"],
                current_price=pos_data["current_price"],
                asset_class=pos_data["asset_class"],
            )

        # 恢复环境状态
        self.current_step = state["current_step"]
        self.current_date = state["current_date"]
        self.done = state["done"]

        logger.info(f"Environment state restored: step={self.current_step}, date={self.current_date}")
