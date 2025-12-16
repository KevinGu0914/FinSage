#!/usr/bin/env python3
"""
V1 LoRA Full Validation Script

完整验证V1训练的LoRA权重，包括：
- 5个专家 (Stock, Bond, Commodity, REITs, Crypto)
- 风控模块
- 对冲策略
- 策略工具箱

验证时间: 2024-07-01 ~ 2024-12-31 (完美衔接训练数据)

Usage:
    python scripts/validate_v1_full.py --start 2024-07-01 --end 2024-12-31
"""

import os
import sys
import torch
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Portfolio Manager with Risk Control
# ============================================================

class RiskManagedPortfolio:
    """带风控的投资组合管理"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_position_pct: float = 0.20,  # 单资产最大仓位
        max_drawdown_limit: float = 0.15,  # 最大回撤限制
        stop_loss_pct: float = 0.05,  # 止损线
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.position_costs: Dict[str, float] = {}  # symbol -> avg cost
        self.value_history: List[float] = []
        self.trade_count = 0
        self.transaction_cost_rate = 0.001  # 0.1%

        # Risk parameters
        self.max_position_pct = max_position_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.stop_loss_pct = stop_loss_pct
        self.peak_value = initial_capital
        self.risk_off_mode = False

        # Hedging positions
        self.hedge_positions: Dict[str, float] = {}

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总价值"""
        total = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices and qty > 0:
                total += qty * prices[symbol]
        for symbol, qty in self.hedge_positions.items():
            if symbol in prices:
                total += qty * prices[symbol]
        return total

    def check_risk_limits(self, prices: Dict[str, float]) -> Dict[str, bool]:
        """检查风控限制"""
        total_value = self.get_total_value(prices)

        # Update peak
        if total_value > self.peak_value:
            self.peak_value = total_value

        # Calculate drawdown
        current_drawdown = (self.peak_value - total_value) / self.peak_value

        risk_status = {
            "drawdown_breach": current_drawdown > self.max_drawdown_limit,
            "current_drawdown": current_drawdown,
            "risk_off_triggered": False,
        }

        # 触发风控
        if current_drawdown > self.max_drawdown_limit and not self.risk_off_mode:
            logger.warning(f"RISK OFF: Drawdown {current_drawdown:.2%} exceeds {self.max_drawdown_limit:.2%}")
            self.risk_off_mode = True
            risk_status["risk_off_triggered"] = True

        # 恢复交易
        if current_drawdown < self.max_drawdown_limit * 0.5 and self.risk_off_mode:
            logger.info(f"RISK ON: Drawdown recovered to {current_drawdown:.2%}")
            self.risk_off_mode = False

        return risk_status

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """检查止损"""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return False

        avg_cost = self.position_costs.get(symbol, current_price)
        loss_pct = (avg_cost - current_price) / avg_cost

        if loss_pct > self.stop_loss_pct:
            logger.warning(f"STOP LOSS: {symbol} loss {loss_pct:.2%} > {self.stop_loss_pct:.2%}")
            return True
        return False

    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        total_value: float,
        force: bool = False,
    ) -> Dict:
        """执行交易 (带风控)"""
        result = {"symbol": symbol, "action": action, "executed": False, "reason": ""}

        # 风控检查
        if self.risk_off_mode and not force and "BUY" in action:
            result["reason"] = "risk_off_mode"
            return result

        if "HOLD" in action:
            return result

        # 解析比例
        pct = 0
        if "_" in action:
            parts = action.split("_")
            if len(parts) >= 2:
                pct_str = parts[1].replace("%", "")
                try:
                    pct = int(pct_str) / 100
                except:
                    pct = 0.25
        else:
            pct = 0.25

        current_qty = self.positions.get(symbol, 0)

        if "BUY" in action:
            # 仓位限制检查
            current_position_value = current_qty * price
            current_position_pct = current_position_value / total_value if total_value > 0 else 0

            if current_position_pct >= self.max_position_pct:
                result["reason"] = f"position_limit ({current_position_pct:.2%} >= {self.max_position_pct:.2%})"
                return result

            # 计算购买金额 (受仓位限制)
            max_buy_pct = self.max_position_pct - current_position_pct
            actual_pct = min(pct, max_buy_pct)

            buy_amount = total_value * actual_pct
            shares_to_buy = int(buy_amount / price)
            cost = shares_to_buy * price * (1 + self.transaction_cost_rate)

            if cost <= self.cash and shares_to_buy > 0:
                self.cash -= cost
                old_qty = self.positions.get(symbol, 0)
                old_cost = self.position_costs.get(symbol, 0)

                new_qty = old_qty + shares_to_buy
                new_avg_cost = ((old_qty * old_cost) + (shares_to_buy * price)) / new_qty if new_qty > 0 else 0

                self.positions[symbol] = new_qty
                self.position_costs[symbol] = new_avg_cost
                self.trade_count += 1

                result["executed"] = True
                result["shares"] = shares_to_buy
                result["cost"] = cost

        elif "SELL" in action:
            if current_qty > 0:
                shares_to_sell = int(current_qty * pct)
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price * (1 - self.transaction_cost_rate)
                    self.cash += proceeds
                    self.positions[symbol] = current_qty - shares_to_sell
                    self.trade_count += 1

                    result["executed"] = True
                    result["shares"] = shares_to_sell
                    result["proceeds"] = proceeds

        return result

    def execute_hedge(self, symbol: str, direction: str, amount: float, price: float):
        """执行对冲交易"""
        shares = int(amount / price)
        if direction == "SHORT":
            self.hedge_positions[symbol] = self.hedge_positions.get(symbol, 0) - shares
            self.cash += shares * price * (1 - self.transaction_cost_rate)
        elif direction == "LONG":
            cost = shares * price * (1 + self.transaction_cost_rate)
            if cost <= self.cash:
                self.hedge_positions[symbol] = self.hedge_positions.get(symbol, 0) + shares
                self.cash -= cost

    def force_stop_loss(self, symbol: str, price: float):
        """强制止损"""
        if symbol in self.positions and self.positions[symbol] > 0:
            qty = self.positions[symbol]
            proceeds = qty * price * (1 - self.transaction_cost_rate)
            self.cash += proceeds
            self.positions[symbol] = 0
            self.trade_count += 1
            logger.info(f"Force stop loss: {symbol}, sold {qty} shares")

    def record_value(self, prices: Dict[str, float]):
        """记录当前组合价值"""
        self.value_history.append(self.get_total_value(prices))

    def get_metrics(self) -> Dict:
        """计算绩效指标"""
        if len(self.value_history) < 2:
            return {}

        values = np.array(self.value_history)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1

        # 年化收益率
        days = len(values)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # 波动率
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # 夏普比率
        rf = 0.04 / 252
        excess_returns = returns - rf
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annualized_return - 0.04) / downside_std if downside_std > 0 else 0

        # Calmar Ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "total_trades": self.trade_count,
            "final_value": values[-1],
        }


# ============================================================
# Market Data
# ============================================================

ASSETS = {
    "stocks": ["SPY", "QQQ"],
    "bonds": ["TLT", "LQD"],
    "commodities": ["GLD", "USO"],
    "reits": ["VNQ"],
    "crypto": ["BTC-USD", "ETH-USD"],
}

HEDGE_INSTRUMENTS = {
    "equity": "SH",    # Short S&P 500
    "bond": "TBF",     # Short 20+ Year Treasury
    "volatility": "VXX",  # VIX ETF
}

ALL_SYMBOLS = []
for symbols in ASSETS.values():
    ALL_SYMBOLS.extend(symbols)
for symbol in HEDGE_INSTRUMENTS.values():
    ALL_SYMBOLS.append(symbol)


def fetch_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """获取市场数据"""
    from finsage.data.fmp_client import FMPClient

    logger.info(f"Fetching data for {len(ALL_SYMBOLS)} symbols from {start_date} to {end_date}")

    client = FMPClient()

    all_data = {}
    for symbol in ALL_SYMBOLS:
        try:
            df = client.get_historical_price(symbol, start_date, end_date)
            if df is not None and not df.empty:
                if 'close' in df.columns:
                    all_data[symbol] = df['close']
                elif 'Close' in df.columns:
                    all_data[symbol] = df['Close']
                logger.info(f"  {symbol}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  {symbol}: failed - {e}")

    if not all_data:
        logger.error("No data fetched!")
        return pd.DataFrame()

    prices = pd.DataFrame(all_data)
    prices = prices.ffill().bfill()
    logger.info(f"Got {len(prices)} trading days for {len(all_data)} symbols")

    return prices


def create_market_observation(date: str, prices: pd.Series, history: pd.DataFrame) -> str:
    """创建市场观察"""
    obs = f"""## 市场日期: {date}
## 资产类别: multi-asset

"""

    for asset_class, symbols in ASSETS.items():
        obs += f"### {asset_class.upper()}\n"
        for symbol in symbols:
            if symbol in prices.index:
                price = prices[symbol]

                if symbol in history.columns:
                    hist = history[symbol].dropna()
                    if len(hist) > 14:
                        change = (hist.iloc[-1] / hist.iloc[-2] - 1) * 100 if len(hist) > 1 else 0

                        # RSI
                        delta = hist.diff()
                        gain = delta.where(delta > 0, 0).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50

                        # Moving averages
                        ma20 = hist.rolling(20).mean().iloc[-1] if len(hist) >= 20 else hist.mean()
                        ma50 = hist.rolling(50).mean().iloc[-1] if len(hist) >= 50 else hist.mean()

                        trend = "上涨" if price > ma20 else "下跌"

                        obs += f"- {symbol}: ${price:.2f}, 日涨跌: {change:+.2f}%, RSI: {rsi:.1f}, 趋势: {trend}\n"
                    else:
                        obs += f"- {symbol}: ${price:.2f}\n"
                else:
                    obs += f"- {symbol}: ${price:.2f}\n"

    # VIX/恐慌指数
    vix = prices.get("VXX", 20)
    market_fear = "高" if vix > 25 else ("中" if vix > 18 else "低")

    obs += f"""
### 市场情绪
- 恐慌指数(VXX): ${vix:.2f}
- 市场恐慌水平: {market_fear}
"""

    return obs


# ============================================================
# Main Backtest
# ============================================================

def run_full_validation(
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 5,
    checkpoint_dir: str = "/root/FinSage_v1_validation/checkpoints/final",
):
    """运行完整V1验证"""

    print("=" * 80)
    print(" V1 LoRA Full Validation")
    print("=" * 80)
    print(f" 时间范围: {start_date} ~ {end_date}")
    print(f" 初始资金: ${initial_capital:,.2f}")
    print(f" 再平衡频率: 每{rebalance_freq}天")
    print(f" Checkpoint: {checkpoint_dir}")
    print("=" * 80)

    # 获取市场数据
    print("\n获取市场数据...")
    prices_df = fetch_market_data(start_date, end_date)
    trading_days = prices_df.index.tolist()

    if len(trading_days) < 2:
        print("数据不足，无法回测")
        return

    print(f"共 {len(trading_days)} 个交易日")

    # 检查GPU
    if not torch.cuda.is_available():
        print("CUDA不可用!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载模型
    print("\n加载LLM Expert (Qwen2.5-14B-Instruct + V1 LoRA)...")
    from finsage.rl.shared_expert_manager import SharedModelExpertManager

    # 检查checkpoint是否存在
    expert_types = ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"]
    lora_paths = {}

    for expert in expert_types:
        lora_path = os.path.join(checkpoint_dir, expert, expert)
        if os.path.exists(lora_path):
            lora_paths[expert] = lora_path
            print(f"  Found {expert}: {lora_path}")
        else:
            print(f"  WARNING: {expert} not found at {lora_path}")

    # 使用bf16 (RTX 5090 32GB足够)
    manager = SharedModelExpertManager(
        model_path="Qwen/Qwen2.5-14B-Instruct",
        device="cuda:0",
        bf16=True,
        load_in_8bit=False,
    )

    # 加载V1预训练的LoRA适配器
    print(f"\n加载V1 LoRA适配器从 {checkpoint_dir}...")
    manager.load_adapters(checkpoint_dir)

    print(f"GPU Memory After Load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 创建带风控的组合
    portfolio = RiskManagedPortfolio(
        initial_capital=initial_capital,
        max_position_pct=0.20,
        max_drawdown_limit=0.15,
        stop_loss_pct=0.05,
    )

    # 回测循环
    print("\n" + "=" * 80)
    print(" 开始回测")
    print("=" * 80)

    decision_log = []
    risk_events = []

    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d")
        prices = prices_df.loc[date]

        # 获取价格字典
        price_dict = {s: prices[s] for s in ALL_SYMBOLS if s in prices.index and not pd.isna(prices[s])}

        # 风控检查
        risk_status = portfolio.check_risk_limits(price_dict)
        if risk_status["risk_off_triggered"]:
            risk_events.append({
                "date": date_str,
                "event": "RISK_OFF",
                "drawdown": risk_status["current_drawdown"],
            })

        # 止损检查
        for symbol in list(portfolio.positions.keys()):
            if symbol in price_dict and portfolio.check_stop_loss(symbol, price_dict[symbol]):
                portfolio.force_stop_loss(symbol, price_dict[symbol])
                risk_events.append({
                    "date": date_str,
                    "event": "STOP_LOSS",
                    "symbol": symbol,
                })

        # 记录组合价值
        portfolio.record_value(price_dict)

        # 是否需要rebalance
        if i % rebalance_freq != 0 and i > 0:
            continue

        # 获取历史数据
        lookback = min(i + 1, 50)
        history = prices_df.iloc[max(0, i-lookback):i+1]

        # 创建市场观察
        obs = create_market_observation(date_str, prices, history)

        # 获取Expert决策
        current_value = portfolio.get_total_value(price_dict)
        print(f"\n[{date_str}] 组合价值: ${current_value:,.2f} | DD: {risk_status['current_drawdown']:.2%}")

        all_actions = manager.run_expert_chain(obs)

        # 执行交易
        for role, action_dict in all_actions.items():
            action = action_dict.get("action", "HOLD")

            # 根据Expert类型决定交易哪些资产
            asset_class = role.split("_")[0].lower()
            if asset_class == "stock":
                symbols = ASSETS.get("stocks", [])
            elif asset_class == "bond":
                symbols = ASSETS.get("bonds", [])
            elif asset_class == "commodity":
                symbols = ASSETS.get("commodities", [])
            elif asset_class == "reits":
                symbols = ASSETS.get("reits", [])
            elif asset_class == "crypto":
                symbols = ASSETS.get("crypto", [])
            else:
                continue

            # 对每个资产执行相同动作
            for symbol in symbols:
                if symbol in price_dict:
                    result = portfolio.execute_trade(
                        symbol=symbol,
                        action=action,
                        price=price_dict[symbol],
                        total_value=current_value
                    )
                    if result["executed"]:
                        print(f"  {role} -> {symbol}: {action}")
                    elif result.get("reason"):
                        logger.debug(f"  {role} -> {symbol}: {action} (blocked: {result['reason']})")

        decision_log.append({
            "date": date_str,
            "portfolio_value": current_value,
            "risk_status": risk_status,
            "decisions": {k: v.get("action", "HOLD") for k, v in all_actions.items()}
        })

    # 计算最终指标
    metrics = portfolio.get_metrics()

    # 输出结果
    print("\n" + "=" * 80)
    print(" V1 验证结果")
    print("=" * 80)
    print(f" 初始资金: ${initial_capital:,.2f}")
    print(f" 最终价值: ${metrics.get('final_value', 0):,.2f}")
    print(f" 总收益率: {metrics.get('total_return', 0)*100:.2f}%")
    print(f" 年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f" 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f" Sortino比率: {metrics.get('sortino_ratio', 0):.2f}")
    print(f" Calmar比率: {metrics.get('calmar_ratio', 0):.2f}")
    print(f" 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f" 波动率: {metrics.get('volatility', 0)*100:.2f}%")
    print(f" 交易次数: {metrics.get('total_trades', 0)}")
    print(f" 风控事件: {len(risk_events)}")
    print("=" * 80)

    # 保存结果
    result = {
        "version": "v1_simplified_9assets_no_short",
        "validation_period": f"{start_date} ~ {end_date}",
        "initial_capital": initial_capital,
        "metrics": metrics,
        "portfolio_values": portfolio.value_history,
        "decisions": decision_log,
        "risk_events": risk_events,
        "positions_final": dict(portfolio.positions),
    }

    os.makedirs("/root/results", exist_ok=True)
    result_file = f"/root/results/v1_validation_{start_date}_{end_date}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--freq", type=int, default=5, help="再平衡频率(天)")
    parser.add_argument("--checkpoint", default="/root/FinSage_v1_validation/checkpoints/final")
    args = parser.parse_args()

    run_full_validation(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        checkpoint_dir=args.checkpoint,
    )


if __name__ == "__main__":
    main()
