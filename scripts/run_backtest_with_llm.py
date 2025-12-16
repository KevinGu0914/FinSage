#!/usr/bin/env python3
"""
LLM Expert 真实回测脚本

使用SharedModelExpertManager进行真实历史数据回测，
计算实际交易收益、夏普比率、最大回撤等指标。

Usage:
    python scripts/run_backtest_with_llm.py --start 2024-10-01 --end 2024-11-30 --capital 1000000
"""

import os
import sys
import torch
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
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
# Portfolio Manager
# ============================================================

class Portfolio:
    """投资组合管理"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.position_costs: Dict[str, float] = {}  # symbol -> avg cost
        self.value_history: List[float] = []
        self.trade_count = 0
        self.transaction_cost_rate = 0.001  # 0.1%

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总价值"""
        total = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices and qty > 0:
                total += qty * prices[symbol]
        return total

    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        total_value: float
    ) -> Dict:
        """执行交易

        action: BUY_25%, BUY_50%, SELL_25%, HOLD 等
        """
        result = {"symbol": symbol, "action": action, "executed": False}

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
                    pct = 0.25  # 默认25%
        else:
            pct = 0.25

        current_qty = self.positions.get(symbol, 0)

        if "BUY" in action:
            # 计算购买金额
            buy_amount = total_value * pct
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

        # 年化收益率 (假设252交易日)
        days = len(values)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # 波动率
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # 夏普比率 (假设无风险利率4%)
        rf = 0.04 / 252
        excess_returns = returns - rf
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
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

ALL_SYMBOLS = []
for symbols in ASSETS.values():
    ALL_SYMBOLS.extend(symbols)


def fetch_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """获取市场数据 (使用 FMP API)"""
    from finsage.data.fmp_client import FMPClient

    logger.info(f"Fetching data for {ALL_SYMBOLS} from {start_date} to {end_date}")

    client = FMPClient()

    # 获取批量历史价格
    all_data = {}
    for symbol in ALL_SYMBOLS:
        try:
            df = client.get_historical_price(symbol, start_date, end_date)
            if df is not None and not df.empty:
                # FMP returns 'close' column
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

    # 合并为DataFrame
    prices = pd.DataFrame(all_data)
    prices = prices.ffill().bfill()
    logger.info(f"Got {len(prices)} trading days of data for {len(all_data)} symbols")

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

                # 计算技术指标
                if symbol in history.columns:
                    hist = history[symbol].dropna()
                    if len(hist) > 14:
                        # 日涨跌
                        change = (hist.iloc[-1] / hist.iloc[-2] - 1) * 100 if len(hist) > 1 else 0

                        # RSI
                        delta = hist.diff()
                        gain = delta.where(delta > 0, 0).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50

                        obs += f"- {symbol}: ${price:.2f}, 日涨跌: {change:+.2f}%, RSI: {rsi:.1f}\n"
                    else:
                        obs += f"- {symbol}: ${price:.2f}\n"
                else:
                    obs += f"- {symbol}: ${price:.2f}\n"

    # 宏观环境 (简化)
    vix_price = prices.get("^VIX", 20)
    spy_trend = "乐观"
    if "SPY" in history.columns and len(history["SPY"]) >= 5:
        spy_trend = "乐观" if prices.get("SPY", 0) > history["SPY"].iloc[-5] else "谨慎"
    obs += f"""
### 宏观环境
- 市场情绪: {spy_trend}
"""

    return obs


# ============================================================
# Main Backtest
# ============================================================

def run_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 5,  # 每N天rebalance
    model_path: str = "Qwen/Qwen2.5-32B-Instruct",
):
    """运行回测"""

    print("=" * 80)
    print(" LLM Expert 真实回测")
    print("=" * 80)
    print(f" 时间范围: {start_date} ~ {end_date}")
    print(f" 初始资金: ${initial_capital:,.2f}")
    print(f" 再平衡频率: 每{rebalance_freq}天")
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

    # 加载模型
    print("\n加载LLM Expert...")
    from finsage.rl.shared_expert_manager import SharedModelExpertManager

    manager = SharedModelExpertManager(
        model_path=model_path,
        device="cuda:0",
        bf16=True,
    )

    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 创建组合
    portfolio = Portfolio(initial_capital)

    # 回测循环
    print("\n" + "=" * 80)
    print(" 开始回测")
    print("=" * 80)

    decision_log = []

    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d")
        prices = prices_df.loc[date]

        # 记录组合价值
        price_dict = {s: prices[s] for s in ALL_SYMBOLS if s in prices.index}
        portfolio.record_value(price_dict)

        # 是否需要rebalance
        if i % rebalance_freq != 0 and i > 0:
            continue

        # 获取历史数据
        lookback = min(i + 1, 30)
        history = prices_df.iloc[max(0, i-lookback):i+1]

        # 创建市场观察
        obs = create_market_observation(date_str, prices, history)

        # 获取Expert决策
        print(f"\n[{date_str}] 组合价值: ${portfolio.get_total_value(price_dict):,.2f}")

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
                        total_value=portfolio.get_total_value(price_dict)
                    )
                    if result["executed"]:
                        print(f"  {role} -> {symbol}: {action}")

        decision_log.append({
            "date": date_str,
            "portfolio_value": portfolio.get_total_value(price_dict),
            "decisions": {k: v.get("action", "HOLD") for k, v in all_actions.items()}
        })

    # 计算最终指标
    metrics = portfolio.get_metrics()

    # 输出结果
    print("\n" + "=" * 80)
    print(" 回测结果")
    print("=" * 80)
    print(f" 初始资金: ${initial_capital:,.2f}")
    print(f" 最终价值: ${metrics.get('final_value', 0):,.2f}")
    print(f" 总收益率: {metrics.get('total_return', 0)*100:.2f}%")
    print(f" 年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f" 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f" 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f" 波动率: {metrics.get('volatility', 0)*100:.2f}%")
    print(f" 交易次数: {metrics.get('total_trades', 0)}")
    print("=" * 80)

    # 保存结果
    result = {
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "metrics": metrics,
        "portfolio_values": portfolio.value_history,
        "decisions": decision_log,
    }

    os.makedirs("/root/results", exist_ok=True)
    result_file = f"/root/results/llm_backtest_{start_date}_{end_date}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-10-01")
    parser.add_argument("--end", default="2024-11-30")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--freq", type=int, default=5, help="再平衡频率(天)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
