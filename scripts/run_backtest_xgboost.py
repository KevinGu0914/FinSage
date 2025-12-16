#!/usr/bin/env python3
"""
XGBoost 基准回测脚本

使用XGBoost模型进行多资产交易决策回测，
作为LLM Expert策略的对比基准。

Usage:
    python scripts/run_backtest_xgboost.py --start 2024-10-01 --end 2024-11-30 --capital 1000000
"""

import os
import sys
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
# Portfolio Manager (same as LLM version)
# ============================================================

class Portfolio:
    """投资组合管理"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.position_costs: Dict[str, float] = {}
        self.value_history: List[float] = []
        self.trade_count = 0
        self.transaction_cost_rate = 0.001  # 0.1%

    def get_total_value(self, prices: Dict[str, float]) -> float:
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
                    pct = 0.25
        else:
            pct = 0.25

        current_qty = self.positions.get(symbol, 0)

        if "BUY" in action:
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

        return result

    def record_value(self, prices: Dict[str, float]):
        self.value_history.append(self.get_total_value(prices))

    def get_metrics(self) -> Dict:
        if len(self.value_history) < 2:
            return {}

        values = np.array(self.value_history)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1
        days = len(values)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        rf = 0.04 / 252
        excess_returns = returns - rf
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

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
    "crypto": ["BTCUSD", "ETHUSD"],  # FMP format (not Yahoo Finance BTC-USD)
}

ALL_SYMBOLS = []
for symbols in ASSETS.values():
    ALL_SYMBOLS.extend(symbols)


def fetch_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """获取市场数据 (使用 FMP API)"""
    from finsage.data.fmp_client import FMPClient

    logger.info(f"Fetching data for {ALL_SYMBOLS} from {start_date} to {end_date}")

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
    logger.info(f"Got {len(prices)} trading days of data for {len(all_data)} symbols")

    return prices


# ============================================================
# Feature Engineering
# ============================================================

def create_features(prices_df: pd.DataFrame, symbol: str, lookback: int = 20) -> pd.DataFrame:
    """创建技术指标特征"""
    if symbol not in prices_df.columns:
        return pd.DataFrame()

    df = pd.DataFrame(index=prices_df.index)
    price = prices_df[symbol]

    # 收益率
    df['return_1d'] = price.pct_change(1)
    df['return_5d'] = price.pct_change(5)
    df['return_10d'] = price.pct_change(10)
    df['return_20d'] = price.pct_change(20)

    # 移动平均
    df['sma_5'] = price.rolling(5).mean() / price - 1
    df['sma_10'] = price.rolling(10).mean() / price - 1
    df['sma_20'] = price.rolling(20).mean() / price - 1

    # 波动率
    df['volatility_5'] = price.pct_change().rolling(5).std()
    df['volatility_20'] = price.pct_change().rolling(20).std()

    # RSI
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = price.ewm(span=12).mean()
    ema26 = price.ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / price

    # Bollinger Bands位置
    sma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    df['bb_position'] = (price - sma20) / (2 * std20)

    # 动量
    df['momentum_10'] = price / price.shift(10) - 1
    df['momentum_20'] = price / price.shift(20) - 1

    # 跨资产特征 - SPY作为市场基准
    if 'SPY' in prices_df.columns and symbol != 'SPY':
        spy = prices_df['SPY']
        df['spy_return_5d'] = spy.pct_change(5)
        df['correlation_spy'] = price.pct_change().rolling(20).corr(spy.pct_change())

    return df.dropna()


def prepare_training_data(prices_df: pd.DataFrame, symbol: str, forward_days: int = 5):
    """准备训练数据"""
    features = create_features(prices_df, symbol)
    if features.empty:
        return None, None

    # 目标: 未来N天收益率
    future_return = prices_df[symbol].pct_change(forward_days).shift(-forward_days)

    # 对齐
    common_idx = features.index.intersection(future_return.dropna().index)
    X = features.loc[common_idx]
    y = future_return.loc[common_idx]

    return X, y


# ============================================================
# XGBoost Model
# ============================================================

class XGBoostTrader:
    """XGBoost交易模型"""

    def __init__(self):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            print("Installing xgboost...")
            os.system("pip install xgboost -q")
            import xgboost as xgb
            self.xgb = xgb

        self.models = {}  # symbol -> model
        self.feature_names = None

    def train(self, prices_df: pd.DataFrame, train_end_idx: int):
        """训练模型"""
        train_prices = prices_df.iloc[:train_end_idx]

        for symbol in ALL_SYMBOLS:
            X, y = prepare_training_data(train_prices, symbol)
            if X is None or len(X) < 50:
                continue

            # 分类目标: 1=买入信号(未来收益>1%), 0=卖出信号(未来收益<-1%), 0.5=持有
            y_class = pd.cut(y, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])

            model = self.xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

            try:
                model.fit(X, y_class)
                self.models[symbol] = model
                self.feature_names = X.columns.tolist()
            except Exception as e:
                logger.warning(f"Failed to train model for {symbol}: {e}")

    def predict(self, prices_df: pd.DataFrame, current_idx: int) -> Dict[str, str]:
        """预测交易信号"""
        actions = {}

        # 获取历史数据到当前
        hist_prices = prices_df.iloc[:current_idx+1]

        for symbol in ALL_SYMBOLS:
            if symbol not in self.models:
                actions[symbol] = "HOLD"
                continue

            features = create_features(hist_prices, symbol)
            if features.empty:
                actions[symbol] = "HOLD"
                continue

            # 最后一行是当前特征
            X_current = features.iloc[[-1]]

            try:
                pred = self.models[symbol].predict(X_current)[0]
                proba = self.models[symbol].predict_proba(X_current)[0]

                # 根据概率和预测决定动作
                confidence = max(proba)

                if pred == 2 and confidence > 0.4:  # 买入
                    if confidence > 0.6:
                        actions[symbol] = "BUY_50%"
                    else:
                        actions[symbol] = "BUY_25%"
                elif pred == 0 and confidence > 0.4:  # 卖出
                    if confidence > 0.6:
                        actions[symbol] = "SELL_50%"
                    else:
                        actions[symbol] = "SELL_25%"
                else:
                    actions[symbol] = "HOLD"

            except Exception as e:
                actions[symbol] = "HOLD"

        return actions


# ============================================================
# Main Backtest
# ============================================================

def run_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 5,
    train_ratio: float = 0.5,  # 前50%数据用于训练
):
    """运行回测"""

    print("=" * 80)
    print(" XGBoost Baseline 回测")
    print("=" * 80)
    print(f" 时间范围: {start_date} ~ {end_date}")
    print(f" 初始资金: ${initial_capital:,.2f}")
    print(f" 再平衡频率: 每{rebalance_freq}天")
    print(f" 训练数据比例: {train_ratio*100:.0f}%")
    print("=" * 80)

    # 获取市场数据 (扩展日期以获取训练数据)
    print("\n获取市场数据...")

    # 向前扩展日期以获取足够的训练数据
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    extended_start = (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")  # 多取一年数据用于训练

    prices_df = fetch_market_data(extended_start, end_date)

    if prices_df.empty or len(prices_df) < 100:
        print("数据不足，无法回测")
        return

    # 找到实际回测开始的索引
    trading_days = prices_df.index.tolist()
    backtest_start_idx = None
    for i, d in enumerate(trading_days):
        if d.strftime("%Y-%m-%d") >= start_date:
            backtest_start_idx = i
            break

    if backtest_start_idx is None or backtest_start_idx < 50:
        print("数据不足，无法回测")
        return

    print(f"共 {len(trading_days)} 天数据，回测开始索引: {backtest_start_idx}")

    # 创建并训练模型
    print("\n训练 XGBoost 模型...")
    trader = XGBoostTrader()
    trader.train(prices_df, backtest_start_idx)
    print(f"已训练 {len(trader.models)} 个资产模型")

    # 创建组合
    portfolio = Portfolio(initial_capital)

    # 回测循环
    print("\n" + "=" * 80)
    print(" 开始回测")
    print("=" * 80)

    decision_log = []
    backtest_days = trading_days[backtest_start_idx:]

    for i, date in enumerate(backtest_days):
        date_str = date.strftime("%Y-%m-%d")
        global_idx = backtest_start_idx + i
        prices = prices_df.iloc[global_idx]

        # 记录组合价值
        price_dict = {s: prices[s] for s in ALL_SYMBOLS if s in prices.index}
        portfolio.record_value(price_dict)

        # 是否需要rebalance
        if i % rebalance_freq != 0 and i > 0:
            continue

        # 获取XGBoost预测
        actions = trader.predict(prices_df, global_idx)

        # 打印进度
        if i % 10 == 0:
            print(f"[{date_str}] 组合价值: ${portfolio.get_total_value(price_dict):,.2f}")

        # 执行交易
        for symbol, action in actions.items():
            if symbol in price_dict and action != "HOLD":
                result = portfolio.execute_trade(
                    symbol=symbol,
                    action=action,
                    price=price_dict[symbol],
                    total_value=portfolio.get_total_value(price_dict)
                )
                if result["executed"]:
                    pass  # 可以打印交易详情

        decision_log.append({
            "date": date_str,
            "portfolio_value": portfolio.get_total_value(price_dict),
            "decisions": actions
        })

    # 计算最终指标
    metrics = portfolio.get_metrics()

    # 输出结果
    print("\n" + "=" * 80)
    print(" XGBoost 回测结果")
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
        "strategy": "XGBoost",
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "metrics": metrics,
        "portfolio_values": portfolio.value_history,
    }

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"xgboost_backtest_{start_date}_{end_date}.json")
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
    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
    )


if __name__ == "__main__":
    main()
