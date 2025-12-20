#!/usr/bin/env python3
"""
FinCon Baseline 评估脚本
模拟论文中 FinCon 的原始形态：单 LLM 直接决策，无 LoRA 微调

FinCon 特点:
1. 使用单个 LLM 进行金融决策
2. 基于 Prompt Engineering 的方法
3. 无强化学习微调
4. 每次决策独立，无记忆
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

# ============================================================
# 配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 资产配置 (与 MARFT 评估相同)
ASSETS = {
    "stocks": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "bonds": ["TLT", "LQD", "HYG", "AGG"],
    "commodities": ["GLD", "SLV", "USO", "UNG"],
    "reits": ["VNQ", "IYR", "XLRE"],
    "crypto": ["BTC-USD", "ETH-USD"],
}

ALL_SYMBOLS = []
for symbols in ASSETS.values():
    ALL_SYMBOLS.extend(symbols)

# FinCon 系统提示词 (模拟原论文风格)
FINCON_SYSTEM_PROMPT = """You are FinCon, a financial consultant AI assistant.
Your task is to analyze market data and provide investment recommendations.

You will receive:
1. Current portfolio holdings
2. Recent price data for various assets
3. Market indicators (moving averages, RSI, etc.)

Based on this information, provide recommendations in the following JSON format:
{
    "recommendations": {
        "SYMBOL": {"action": "BUY/SELL/HOLD", "confidence": 0.0-1.0, "reasoning": "..."},
        ...
    },
    "portfolio_allocation": {
        "stocks": 0.0-1.0,
        "bonds": 0.0-1.0,
        "commodities": 0.0-1.0,
        "reits": 0.0-1.0,
        "crypto": 0.0-1.0
    },
    "overall_reasoning": "..."
}

Important:
- Be conservative with recommendations
- Consider market conditions and diversification
- Allocations should sum to 1.0
"""


class SimplePortfolio:
    """简单的投资组合管理器"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: Dict[str, float] = {}  # symbol -> shares
        self.history: List[Dict] = []

    def get_value(self, prices: Dict[str, float]) -> float:
        """计算总价值"""
        holdings_value = sum(
            self.holdings.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.holdings
        )
        return self.cash + holdings_value

    def execute_trade(self, symbol: str, action: str, price: float,
                      allocation: float, total_value: float):
        """执行交易"""
        if action == "HOLD" or price <= 0:
            return

        target_value = total_value * allocation
        current_value = self.holdings.get(symbol, 0) * price

        if action == "BUY":
            buy_value = min(target_value - current_value, self.cash * 0.1)  # 最多用 10% 现金
            if buy_value > 0:
                shares = buy_value / price
                self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
                self.cash -= buy_value

        elif action == "SELL":
            sell_shares = self.holdings.get(symbol, 0) * 0.5  # 卖出 50%
            if sell_shares > 0:
                self.holdings[symbol] -= sell_shares
                self.cash += sell_shares * price


def get_fincon_decision(
    prices: Dict[str, float],
    history_data: pd.DataFrame,
    portfolio: SimplePortfolio,
    use_openai: bool = True
) -> Dict[str, Any]:
    """
    获取 FinCon 风格的决策

    如果无法使用 OpenAI API，则使用基于规则的回退策略
    """

    if use_openai:
        try:
            import openai
            client = openai.OpenAI()

            # 准备市场数据摘要
            market_summary = []
            for symbol in ALL_SYMBOLS:
                if symbol in history_data.columns:
                    recent = history_data[symbol].dropna().tail(20)
                    if len(recent) >= 5:
                        current = recent.iloc[-1]
                        ma5 = recent.tail(5).mean()
                        ma20 = recent.mean()
                        change_5d = (recent.iloc[-1] / recent.iloc[-5] - 1) * 100 if len(recent) >= 5 else 0

                        market_summary.append(
                            f"{symbol}: ${current:.2f}, MA5={ma5:.2f}, MA20={ma20:.2f}, 5D Change={change_5d:.1f}%"
                        )

            user_prompt = f"""Current Portfolio:
Cash: ${portfolio.cash:,.2f}
Holdings: {json.dumps({k: f"{v:.2f} shares" for k, v in portfolio.holdings.items() if v > 0})}

Market Data:
{chr(10).join(market_summary)}

Please analyze and provide your investment recommendations."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": FINCON_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # 尝试解析 JSON
            try:
                # 提取 JSON 部分
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

        except Exception as e:
            logger.warning(f"OpenAI API 调用失败: {e}, 使用规则回退")

    # 回退: 基于简单规则的策略
    return get_rule_based_decision(prices, history_data)


def get_rule_based_decision(prices: Dict[str, float], history_data: pd.DataFrame) -> Dict:
    """基于规则的回退策略 (模拟简单的 FinCon)"""

    recommendations = {}

    for symbol in ALL_SYMBOLS:
        if symbol not in history_data.columns:
            continue

        recent = history_data[symbol].dropna().tail(20)
        if len(recent) < 10:
            recommendations[symbol] = {"action": "HOLD", "confidence": 0.5}
            continue

        current = recent.iloc[-1]
        ma5 = recent.tail(5).mean()
        ma20 = recent.mean()

        # 简单的均线策略
        if current > ma5 > ma20:
            action = "BUY"
            confidence = 0.7
        elif current < ma5 < ma20:
            action = "SELL"
            confidence = 0.7
        else:
            action = "HOLD"
            confidence = 0.5

        recommendations[symbol] = {"action": action, "confidence": confidence}

    return {
        "recommendations": recommendations,
        "portfolio_allocation": {
            "stocks": 0.5,
            "bonds": 0.2,
            "commodities": 0.15,
            "reits": 0.1,
            "crypto": 0.05
        }
    }


def run_fincon_evaluation(
    start_date: str = "2024-06-03",
    end_date: str = "2024-11-29",
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 1,
    use_openai: bool = True
):
    """运行 FinCon 基准评估"""

    logger.info("=" * 60)
    logger.info("FinCon Baseline 评估")
    logger.info("=" * 60)
    logger.info(f"测试期间: {start_date} ~ {end_date}")
    logger.info(f"初始资金: ${initial_capital:,.0f}")
    logger.info(f"决策频率: 每{rebalance_freq}天")
    logger.info(f"使用 OpenAI: {use_openai}")

    # 加载数据 (使用 yfinance)
    logger.info("获取市场数据...")
    price_data = pd.DataFrame()
    for symbol in ALL_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                price_data[symbol] = hist['Close']
        except Exception as e:
            logger.warning(f"无法获取 {symbol}: {e}")

    if price_data.empty:
        logger.error("无法获取市场数据")
        return

    # 初始化组合
    portfolio = SimplePortfolio(initial_capital)

    # 获取交易日
    trading_days = price_data.index.tolist()

    logger.info(f"交易日数: {len(trading_days)}")

    # 开始回测
    decision_count = 0

    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

        # 获取当天价格
        prices = {}
        for symbol in ALL_SYMBOLS:
            if symbol in price_data.columns:
                price = price_data.loc[date, symbol]
                if pd.notna(price) and price > 0:
                    prices[symbol] = price

        if not prices:
            continue

        # 计算当前价值
        current_value = portfolio.get_value(prices)
        returns = (current_value / initial_capital - 1) * 100

        # 决策日
        if i % rebalance_freq == 0:
            decision_count += 1

            # 获取历史数据
            history = price_data.iloc[max(0, i-60):i+1]

            # 获取 FinCon 决策
            decision = get_fincon_decision(prices, history, portfolio, use_openai)

            # 执行交易
            recommendations = decision.get("recommendations", {})
            allocations = decision.get("portfolio_allocation", {})

            for symbol, rec in recommendations.items():
                action = rec.get("action", "HOLD")
                if symbol in prices:
                    # 确定资产类别
                    asset_class = None
                    for cls, symbols in ASSETS.items():
                        if symbol in symbols:
                            asset_class = cls
                            break

                    allocation = allocations.get(asset_class, 0.1) / len(ASSETS.get(asset_class, [symbol]))
                    portfolio.execute_trade(symbol, action, prices[symbol], allocation, current_value)

        # 记录
        if i % 5 == 0 or i == len(trading_days) - 1:
            logger.info(f"[{date_str}] 组合价值: ${current_value:,.2f} (收益: {returns:+.2f}%)")

    # 最终结果
    final_prices = {}
    for symbol in ALL_SYMBOLS:
        if symbol in price_data.columns:
            final_prices[symbol] = price_data.iloc[-1][symbol]

    final_value = portfolio.get_value(final_prices)
    total_return = (final_value / initial_capital - 1) * 100

    # 计算年化收益
    days = (trading_days[-1] - trading_days[0]).days
    annualized_return = ((final_value / initial_capital) ** (365 / days) - 1) * 100

    logger.info("=" * 60)
    logger.info("FinCon Baseline 评估结果")
    logger.info("=" * 60)
    logger.info(f"最终价值: ${final_value:,.2f}")
    logger.info(f"累计收益: {total_return:+.2f}%")
    logger.info(f"年化收益: {annualized_return:+.2f}%")
    logger.info(f"决策次数: {decision_count}")
    logger.info("=" * 60)

    return {
        "strategy": "FinCon",
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "decision_count": decision_count
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinCon Baseline 评估")
    parser.add_argument("--start", default="2024-06-03", help="开始日期")
    parser.add_argument("--end", default="2024-11-29", help="结束日期")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--freq", type=int, default=1, help="决策频率(天)")
    parser.add_argument("--no-openai", action="store_true", help="不使用 OpenAI API")
    args = parser.parse_args()

    result = run_fincon_evaluation(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        use_openai=not args.no_openai
    )

    # 保存结果
    if result:
        output_file = f"results/fincon_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
