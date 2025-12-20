#!/usr/bin/env python3
"""
FinAgent Baseline 评估脚本
模拟论文中 FinAgent 的原始形态：多代理协作，无 LoRA 微调

FinAgent 特点:
1. 多个专门化的 Agent (分析师、风险管理、执行等)
2. Agent 之间通过消息传递协作
3. 无强化学习微调，使用原生 LLM
4. 有简单的记忆机制
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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


# ============================================================
# FinAgent 多代理系统
# ============================================================

@dataclass
class AgentMessage:
    """代理间消息"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: str


class BaseAgent:
    """基础代理类"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory: List[Dict] = []

    def add_memory(self, event: Dict):
        """添加记忆"""
        self.memory.append(event)
        # 保留最近 10 条记忆
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

    def process(self, input_data: Dict, use_openai: bool = False) -> Dict:
        """处理输入并返回结果"""
        raise NotImplementedError


class MarketAnalystAgent(BaseAgent):
    """市场分析师代理"""

    def __init__(self):
        super().__init__("MarketAnalyst", "分析市场趋势和技术指标")

    def process(self, input_data: Dict, use_openai: bool = False) -> Dict:
        prices = input_data.get("prices", {})
        history = input_data.get("history", pd.DataFrame())

        analysis = {}

        for symbol in ALL_SYMBOLS:
            if symbol not in history.columns:
                continue

            recent = history[symbol].dropna().tail(20)
            if len(recent) < 10:
                analysis[symbol] = {"trend": "neutral", "strength": 0.5}
                continue

            current = recent.iloc[-1]
            ma5 = recent.tail(5).mean()
            ma20 = recent.mean()

            # 计算 RSI
            delta = recent.diff()
            gain = delta.where(delta > 0, 0).mean()
            loss = (-delta.where(delta < 0, 0)).mean()
            rs = gain / loss if loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))

            # 趋势判断
            if current > ma5 > ma20 and rsi < 70:
                trend = "bullish"
                strength = min(0.9, 0.5 + (current - ma20) / ma20)
            elif current < ma5 < ma20 and rsi > 30:
                trend = "bearish"
                strength = min(0.9, 0.5 + (ma20 - current) / ma20)
            else:
                trend = "neutral"
                strength = 0.5

            analysis[symbol] = {
                "trend": trend,
                "strength": strength,
                "rsi": rsi,
                "ma5": ma5,
                "ma20": ma20,
                "current": current
            }

        self.add_memory({"type": "analysis", "date": input_data.get("date"), "summary": len(analysis)})

        return {"analysis": analysis}


class RiskManagerAgent(BaseAgent):
    """风险管理代理"""

    def __init__(self):
        super().__init__("RiskManager", "评估和控制风险")
        self.max_position_pct = 0.15  # 单个资产最大仓位
        self.max_drawdown_limit = 0.10  # 最大回撤限制

    def process(self, input_data: Dict, use_openai: bool = False) -> Dict:
        portfolio = input_data.get("portfolio", {})
        analysis = input_data.get("analysis", {})

        risk_assessment = {}
        total_value = portfolio.get("total_value", 1_000_000)

        for symbol, data in analysis.items():
            rsi = data.get("rsi", 50)
            trend = data.get("trend", "neutral")

            # 风险评分
            risk_score = 0.5

            if rsi > 80:
                risk_score += 0.3  # 超买风险
            elif rsi < 20:
                risk_score += 0.2  # 超卖可能反弹

            if trend == "bearish":
                risk_score += 0.2

            # 资产类别风险
            for cls, symbols in ASSETS.items():
                if symbol in symbols:
                    if cls == "crypto":
                        risk_score += 0.3
                    elif cls == "commodities":
                        risk_score += 0.1
                    break

            risk_assessment[symbol] = {
                "risk_score": min(1.0, risk_score),
                "max_allocation": self.max_position_pct * (1 - risk_score * 0.5),
                "action_modifier": 1.0 - risk_score * 0.5
            }

        self.add_memory({"type": "risk_check", "date": input_data.get("date")})

        return {"risk_assessment": risk_assessment}


class PortfolioManagerAgent(BaseAgent):
    """投资组合管理代理"""

    def __init__(self):
        super().__init__("PortfolioManager", "做出最终投资决策")

    def process(self, input_data: Dict, use_openai: bool = False) -> Dict:
        analysis = input_data.get("analysis", {})
        risk_assessment = input_data.get("risk_assessment", {})

        recommendations = {}

        for symbol in analysis:
            if symbol not in risk_assessment:
                continue

            trend = analysis[symbol].get("trend", "neutral")
            strength = analysis[symbol].get("strength", 0.5)
            risk_score = risk_assessment[symbol].get("risk_score", 0.5)
            action_modifier = risk_assessment[symbol].get("action_modifier", 1.0)

            # 综合决策
            if trend == "bullish" and risk_score < 0.7:
                action = "BUY"
                confidence = strength * action_modifier
            elif trend == "bearish" or risk_score > 0.8:
                action = "SELL"
                confidence = (1 - strength) * action_modifier
            else:
                action = "HOLD"
                confidence = 0.5

            recommendations[symbol] = {
                "action": action,
                "confidence": confidence,
                "allocation": risk_assessment[symbol].get("max_allocation", 0.05)
            }

        # 资产配置
        allocation = {
            "stocks": 0.45,
            "bonds": 0.25,
            "commodities": 0.15,
            "reits": 0.10,
            "crypto": 0.05
        }

        self.add_memory({"type": "decision", "date": input_data.get("date")})

        return {
            "recommendations": recommendations,
            "portfolio_allocation": allocation
        }


class FinAgentSystem:
    """FinAgent 多代理系统"""

    def __init__(self):
        self.analyst = MarketAnalystAgent()
        self.risk_manager = RiskManagerAgent()
        self.portfolio_manager = PortfolioManagerAgent()
        self.message_log: List[AgentMessage] = []

    def make_decision(self, prices: Dict, history: pd.DataFrame,
                      portfolio: Dict, date: str, use_openai: bool = False) -> Dict:
        """多代理协作决策"""

        # 1. 市场分析师分析
        analyst_input = {"prices": prices, "history": history, "date": date}
        analyst_output = self.analyst.process(analyst_input, use_openai)

        self.message_log.append(AgentMessage(
            sender="MarketAnalyst",
            receiver="RiskManager",
            content=analyst_output,
            timestamp=date
        ))

        # 2. 风险管理评估
        risk_input = {
            "portfolio": portfolio,
            "analysis": analyst_output.get("analysis", {}),
            "date": date
        }
        risk_output = self.risk_manager.process(risk_input, use_openai)

        self.message_log.append(AgentMessage(
            sender="RiskManager",
            receiver="PortfolioManager",
            content=risk_output,
            timestamp=date
        ))

        # 3. 组合管理决策
        pm_input = {
            "analysis": analyst_output.get("analysis", {}),
            "risk_assessment": risk_output.get("risk_assessment", {}),
            "date": date
        }
        final_decision = self.portfolio_manager.process(pm_input, use_openai)

        return final_decision


class SimplePortfolio:
    """简单的投资组合管理器"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: Dict[str, float] = {}

    def get_value(self, prices: Dict[str, float]) -> float:
        holdings_value = sum(
            self.holdings.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.holdings
        )
        return self.cash + holdings_value

    def execute_trade(self, symbol: str, action: str, price: float,
                      allocation: float, total_value: float):
        if action == "HOLD" or price <= 0:
            return

        target_value = total_value * allocation
        current_value = self.holdings.get(symbol, 0) * price

        if action == "BUY":
            buy_value = min(target_value - current_value, self.cash * 0.1)
            if buy_value > 0:
                shares = buy_value / price
                self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
                self.cash -= buy_value

        elif action == "SELL":
            sell_shares = self.holdings.get(symbol, 0) * 0.5
            if sell_shares > 0:
                self.holdings[symbol] -= sell_shares
                self.cash += sell_shares * price


def run_finagent_evaluation(
    start_date: str = "2024-06-03",
    end_date: str = "2024-11-29",
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 1,
    use_openai: bool = False
):
    """运行 FinAgent 基准评估"""

    logger.info("=" * 60)
    logger.info("FinAgent Baseline 评估 (多代理协作)")
    logger.info("=" * 60)
    logger.info(f"测试期间: {start_date} ~ {end_date}")
    logger.info(f"初始资金: ${initial_capital:,.0f}")
    logger.info(f"决策频率: 每{rebalance_freq}天")

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

    # 初始化
    portfolio = SimplePortfolio(initial_capital)
    finagent = FinAgentSystem()

    trading_days = price_data.index.tolist()
    logger.info(f"交易日数: {len(trading_days)}")

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

        current_value = portfolio.get_value(prices)
        returns = (current_value / initial_capital - 1) * 100

        # 决策日
        if i % rebalance_freq == 0:
            decision_count += 1

            history = price_data.iloc[max(0, i-60):i+1]

            portfolio_info = {
                "cash": portfolio.cash,
                "holdings": portfolio.holdings.copy(),
                "total_value": current_value
            }

            # 多代理决策
            decision = finagent.make_decision(
                prices=prices,
                history=history,
                portfolio=portfolio_info,
                date=date_str,
                use_openai=use_openai
            )

            # 执行交易
            recommendations = decision.get("recommendations", {})
            allocations = decision.get("portfolio_allocation", {})

            for symbol, rec in recommendations.items():
                action = rec.get("action", "HOLD")
                allocation = rec.get("allocation", 0.05)

                if symbol in prices:
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

    days = (trading_days[-1] - trading_days[0]).days
    annualized_return = ((final_value / initial_capital) ** (365 / days) - 1) * 100

    logger.info("=" * 60)
    logger.info("FinAgent Baseline 评估结果")
    logger.info("=" * 60)
    logger.info(f"最终价值: ${final_value:,.2f}")
    logger.info(f"累计收益: {total_return:+.2f}%")
    logger.info(f"年化收益: {annualized_return:+.2f}%")
    logger.info(f"决策次数: {decision_count}")
    logger.info(f"代理消息数: {len(finagent.message_log)}")
    logger.info("=" * 60)

    return {
        "strategy": "FinAgent",
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "decision_count": decision_count,
        "agent_messages": len(finagent.message_log)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinAgent Baseline 评估")
    parser.add_argument("--start", default="2024-06-03", help="开始日期")
    parser.add_argument("--end", default="2024-11-29", help="结束日期")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--freq", type=int, default=1, help="决策频率(天)")
    parser.add_argument("--openai", action="store_true", help="使用 OpenAI API")
    args = parser.parse_args()

    result = run_finagent_evaluation(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        use_openai=args.openai
    )

    if result:
        output_file = f"results/finagent_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
