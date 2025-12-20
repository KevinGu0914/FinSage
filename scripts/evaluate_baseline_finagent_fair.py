#!/usr/bin/env python3
"""
FinAgent Baseline Fair Evaluation - Using Qwen2.5-14B-Instruct (Same as MARFT)

This script implements the original FinAgent multi-agent approach:
1. Multiple specialized agents (Analyst, Risk Manager, Portfolio Manager)
2. Agents communicate via message passing
3. NO reinforcement learning fine-tuning
4. Simple memory mechanism for each agent

Key Difference from MARFT:
- MARFT: Uses 9 LoRA-finetuned expert adapters with RL training
- FinAgent: Uses same base model with multi-agent prompting only

This ensures a fair comparison - same base model, different methodology.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

# Try to import transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")

# ============================================================
# Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finagent_fair_eval.log')
    ]
)
logger = logging.getLogger(__name__)

# Same assets as MARFT
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
# Agent Prompts (Following FinAgent paper style)
# ============================================================

ANALYST_PROMPT = """You are MarketAnalyst, a specialist in analyzing financial markets.

Your role is to analyze market data and identify trends, opportunities, and risks.

Given the market data, provide your analysis in JSON format:
{
    "market_overview": "Brief market summary",
    "asset_analysis": {
        "SYMBOL": {
            "trend": "bullish/bearish/neutral",
            "strength": 0.0-1.0,
            "signals": ["signal1", "signal2"],
            "recommendation": "Brief analysis"
        }
    },
    "sector_rotation": "Which sectors are performing well/poorly"
}

Focus on technical indicators, momentum, and relative strength."""


RISK_MANAGER_PROMPT = """You are RiskManager, responsible for portfolio risk control.

Your role is to assess risks and set position limits based on market analysis.

Given the analyst's assessment, provide risk controls in JSON format:
{
    "overall_risk_level": "low/medium/high",
    "risk_factors": ["factor1", "factor2"],
    "position_limits": {
        "SYMBOL": {
            "max_allocation": 0.0-0.15,
            "risk_score": 0.0-1.0,
            "action_modifier": 0.0-1.0
        }
    },
    "hedge_recommendations": ["hedge1", "hedge2"]
}

Consider volatility, correlation, and tail risks."""


PORTFOLIO_MANAGER_PROMPT = """You are PortfolioManager, making final investment decisions.

Your role is to synthesize analyst insights and risk controls into actionable trades.

Given the analysis and risk assessment, provide final decisions in JSON format:
{
    "decisions": {
        "SYMBOL": {
            "action": "BUY/SELL/HOLD",
            "size": "small/medium/large",
            "confidence": 0.0-1.0,
            "rationale": "Brief explanation"
        }
    },
    "target_allocation": {
        "stocks": 0.0-1.0,
        "bonds": 0.0-1.0,
        "commodities": 0.0-1.0,
        "reits": 0.0-1.0,
        "crypto": 0.0-1.0
    },
    "execution_notes": "Any special considerations"
}

Balance returns with risk management."""


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: str


class QwenModel:
    """Shared Qwen model for all agents"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        logger.info(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("Model loaded!")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        if self.model is None:
            self.load_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response


class BaseAgent:
    """Base agent class"""

    def __init__(self, name: str, role: str, system_prompt: str, llm: QwenModel):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.memory: List[Dict] = []

    def add_memory(self, event: Dict):
        self.memory.append(event)
        if len(self.memory) > 5:  # Keep last 5 memories
            self.memory = self.memory[-5:]

    def process(self, user_prompt: str) -> Dict:
        """Process input and return structured response"""
        try:
            response = self.llm.generate(self.system_prompt, user_prompt)
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"{self.name} error: {e}")

        return {}


class MarketAnalystAgent(BaseAgent):
    """Market analysis specialist"""

    def __init__(self, llm: QwenModel):
        super().__init__(
            "MarketAnalyst",
            "Analyze market trends and opportunities",
            ANALYST_PROMPT,
            llm
        )

    def analyze(self, prices: Dict, history: pd.DataFrame, date: str) -> Dict:
        """Analyze market data"""
        # Build market summary
        lines = []
        for asset_class, symbols in ASSETS.items():
            lines.append(f"\n{asset_class.upper()}:")
            for symbol in symbols:
                if symbol in history.columns:
                    recent = history[symbol].dropna().tail(20)
                    if len(recent) >= 5:
                        current = recent.iloc[-1]
                        ma5 = recent.tail(5).mean()
                        ma20 = recent.mean()
                        ret_5d = (recent.iloc[-1] / recent.iloc[-5] - 1) * 100

                        # RSI
                        delta = recent.diff()
                        gain = delta.where(delta > 0, 0).mean()
                        loss = (-delta.where(delta < 0, 0)).mean()
                        rsi = 100 - (100 / (1 + gain / loss)) if loss > 0 else 50

                        trend = "UP" if current > ma20 else "DOWN"
                        lines.append(
                            f"  {symbol}: ${current:.2f} | MA5=${ma5:.2f} | "
                            f"MA20=${ma20:.2f} | 5D={ret_5d:+.1f}% | RSI={rsi:.0f} | {trend}"
                        )

        prompt = f"""Date: {date}

MARKET DATA:
{chr(10).join(lines)}

Recent memory: {json.dumps(self.memory[-2:]) if self.memory else 'None'}

Please provide your market analysis."""

        result = self.process(prompt)
        self.add_memory({"date": date, "type": "analysis", "summary": len(result.get("asset_analysis", {}))})

        logger.info(f"  [{self.name}] Analyzed {len(result.get('asset_analysis', {}))} assets")
        return result


class RiskManagerAgent(BaseAgent):
    """Risk management specialist"""

    def __init__(self, llm: QwenModel):
        super().__init__(
            "RiskManager",
            "Assess and control portfolio risk",
            RISK_MANAGER_PROMPT,
            llm
        )

    def assess(self, analysis: Dict, portfolio_value: float, date: str) -> Dict:
        """Assess risks based on analyst report"""
        prompt = f"""Date: {date}
Portfolio Value: ${portfolio_value:,.0f}

ANALYST REPORT:
{json.dumps(analysis, indent=2)}

Recent memory: {json.dumps(self.memory[-2:]) if self.memory else 'None'}

Please provide your risk assessment."""

        result = self.process(prompt)
        self.add_memory({"date": date, "type": "risk_assessment", "level": result.get("overall_risk_level", "unknown")})

        logger.info(f"  [{self.name}] Risk Level: {result.get('overall_risk_level', 'unknown')}")
        return result


class PortfolioManagerAgent(BaseAgent):
    """Portfolio management and execution"""

    def __init__(self, llm: QwenModel):
        super().__init__(
            "PortfolioManager",
            "Make final investment decisions",
            PORTFOLIO_MANAGER_PROMPT,
            llm
        )

    def decide(self, analysis: Dict, risk_assessment: Dict,
               portfolio_info: Dict, date: str) -> Dict:
        """Make final trading decisions"""
        prompt = f"""Date: {date}

PORTFOLIO STATUS:
Cash: ${portfolio_info.get('cash', 0):,.0f}
Total Value: ${portfolio_info.get('total_value', 0):,.0f}
Holdings: {json.dumps({k: f"{v:.2f}" for k, v in portfolio_info.get('holdings', {}).items() if v > 0})}

ANALYST ASSESSMENT:
{json.dumps(analysis.get('asset_analysis', {}), indent=2)[:1000]}

RISK ASSESSMENT:
Overall Risk: {risk_assessment.get('overall_risk_level', 'unknown')}
Position Limits: {json.dumps(risk_assessment.get('position_limits', {}), indent=2)[:500]}

Recent memory: {json.dumps(self.memory[-2:]) if self.memory else 'None'}

Please provide your final trading decisions."""

        result = self.process(prompt)
        decisions = result.get("decisions", {})
        self.add_memory({"date": date, "type": "decisions", "count": len(decisions)})

        logger.info(f"  [{self.name}] Made {len(decisions)} decisions")
        return result


class FinAgentSystem:
    """Multi-agent collaboration system"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.llm = QwenModel(model_name)
        self.analyst = MarketAnalystAgent(self.llm)
        self.risk_manager = RiskManagerAgent(self.llm)
        self.portfolio_manager = PortfolioManagerAgent(self.llm)
        self.message_log: List[AgentMessage] = []

    def make_decision(self, prices: Dict, history: pd.DataFrame,
                      portfolio_info: Dict, date: str) -> Dict:
        """Multi-agent collaborative decision making"""
        logger.info(f"  [FinAgent] Starting multi-agent collaboration...")

        # Step 1: Market Analysis
        analysis = self.analyst.analyze(prices, history, date)
        self.message_log.append(AgentMessage(
            sender="MarketAnalyst",
            receiver="RiskManager",
            content=analysis,
            timestamp=date
        ))

        # Step 2: Risk Assessment
        risk_assessment = self.risk_manager.assess(
            analysis,
            portfolio_info.get("total_value", 1_000_000),
            date
        )
        self.message_log.append(AgentMessage(
            sender="RiskManager",
            receiver="PortfolioManager",
            content=risk_assessment,
            timestamp=date
        ))

        # Step 3: Final Decision
        decision = self.portfolio_manager.decide(
            analysis,
            risk_assessment,
            portfolio_info,
            date
        )
        self.message_log.append(AgentMessage(
            sender="PortfolioManager",
            receiver="Execution",
            content=decision,
            timestamp=date
        ))

        return decision


class SimplePortfolio:
    """Simple portfolio manager"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: Dict[str, float] = {}
        self.trade_history: List[Dict] = []

    def get_value(self, prices: Dict[str, float]) -> float:
        holdings_value = sum(
            self.holdings.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.holdings
        )
        return self.cash + holdings_value

    def execute_trade(self, symbol: str, action: str, size: str,
                      price: float, total_value: float, date: str):
        if action == "HOLD" or price <= 0:
            return

        size_pct = {"small": 0.03, "medium": 0.06, "large": 0.10}.get(size, 0.05)

        if action == "BUY":
            buy_value = min(total_value * size_pct, self.cash * 0.95)
            if buy_value > 100:
                shares = buy_value / price
                self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
                self.cash -= buy_value
                self.trade_history.append({
                    "date": date, "symbol": symbol, "action": "BUY",
                    "shares": shares, "price": price
                })
                logger.info(f"    [TRADE] BUY {symbol}: {shares:.2f} @ ${price:.2f}")

        elif action == "SELL":
            current_shares = self.holdings.get(symbol, 0)
            sell_shares = current_shares * 0.5
            if sell_shares > 0:
                self.holdings[symbol] -= sell_shares
                self.cash += sell_shares * price
                self.trade_history.append({
                    "date": date, "symbol": symbol, "action": "SELL",
                    "shares": sell_shares, "price": price
                })
                logger.info(f"    [TRADE] SELL {symbol}: {sell_shares:.2f} @ ${price:.2f}")


def run_finagent_fair_evaluation(
    start_date: str = "2024-06-03",
    end_date: str = "2024-11-29",
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 1,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
):
    """Run FinAgent fair baseline evaluation"""

    logger.info("=" * 70)
    logger.info("FinAgent FAIR Baseline Evaluation")
    logger.info("Multi-Agent Collaboration using Qwen2.5-14B-Instruct")
    logger.info("=" * 70)
    logger.info(f"Test Period: {start_date} ~ {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.0f}")
    logger.info(f"Decision Frequency: Every {rebalance_freq} day(s)")
    logger.info(f"Model: {model_name}")

    if not HAS_TRANSFORMERS:
        logger.error("Transformers not available!")
        return None

    # Initialize multi-agent system
    finagent = FinAgentSystem(model_name)

    # Load market data
    logger.info("Fetching market data...")
    price_data = pd.DataFrame()
    for symbol in ALL_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                price_data[symbol] = hist['Close']
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    if price_data.empty:
        logger.error("No market data!")
        return None

    # Initialize
    portfolio = SimplePortfolio(initial_capital)
    trading_days = price_data.index.tolist()
    logger.info(f"Trading days: {len(trading_days)}")

    # Metrics
    decision_count = 0
    value_history = []
    max_value = initial_capital
    max_drawdown = 0

    # Main loop
    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

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

        if current_value > max_value:
            max_value = current_value
        drawdown = (max_value - current_value) / max_value * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        value_history.append({"date": date_str, "value": current_value, "return": returns})

        # Decision day
        if i % rebalance_freq == 0:
            decision_count += 1
            logger.info(f"\n[{date_str}] Decision #{decision_count}")
            logger.info(f"  Portfolio: ${current_value:,.2f} (Return: {returns:+.2f}%)")

            history = price_data.iloc[max(0, i-60):i+1]

            portfolio_info = {
                "cash": portfolio.cash,
                "holdings": portfolio.holdings.copy(),
                "total_value": current_value
            }

            try:
                start_time = time.time()
                decision = finagent.make_decision(prices, history, portfolio_info, date_str)
                elapsed = time.time() - start_time
                logger.info(f"  Total Decision Time: {elapsed:.1f}s")

                # Execute trades
                decisions = decision.get("decisions", {})
                for symbol, rec in decisions.items():
                    action = rec.get("action", "HOLD")
                    size = rec.get("size", "medium")
                    confidence = rec.get("confidence", 0.5)

                    if symbol in prices and action != "HOLD" and confidence > 0.4:
                        portfolio.execute_trade(
                            symbol, action, size,
                            prices[symbol], current_value, date_str
                        )

            except Exception as e:
                logger.error(f"  Decision Error: {e}")

        # Log progress
        if i % 5 == 0:
            logger.info(f"[{date_str}] Value: ${current_value:,.2f} (Return: {returns:+.2f}%)")

    # Final results
    final_prices = {s: price_data.iloc[-1][s] for s in ALL_SYMBOLS if s in price_data.columns}
    final_value = portfolio.get_value(final_prices)
    total_return = (final_value / initial_capital - 1) * 100

    days = (trading_days[-1] - trading_days[0]).days
    annualized_return = ((final_value / initial_capital) ** (365 / days) - 1) * 100

    # Sharpe
    if len(value_history) > 1:
        returns_series = pd.Series([v['return'] for v in value_history])
        daily_returns = returns_series.diff().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    logger.info("\n" + "=" * 70)
    logger.info("FinAgent FAIR Baseline Results")
    logger.info("=" * 70)
    logger.info(f"Final Value: ${final_value:,.2f}")
    logger.info(f"Total Return: {total_return:+.2f}%")
    logger.info(f"Annualized Return: {annualized_return:+.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Total Decisions: {decision_count}")
    logger.info(f"Agent Messages: {len(finagent.message_log)}")
    logger.info(f"Total Trades: {len(portfolio.trade_history)}")
    logger.info("=" * 70)

    result = {
        "strategy": "FinAgent_Fair",
        "model": model_name,
        "note": "Multi-agent with same base model as MARFT, NO LoRA",
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "decision_count": decision_count,
        "agent_messages": len(finagent.message_log),
        "trade_count": len(portfolio.trade_history),
        "test_period": f"{start_date} ~ {end_date}"
    }

    output_file = f"results/finagent_fair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinAgent Fair Baseline Evaluation")
    parser.add_argument("--start", default="2024-06-03", help="Start date")
    parser.add_argument("--end", default="2024-11-29", help="End date")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--freq", type=int, default=1, help="Decision frequency (days)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model name")
    args = parser.parse_args()

    result = run_finagent_fair_evaluation(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        model_name=args.model
    )

    if result:
        print("\n=== FINAL RESULTS ===")
        print(f"Annualized Return: {result['annualized_return']:+.2f}%")
        print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
