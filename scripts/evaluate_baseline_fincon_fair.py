#!/usr/bin/env python3
"""
FinCon Baseline Fair Evaluation - Using Qwen2.5-14B-Instruct (Same as MARFT)

This script implements the original FinCon approach:
1. Single LLM for financial decision making
2. Prompt engineering based approach
3. NO reinforcement learning fine-tuning
4. Each decision is independent (no memory)

Key Difference from MARFT:
- MARFT: Uses 9 LoRA-finetuned expert adapters with RL
- FinCon: Uses the same base model but with prompting only

This ensures a fair comparison - same base model, different methodology.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import time

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

# Try to import transformers for local model
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, will use rule-based fallback")

# ============================================================
# Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fincon_fair_eval.log')
    ]
)
logger = logging.getLogger(__name__)

# Same assets as MARFT evaluation
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

# FinCon System Prompt (Following original paper style)
FINCON_SYSTEM_PROMPT = """You are FinCon, a financial consultant AI assistant specialized in portfolio management.

Your role is to analyze market data and provide investment recommendations for a multi-asset portfolio.

You will receive:
1. Current portfolio holdings and cash
2. Recent price data and technical indicators
3. Market conditions summary

Based on this analysis, provide recommendations in JSON format:
{
    "recommendations": {
        "SYMBOL": {"action": "BUY/SELL/HOLD", "confidence": 0.0-1.0, "size": "small/medium/large"},
        ...
    },
    "asset_allocation": {
        "stocks": 0.0-1.0,
        "bonds": 0.0-1.0,
        "commodities": 0.0-1.0,
        "reits": 0.0-1.0,
        "crypto": 0.0-1.0
    },
    "reasoning": "Brief explanation of the decision"
}

Guidelines:
- Be data-driven in your analysis
- Consider diversification and risk management
- Asset allocations should sum to 1.0
- Confidence reflects conviction level
"""


class QwenFinConModel:
    """FinCon using Qwen2.5-14B-Instruct (same as MARFT base model)"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load model with memory optimization"""
        logger.info(f"Loading {self.model_name}...")
        logger.info(f"Device: {self.device}")

        if self.device == "cuda":
            import torch
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {gpu_mem:.1f}GB")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load with 4-bit quantization for memory efficiency
        from transformers import BitsAndBytesConfig

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

        logger.info("Model loaded successfully!")

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from model"""
        if self.model is None:
            self.load_model()

        messages = [
            {"role": "system", "content": FINCON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
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

        # Size mapping
        size_pct = {"small": 0.03, "medium": 0.06, "large": 0.10}.get(size, 0.05)

        if action == "BUY":
            buy_value = min(total_value * size_pct, self.cash * 0.95)
            if buy_value > 100:
                shares = buy_value / price
                self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
                self.cash -= buy_value
                self.trade_history.append({
                    "date": date, "symbol": symbol, "action": "BUY",
                    "shares": shares, "price": price, "value": buy_value
                })
                logger.info(f"  [TRADE] BUY {symbol}: {shares:.2f} shares @ ${price:.2f}")

        elif action == "SELL":
            current_shares = self.holdings.get(symbol, 0)
            sell_shares = current_shares * 0.5  # Sell 50%
            if sell_shares > 0:
                self.holdings[symbol] -= sell_shares
                sell_value = sell_shares * price
                self.cash += sell_value
                self.trade_history.append({
                    "date": date, "symbol": symbol, "action": "SELL",
                    "shares": sell_shares, "price": price, "value": sell_value
                })
                logger.info(f"  [TRADE] SELL {symbol}: {sell_shares:.2f} shares @ ${price:.2f}")


def calculate_technical_indicators(history: pd.DataFrame, symbol: str) -> Dict:
    """Calculate technical indicators for a symbol"""
    if symbol not in history.columns:
        return {}

    recent = history[symbol].dropna().tail(30)
    if len(recent) < 10:
        return {}

    current = recent.iloc[-1]
    ma5 = recent.tail(5).mean()
    ma20 = recent.tail(20).mean() if len(recent) >= 20 else recent.mean()

    # RSI calculation
    delta = recent.diff()
    gain = delta.where(delta > 0, 0).mean()
    loss = (-delta.where(delta < 0, 0)).mean()
    rs = gain / loss if loss != 0 else 1
    rsi = 100 - (100 / (1 + rs))

    # Volatility
    volatility = recent.pct_change().std() * np.sqrt(252) * 100

    # 5-day return
    ret_5d = (recent.iloc[-1] / recent.iloc[-5] - 1) * 100 if len(recent) >= 5 else 0

    return {
        "current": current,
        "ma5": ma5,
        "ma20": ma20,
        "rsi": rsi,
        "volatility": volatility,
        "return_5d": ret_5d
    }


def build_market_prompt(prices: Dict, history: pd.DataFrame,
                        portfolio: SimplePortfolio, date: str) -> str:
    """Build prompt for FinCon model"""

    # Portfolio summary
    total_value = portfolio.get_value(prices)
    holdings_summary = []
    for symbol, shares in portfolio.holdings.items():
        if shares > 0 and symbol in prices:
            value = shares * prices[symbol]
            pct = value / total_value * 100
            holdings_summary.append(f"  {symbol}: {shares:.2f} shares (${value:,.0f}, {pct:.1f}%)")

    # Market data summary
    market_lines = []
    for asset_class, symbols in ASSETS.items():
        class_lines = [f"\n{asset_class.upper()}:"]
        for symbol in symbols:
            if symbol in prices:
                indicators = calculate_technical_indicators(history, symbol)
                if indicators:
                    trend = "UP" if indicators['current'] > indicators['ma20'] else "DOWN"
                    class_lines.append(
                        f"  {symbol}: ${indicators['current']:.2f} | "
                        f"RSI={indicators['rsi']:.1f} | "
                        f"5D={indicators['return_5d']:+.1f}% | "
                        f"Vol={indicators['volatility']:.1f}% | "
                        f"Trend={trend}"
                    )
        market_lines.extend(class_lines)

    prompt = f"""Date: {date}

PORTFOLIO STATUS:
Total Value: ${total_value:,.2f}
Cash: ${portfolio.cash:,.2f} ({portfolio.cash/total_value*100:.1f}%)
Holdings:
{chr(10).join(holdings_summary) if holdings_summary else "  (No holdings)"}

MARKET DATA:
{chr(10).join(market_lines)}

Please analyze the market conditions and provide your investment recommendations.
Return your response in the specified JSON format."""

    return prompt


def parse_llm_response(response: str) -> Dict:
    """Parse LLM response to extract recommendations"""
    import re

    # Try to extract JSON
    try:
        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: parse text-based recommendations
    recommendations = {}
    for symbol in ALL_SYMBOLS:
        if f"{symbol}" in response.upper():
            if "BUY" in response.upper():
                recommendations[symbol] = {"action": "BUY", "confidence": 0.6, "size": "medium"}
            elif "SELL" in response.upper():
                recommendations[symbol] = {"action": "SELL", "confidence": 0.6, "size": "medium"}

    return {
        "recommendations": recommendations,
        "asset_allocation": {
            "stocks": 0.5, "bonds": 0.2, "commodities": 0.15,
            "reits": 0.1, "crypto": 0.05
        }
    }


def run_fincon_fair_evaluation(
    start_date: str = "2024-06-03",
    end_date: str = "2024-11-29",
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 1,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
):
    """Run FinCon fair baseline evaluation using same model as MARFT"""

    logger.info("=" * 70)
    logger.info("FinCon FAIR Baseline Evaluation")
    logger.info("Using: Qwen2.5-14B-Instruct (Same as MARFT, NO LoRA)")
    logger.info("=" * 70)
    logger.info(f"Test Period: {start_date} ~ {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.0f}")
    logger.info(f"Decision Frequency: Every {rebalance_freq} day(s)")
    logger.info(f"Model: {model_name}")

    # Initialize model
    if HAS_TRANSFORMERS:
        model = QwenFinConModel(model_name)
        model.load_model()
    else:
        logger.error("Transformers not available!")
        return None

    # Load market data
    logger.info("Fetching market data...")
    price_data = pd.DataFrame()
    for symbol in ALL_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                price_data[symbol] = hist['Close']
                logger.info(f"  Loaded {symbol}: {len(hist)} days")
        except Exception as e:
            logger.warning(f"  Failed to load {symbol}: {e}")

    if price_data.empty:
        logger.error("No market data available!")
        return None

    # Initialize portfolio
    portfolio = SimplePortfolio(initial_capital)
    trading_days = price_data.index.tolist()
    logger.info(f"Trading days: {len(trading_days)}")

    # Track metrics
    decision_count = 0
    value_history = []
    max_value = initial_capital
    max_drawdown = 0

    # Main evaluation loop
    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

        # Get current prices
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

        # Track drawdown
        if current_value > max_value:
            max_value = current_value
        drawdown = (max_value - current_value) / max_value * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        value_history.append({
            "date": date_str,
            "value": current_value,
            "return": returns
        })

        # Decision day
        if i % rebalance_freq == 0:
            decision_count += 1

            logger.info(f"\n[{date_str}] Decision #{decision_count}")
            logger.info(f"  Portfolio: ${current_value:,.2f} (Return: {returns:+.2f}%)")

            # Build prompt and get LLM decision
            history = price_data.iloc[max(0, i-60):i+1]
            prompt = build_market_prompt(prices, history, portfolio, date_str)

            try:
                start_time = time.time()
                response = model.generate(prompt)
                elapsed = time.time() - start_time

                logger.info(f"  LLM Response Time: {elapsed:.1f}s")
                logger.debug(f"  Raw Response: {response[:500]}...")

                # Parse and execute
                decision = parse_llm_response(response)
                recommendations = decision.get("recommendations", {})

                logger.info(f"  Recommendations: {len(recommendations)} symbols")

                for symbol, rec in recommendations.items():
                    action = rec.get("action", "HOLD")
                    size = rec.get("size", "medium")
                    confidence = rec.get("confidence", 0.5)

                    if symbol in prices and action != "HOLD" and confidence > 0.4:
                        portfolio.execute_trade(
                            symbol, action, size,
                            prices[symbol], current_value, date_str
                        )

            except Exception as e:
                logger.error(f"  LLM Error: {e}")

        # Log progress every 5 days
        if i % 5 == 0:
            logger.info(f"[{date_str}] Value: ${current_value:,.2f} (Return: {returns:+.2f}%, DD: {drawdown:.2f}%)")

    # Final results
    final_prices = {s: price_data.iloc[-1][s] for s in ALL_SYMBOLS if s in price_data.columns}
    final_value = portfolio.get_value(final_prices)
    total_return = (final_value / initial_capital - 1) * 100

    days = (trading_days[-1] - trading_days[0]).days
    annualized_return = ((final_value / initial_capital) ** (365 / days) - 1) * 100

    # Calculate Sharpe (simplified)
    if len(value_history) > 1:
        returns_series = pd.Series([v['return'] for v in value_history])
        daily_returns = returns_series.diff().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    logger.info("\n" + "=" * 70)
    logger.info("FinCon FAIR Baseline Results")
    logger.info("=" * 70)
    logger.info(f"Final Value: ${final_value:,.2f}")
    logger.info(f"Total Return: {total_return:+.2f}%")
    logger.info(f"Annualized Return: {annualized_return:+.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Total Decisions: {decision_count}")
    logger.info(f"Total Trades: {len(portfolio.trade_history)}")
    logger.info("=" * 70)

    result = {
        "strategy": "FinCon_Fair",
        "model": model_name,
        "note": "Same base model as MARFT, NO LoRA fine-tuning",
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "decision_count": decision_count,
        "trade_count": len(portfolio.trade_history),
        "test_period": f"{start_date} ~ {end_date}"
    }

    # Save results
    output_file = f"results/fincon_fair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinCon Fair Baseline Evaluation")
    parser.add_argument("--start", default="2024-06-03", help="Start date")
    parser.add_argument("--end", default="2024-11-29", help="End date")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--freq", type=int, default=1, help="Decision frequency (days)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model name")
    args = parser.parse_args()

    result = run_fincon_fair_evaluation(
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
