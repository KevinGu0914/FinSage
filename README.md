# FinSage

AI-Powered Multi-Asset Portfolio Management System

FinSage is a sophisticated multi-agent financial portfolio management system that leverages Large Language Models (LLMs) for intelligent investment decision-making across multiple asset classes.

## Features

- **Multi-Agent Architecture**: Coordinated agents for market analysis, portfolio management, position sizing, and hedging
- **Multi-Asset Support**: Stocks, Bonds, Commodities, REITs, and Cryptocurrencies
- **Advanced Risk Management**: Dynamic risk control, drawdown protection, and tail risk hedging
- **11 Hedging Strategies**: Including Risk Parity, Black-Litterman, CVaR Optimization, and more
- **Factor-Based Screening**: Dynamic asset universe with momentum, quality, and value factors
- **Checkpoint/Resume**: Interrupt and resume long-running backtests

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finsage.git
cd finsage

# Install dependencies
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
FMP_API_KEY=your_fmp_api_key  # Optional: for fundamental data
```

## Usage

### Command Line

```bash
# Basic backtest
finsage --start 2024-01-01 --end 2024-12-31 --frequency daily

# With custom configuration
finsage -s 2024-01-01 -e 2024-06-30 -f weekly --config conservative --capital 500000

# Resume from checkpoint
finsage -s 2024-01-01 -e 2024-12-31 -f daily --resume
```

### Python API

```python
from finsage.config import FinSageConfig
from finsage.core.orchestrator import FinSageOrchestrator

config = FinSageConfig()
config.trading.initial_capital = 1_000_000
config.llm.model = "gpt-4o-mini"

orchestrator = FinSageOrchestrator(config=config)
results = orchestrator.run(
    start_date="2024-01-01",
    end_date="2024-12-31",
    rebalance_frequency="daily"
)

print(f"Cumulative Return: {results['final_metrics']['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {results['final_metrics']['sharpe_ratio']:.2f}")
```

## Architecture

```
FinSage
├── Expert Layer (5 Experts)
│   ├── Macro Expert - Global economic analysis
│   ├── Technical Expert - Price and trend analysis
│   ├── Fundamental Expert - Company/asset fundamentals
│   ├── Sentiment Expert - News and market sentiment
│   └── Commodity Expert - Commodity-specific analysis
│
├── Manager Layer (3 Managers)
│   ├── Portfolio Manager - Asset allocation decisions
│   ├── Position Sizing Agent - Risk-based position sizing
│   └── Hedging Agent - Tail risk management
│
├── Risk Control Layer
│   └── Risk Controller - Drawdown limits, position limits
│
└── Execution Layer
    └── Portfolio Executor - Trade execution simulation
```

## Project Structure

```
finsage/
├── agents/           # AI agents (experts and managers)
├── core/             # Orchestrator and execution engine
├── data/             # Data fetching and processing
├── hedging/          # Hedging strategy tools
├── llm/              # LLM provider abstraction
├── utils/            # Utilities and helpers
├── cli.py            # CLI entry point
└── config.py         # Configuration management
```

## Configuration Templates

| Template | Risk Level | Target Volatility | Max Drawdown |
|----------|-----------|-------------------|--------------|
| default | Moderate | 15% | 15% |
| conservative | Low | 10% | 10% |
| aggressive | High | 25% | 25% |

## Hedging Strategies

1. **Minimum Variance** - Minimize portfolio variance
2. **Risk Parity** - Equal risk contribution
3. **Black-Litterman** - Views-based allocation
4. **Mean-Variance** - Classic Markowitz optimization
5. **DCC-GARCH** - Dynamic correlation modeling
6. **Hierarchical Risk Parity** - Clustering-based allocation
7. **CVaR Optimization** - Conditional Value-at-Risk
8. **Robust Optimization** - Uncertainty-aware allocation
9. **Copula Hedging** - Tail dependency modeling
10. **Factor Hedging** - Factor exposure management
11. **Regime Switching** - Market regime detection

## Output Metrics

- Cumulative Return
- Annualized Return
- Volatility
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Total Trades

## Requirements

- Python >= 3.9
- OpenAI API key (or compatible LLM provider)
- See `requirements.txt` for full dependencies

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with qualified financial advisors before making investment decisions.
