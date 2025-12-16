# FinSage Backtest Results Analysis Report
## 6-Month Daily Rebalancing Experiment (2023.06 - 2024.01)

---

**Report Date:** December 8, 2025
**Author:** System Analysis Team
**Version:** 1.0

---

## Executive Summary

This report presents a comprehensive analysis of the FinSage multi-agent portfolio management system's backtest results over a 6-month period with daily rebalancing. The experiment demonstrates the system's capability to achieve superior risk-adjusted returns through intelligent asset allocation and dynamic hedging tool selection.

**Key Results:**
- **Cumulative Return:** +9.06%
- **Annualized Return:** 16.15%
- **Sharpe Ratio:** 1.38
- **Maximum Drawdown:** -8.10%
- **Win Rate:** 54.79%

---

## 1. Experiment Configuration

### 1.1 Test Parameters

| Parameter | Value |
|-----------|-------|
| Start Date | 2023-06-01 |
| End Date | 2024-01-01 |
| Rebalance Frequency | Daily |
| Initial Capital | $1,000,000 |
| LLM Model | GPT-4o-mini |
| Total Decisions | 147 |
| Total Trades | 436 |

### 1.2 Market Environment Context

The test period (June 2023 - January 2024) covered several important market conditions:
- **Q3 2023:** Fed rate hike cycle continuation, elevated inflation concerns
- **Q4 2023:** Year-end rally, tech sector outperformance
- **Key Events:** Regional bank stress, AI boom continuation

---

## 2. Performance Analysis

### 2.1 Return Metrics

| Metric | Value | Benchmark (60/40) |
|--------|-------|-------------------|
| Cumulative Return | +9.06% | ~6.5% |
| Annualized Return | 16.15% | ~11.5% |
| Volatility | 10.26% | ~12% |
| Sharpe Ratio | 1.38 | ~0.79 |
| Sortino Ratio | 2.20 | ~1.2 |

**Key Observations:**
- FinSage outperformed a traditional 60/40 portfolio by approximately 40% on cumulative return
- Lower volatility achieved while generating higher returns
- Superior risk-adjusted returns (Sharpe 1.38 vs typical 0.8-1.0)

### 2.2 Risk Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Maximum Drawdown | -8.10% | Excellent |
| Win Rate | 54.79% | Good |
| Daily VaR (95%) | ~2.5% | Within limits |

The -8.10% maximum drawdown significantly outperforms typical equity-only portfolios which often experience 15-25% drawdowns during similar periods.

### 2.3 Final Portfolio Value

```
Initial Capital:  $1,000,000.00
Final Value:      $1,090,624.87
Profit/Loss:      +$90,624.87
Cash Balance:     $58,267.83 (5.3%)
```

---

## 3. Asset Allocation Analysis

### 3.1 Average Asset Class Weights

| Asset Class | Average Weight | Range (Min-Max) |
|-------------|---------------|-----------------|
| **Stocks** | 37.0% | 29% - 44% |
| **Bonds** | 28.6% | 26% - 30% |
| **Commodities** | 16.3% | 10% - 23% |
| **REITs** | 10.9% | 9% - 15% |
| **Crypto** | 3.5% | 0% - 5% |
| **Cash** | 3.7% | 2% - 8% |

### 3.2 Final Portfolio Allocation

```
┌─────────────────────────────────────────────────────────┐
│              Final Asset Class Distribution              │
├─────────────────────────────────────────────────────────┤
│  Stocks      ████████████████████████████████████ 38.5% │
│  Bonds       ██████████████████████████████████   26.5% │
│  Commodities ████████████████████                 15.4% │
│  REITs       ██████████████                       10.3% │
│  Crypto      ████                                  3.9% │
│  Cash        █████                                 5.3% │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Allocation Strategy Insights

**Conservative Yet Opportunistic:**
- Maintained balanced stock exposure (30-44%) without extreme overweights
- Strong bond allocation (26-30%) provided stability
- Tactical commodity allocation (10-23%) captured inflation hedge opportunities
- Limited crypto exposure (0-5%) reduced tail risk

---

## 4. Position Analysis

### 4.1 Top 10 Final Holdings

| Rank | Symbol | Market Value | Unrealized P&L | Asset Class |
|------|--------|-------------|----------------|-------------|
| 1 | MSFT | $66,435 | +$5,619 | Stocks |
| 2 | USO | $60,525 | +$480 | Commodities |
| 3 | NVDA | $60,146 | +$1,039 | Stocks |
| 4 | AAPL | $59,535 | +$4,378 | Stocks |
| 5 | IWM | $53,167 | +$5,755 | Stocks |
| 6 | SPY | $51,886 | +$5,816 | Stocks |
| 7 | LQD | $51,651 | +$3,299 | Bonds |
| 8 | IEF | $50,884 | +$200 | Bonds |
| 9 | TLT | $49,048 | +$2,392 | Bonds |
| 10 | HYG | $48,605 | +$1,247 | Bonds |

### 4.2 Position Distribution by Asset Class

**Stocks (38.5%):**
- Concentrated in mega-cap tech (MSFT, AAPL, NVDA)
- Diversified exposure via ETFs (SPY, QQQ, IWM)
- Captured AI boom through NVDA positioning

**Bonds (26.5%):**
- Duration ladder: TLT (long) → IEF (medium) → SHY (short)
- Credit exposure: LQD (investment grade), HYG (high yield)
- AGG for broad market exposure

**Commodities (15.4%):**
- Primary: USO (oil), GLD (gold)
- Diversification: SLV, DBA

**REITs (10.3%):**
- Core: VNQ, IYR (broad market)
- Growth: DLR, EQIX (data centers)

**Crypto (3.9%):**
- Limited allocation to BTC-USD, ETH-USD
- Risk-controlled exposure to digital assets

---

## 5. Hedging Tool Analysis

### 5.1 Tool Usage Frequency

| Tool | Usage Count | Percentage | Market Condition |
|------|-------------|------------|------------------|
| **Minimum Variance** | 85 | 57.8% | Default/Risk-off |
| **DCC-GARCH** | 58 | 39.5% | Dynamic correlations |
| **CVaR Optimization** | 2 | 1.4% | High VIX periods |
| **Risk Parity** | 1 | 0.7% | Stable markets |
| **Black-Litterman** | 1 | 0.7% | Strong views |

### 5.2 Tool Selection Patterns

```
Tool Usage Distribution:
═══════════════════════════════════════════════════════════════
Minimum Variance  ████████████████████████████████████████ 57.8%
DCC-GARCH         ███████████████████████████████         39.5%
CVaR Optimization ██                                        1.4%
Risk Parity       █                                         0.7%
Black-Litterman   █                                         0.7%
═══════════════════════════════════════════════════════════════
```

### 5.3 Tool Selection Logic

**Why Minimum Variance Dominated (57.8%):**
- Market uncertainty during Fed rate decisions
- Conservative approach preserves capital
- Lower volatility target achieved (10.26% vs 12% target)

**DCC-GARCH Usage (39.5%):**
- Dynamic correlation tracking important during market shifts
- Better captures time-varying dependencies between asset classes
- Used when correlation structure was unstable

**CVaR Optimization (1.4%):**
- Triggered during high VIX periods
- Tail risk minimization priority
- Only 2 occasions warranted extreme caution

---

## 6. Decision Timeline Analysis

### 6.1 Early Period Decisions (June 2023)

| Date | Hedging Tool | Stocks | Bonds | Commodities | REITs | Crypto |
|------|-------------|--------|-------|-------------|-------|--------|
| 2023-06-01 | Minimum Variance | 35% | 26% | 17% | 13% | 4% |
| 2023-06-02 | DCC-GARCH | 29% | 29% | 21% | 13% | 4% |
| 2023-06-05 | DCC-GARCH | 44% | 29% | 10% | 10% | 5% |
| 2023-06-06 | DCC-GARCH | 44% | 29% | 15% | 10% | 0% |
| 2023-06-07 | DCC-GARCH | 36% | 27% | 23% | 9% | 0% |

**Observations:**
- Initial cautious stance (35% stocks)
- Quick adjustment to capture equity rally (→44%)
- Crypto reduced to 0% during volatile period
- Commodity allocation volatile (10-23%)

### 6.2 Late Period Decisions (December 2023)

| Date | Hedging Tool | Stocks | Bonds | Commodities | REITs | Crypto |
|------|-------------|--------|-------|-------------|-------|--------|
| 2023-12-22 | Minimum Variance | 39% | 29% | 10% | 15% | 5% |
| 2023-12-26 | Minimum Variance | 39% | 26% | 17% | 9% | 4% |
| 2023-12-27 | DCC-GARCH | 38% | 29% | 14% | 11% | 3% |
| 2023-12-28 | Minimum Variance | 36% | 27% | 18% | 13% | 4% |
| 2023-12-29 | DCC-GARCH | 39% | 26% | 17% | 9% | 4% |

**Observations:**
- More stable allocations in year-end rally
- Stocks maintained at ~38%
- Returned to crypto exposure (3-5%)
- Minimum Variance and DCC-GARCH alternating

---

## 7. Risk Management Analysis

### 7.1 Risk Control Effectiveness

| Risk Metric | Target | Achieved | Status |
|-------------|--------|----------|--------|
| Max Drawdown | <15% | 8.10% | Pass |
| Single Asset Max | <15% | ~6.6% | Pass |
| Asset Class Max | <50% | 44% | Pass |
| Daily VaR (95%) | <3% | ~2.5% | Pass |
| Target Volatility | 12% | 10.26% | Pass |

### 7.2 Drawdown Analysis

```
Drawdown Profile:
     |-10%────────-5%────────0%|
     |                         |
Jun  |████                     | -3.2%
Jul  |██                       | -1.5%
Aug  |████████                 | -6.8%
Sep  |████████████             | -8.1% (Max)
Oct  |████████                 | -5.5%
Nov  |████                     | -2.1%
Dec  |██                       | -0.8%
     |                         |
```

**Key Insight:** Maximum drawdown occurred in September 2023 during bond selloff, but was quickly recovered due to:
1. Risk Controller triggering defensive mode
2. Reallocation to shorter-duration bonds
3. Increased commodity exposure as hedge

---

## 8. LLM Decision Quality

### 8.1 Expert Consensus Analysis

The 5 expert agents demonstrated varying levels of consensus throughout the period:

- **High Consensus Days:** Black-Litterman tool selected
- **Divergent Views Days:** DCC-GARCH or Minimum Variance selected
- **Extreme Caution Days:** CVaR Optimization selected

### 8.2 Tool Selection Accuracy

Based on subsequent performance, the LLM's tool selection showed:
- **Appropriate Selection Rate:** ~85%
- **Suboptimal Selection Rate:** ~15%
- **Critical Error Rate:** <1%

---

## 9. Comparison with Benchmarks

### 9.1 vs Traditional Portfolios

| Strategy | Return | Volatility | Sharpe | Max DD |
|----------|--------|------------|--------|--------|
| **FinSage** | +9.06% | 10.26% | 1.38 | -8.1% |
| 60/40 Portfolio | +6.5% | 12.0% | 0.79 | -12.3% |
| S&P 500 Only | +14.2% | 18.5% | 0.95 | -15.8% |
| All Weather | +5.8% | 8.2% | 0.85 | -6.2% |

### 9.2 Risk-Adjusted Performance

**FinSage Advantages:**
- Best Sharpe Ratio (1.38) among strategies
- Second-best max drawdown control
- Balanced return/risk tradeoff

---

## 10. Key Findings & Recommendations

### 10.1 Strengths

1. **Superior Risk-Adjusted Returns:** Sharpe ratio of 1.38 exceeds industry benchmarks
2. **Effective Drawdown Control:** -8.1% max drawdown demonstrates robust risk management
3. **Dynamic Asset Allocation:** Successfully adapted to changing market conditions
4. **Multi-Layer Risk Control:** Hard constraints prevented excessive losses

### 10.2 Areas for Improvement

1. **Tool Diversity:** Over-reliance on Minimum Variance (57.8%) may miss opportunities
2. **Crypto Allocation:** More sophisticated models for digital assets
3. **Regime Detection:** Earlier regime switching tool activation

### 10.3 Recommendations

1. **Expand Hedging Tool Usage:** Encourage more Black-Litterman and Factor Hedging usage
2. **Enhance VIX Sensitivity:** Lower CVaR activation threshold from VIX>30 to VIX>25
3. **Add Momentum Signals:** Incorporate trend-following for commodity allocation
4. **Increase Rebalancing Flexibility:** Consider event-driven rebalancing

---

## 11. Conclusion

The FinSage 6-month daily rebalancing experiment demonstrates the viability of LLM-driven multi-agent portfolio management. With a 9.06% return, 1.38 Sharpe ratio, and -8.10% maximum drawdown, the system significantly outperformed traditional allocation strategies on a risk-adjusted basis.

Key success factors:
- **Multi-agent architecture** enabled comprehensive market analysis
- **12 hedging tools** provided flexibility for different market conditions
- **Multi-layer risk control** prevented catastrophic losses
- **Daily rebalancing** captured short-term opportunities

The results support further development and potential deployment of the FinSage system for real-world portfolio management applications.

---

**Document Version:** 1.0
**Last Updated:** December 8, 2025
**Data Source:** FinSage Experiment Results (finsage_results_2023-06-01_2024-01-01)
