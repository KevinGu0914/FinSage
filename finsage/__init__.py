"""
FinSage: Multi-Asset Multi-Agent Trading System
金融智者 - 多资产多智能体交易系统

A sophisticated AI-powered trading framework that combines:
- 5 Asset Class Expert Agents (Stocks, Bonds, Commodities, REITs, Crypto)
- Portfolio Manager Agent with rule-based + LLM hybrid decision making
- Risk Controller Agent for comprehensive risk management
- 11 Hedging Tools from academic financial journals
- Intraday Risk Monitor for real-time anomaly detection

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    FinSageOrchestrator                  │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
    │  │ Stock   │  │ Bond    │  │Commodity│  │ REITs   │    │
    │  │ Expert  │  │ Expert  │  │ Expert  │  │ Expert  │    │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
    │       └────────────┼────────────┼────────────┘         │
    │                    ▼                                    │
    │            ┌───────────────┐                           │
    │            │   Portfolio   │◄── Rule Engine + LLM      │
    │            │    Manager    │                           │
    │            └───────┬───────┘                           │
    │                    │                                    │
    │       ┌────────────┼────────────┐                      │
    │       ▼            ▼            ▼                      │
    │  ┌─────────┐ ┌──────────┐ ┌──────────┐                │
    │  │Hedging  │ │ Risk     │ │ Intraday │                │
    │  │Toolkit  │ │Controller│ │ Monitor  │                │
    │  └─────────┘ └──────────┘ └──────────┘                │
    └─────────────────────────────────────────────────────────┘

Author: Boyang Gu
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Boyang Gu"

# Core Components
from finsage.config import (
    FinSageConfig,
    LLMConfig,
    TradingConfig,
    RiskConfig,
    AssetConfig,
    DataConfig,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
)

# Agents - Core
from finsage.agents.portfolio_manager import PortfolioManager, PortfolioDecision
from finsage.agents.risk_controller import RiskController, RiskAssessment

# Agents - Advanced Components
from finsage.agents.base_expert import BaseExpert, ExpertReport, ExpertRecommendation
from finsage.agents.position_sizing_agent import PositionSizingAgent, PositionSizingDecision
from finsage.agents.hedging_agent import HedgingAgent, HedgingDecision
from finsage.agents.manager_coordinator import ManagerCoordinator, IntegratedDecision

# Core Orchestration
from finsage.core.orchestrator import FinSageOrchestrator

# LLM Provider
from finsage.llm.llm_provider import LLMProvider

# Data Components
from finsage.data.data_loader import DataLoader
from finsage.data.market_data import MarketDataProvider
from finsage.data.dynamic_screener import DynamicAssetScreener

# Hedging Tools
from finsage.hedging.toolkit import HedgingToolkit

# Risk Monitoring
from finsage.risk.intraday_monitor import (
    IntradayRiskMonitor,
    IntradayAlert,
    IntradayRiskReport,
    AlertLevel,
    AlertType,
    EmergencyAction,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Configuration
    "FinSageConfig",
    "LLMConfig",
    "TradingConfig",
    "RiskConfig",
    "AssetConfig",
    "DataConfig",
    "CONSERVATIVE_CONFIG",
    "AGGRESSIVE_CONFIG",
    # Agents - Core
    "PortfolioManager",
    "PortfolioDecision",
    "RiskController",
    "RiskAssessment",
    # Agents - Advanced
    "BaseExpert",
    "ExpertReport",
    "ExpertRecommendation",
    "PositionSizingAgent",
    "PositionSizingDecision",
    "HedgingAgent",
    "HedgingDecision",
    "ManagerCoordinator",
    "IntegratedDecision",
    # Core
    "FinSageOrchestrator",
    # LLM
    "LLMProvider",
    # Data
    "DataLoader",
    "MarketDataProvider",
    "DynamicAssetScreener",
    # Hedging
    "HedgingToolkit",
    # Risk Monitoring
    "IntradayRiskMonitor",
    "IntradayAlert",
    "IntradayRiskReport",
    "AlertLevel",
    "AlertType",
    "EmergencyAction",
]
