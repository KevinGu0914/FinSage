"""
FinSage Configuration
配置文件
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from urllib.parse import quote
import os

# Import FMPClient for centralized URL constants (避免循环导入，延迟在 __post_init__ 中使用)
# URL 常量定义在 finsage/data/fmp_client.py 中


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"  # "openai", "anthropic", "local"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60

    def __post_init__(self):
        if self.api_key is None:
            # Try multiple env var names
            self.api_key = os.environ.get("OA_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")


@dataclass
class TradingConfig:
    """交易配置"""
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005
    min_trade_value: float = 100.0
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"
    rebalance_threshold: float = 0.02  # 降低至2%，更频繁触发再平衡


@dataclass
class RiskConfig:
    """风控配置"""
    # 硬性约束
    max_single_asset: float = 0.15
    max_asset_class: float = 0.50
    max_drawdown_trigger: float = 0.15
    max_portfolio_var_95: float = 0.03

    # 软性约束
    target_volatility: float = 0.12
    max_correlation_cluster: float = 0.60
    min_diversification_ratio: float = 1.2


@dataclass
class AssetConfig:
    """资产配置"""
    # 资产类别权重范围
    allocation_bounds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "stocks": {"min": 0.30, "max": 0.50, "default": 0.40},
        "bonds": {"min": 0.15, "max": 0.35, "default": 0.25},
        "commodities": {"min": 0.10, "max": 0.25, "default": 0.15},
        "reits": {"min": 0.05, "max": 0.15, "default": 0.10},
        "crypto": {"min": 0.00, "max": 0.10, "default": 0.05},
        "cash": {"min": 0.02, "max": 0.15, "default": 0.05},
    })

    # 默认资产池
    default_universe: Dict[str, List[str]] = field(default_factory=lambda: {
        "stocks": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG"],
        "commodities": ["GLD", "SLV", "USO", "DBA", "COPX"],
        "reits": ["VNQ", "IYR", "DLR", "EQIX"],
        "crypto": ["BTC-USD", "ETH-USD"],
    })


@dataclass
class FMPConfig:
    """
    FMP Ultra API 配置
    Documentation: https://site.financialmodelingprep.com/developer/docs#chart
    """
    api_key: Optional[str] = None
    # 默认值会被 __post_init__ 覆盖为 FMPClient.BASE_URL
    base_url: str = "https://financialmodelingprep.com/stable"
    tier: str = "ultra"  # "starter", "premium", "ultimate", "ultra"

    # API 端点配置
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        # Stock Screener - 全市场筛选
        "company_screener": "/company-screener",

        # Market Data - 市场数据
        "quote": "/quote",
        "batch_quote": "/batch-quote",
        "batch_exchange_quote": "/batch-exchange-quote",
        "stock_list": "/stock-list",
        "etf_list": "/etf-list",
        "profile": "/profile",

        # Multi-Asset - 多资产类别
        "cryptocurrency_list": "/cryptocurrency-list",  # 4,786+ 加密货币
        "commodities_list": "/commodities-list",  # 40+ 商品期货

        # Financial Statements - 财务报表
        "income_statement": "/income-statement",
        "balance_sheet": "/balance-sheet-statement",
        "cash_flow": "/cash-flow-statement",
        "latest_financials": "/latest-financial-statements",

        # Key Metrics & Ratios - 关键指标
        "key_metrics": "/key-metrics",
        "key_metrics_ttm": "/key-metrics-ttm",
        "ratios": "/ratios",
        "ratios_ttm": "/ratios-ttm",
        "financial_scores": "/financial-scores",
        "enterprise_values": "/enterprise-values",

        # Historical Data - 历史数据
        "historical_price": "/historical-price-eod/full",
        "historical_chart_1min": "/historical-chart/1min",
        "historical_chart_5min": "/historical-chart/5min",
        "historical_chart_15min": "/historical-chart/15min",
        "historical_chart_1hour": "/historical-chart/1hour",

        # Batch Operations - 批量操作
        "market_cap_batch": "/market-capitalization-batch",
        "shares_float_all": "/shares-float-all",

        # News & Sentiment - 新闻情绪 (stable endpoints)
        "stock_news": "/news/stock",  # 使用 ?symbols=XXX 参数
        "general_news": "/news/general",

        # Additional stable endpoints - 补充端点
        "profile": "/profile",  # 使用 ?symbol=XXX 参数
        "etf_holdings": "/etf/holdings",  # 使用 ?symbol=XXX 参数
        "institutional_ownership": "/institutional-ownership/symbol-positions-summary",  # 需要 ?symbol=&year=&quarter=
        "economic_calendar": "/economic-calendar/event",
        "earnings_calendar": "/earnings-calendar",
    })

    # 筛选参数默认值
    screener_defaults: Dict[str, Any] = field(default_factory=lambda: {
        "exchange": "NYSE,NASDAQ,AMEX",  # 默认美股交易所
        "country": "US",
        "is_actively_trading": True,
        "limit": 1000,
    })

    # 速率限制配置 (Ultra tier)
    rate_limit: Dict[str, int] = field(default_factory=lambda: {
        "requests_per_minute": 750,  # Ultra tier
        "batch_size": 100,
        "delay_between_batches": 0.1,  # seconds
    })

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("FMP_API_KEY") or os.environ.get("OA_FMP_KEY")
        # 优先使用环境变量，否则使用 FMPClient 中定义的统一常量
        if os.environ.get("FMP_BASE_URL"):
            self.base_url = os.environ.get("FMP_BASE_URL")
        else:
            # 延迟导入避免循环依赖
            from finsage.data.fmp_client import FMPClient
            self.base_url = FMPClient.BASE_URL
        if os.environ.get("FMP_API_TIER"):
            self.tier = os.environ.get("FMP_API_TIER")

    def get_url(self, endpoint: str, **params) -> str:
        """构建完整的 API URL (带参数验证和URL编码)"""
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        url = f"{self.base_url}{self.endpoints[endpoint]}"
        params["apikey"] = self.api_key

        # 对参数值进行 URL 编码，防止注入和特殊字符问题
        query_string = "&".join(
            f"{k}={quote(str(v), safe='')}"
            for k, v in params.items()
            if v is not None
        )
        return f"{url}?{query_string}"


@dataclass
class DataConfig:
    """数据配置"""
    data_source: str = "fmp"  # "yfinance", "fmp", "local" - 使用FMP stable API
    cache_dir: str = "./data/cache"
    fmp_api_key: Optional[str] = None
    lookback_days: int = 252  # 一年
    news_limit: int = 50

    # FMP Ultra 配置
    fmp: FMPConfig = field(default_factory=FMPConfig)

    # 全市场筛选配置
    enable_full_market_screening: bool = True
    screening_cache_hours: int = 24  # 筛选缓存有效期
    max_screening_candidates: int = 500  # 最大候选数量

    def __post_init__(self):
        if self.fmp_api_key is None:
            # Try multiple env var names
            self.fmp_api_key = os.environ.get("OA_FMP_KEY") or os.environ.get("FMP_API_KEY")
        # Sync with FMP config
        if self.fmp.api_key is None:
            self.fmp.api_key = self.fmp_api_key


@dataclass
class FinSageConfig:
    """FinSage主配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    assets: AssetConfig = field(default_factory=AssetConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # 日志和输出
    log_level: str = "INFO"
    output_dir: str = "./results"
    save_trades: bool = True
    save_reports: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FinSageConfig":
        """从字典创建配置"""
        return cls(
            llm=LLMConfig(**config_dict.get("llm", {})),
            trading=TradingConfig(**config_dict.get("trading", {})),
            risk=RiskConfig(**config_dict.get("risk", {})),
            assets=AssetConfig(**config_dict.get("assets", {})),
            data=DataConfig(**config_dict.get("data", {})),
            log_level=config_dict.get("log_level", "INFO"),
            output_dir=config_dict.get("output_dir", "./results"),
            save_trades=config_dict.get("save_trades", True),
            save_reports=config_dict.get("save_reports", True),
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "transaction_cost": self.trading.transaction_cost,
                "rebalance_frequency": self.trading.rebalance_frequency,
            },
            "risk": {
                "max_single_asset": self.risk.max_single_asset,
                "max_asset_class": self.risk.max_asset_class,
                "target_volatility": self.risk.target_volatility,
            },
            "assets": {
                "allocation_bounds": self.assets.allocation_bounds,
            },
            "data": {
                "data_source": self.data.data_source,
            },
        }


# 预定义配置模板
CONSERVATIVE_CONFIG = FinSageConfig(
    risk=RiskConfig(
        max_single_asset=0.10,
        max_asset_class=0.40,
        target_volatility=0.08,
    ),
    assets=AssetConfig(
        allocation_bounds={
            "stocks": {"min": 0.20, "max": 0.40, "default": 0.30},
            "bonds": {"min": 0.30, "max": 0.50, "default": 0.40},
            "commodities": {"min": 0.05, "max": 0.15, "default": 0.10},
            "reits": {"min": 0.05, "max": 0.10, "default": 0.08},
            "crypto": {"min": 0.00, "max": 0.02, "default": 0.02},
            "cash": {"min": 0.05, "max": 0.20, "default": 0.10},
        }
    ),
)

AGGRESSIVE_CONFIG = FinSageConfig(
    risk=RiskConfig(
        max_single_asset=0.20,
        max_asset_class=0.60,
        target_volatility=0.18,
    ),
    assets=AssetConfig(
        allocation_bounds={
            "stocks": {"min": 0.40, "max": 0.70, "default": 0.55},
            "bonds": {"min": 0.05, "max": 0.20, "default": 0.10},
            "commodities": {"min": 0.10, "max": 0.25, "default": 0.15},
            "reits": {"min": 0.05, "max": 0.15, "default": 0.10},
            "crypto": {"min": 0.00, "max": 0.15, "default": 0.07},
            "cash": {"min": 0.00, "max": 0.05, "default": 0.03},
        }
    ),
)
