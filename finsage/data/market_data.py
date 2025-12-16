"""
Market Data Provider
市场数据提供者 - 整合多种数据并提供统一接口

使用真实 FMP API 数据替代硬编码宏观数据
增强数据模态: 情绪、资金流、经济日历、期权、财报
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from finsage.data.data_loader import DataLoader
from finsage.data.macro_loader import MacroDataLoader
from finsage.data.enhanced_data_loader import EnhancedDataLoader

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    市场数据提供者

    统一接口获取:
    - 价格数据
    - 新闻数据
    - 技术指标
    - 宏观数据
    - 链上数据 (加密货币)
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        macro_loader: Optional[MacroDataLoader] = None,
        enhanced_loader: Optional[EnhancedDataLoader] = None,
        config: Optional[Dict] = None,
    ):
        """
        初始化

        Args:
            data_loader: 数据加载器
            macro_loader: 宏观数据加载器 (FMP API)
            enhanced_loader: 增强数据加载器 (情绪、资金流等)
            config: 配置
        """
        self.loader = data_loader or DataLoader()
        self.macro_loader = macro_loader or MacroDataLoader()
        self.enhanced_loader = enhanced_loader or EnhancedDataLoader()
        self.config = config or {}

        # 缓存
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._news_cache: Dict[str, List[Dict]] = {}

        logger.info("MarketDataProvider initialized (with FMP macro data + enhanced data)")

    def get_market_snapshot(
        self,
        symbols: List[str],
        date: str,
        lookback_days: int = 30,
        include_news: bool = True,
        include_technicals: bool = True,
        include_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        获取市场快照

        Args:
            symbols: 股票代码列表
            date: 日期
            lookback_days: 回看天数
            include_news: 是否包含新闻
            include_technicals: 是否包含技术指标
            include_sentiment: 是否包含市场情绪数据

        Returns:
            市场数据快照
        """
        end_date = date
        start_date = (
            datetime.strptime(date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        # 加载价格数据
        price_data = self._get_price_data(symbols, start_date, end_date)

        # 构建快照
        snapshot = {}
        for symbol in symbols:
            symbol_data = self._extract_symbol_data(price_data, symbol)
            if symbol_data is not None:
                snapshot[symbol] = {
                    "price": symbol_data.get("close", 0),
                    "close": symbol_data.get("close", 0),
                    "open": symbol_data.get("open", 0),
                    "high": symbol_data.get("high", 0),
                    "low": symbol_data.get("low", 0),
                    "volume": symbol_data.get("volume", 0),
                    "change_pct": symbol_data.get("change_pct", 0),
                }

                if include_technicals:
                    technicals = self._get_technicals(price_data, symbol)
                    snapshot[symbol].update(technicals)

        # 添加新闻
        if include_news:
            news = self._get_news(symbols, start_date, end_date)
            snapshot["news"] = news

        # 添加宏观数据
        snapshot["macro"] = self._get_macro_data(date)

        # 添加市场情绪数据
        if include_sentiment:
            sentiment = self._get_sentiment_data(symbols)
            snapshot["sentiment"] = sentiment
            # 将情绪数据合并到新闻中，方便专家使用
            if "news" in snapshot:
                snapshot["news"] = self._enrich_news_with_sentiment(
                    snapshot["news"], sentiment
                )

        return snapshot

    def get_returns_matrix(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取收益率矩阵

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收益率DataFrame
        """
        price_data = self._get_price_data(symbols, start_date, end_date)

        # 提取收盘价
        if isinstance(price_data.columns, pd.MultiIndex):
            close_prices = price_data["Close"] if "Close" in price_data.columns.get_level_values(0) else price_data["Adj Close"]
        else:
            close_prices = price_data

        # 计算收益率
        returns = close_prices.pct_change().dropna()

        return returns

    def get_covariance_matrix(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """
        获取协方差矩阵

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            annualize: 是否年化

        Returns:
            协方差矩阵
        """
        returns = self.get_returns_matrix(symbols, start_date, end_date)
        cov = returns.cov()

        if annualize:
            cov = cov * 252

        return cov

    def _get_price_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取价格数据 (带缓存)"""
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date}_{end_date}"

        if cache_key not in self._price_cache:
            self._price_cache[cache_key] = self.loader.load_price_data(
                symbols, start_date, end_date
            )

        return self._price_cache[cache_key]

    def _extract_symbol_data(
        self,
        price_data: pd.DataFrame,
        symbol: str,
    ) -> Optional[Dict]:
        """提取单个股票的最新数据"""
        if price_data.empty:
            return None

        try:
            if isinstance(price_data.columns, pd.MultiIndex):
                # 多股票数据 - FMP数据的MultiIndex: level 0 = symbol, level 1 = column name
                # 先检查 level 0 (symbol 作为第一级)
                if symbol in price_data.columns.get_level_values(0):
                    latest = price_data[symbol].iloc[-1]
                # 再检查 level 1 (兼容其他数据源格式)
                elif symbol in price_data.columns.get_level_values(1):
                    latest = price_data.xs(symbol, axis=1, level=1).iloc[-1]
                else:
                    return None
            else:
                latest = price_data.iloc[-1]

            # 处理不同列名格式
            close = latest.get("Close", latest.get("close", latest.get("Adj Close", 0)))
            open_price = latest.get("Open", latest.get("open", 0))
            high = latest.get("High", latest.get("high", 0))
            low = latest.get("Low", latest.get("low", 0))
            volume = latest.get("Volume", latest.get("volume", 0))

            # 计算涨跌幅
            if len(price_data) >= 2:
                if isinstance(price_data.columns, pd.MultiIndex):
                    # FMP数据的MultiIndex: level 0 = symbol, level 1 = column name
                    if symbol in price_data.columns.get_level_values(0):
                        prev_close = price_data[symbol]["Close"].iloc[-2]
                    else:
                        prev_close = price_data.xs(symbol, axis=1, level=1)["Close"].iloc[-2]
                else:
                    prev_close = price_data["Close"].iloc[-2]
                change_pct = (close - prev_close) / prev_close if prev_close > 0 else 0
            else:
                change_pct = 0

            return {
                "close": float(close),
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "volume": float(volume),
                "change_pct": float(change_pct),
            }

        except Exception as e:
            logger.warning(f"Failed to extract data for {symbol}: {e}")
            return None

    def _get_technicals(
        self,
        price_data: pd.DataFrame,
        symbol: str,
    ) -> Dict[str, Any]:
        """获取技术指标 (包括 MACD, RSI, Bollinger Bands)"""
        try:
            if isinstance(price_data.columns, pd.MultiIndex):
                # FMP数据的MultiIndex: level 0 = symbol, level 1 = column name
                if symbol in price_data.columns.get_level_values(0):
                    close = price_data[symbol]["Close"]
                elif symbol in price_data.columns.get_level_values(1):
                    close = price_data.xs(symbol, axis=1, level=1)["Close"]
                else:
                    return self._get_default_technicals()
            else:
                close = price_data["Close"]

            # 计算完整技术指标 (包括 MACD, Bollinger Bands)
            indicators = self.loader.get_technical_indicators(
                pd.DataFrame({"Close": close}),
                indicators=["sma_20", "sma_50", "rsi_14", "macd", "bb"]
            )

            # 获取最新值
            latest_close = float(close.iloc[-1]) if not close.empty else 0
            sma_20 = float(indicators["sma_20"].iloc[-1]) if "sma_20" in indicators and not indicators["sma_20"].empty else 0
            sma_50 = float(indicators["sma_50"].iloc[-1]) if "sma_50" in indicators and not indicators["sma_50"].empty else 0
            rsi_14 = float(indicators["rsi_14"].iloc[-1]) if "rsi_14" in indicators and not indicators["rsi_14"].empty else 50

            # MACD 指标
            macd_line = float(indicators["macd"].iloc[-1]) if "macd" in indicators and not indicators["macd"].empty else 0
            macd_signal = float(indicators["macd_signal"].iloc[-1]) if "macd_signal" in indicators and not indicators["macd_signal"].empty else 0
            macd_hist = float(indicators["macd_hist"].iloc[-1]) if "macd_hist" in indicators and not indicators["macd_hist"].empty else 0

            # Bollinger Bands
            bb_upper = float(indicators["bb_upper"].iloc[-1]) if "bb_upper" in indicators and not indicators["bb_upper"].empty else 0
            bb_middle = float(indicators["bb_middle"].iloc[-1]) if "bb_middle" in indicators and not indicators["bb_middle"].empty else 0
            bb_lower = float(indicators["bb_lower"].iloc[-1]) if "bb_lower" in indicators and not indicators["bb_lower"].empty else 0

            # 计算信号
            macd_cross = "bullish" if macd_hist > 0 and macd_line > macd_signal else "bearish" if macd_hist < 0 else "neutral"
            bb_position = "overbought" if latest_close > bb_upper else "oversold" if latest_close < bb_lower else "neutral"
            trend = "uptrend" if latest_close > sma_20 > sma_50 else "downtrend" if latest_close < sma_20 < sma_50 else "sideways"

            return {
                # 均线
                "sma_20": sma_20,
                "sma_50": sma_50,
                "ma_20": sma_20,  # 兼容旧格式
                "ma_50": sma_50,  # 兼容旧格式
                # RSI
                "rsi_14": rsi_14,
                "rsi": rsi_14,  # 兼容旧格式
                # MACD
                "macd": macd_line,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "macd_cross": macd_cross,
                # Bollinger Bands
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "bb_position": bb_position,
                # 趋势信号
                "trend": trend,
            }

        except Exception as e:
            logger.warning(f"Failed to get technicals for {symbol}: {e}")
            return self._get_default_technicals()

    def _get_default_technicals(self) -> Dict[str, Any]:
        """返回默认技术指标值"""
        return {
            "sma_20": 0, "sma_50": 0, "ma_20": 0, "ma_50": 0,
            "rsi_14": 50, "rsi": 50,
            "macd": 0, "macd_signal": 0, "macd_hist": 0, "macd_cross": "neutral",
            "bb_upper": 0, "bb_middle": 0, "bb_lower": 0, "bb_position": "neutral",
            "trend": "sideways",
        }

    def _get_news(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """获取新闻"""
        cache_key = f"news_{'-'.join(sorted(symbols))}_{start_date}_{end_date}"

        if cache_key not in self._news_cache:
            self._news_cache[cache_key] = self.loader.load_news(
                symbols, start_date, end_date, limit=50
            )

        return self._news_cache[cache_key]

    def _get_sentiment_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        获取市场情绪数据

        Args:
            symbols: 股票代码列表

        Returns:
            市场情绪数据
        """
        try:
            # 获取市场整体情绪
            market_sentiment = self.enhanced_loader.get_market_sentiment(symbols)

            # 获取各个股票的个别情绪
            individual_sentiments = {}
            for symbol in symbols[:5]:  # 限制API调用数量
                sent = self.enhanced_loader.get_stock_sentiment(symbol)
                if sent:
                    individual_sentiments[symbol] = sent

            return {
                "market": market_sentiment,
                "individual": individual_sentiments,
            }

        except Exception as e:
            logger.warning(f"Failed to get sentiment data: {e}")
            return {
                "market": {
                    "market_sentiment_score": 0,
                    "market_sentiment_label": "neutral",
                    "bullish_percent": 50,
                    "bearish_percent": 50,
                },
                "individual": {},
            }

    def _enrich_news_with_sentiment(
        self,
        news: List[Dict],
        sentiment_data: Dict[str, Any]
    ) -> List[Dict]:
        """
        将情绪数据与新闻合并，方便专家分析

        Args:
            news: 新闻列表
            sentiment_data: 情绪数据

        Returns:
            增强后的新闻列表
        """
        if not news or not sentiment_data:
            return news

        # 获取市场情绪标签
        market_sentiment = sentiment_data.get("market", {})
        market_label = market_sentiment.get("market_sentiment_label", "neutral")
        market_score = market_sentiment.get("market_sentiment_score", 0)

        # 为新闻添加整体市场情绪上下文
        enriched_news = []
        for n in news:
            enriched = n.copy()
            # 如果新闻没有情绪标签，使用市场整体情绪
            if "sentiment" not in enriched or enriched.get("sentiment") == "neutral":
                # 根据市场情绪推断新闻情绪
                if market_score > 0.2:
                    enriched["sentiment"] = "positive"
                elif market_score < -0.2:
                    enriched["sentiment"] = "negative"
                else:
                    enriched["sentiment"] = "neutral"

            # 添加市场上下文
            enriched["market_context"] = {
                "overall_sentiment": market_label,
                "sentiment_score": market_score,
            }
            enriched_news.append(enriched)

        # 添加一条市场情绪汇总新闻
        sentiment_summary = {
            "title": f"Market Sentiment: {market_label.upper()} (Score: {market_score:.2f})",
            "sentiment": market_label,
            "symbols": [],
            "source": "sentiment_analysis",
            "is_sentiment_summary": True,
            "bullish_percent": market_sentiment.get("bullish_percent", 50),
            "bearish_percent": market_sentiment.get("bearish_percent", 50),
        }
        enriched_news.insert(0, sentiment_summary)

        return enriched_news

    def _get_macro_data(self, date: str) -> Dict[str, Any]:
        """
        获取宏观数据 - 使用 FMP API 真实数据

        Args:
            date: 日期 (当前获取最新数据，历史数据需要另行处理)

        Returns:
            宏观数据字典
        """
        try:
            # 使用 MacroDataLoader 获取真实数据
            macro = self.macro_loader.get_macro_for_experts(date)
            logger.debug(f"Loaded real macro data: VIX={macro.get('vix')}, DXY={macro.get('dxy')}")
            return macro
        except Exception as e:
            logger.warning(f"Failed to get real macro data, using fallback: {e}")
            # 降级为默认值
            return {
                "vix": 20.0,
                "dxy": 103.5,
                "treasury_10y": 4.2,
                "treasury_2y": 4.5,
                "treasury_30y": 4.5,
                "inflation_expectation": 2.5,
                "real_rate": 1.7,
                "yield_curve_spread": -0.3,
            }

    def get_full_macro_snapshot(self) -> Dict[str, Any]:
        """
        获取完整的宏观数据快照

        Returns:
            包含所有宏观指标的完整数据
        """
        return self.macro_loader.get_full_macro_snapshot()

    def get_crypto_onchain_data(self) -> Dict[str, Dict[str, Any]]:
        """
        获取加密货币链上数据 (通过 MacroDataLoader)

        Returns:
            加密货币数据
        """
        return self.macro_loader.get_crypto_data()

    def get_commodities_data(self) -> Dict[str, Dict[str, Any]]:
        """
        获取商品数据

        Returns:
            商品数据
        """
        return self.macro_loader.get_commodities()

    def get_sector_performance(self) -> Dict[str, float]:
        """
        获取板块表现

        Returns:
            板块涨跌幅
        """
        return self.macro_loader.get_sector_performance()

    def get_bond_data(self) -> Dict[str, Any]:
        """
        获取 Bond Expert 需要的利率数据

        Returns:
            包含 Fed Funds, 国债收益率, 利差等数据
        """
        return self.macro_loader.get_bond_expert_data()

    def get_rates_data(self) -> Dict[str, Any]:
        """
        获取利率数据 (供 Bond Expert 的 _format_rate_data 使用)

        Returns:
            兼容 Bond Expert 格式的利率数据
        """
        bond_data = self.macro_loader.get_bond_expert_data()
        return {
            "fed_funds": bond_data.get("fed_funds", 5.25),
            "treasury_2y": bond_data.get("treasury_2y", 4.5),
            "treasury_10y": bond_data.get("treasury_10y", 4.2),
            "treasury_30y": bond_data.get("treasury_30y", 4.5),
            "spread_2s10s": bond_data.get("spread_2s10s", -30),
        }

    def get_reits_data(self) -> Dict[str, Any]:
        """
        获取 REITs Expert 需要的数据

        Returns:
            包含 Cap Rate, Cap Rate Spread, 利率环境等数据
        """
        return self.macro_loader.get_reits_expert_data()

    def get_reits_cap_rate(self, symbol: str = None) -> Dict[str, Any]:
        """
        获取 REITs Cap Rate

        Args:
            symbol: 可选，单个 REIT 的代码

        Returns:
            Cap Rate 数据
        """
        if symbol:
            cap_rate = self.macro_loader.get_reit_cap_rate(symbol)
            return {"symbol": symbol, "cap_rate": cap_rate}
        else:
            return self.macro_loader.get_reits_average_cap_rate()

    def clear_cache(self):
        """清除缓存"""
        self._price_cache.clear()
        self._news_cache.clear()
        logger.info("Cache cleared")

    # ==================== 增强数据模态 ====================

    def get_market_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        获取市场情绪数据 (RapidAPI)

        Args:
            symbols: 股票代码列表

        Returns:
            市场情绪数据
        """
        return self.enhanced_loader.get_market_sentiment(symbols)

    def get_stock_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取单个股票的情绪数据

        Args:
            symbol: 股票代码

        Returns:
            情绪数据
        """
        return self.enhanced_loader.get_stock_sentiment(symbol)

    def get_fund_flows(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取 ETF 资金流向数据 (FMP)

        Args:
            symbol: ETF 代码

        Returns:
            资金流向数据
        """
        return self.enhanced_loader.get_etf_fund_flows(symbol)

    def get_institutional_holdings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取机构持仓数据 (FMP)

        Args:
            symbol: 股票代码

        Returns:
            机构持仓数据
        """
        return self.enhanced_loader.get_institutional_ownership(symbol)

    def get_economic_calendar(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        获取经济日历 (FMP)

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            经济事件列表
        """
        return self.enhanced_loader.get_economic_calendar(start_date, end_date)

    def get_fomc_schedule(self) -> Optional[Dict[str, Any]]:
        """
        获取下一次 FOMC 会议信息

        Returns:
            FOMC 会议信息
        """
        return self.enhanced_loader.get_upcoming_fomc()

    def get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取期权数据 (FMP)

        Args:
            symbol: 股票代码

        Returns:
            期权链数据
        """
        return self.enhanced_loader.get_options_chain(symbol)

    def get_put_call_ratio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取看跌/看涨比率

        Args:
            symbol: 股票代码

        Returns:
            PCR 数据
        """
        return self.enhanced_loader.get_put_call_ratio(symbol)

    def get_earnings_calendar(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        获取财报日历 (FMP)

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            财报事件列表
        """
        return self.enhanced_loader.get_earnings_calendar(start_date, end_date)

    def get_earnings_surprise(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取财报惊喜数据

        Args:
            symbol: 股票代码

        Returns:
            财报惊喜数据
        """
        return self.enhanced_loader.get_earnings_surprise(symbol)

    def get_enhanced_market_data(
        self,
        symbols: List[str],
        include_sentiment: bool = True,
        include_fund_flows: bool = True,
        include_economic_calendar: bool = True,
        include_options: bool = False,
        include_earnings: bool = True,
    ) -> Dict[str, Any]:
        """
        获取增强市场数据 (整合所有新增数据模态)

        Args:
            symbols: 股票代码列表
            include_sentiment: 是否包含情绪数据
            include_fund_flows: 是否包含资金流向
            include_economic_calendar: 是否包含经济日历
            include_options: 是否包含期权数据
            include_earnings: 是否包含财报数据

        Returns:
            增强市场数据
        """
        return self.enhanced_loader.get_enhanced_market_data(
            symbols=symbols,
            include_sentiment=include_sentiment,
            include_fund_flows=include_fund_flows,
            include_economic_calendar=include_economic_calendar,
            include_options=include_options,
            include_earnings=include_earnings,
        )
