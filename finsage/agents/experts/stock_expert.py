"""
Stock Expert Agent
股票投资专家
"""

import json
from typing import Dict, List, Any, Optional
from finsage.agents.base_expert import (
    BaseExpert,
    ExpertRecommendation,
    Action,
)
from finsage.config import AssetConfig
import logging

logger = logging.getLogger(__name__)


class StockExpert(BaseExpert):
    """
    股票投资专家

    专注领域:
    - 美股大盘股 (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
    - 股票ETF (SPY, QQQ, IWM, VTI)
    - 基本面分析
    - 技术面分析
    - 新闻情绪分析
    """

    @classmethod
    def _get_default_symbols(cls) -> List[str]:
        """从 config.py 获取默认股票符号列表"""
        return AssetConfig().default_universe.get("stocks", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "SPY", "QQQ", "IWM", "VTI"
        ])

    def __init__(
        self,
        llm_provider: Any,
        symbols: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(
            llm_provider=llm_provider,
            asset_class="stocks",
            symbols=symbols or self._get_default_symbols(),
            config=config
        )

    @property
    def name(self) -> str:
        return "Stock Expert"

    @property
    def description(self) -> str:
        return """股票投资专家，专注于美股市场分析。
擅长基本面分析(财报、估值)、技术面分析(趋势、指标)和新闻情绪分析。
覆盖科技股、大盘ETF等主要标的。"""

    @property
    def expertise(self) -> List[str]:
        return [
            "基本面分析 (P/E, P/B, ROE, 营收增长)",
            "技术面分析 (MACD, RSI, 布林带, 均线)",
            "财报解读 (EPS, 营收, 指引)",
            "行业轮动分析",
            "新闻情绪分析",
            "美股市场结构",
        ]

    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建股票分析Prompt"""

        # 格式化价格数据
        price_summary = self._format_price_data(market_data)

        # 格式化技术指标
        tech_summary = self._format_technical_indicators(technical_indicators)

        # 格式化新闻
        news_summary = self._format_news(news_data)

        prompt = f"""## 股票市场分析任务

### 当前持仓资产
{', '.join(self.symbols)}

### 价格数据摘要
{price_summary}

### 技术指标
{tech_summary}

### 近期新闻
{news_summary}

### 分析要求
1. 评估每只股票的短期(1-5天)走势
2. 结合基本面和技术面给出交易建议
3. 考虑整体市场环境(风险偏好、板块轮动)
4. 给出具体的仓位建议和置信度

请给出你的专业分析和建议。
"""
        return prompt

    def _format_price_data(self, market_data: Dict[str, Any]) -> str:
        """格式化价格数据"""
        if not market_data:
            return "暂无价格数据"

        lines = []
        for symbol, data in market_data.items():
            if symbol in self.symbols:
                price = data.get("close", data.get("price", "N/A"))
                change = data.get("change_pct", 0)
                volume = data.get("volume", "N/A")
                lines.append(f"- {symbol}: ${price:.2f} ({change:+.2f}%), 成交量: {volume}")

        return "\n".join(lines) if lines else "暂无价格数据"

    def _format_technical_indicators(self, indicators: Dict[str, Any]) -> str:
        """格式化技术指标 (包含信号解读)"""
        if not indicators:
            return "暂无技术指标"

        lines = []
        bearish_signals = []  # 追踪看空信号
        bullish_signals = []  # 追踪看多信号

        for symbol, ind in indicators.items():
            if symbol in self.symbols:
                # 基础指标
                rsi = ind.get("rsi", ind.get("rsi_14", 50))
                macd = ind.get("macd", 0)
                macd_signal = ind.get("macd_signal", 0)
                macd_hist = ind.get("macd_hist", 0)
                macd_cross = ind.get("macd_cross", "neutral")
                ma20 = ind.get("ma_20", ind.get("sma_20", 0))
                ma50 = ind.get("ma_50", ind.get("sma_50", 0))
                price = ind.get("price", 0)
                trend = ind.get("trend", "sideways")
                bb_position = ind.get("bb_position", "neutral")

                # 格式化数值
                rsi_str = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else str(rsi)
                macd_str = f"{macd:.3f}" if isinstance(macd, (int, float)) else str(macd)
                macd_hist_str = f"{macd_hist:.3f}" if isinstance(macd_hist, (int, float)) else str(macd_hist)

                # 生成信号解读
                signals = []

                # RSI 信号
                if isinstance(rsi, (int, float)):
                    if rsi > 70:
                        signals.append("RSI超买")
                        bearish_signals.append(f"{symbol} RSI={rsi_str} 超买")
                    elif rsi < 30:
                        signals.append("RSI超卖")
                        bullish_signals.append(f"{symbol} RSI={rsi_str} 超卖")

                # MACD 信号
                if macd_cross == "bearish":
                    signals.append("MACD死叉")
                    bearish_signals.append(f"{symbol} MACD死叉 (hist={macd_hist_str})")
                elif macd_cross == "bullish":
                    signals.append("MACD金叉")
                    bullish_signals.append(f"{symbol} MACD金叉 (hist={macd_hist_str})")

                # 趋势信号
                if trend == "downtrend":
                    signals.append("下跌趋势")
                    bearish_signals.append(f"{symbol} 下跌趋势 (价格<MA20<MA50)")
                elif trend == "uptrend":
                    signals.append("上涨趋势")

                # Bollinger Bands 信号
                if bb_position == "overbought":
                    signals.append("BB超买")
                    bearish_signals.append(f"{symbol} 突破布林带上轨")
                elif bb_position == "oversold":
                    signals.append("BB超卖")
                    bullish_signals.append(f"{symbol} 跌破布林带下轨")

                signal_str = ", ".join(signals) if signals else "无明显信号"
                lines.append(
                    f"- {symbol}: RSI={rsi_str}, MACD={macd_str}, MACD_HIST={macd_hist_str}, "
                    f"MA20={ma20:.2f}, MA50={ma50:.2f}, 趋势={trend} | 信号: {signal_str}"
                )

        # 添加信号汇总
        result = "\n".join(lines) if lines else "暂无技术指标"

        if bearish_signals:
            result += f"\n\n⚠️ 做空信号汇总:\n" + "\n".join(f"  - {s}" for s in bearish_signals)

        if bullish_signals:
            result += f"\n\n📈 做多信号汇总:\n" + "\n".join(f"  - {s}" for s in bullish_signals)

        return result

    def _format_news(self, news_data: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_data:
            return "暂无相关新闻"

        # 过滤与本专家相关的新闻
        relevant_news = []
        for news in news_data[:10]:  # 最多10条
            title = news.get("title", "")
            sentiment = news.get("sentiment", "neutral")
            symbols = news.get("symbols", [])

            # 检查是否与本专家的资产相关
            if any(s in self.symbols for s in symbols) or not symbols:
                relevant_news.append(f"- [{sentiment}] {title}")

        return "\n".join(relevant_news) if relevant_news else "暂无相关新闻"

    def _parse_llm_response(self, response: str) -> List[ExpertRecommendation]:
        """解析LLM响应"""
        recommendations = []

        try:
            # 尝试提取JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response

            data = json.loads(json_str.strip())

            for rec in data.get("recommendations", []):
                action_str = rec.get("action", "HOLD")
                action = self._parse_action(action_str)

                recommendation = ExpertRecommendation(
                    asset_class=self.asset_class,
                    symbol=rec.get("symbol", ""),
                    action=action,
                    confidence=float(rec.get("confidence", 0.5)),
                    target_weight=float(rec.get("target_weight", 0.0)),
                    reasoning=rec.get("reasoning", ""),
                    market_view={
                        "trend": rec.get("trend", "neutral"),
                        "risk_level": rec.get("risk_level", "medium"),
                    },
                    risk_assessment={
                        "volatility": rec.get("volatility", 0.2),
                        "downside_risk": rec.get("downside_risk", 0.1),
                    }
                )
                recommendations.append(recommendation)

        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {type(e).__name__}: {e}")
            # 返回默认HOLD建议，但标记为解析失败
            for symbol in self.symbols[:3]:
                recommendations.append(ExpertRecommendation(
                    asset_class=self.asset_class,
                    symbol=symbol,
                    action=Action.HOLD,
                    confidence=0.3,  # 降低置信度表示不确定性
                    target_weight=0.05,  # 降低权重
                    reasoning="[PARSE_FAILED] Unable to parse LLM response, defaulting to conservative HOLD",
                    market_view={"trend": "neutral", "parse_error": True},
                    risk_assessment={"volatility": 0.25, "uncertainty": "high"},
                ))
            logger.error(f"LLM response parse failed for {self.asset_class}, using fallback recommendations")

        return recommendations

    def _parse_action(self, action_str: str) -> Action:
        """解析动作字符串 (支持做空)"""
        action_map = {
            # 做空动作
            "SHORT_100%": Action.SHORT_100,
            "SHORT_75%": Action.SHORT_75,
            "SHORT_50%": Action.SHORT_50,
            "SHORT_25%": Action.SHORT_25,
            # 卖出/减持
            "SELL_100%": Action.SELL_100,
            "SELL_75%": Action.SELL_75,
            "SELL_50%": Action.SELL_50,
            "SELL_25%": Action.SELL_25,
            # 持有
            "HOLD": Action.HOLD,
            # 买入/加仓
            "BUY_25%": Action.BUY_25,
            "BUY_50%": Action.BUY_50,
            "BUY_75%": Action.BUY_75,
            "BUY_100%": Action.BUY_100,
        }
        return action_map.get(action_str, Action.HOLD)
