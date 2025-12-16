"""
Bond Expert Agent
债券投资专家
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


class BondExpert(BaseExpert):
    """
    债券投资专家

    专注领域:
    - 美国国债 ETF (TLT, IEF, SHY)
    - 企业债 ETF (LQD, HYG)
    - 综合债券 ETF (AGG, BND)
    - 利率分析
    - 信用分析
    """

    @classmethod
    def _get_default_symbols(cls) -> List[str]:
        """从 config.py 获取默认债券符号列表"""
        return AssetConfig().default_universe.get("bonds", [
            "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND"
        ])

    def __init__(
        self,
        llm_provider: Any,
        symbols: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(
            llm_provider=llm_provider,
            asset_class="bonds",
            symbols=symbols or self._get_default_symbols(),
            config=config
        )

    @property
    def name(self) -> str:
        return "Bond Expert"

    @property
    def description(self) -> str:
        return """债券投资专家，专注于固定收益市场分析。
擅长利率走势判断、久期管理、信用分析和收益率曲线分析。
覆盖国债、企业债、高收益债等主要品种。"""

    @property
    def expertise(self) -> List[str]:
        return [
            "利率走势分析 (Fed Funds, 10Y, 2Y)",
            "收益率曲线分析 (陡峭/平坦/倒挂)",
            "久期和凸性管理",
            "信用利差分析 (IG/HY Spread)",
            "美联储政策解读",
            "通胀预期分析",
        ]

    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建债券分析Prompt"""

        price_summary = self._format_price_data(market_data)
        rate_summary = self._format_rate_data(market_data)
        news_summary = self._format_news(news_data)

        prompt = f"""## 债券市场分析任务

### 当前持仓资产
{', '.join(self.symbols)}

### 价格数据摘要
{price_summary}

### 利率环境
{rate_summary}

### 近期新闻与政策
{news_summary}

### 分析要求
1. 评估利率走势方向
2. 分析收益率曲线形态变化
3. 评估信用风险环境
4. 给出久期配置建议
5. 给出具体的交易建议和置信度

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
                yield_val = data.get("yield", "N/A")
                lines.append(f"- {symbol}: ${price:.2f} ({change:+.2f}%), 收益率: {yield_val}")

        return "\n".join(lines) if lines else "暂无价格数据"

    def _format_rate_data(self, market_data: Dict[str, Any]) -> str:
        """格式化利率数据"""
        rates = market_data.get("rates", {})
        if not rates:
            return "暂无利率数据"

        lines = [
            f"- Fed Funds Rate: {rates.get('fed_funds', 'N/A')}%",
            f"- 2Y Treasury: {rates.get('treasury_2y', 'N/A')}%",
            f"- 10Y Treasury: {rates.get('treasury_10y', 'N/A')}%",
            f"- 30Y Treasury: {rates.get('treasury_30y', 'N/A')}%",
            f"- 2s10s Spread: {rates.get('spread_2s10s', 'N/A')} bps",
        ]
        return "\n".join(lines)

    def _format_news(self, news_data: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_data:
            return "暂无相关新闻"

        # 过滤债券/利率相关新闻
        keywords = ["fed", "rate", "bond", "treasury", "yield", "inflation", "fomc"]
        relevant_news = []

        for news in news_data[:10]:
            title = news.get("title", "").lower()
            if any(kw in title for kw in keywords):
                sentiment = news.get("sentiment", "neutral")
                relevant_news.append(f"- [{sentiment}] {news.get('title', '')}")

        return "\n".join(relevant_news) if relevant_news else "暂无相关新闻"

    def _parse_llm_response(self, response: str) -> List[ExpertRecommendation]:
        """解析LLM响应"""
        recommendations = []

        try:
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
                        "rate_view": rec.get("rate_view", "neutral"),
                        "duration_preference": rec.get("duration_preference", "neutral"),
                        "credit_view": rec.get("credit_view", "neutral"),
                    },
                    risk_assessment={
                        "duration_risk": rec.get("duration_risk", 0.2),
                        "credit_risk": rec.get("credit_risk", 0.1),
                    }
                )
                recommendations.append(recommendation)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            for symbol in self.symbols[:2]:
                recommendations.append(ExpertRecommendation(
                    asset_class=self.asset_class,
                    symbol=symbol,
                    action=Action.HOLD,
                    confidence=0.3,  # 降低置信度表示不确定性
                    target_weight=0.05,  # 降低权重
                    reasoning="[PARSE_FAILED] Unable to parse LLM response, defaulting to conservative HOLD",
                    market_view={"rate_view": "neutral", "parse_error": True},
                    risk_assessment={"duration_risk": 0.25, "uncertainty": "high"},
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
