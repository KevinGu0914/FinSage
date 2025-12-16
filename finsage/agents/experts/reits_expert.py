"""
REITs Expert Agent
房地产投资信托专家
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


class REITsExpert(BaseExpert):
    """
    REITs投资专家

    专注领域:
    - 综合REITs ETF (VNQ, IYR, SCHH)
    - 数据中心 (DLR, EQIX)
    - 物流仓储 (PLD)
    - 住宅 (EQR, AVB)
    """

    @classmethod
    def _get_default_symbols(cls) -> List[str]:
        """从 config.py 获取默认 REITs 符号列表"""
        return AssetConfig().default_universe.get("reits", [
            "VNQ", "IYR", "SCHH", "DLR", "EQIX", "PLD", "EQR", "AVB"
        ])

    def __init__(
        self,
        llm_provider: Any,
        symbols: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(
            llm_provider=llm_provider,
            asset_class="reits",
            symbols=symbols or self._get_default_symbols(),
            config=config
        )

    @property
    def name(self) -> str:
        return "REITs Expert"

    @property
    def description(self) -> str:
        return """REITs投资专家，专注于房地产投资信托分析。
擅长利率敏感性分析、NAV估值、细分行业趋势和收益率分析。
覆盖综合REITs、数据中心、物流、住宅等细分领域。"""

    @property
    def expertise(self) -> List[str]:
        return [
            "REITs估值分析 (NAV, P/FFO, P/AFFO)",
            "利率敏感性分析",
            "细分行业趋势 (数据中心, 物流, 住宅)",
            "空置率和租金趋势",
            "分红收益率分析",
            "Cap Rate vs Treasury分析",
        ]

    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建REITs分析Prompt"""

        price_summary = self._format_price_data(market_data)
        rate_summary = self._format_rate_environment(market_data)
        news_summary = self._format_news(news_data)

        prompt = f"""## REITs市场分析任务

### 当前持仓资产
{', '.join(self.symbols)}

### 价格数据摘要
{price_summary}

### 利率环境
{rate_summary}

### 近期新闻
{news_summary}

### 分析要求
1. 评估利率环境对REITs的影响
2. 分析各细分行业的基本面
3. 评估分红收益率的吸引力
4. 识别结构性趋势受益者
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
                div_yield = data.get("dividend_yield", "N/A")
                lines.append(f"- {symbol}: ${price:.2f} ({change:+.2f}%), 股息率: {div_yield}")

        return "\n".join(lines) if lines else "暂无价格数据"

    def _format_rate_environment(self, market_data: Dict[str, Any]) -> str:
        """格式化利率环境"""
        rates = market_data.get("rates", {})
        lines = [
            f"- 10Y Treasury: {rates.get('treasury_10y', 'N/A')}%",
            f"- 平均Cap Rate: {rates.get('avg_cap_rate', 'N/A')}%",
            f"- Cap Rate Spread: {rates.get('cap_rate_spread', 'N/A')} bps",
            f"- 利率预期: {rates.get('rate_expectation', 'stable')}",
        ]
        return "\n".join(lines)

    def _format_news(self, news_data: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_data:
            return "暂无相关新闻"

        keywords = ["reit", "real estate", "property", "housing", "rental", "data center", "warehouse"]
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
                        "rate_sensitivity": rec.get("rate_sensitivity", "high"),
                        "sector_outlook": rec.get("sector_outlook", "neutral"),
                        "yield_attractiveness": rec.get("yield_attractiveness", "moderate"),
                    },
                    risk_assessment={
                        "interest_rate_risk": rec.get("interest_rate_risk", 0.3),
                        "vacancy_risk": rec.get("vacancy_risk", 0.1),
                    }
                )
                recommendations.append(recommendation)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            recommendations.append(ExpertRecommendation(
                asset_class=self.asset_class,
                symbol="VNQ",
                action=Action.HOLD,
                confidence=0.3,  # 降低置信度表示不确定性
                target_weight=0.03,  # 降低权重
                reasoning="[PARSE_FAILED] Unable to parse LLM response, defaulting to conservative HOLD",
                market_view={"rate_sensitivity": "high", "parse_error": True},
                risk_assessment={"interest_rate_risk": 0.35, "uncertainty": "high"},
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
