"""
Commodity Expert Agent
大宗商品投资专家
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


class CommodityExpert(BaseExpert):
    """
    大宗商品投资专家

    专注领域:
    - 贵金属 (GLD, SLV, IAU)
    - 能源 (USO, UNG, XLE)
    - 农产品 (DBA)
    - 工业金属 (COPX)
    """

    @classmethod
    def _get_default_symbols(cls) -> List[str]:
        """从 config.py 获取默认大宗商品符号列表"""
        return AssetConfig().default_universe.get("commodities", [
            "GLD", "SLV", "IAU", "USO", "UNG", "DBA", "COPX", "XLE"
        ])

    def __init__(
        self,
        llm_provider: Any,
        symbols: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(
            llm_provider=llm_provider,
            asset_class="commodities",
            symbols=symbols or self._get_default_symbols(),
            config=config
        )

    @property
    def name(self) -> str:
        return "Commodity Expert"

    @property
    def description(self) -> str:
        return """大宗商品投资专家，专注于商品市场分析。
擅长供需分析、库存跟踪、美元相关性分析和地缘政治影响评估。
覆盖贵金属、能源、农产品等主要商品。"""

    @property
    def expertise(self) -> List[str]:
        return [
            "供需平衡分析",
            "库存数据解读 (EIA, COMEX)",
            "美元指数相关性分析",
            "地缘政治风险评估",
            "期货曲线结构分析 (Contango/Backwardation)",
            "通胀对冲策略",
        ]

    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建商品分析Prompt"""

        price_summary = self._format_price_data(market_data)
        macro_summary = self._format_macro_data(market_data)
        news_summary = self._format_news(news_data)

        prompt = f"""## 大宗商品市场分析任务

### 当前持仓资产
{', '.join(self.symbols)}

### 价格数据摘要
{price_summary}

### 宏观环境
{macro_summary}

### 近期新闻
{news_summary}

### 分析要求
1. 评估各商品的供需状况
2. 分析美元走势对商品的影响
3. 评估地缘政治风险
4. 判断商品作为对冲工具的价值
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
                lines.append(f"- {symbol}: ${price:.2f} ({change:+.2f}%)")

        return "\n".join(lines) if lines else "暂无价格数据"

    def _format_macro_data(self, market_data: Dict[str, Any]) -> str:
        """格式化宏观数据"""
        macro = market_data.get("macro", {})
        lines = [
            f"- 美元指数 (DXY): {macro.get('dxy', 'N/A')}",
            f"- 实际利率: {macro.get('real_rate', 'N/A')}%",
            f"- 通胀预期: {macro.get('inflation_expectation', 'N/A')}%",
            f"- VIX恐慌指数: {macro.get('vix', 'N/A')}",
        ]
        return "\n".join(lines)

    def _format_news(self, news_data: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_data:
            return "暂无相关新闻"

        keywords = ["gold", "oil", "commodity", "opec", "mining", "metal", "energy"]
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
                        "supply_demand": rec.get("supply_demand", "balanced"),
                        "dollar_impact": rec.get("dollar_impact", "neutral"),
                        "geopolitical_risk": rec.get("geopolitical_risk", "low"),
                    },
                    risk_assessment={
                        "volatility": rec.get("volatility", 0.25),
                        "correlation_to_equity": rec.get("correlation_to_equity", -0.2),
                    }
                )
                recommendations.append(recommendation)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # 默认推荐黄金作为避险，但降低置信度
            recommendations.append(ExpertRecommendation(
                asset_class=self.asset_class,
                symbol="GLD",
                action=Action.HOLD,
                confidence=0.3,  # 降低置信度表示不确定性
                target_weight=0.05,  # 降低权重
                reasoning="[PARSE_FAILED] Unable to parse LLM response, defaulting to conservative HOLD gold",
                market_view={"supply_demand": "balanced", "parse_error": True},
                risk_assessment={"volatility": 0.20, "uncertainty": "high"},
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
