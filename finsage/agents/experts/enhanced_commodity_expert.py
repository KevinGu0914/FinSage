"""
Enhanced Commodity Expert Agent
增强版大宗商品投资专家

基于 FMP 现有数据的增强版本:
- COT (交易商持仓报告) 数据分析
- 商品市场环境识别 (美元环境、通胀对冲需求、投机者仓位)
- 能源经济事件追踪 (EIA、OPEC、库存报告)
- 实际利率与商品相关性分析
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


class EnhancedCommodityExpert(BaseExpert):
    """
    增强版大宗商品投资专家

    专注领域:
    - 贵金属 (GLD, SLV, IAU) - 通胀对冲、避险需求
    - 能源 (USO, UNG, XLE) - 供需、地缘政治
    - 农产品 (DBA) - 天气、供给冲击
    - 工业金属 (COPX) - 经济周期、中国需求

    增强功能:
    - COT 仓位分析 (商业 vs 投机持仓)
    - 市场环境识别
    - 能源经济事件整合
    """

    @classmethod
    def _get_default_symbols(cls) -> List[str]:
        """从 config.py 获取默认大宗商品符号列表"""
        return AssetConfig().default_universe.get("commodities", [
            "GLD", "SLV", "IAU", "USO", "UNG", "DBA", "COPX", "XLE"
        ])

    # COT 数据符号映射
    COT_MAPPING = {
        "GLD": "gold",
        "SLV": "silver",
        "IAU": "gold",
        "USO": "oil",
        "UNG": "natural_gas",
        "COPX": "copper",
        "DBA": "wheat",  # 农产品 ETF 用小麦作为代理
    }

    def __init__(
        self,
        llm_provider: Any,
        symbols: Optional[List[str]] = None,
        config: Optional[Dict] = None,
        macro_loader: Optional[Any] = None,
    ):
        super().__init__(
            llm_provider=llm_provider,
            asset_class="commodities",
            symbols=symbols or self._get_default_symbols(),
            config=config
        )
        self.macro_loader = macro_loader

    @property
    def name(self) -> str:
        return "Enhanced Commodity Expert"

    @property
    def description(self) -> str:
        return """增强版大宗商品投资专家，专注于商品市场深度分析。
集成 COT 持仓数据、市场环境识别、能源经济事件追踪。
擅长供需分析、投机者仓位解读、美元相关性分析和地缘政治影响评估。"""

    @property
    def expertise(self) -> List[str]:
        return [
            "COT持仓报告解读 (商业vs投机)",
            "供需平衡分析",
            "投机者仓位情绪分析",
            "美元指数相关性分析",
            "实际利率对贵金属影响",
            "地缘政治风险评估",
            "期货曲线结构分析 (Contango/Backwardation)",
            "通胀对冲策略",
            "能源经济事件解读 (EIA/OPEC)",
        ]

    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建增强版商品分析Prompt"""

        price_summary = self._format_price_data(market_data)
        macro_summary = self._format_macro_data(market_data)
        cot_summary = self._format_cot_data(market_data)
        regime_summary = self._format_regime_data(market_data)
        events_summary = self._format_economic_events(market_data)
        news_summary = self._format_news(news_data)

        prompt = f"""## 增强版大宗商品市场分析任务

### 当前持仓资产
{', '.join(self.symbols)}

### 价格数据摘要
{price_summary}

### 宏观环境
{macro_summary}

### COT 持仓分析 (交易商持仓报告)
{cot_summary}

### 市场环境判断
{regime_summary}

### 近期能源经济事件
{events_summary}

### 近期新闻
{news_summary}

### 分析要求
1. **COT 仓位解读**: 分析商业与投机持仓的变化趋势，判断市场情绪
2. **供需状况评估**: 结合库存数据和产能利用率
3. **美元与实际利率影响**: 分析美元走势和实际利率对贵金属的影响
4. **市场环境判断**: 根据环境判断给出配置倾向
5. **地缘政治风险**: 评估OPEC决策、地缘冲突等影响
6. **交易建议**: 给出具体的交易建议、目标权重和置信度

### 输出格式要求
请以JSON格式输出，包含以下字段:
```json
{{
    "market_regime_analysis": "对当前商品市场环境的总体判断",
    "cot_interpretation": "COT数据的解读和信号",
    "recommendations": [
        {{
            "symbol": "资产代码",
            "action": "动作 (SELL_100%/SELL_75%/SELL_50%/SELL_25%/HOLD/BUY_25%/BUY_50%/BUY_75%/BUY_100%)",
            "confidence": 0.0-1.0,
            "target_weight": 0.0-1.0,
            "reasoning": "具体理由",
            "supply_demand": "tight/balanced/oversupply",
            "dollar_impact": "bullish/neutral/bearish",
            "cot_signal": "bullish/neutral/bearish",
            "geopolitical_risk": "high/medium/low",
            "volatility": 0.0-1.0,
            "correlation_to_equity": -1.0 到 1.0
        }}
    ]
}}
```
"""
        return prompt

    def _format_price_data(self, market_data: Dict[str, Any]) -> str:
        """格式化价格数据"""
        if not market_data:
            return "暂无价格数据"

        lines = []

        # 从 commodity_data 获取商品价格
        commodity_data = market_data.get("commodity_data", {})
        commodities = commodity_data.get("commodities", {})

        if commodities:
            lines.append("**期货商品价格:**")
            for name, data in commodities.items():
                if isinstance(data, dict):
                    price = data.get("price", "N/A")
                    change = data.get("change_pct", 0)
                    lines.append(f"- {name}: ${price:.2f} ({change:+.2f}%)")

        # ETF 价格
        lines.append("\n**商品ETF价格:**")
        for symbol, data in market_data.items():
            if symbol in self.symbols and isinstance(data, dict):
                price = data.get("close", data.get("price", "N/A"))
                change = data.get("change_pct", 0)
                if isinstance(price, (int, float)):
                    lines.append(f"- {symbol}: ${price:.2f} ({change:+.2f}%)")

        return "\n".join(lines) if lines else "暂无价格数据"

    def _format_macro_data(self, market_data: Dict[str, Any]) -> str:
        """格式化宏观数据"""
        # 优先从 commodity_data 获取
        commodity_data = market_data.get("commodity_data", {})

        if commodity_data:
            dxy = commodity_data.get("dxy", "N/A")
            vix = commodity_data.get("vix", "N/A")
            real_rate = commodity_data.get("real_rate", "N/A")
        else:
            macro = market_data.get("macro", {})
            dxy = macro.get("dxy", "N/A")
            vix = macro.get("vix", "N/A")
            real_rate = macro.get("real_rate", "N/A")

        lines = [
            f"- 美元指数 (DXY): {dxy}",
            f"- 实际利率 (10Y-通胀): {real_rate}%" if isinstance(real_rate, (int, float)) else f"- 实际利率: {real_rate}",
            f"- VIX恐慌指数: {vix}",
        ]

        # 美元对商品的影响说明
        if isinstance(dxy, (int, float)):
            if dxy > 105:
                lines.append("  → 美元走强，商品价格承压")
            elif dxy < 100:
                lines.append("  → 美元走弱，支撑商品价格")

        # 实际利率对黄金的影响
        if isinstance(real_rate, (int, float)):
            if real_rate > 2.0:
                lines.append("  → 实际利率高，黄金机会成本高")
            elif real_rate < 0:
                lines.append("  → 负实际利率，利好黄金")

        return "\n".join(lines)

    def _format_cot_data(self, market_data: Dict[str, Any]) -> str:
        """格式化 COT 持仓数据"""
        commodity_data = market_data.get("commodity_data", {})
        cot_data = commodity_data.get("cot_data", {})

        if not cot_data:
            return "暂无 COT 持仓数据"

        lines = ["**主要商品 COT 持仓分析:**"]

        for commodity, data in cot_data.items():
            if not data or not isinstance(data, dict):
                continue

            commercial_net = data.get("commercial_net", 0)
            speculator_net = data.get("speculator_net", 0)
            speculator_sentiment = data.get("speculator_sentiment", "N/A")
            report_date = data.get("report_date", "")

            # 生成解读
            sentiment_emoji = {
                "bullish": "🟢",
                "bearish": "🔴",
                "neutral": "🟡"
            }.get(speculator_sentiment, "⚪")

            lines.append(f"\n**{commodity.upper()}** ({report_date})")
            lines.append(f"  - 商业持仓净头寸: {commercial_net:,.0f}")
            lines.append(f"  - 投机者净头寸: {speculator_net:,.0f}")
            lines.append(f"  - 投机者情绪: {sentiment_emoji} {speculator_sentiment}")

            # COT 信号解读
            if speculator_net > 0 and commercial_net < 0:
                lines.append("  → 投机者做多 vs 商业做空 (可能顶部信号)")
            elif speculator_net < 0 and commercial_net > 0:
                lines.append("  → 投机者做空 vs 商业做多 (可能底部信号)")

        return "\n".join(lines)

    def _format_regime_data(self, market_data: Dict[str, Any]) -> str:
        """格式化市场环境数据"""
        commodity_data = market_data.get("commodity_data", {})
        regime = commodity_data.get("regime", {})

        if not regime:
            return "暂无市场环境判断"

        lines = ["**商品市场环境判断:**"]

        dollar_env = regime.get("dollar_environment", "unknown")
        inflation_hedge = regime.get("inflation_hedge_demand", "unknown")
        speculator_pos = regime.get("speculator_positioning", "unknown")
        overall_bias = regime.get("overall_bias", "unknown")

        # 美元环境
        dollar_emoji = {"strong": "💪", "weak": "📉", "neutral": "↔️"}.get(dollar_env, "❓")
        lines.append(f"- 美元环境: {dollar_emoji} {dollar_env}")

        # 通胀对冲需求
        inflation_emoji = {"high": "🔥", "low": "❄️", "moderate": "🌤️"}.get(inflation_hedge, "❓")
        lines.append(f"- 通胀对冲需求: {inflation_emoji} {inflation_hedge}")

        # 投机者仓位
        spec_emoji = {"net_long": "📈", "net_short": "📉", "mixed": "🔀"}.get(speculator_pos, "❓")
        lines.append(f"- 投机者仓位: {spec_emoji} {speculator_pos}")

        # 总体偏向
        bias_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(overall_bias, "⚪")
        lines.append(f"- 总体偏向: {bias_emoji} {overall_bias}")

        # 策略建议
        lines.append("\n**环境对应策略:**")
        if overall_bias == "bullish":
            lines.append("  → 增加商品配置，尤其是贵金属和能源")
        elif overall_bias == "bearish":
            lines.append("  → 减少商品配置，保持防御")
        else:
            lines.append("  → 维持均衡配置，关注结构性机会")

        return "\n".join(lines)

    def _format_economic_events(self, market_data: Dict[str, Any]) -> str:
        """格式化经济事件"""
        commodity_data = market_data.get("commodity_data", {})
        events = commodity_data.get("economic_events", [])

        if not events:
            return "暂无近期能源经济事件"

        lines = ["**近期能源经济事件:**"]

        for event in events[:5]:  # 只显示前5个
            date = event.get("date", "")
            name = event.get("event", "")
            impact = event.get("impact", "low")

            impact_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(impact, "⚪")
            lines.append(f"- [{date}] {impact_emoji} {name}")

        return "\n".join(lines)

    def _format_news(self, news_data: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_data:
            return "暂无相关新闻"

        keywords = [
            "gold", "oil", "commodity", "opec", "mining", "metal", "energy",
            "copper", "silver", "natural gas", "crude", "eia", "inventory",
            "inflation", "hedge", "precious"
        ]
        relevant_news = []

        for news in news_data[:15]:
            title = news.get("title", "").lower()
            if any(kw in title for kw in keywords):
                sentiment = news.get("sentiment", "neutral")
                sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(sentiment, "⚪")
                relevant_news.append(f"- {sentiment_emoji} {news.get('title', '')}")

        return "\n".join(relevant_news[:8]) if relevant_news else "暂无相关新闻"

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
                        "cot_signal": rec.get("cot_signal", "neutral"),
                        "geopolitical_risk": rec.get("geopolitical_risk", "low"),
                        "market_regime": data.get("market_regime_analysis", ""),
                        "cot_interpretation": data.get("cot_interpretation", ""),
                    },
                    risk_assessment={
                        "volatility": rec.get("volatility", 0.25),
                        "correlation_to_equity": rec.get("correlation_to_equity", -0.2),
                    }
                )
                recommendations.append(recommendation)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # 默认推荐黄金作为避险
            recommendations.append(ExpertRecommendation(
                asset_class=self.asset_class,
                symbol="GLD",
                action=Action.HOLD,
                confidence=0.3,
                target_weight=0.05,
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

    def get_enhanced_market_data(self) -> Dict[str, Any]:
        """
        获取增强版市场数据

        如果有 macro_loader，使用它来获取 COT 和商品环境数据
        """
        if self.macro_loader is None:
            return {}

        try:
            commodity_data = self.macro_loader.get_commodity_expert_data()
            return {"commodity_data": commodity_data}
        except Exception as e:
            logger.warning(f"Failed to get enhanced commodity data: {e}")
            return {}
