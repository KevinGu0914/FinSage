"""
Crypto Expert Agent
加密货币投资专家
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


class CryptoExpert(BaseExpert):
    """
    加密货币投资专家

    专注领域:
    - 主流币 (BTC, ETH)
    - Layer1 (SOL, AVAX)
    - Crypto ETF (BITO, GBTC)
    - 链上分析
    """

    @classmethod
    def _get_default_symbols(cls) -> List[str]:
        """从 config.py 获取默认加密货币符号列表"""
        return AssetConfig().default_universe.get("crypto", [
            "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "BITO"
        ])

    def __init__(
        self,
        llm_provider: Any,
        symbols: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        # 加密货币默认配置更保守
        default_config = {
            "max_single_weight": 0.05,  # 单一加密资产最大5%
            "max_class_weight": 0.10,    # 加密类别最大10%
        }
        if config:
            default_config.update(config)

        super().__init__(
            llm_provider=llm_provider,
            asset_class="crypto",
            symbols=symbols or self._get_default_symbols(),
            config=default_config
        )

    @property
    def name(self) -> str:
        return "Crypto Expert"

    @property
    def description(self) -> str:
        return """加密货币投资专家，专注于数字资产市场分析。
擅长链上数据分析、周期判断、叙事追踪和监管环境评估。
覆盖BTC、ETH等主流币种及相关ETF。
注意：加密货币波动性高，建议配置比例控制在10%以内。"""

    @property
    def expertise(self) -> List[str]:
        return [
            "链上数据分析 (Active Addresses, NVT, MVRV)",
            "交易所流入/流出分析",
            "期货资金费率和未平仓合约",
            "减半周期和供给分析",
            "监管环境评估",
            "宏观叙事追踪",
        ]

    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建加密货币分析Prompt"""

        price_summary = self._format_price_data(market_data)
        onchain_summary = self._format_onchain_data(market_data)
        news_summary = self._format_news(news_data)

        prompt = f"""## 加密货币市场分析任务

### 当前持仓资产
{', '.join(self.symbols)}

### 价格数据摘要
{price_summary}

### 链上数据
{onchain_summary}

### 近期新闻
{news_summary}

### 风险提示
加密货币波动性极高，建议总配置不超过组合的10%。

### 分析要求
1. 评估市场周期阶段
2. 分析链上活跃度和资金流向
3. 评估监管环境变化
4. 判断短期趋势和关键价位
5. 给出保守的交易建议和置信度

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
                change_7d = data.get("change_7d", 0)
                lines.append(f"- {symbol}: ${price:,.2f} (24h: {change:+.2f}%, 7d: {change_7d:+.2f}%)")

        return "\n".join(lines) if lines else "暂无价格数据"

    def _format_onchain_data(self, market_data: Dict[str, Any]) -> str:
        """格式化链上数据"""
        onchain = market_data.get("onchain", {})
        if not onchain:
            return "暂无链上数据"

        lines = [
            f"- BTC Active Addresses: {onchain.get('btc_active_addresses', 'N/A')}",
            f"- BTC Exchange Netflow: {onchain.get('btc_exchange_netflow', 'N/A')}",
            f"- Funding Rate: {onchain.get('funding_rate', 'N/A')}%",
            f"- Open Interest: ${onchain.get('open_interest', 'N/A')}B",
            f"- Fear & Greed Index: {onchain.get('fear_greed', 'N/A')}",
        ]
        return "\n".join(lines)

    def _format_news(self, news_data: List[Dict]) -> str:
        """格式化新闻数据"""
        if not news_data:
            return "暂无相关新闻"

        keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "sec", "etf", "halving"]
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
            json_str = response
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    inner_parts = parts[1].split("```")
                    if len(inner_parts) > 0:
                        json_str = inner_parts[0]
            elif "```" in response:
                parts = response.split("```")
                if len(parts) > 1:
                    json_str = parts[1]

            data = json.loads(json_str.strip())

            for rec in data.get("recommendations", []):
                action_str = rec.get("action", "HOLD")
                action = self._parse_action(action_str)

                # 强制限制加密货币权重
                target_weight = min(
                    float(rec.get("target_weight", 0.0)),
                    self.config.get("max_single_weight", 0.05)
                )

                recommendation = ExpertRecommendation(
                    asset_class=self.asset_class,
                    symbol=rec.get("symbol", ""),
                    action=action,
                    confidence=float(rec.get("confidence", 0.5)),
                    target_weight=target_weight,
                    reasoning=rec.get("reasoning", ""),
                    market_view={
                        "cycle_phase": rec.get("cycle_phase", "uncertain"),
                        "sentiment": rec.get("sentiment", "neutral"),
                        "regulatory_risk": rec.get("regulatory_risk", "medium"),
                    },
                    risk_assessment={
                        "volatility": rec.get("volatility", 0.6),
                        "liquidity_risk": rec.get("liquidity_risk", 0.2),
                    }
                )
                recommendations.append(recommendation)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # 保守默认：不建议加仓，进一步降低置信度和权重
            recommendations.append(ExpertRecommendation(
                asset_class=self.asset_class,
                symbol="BTC-USD",
                action=Action.HOLD,
                confidence=0.2,  # 非常低的置信度
                target_weight=0.01,  # 极小权重
                reasoning="[PARSE_FAILED] Unable to parse LLM response, defaulting to minimal HOLD",
                market_view={"cycle_phase": "uncertain", "parse_error": True},
                risk_assessment={"volatility": 0.7, "uncertainty": "very_high"},
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
