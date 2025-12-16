"""
Base Expert Agent Class
所有资产类别专家的基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class Action(Enum):
    """
    13-Action Trading Space (支持做空)

    动作空间:
    - SHORT_100% ~ SHORT_25%: 做空 (卖空借入的股票)
    - SELL_100% ~ SELL_25%: 减持或平仓多头
    - HOLD: 维持现有仓位
    - BUY_25% ~ BUY_100%: 买入或加仓多头
    """
    # 做空动作 (开空仓或加空仓)
    SHORT_100 = "SHORT_100%"
    SHORT_75 = "SHORT_75%"
    SHORT_50 = "SHORT_50%"
    SHORT_25 = "SHORT_25%"
    # 卖出/减持动作
    SELL_100 = "SELL_100%"
    SELL_75 = "SELL_75%"
    SELL_50 = "SELL_50%"
    SELL_25 = "SELL_25%"
    # 持有
    HOLD = "HOLD"
    # 买入/加仓动作
    BUY_25 = "BUY_25%"
    BUY_50 = "BUY_50%"
    BUY_75 = "BUY_75%"
    BUY_100 = "BUY_100%"


@dataclass
class ExpertRecommendation:
    """专家建议数据结构"""
    asset_class: str                    # 资产类别
    symbol: str                         # 资产代码
    action: Action                      # 建议动作
    confidence: float                   # 置信度 [0, 1]
    target_weight: float                # 建议权重
    reasoning: str                      # 决策理由
    market_view: Dict[str, Any]         # 市场观点
    risk_assessment: Dict[str, float]   # 风险评估

    def to_dict(self) -> Dict:
        return {
            "asset_class": self.asset_class,
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": self.confidence,
            "target_weight": self.target_weight,
            "reasoning": self.reasoning,
            "market_view": self.market_view,
            "risk_assessment": self.risk_assessment,
        }


@dataclass
class ExpertReport:
    """专家完整报告"""
    expert_name: str
    asset_class: str
    timestamp: str
    recommendations: List[ExpertRecommendation]
    overall_view: str                   # bullish/bearish/neutral
    sector_allocation: Dict[str, float] # 细分配置建议
    key_factors: List[str]              # 关键影响因素

    def to_dict(self) -> Dict:
        return {
            "expert_name": self.expert_name,
            "asset_class": self.asset_class,
            "timestamp": self.timestamp,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "overall_view": self.overall_view,
            "sector_allocation": self.sector_allocation,
            "key_factors": self.key_factors,
        }


class BaseExpert(ABC):
    """
    专家Agent基类

    每个专家负责特定资产类别的分析和建议:
    - Stock Expert: 股票
    - Bond Expert: 债券
    - Commodity Expert: 大宗商品
    - REITs Expert: 房地产投资信托
    - Crypto Expert: 加密货币
    """

    def __init__(
        self,
        llm_provider: Any,
        asset_class: str,
        symbols: List[str],
        config: Optional[Dict] = None
    ):
        """
        初始化专家Agent

        Args:
            llm_provider: LLM服务提供者
            asset_class: 资产类别名称
            symbols: 该专家负责的资产列表
            config: 专家配置参数
        """
        self.llm = llm_provider
        self.asset_class = asset_class
        self.symbols = symbols
        self.config = config or {}

        # 默认配置
        self.max_single_weight = self.config.get("max_single_weight", 0.15)
        self.min_confidence = self.config.get("min_confidence", 0.5)

        logger.info(f"Initialized {self.name} for {asset_class} with symbols: {symbols}")

    def update_symbols(self, new_symbols: List[str]) -> None:
        """
        动态更新资产候选池

        Args:
            new_symbols: 新的资产符号列表
        """
        old_symbols = self.symbols
        self.symbols = new_symbols
        logger.info(f"{self.name} symbols updated: {len(old_symbols)} -> {len(new_symbols)} symbols")

    @property
    @abstractmethod
    def name(self) -> str:
        """专家名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """专家描述 - 用于Prompt"""
        pass

    @property
    @abstractmethod
    def expertise(self) -> List[str]:
        """专业领域列表"""
        pass

    @abstractmethod
    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        technical_indicators: Dict[str, Any],
    ) -> str:
        """构建分析Prompt"""
        pass

    @abstractmethod
    def _parse_llm_response(self, response: str) -> List[ExpertRecommendation]:
        """解析LLM响应"""
        pass

    def analyze(
        self,
        market_data: Dict[str, Any],
        news_data: Optional[List[Dict]] = None,
        technical_indicators: Optional[Dict[str, Any]] = None,
        macro_data: Optional[Dict[str, Any]] = None,
    ) -> ExpertReport:
        """
        执行分析并生成报告

        Args:
            market_data: 市场数据 (价格, 成交量等)
            news_data: 新闻数据
            technical_indicators: 技术指标
            macro_data: 宏观经济数据

        Returns:
            ExpertReport: 专家报告
        """
        from datetime import datetime

        # 构建Prompt
        prompt = self._build_analysis_prompt(
            market_data=market_data,
            news_data=news_data or [],
            technical_indicators=technical_indicators or {},
        )

        # 调用LLM
        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000,  # Increased from 2000 to handle large stock universes (27+ symbols)
            )

            # 解析响应
            recommendations = self._parse_llm_response(response)

            # 构建报告
            report = ExpertReport(
                expert_name=self.name,
                asset_class=self.asset_class,
                timestamp=datetime.now().isoformat(),
                recommendations=recommendations,
                overall_view=self._determine_overall_view(recommendations),
                sector_allocation=self._calculate_sector_allocation(recommendations),
                key_factors=self._extract_key_factors(response),
            )

            logger.info(f"{self.name} generated report with {len(recommendations)} recommendations")
            return report

        except Exception as e:
            logger.error(f"{self.name} analysis failed: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """获取系统Prompt"""
        return f"""你是一位专业的{self.asset_class}投资专家。

## 你的专业领域
{chr(10).join(f'- {e}' for e in self.expertise)}

## 你的职责
1. 分析提供的市场数据和新闻
2. 评估各资产的投资价值
3. 给出具体的交易建议和置信度
4. 解释你的决策逻辑

## 交易动作说明
- BUY_25%~100%: 买入或加仓多头
- HOLD: 维持现有仓位
- SELL_25%~100%: 减持或平仓多头
- SHORT_25%~100%: 做空 (当预期价格下跌时使用)

## 做空时机 (重要!)
你必须在以下情况积极使用SHORT动作，而不是仅仅使用SELL:
1. 强烈看空信号 (如利率上升对债券的负面影响，经济衰退预期)
2. 技术面显示明确下跌趋势 (如跌破关键支撑位，MACD死叉)
3. 基本面恶化 (盈利预警，行业下行周期)
4. 估值过高且有回调风险
5. 市场情绪极度乐观时的逆向操作
6. 对冲组合风险的需要

注意: SHORT是主动做空获利，SELL是减持已有多头仓位。
当你认为某资产会下跌时，应该使用SHORT而非仅SELL。

## 输出格式
请以JSON格式输出你的分析结果:
{{
    "overall_view": "bullish/bearish/neutral",
    "recommendations": [
        {{
            "symbol": "资产代码",
            "action": "BUY_25%/BUY_50%/BUY_75%/BUY_100%/HOLD/SELL_25%/SELL_50%/SELL_75%/SELL_100%/SHORT_25%/SHORT_50%/SHORT_75%/SHORT_100%",
            "confidence": 0.0-1.0,
            "target_weight": 0.0-1.0,
            "reasoning": "决策理由",
            "risk_level": "low/medium/high"
        }}
    ],
    "key_factors": ["关键因素1", "关键因素2"],
    "market_analysis": "市场分析总结"
}}
"""

    def _determine_overall_view(self, recommendations: List[ExpertRecommendation]) -> str:
        """根据建议确定整体观点"""
        if not recommendations:
            return "neutral"

        buy_weight = sum(
            r.confidence for r in recommendations
            if "BUY" in r.action.value
        )
        # SELL + SHORT 都算做空观点
        sell_weight = sum(
            r.confidence for r in recommendations
            if "SELL" in r.action.value or "SHORT" in r.action.value
        )

        if buy_weight > sell_weight * 1.5:
            return "bullish"
        elif sell_weight > buy_weight * 1.5:
            return "bearish"
        return "neutral"

    def _calculate_sector_allocation(
        self,
        recommendations: List[ExpertRecommendation]
    ) -> Dict[str, float]:
        """计算细分配置"""
        allocation = {}
        total_weight = sum(r.target_weight for r in recommendations)

        if total_weight > 0:
            for r in recommendations:
                allocation[r.symbol] = r.target_weight / total_weight

        return allocation

    def _extract_key_factors(self, response: str) -> List[str]:
        """提取关键因素"""
        try:
            data = json.loads(response)
            return data.get("key_factors", [])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return []
