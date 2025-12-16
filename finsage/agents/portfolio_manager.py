"""
Portfolio Manager Agent
组合管理智能体 - 负责综合各专家意见并调用对冲工具
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from finsage.agents.base_expert import ExpertReport

logger = logging.getLogger(__name__)


@dataclass
class PortfolioDecision:
    """组合决策结果"""
    timestamp: str
    target_allocation: Dict[str, float]      # 目标配置 {asset: weight}
    trades: List[Dict[str, Any]]             # 交易指令
    hedging_tool_used: str                   # 使用的对冲工具
    reasoning: str                           # 决策理由
    risk_metrics: Dict[str, float]           # 风险指标
    expert_summary: Dict[str, str]           # 各专家意见摘要

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "target_allocation": self.target_allocation,
            "trades": self.trades,
            "hedging_tool_used": self.hedging_tool_used,
            "reasoning": self.reasoning,
            "risk_metrics": self.risk_metrics,
            "expert_summary": self.expert_summary,
        }


class PortfolioManager:
    """
    组合管理智能体

    核心职责:
    1. 收集5位专家的建议
    2. 选择合适的对冲工具
    3. 计算最优资产配置
    4. 生成交易指令
    """

    # 默认资产配置范围
    DEFAULT_ALLOCATION_BOUNDS = {
        "stocks": {"min": 0.30, "max": 0.50, "default": 0.40},
        "bonds": {"min": 0.15, "max": 0.35, "default": 0.25},
        "commodities": {"min": 0.10, "max": 0.25, "default": 0.15},
        "reits": {"min": 0.05, "max": 0.15, "default": 0.10},
        "crypto": {"min": 0.00, "max": 0.10, "default": 0.05},
        "cash": {"min": 0.02, "max": 0.15, "default": 0.05},
    }

    # 工具选择触发条件配置 (v2.1 - 降低门槛增加多样性)
    TOOL_SELECTION_RULES = {
        "cvar_optimization": {
            "description": "CVaR优化 - 尾部风险管理",
            "triggers": {
                "vix_threshold": 18,  # VIX > 18 时触发 (降低自25)
                "drawdown_threshold": -0.03,  # 当前回撤 > 3% 时触发 (降低自5%)
                "bearish_majority": True,  # 多数专家看空时触发
            },
            "priority": 1,
        },
        "dcc_garch": {
            "description": "DCC-GARCH动态相关 - 波动率变化时使用",
            "triggers": {
                "vix_threshold": 15,  # VIX > 15 时倾向使用 (降低自20)
                "volatility_change_threshold": 0.08,  # 波动率变化 > 8% 时触发 (降低自15%)
                "correlation_change_threshold": 0.05,  # 相关性变化 > 5% 时触发 (降低自10%)
            },
            "priority": 2,
        },
        "robust_optimization": {
            "description": "鲁棒优化 - 参数不确定时使用",
            "triggers": {
                "expert_disagreement_threshold": 0.3,  # 专家分歧度 > 30% 时触发 (降低自60%)
                "vix_threshold": 16,  # VIX > 16 时倾向使用 (降低自22)
            },
            "priority": 3,
        },
        "risk_parity": {
            "description": "风险平价 - 均衡风险分配",
            "triggers": {
                "neutral_experts": 2,  # 至少2个专家中性 (降低自3)
                "low_volatility": True,  # 低波动市场
            },
            "priority": 4,
        },
        "minimum_variance": {
            "description": "最小方差 - 稳定市场基准方法",
            "triggers": {
                "vix_threshold": 14,  # VIX < 14 时默认使用 (降低自18)
                "stable_market": True,
            },
            "priority": 5,
        },
        "black_litterman": {
            "description": "Black-Litterman - 结合市场均衡和专家观点",
            "triggers": {
                "bullish_majority": True,  # 多数专家看多时使用
                "moderate_vix": True,  # VIX 14-20 区间
            },
            "priority": 6,
        },
        "mean_variance": {
            "description": "均值方差优化 - 经典马科维茨",
            "triggers": {
                "low_vix": True,  # VIX < 14
                "high_confidence": True,  # 专家信心高
            },
            "priority": 7,
        },
    }

    def __init__(
        self,
        llm_provider: Any,
        hedging_toolkit: Any,
        config: Optional[Dict] = None
    ):
        """
        初始化组合管理器

        Args:
            llm_provider: LLM服务提供者
            hedging_toolkit: 对冲工具箱
            config: 配置参数
        """
        self.llm = llm_provider
        self.toolkit = hedging_toolkit
        self.config = config or {}

        # 配置参数
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.05)
        self.min_trade_value = self.config.get("min_trade_value", 100)
        self.allocation_bounds = self.config.get(
            "allocation_bounds",
            self.DEFAULT_ALLOCATION_BOUNDS
        )

        logger.info("Portfolio Manager initialized")

    def decide(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        current_portfolio: Dict[str, float],
        risk_constraints: Dict[str, float],
    ) -> PortfolioDecision:
        """
        综合各专家意见，做出组合决策

        Args:
            expert_reports: 5位专家的报告
            market_data: 市场数据
            current_portfolio: 当前持仓
            risk_constraints: 风控约束

        Returns:
            PortfolioDecision: 组合决策
        """
        # Step 1: 汇总专家建议
        expert_summary = self._summarize_expert_views(expert_reports)

        # Step 2: 选择对冲工具
        selected_tool = self._select_hedging_tool(
            expert_reports, market_data, risk_constraints
        )

        # Step 3: 计算目标配置
        target_allocation = self._compute_target_allocation(
            expert_reports,
            market_data,
            selected_tool,
            risk_constraints
        )

        # Step 4: 生成交易指令
        trades = self._generate_trades(current_portfolio, target_allocation)

        # Step 5: 计算风险指标
        risk_metrics = self._compute_risk_metrics(target_allocation, market_data)

        # Step 6: 生成决策理由
        reasoning = self._generate_reasoning(
            expert_summary, selected_tool, target_allocation
        )

        decision = PortfolioDecision(
            timestamp=datetime.now().isoformat(),
            target_allocation=target_allocation,
            trades=trades,
            hedging_tool_used=selected_tool,
            reasoning=reasoning,
            risk_metrics=risk_metrics,
            expert_summary=expert_summary,
        )

        logger.info(f"Portfolio decision made: {len(trades)} trades, tool: {selected_tool}")
        return decision

    def _summarize_expert_views(
        self,
        expert_reports: Dict[str, ExpertReport]
    ) -> Dict[str, str]:
        """汇总各专家观点"""
        summary = {}
        for asset_class, report in expert_reports.items():
            view = report.overall_view
            top_picks = [r.symbol for r in report.recommendations[:2]]
            summary[asset_class] = f"{view} ({', '.join(top_picks)})"
        return summary

    def _select_hedging_tool(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> str:
        """
        使用规则引擎 + LLM 选择合适的对冲工具

        决策流程:
        1. 规则引擎预筛选 - 根据市场环境确定候选工具
        2. LLM 最终决策 - 在候选工具中选择最优

        基于:
        - 市场波动率 (VIX)
        - 资产相关性变化
        - 专家观点分歧度
        - 当前回撤水平
        """
        # Step 1: 规则引擎预筛选
        market_conditions = self._analyze_market_conditions(expert_reports, market_data, risk_constraints)
        candidate_tools, rule_reasoning = self._rule_based_tool_preselection(market_conditions)

        logger.info(f"Rule-based candidates: {candidate_tools}, reason: {rule_reasoning}")

        # 获取可用工具（过滤为候选工具）
        available_tools = self.toolkit.list_tools()
        filtered_tools = [t for t in available_tools if t['name'] in candidate_tools]

        # 如果没有匹配的候选工具，使用所有可用工具
        if not filtered_tools:
            filtered_tools = available_tools
            candidate_tools = [t['name'] for t in available_tools]

        # Step 2: LLM 最终决策
        prompt = self._build_tool_selection_prompt(
            expert_reports, market_data, filtered_tools, market_conditions, rule_reasoning
        )

        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": self._get_tool_selection_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度，增加决策稳定性
                max_tokens=500,
            )

            # 解析响应 - 提取 JSON 内容
            json_str = self._extract_json_from_response(response)
            data = json.loads(json_str)
            selected_tool = data.get("tool_name", candidate_tools[0])

            # 验证选择的工具在候选列表中
            if selected_tool not in candidate_tools:
                logger.warning(f"LLM selected {selected_tool} not in candidates, using first candidate")
                selected_tool = candidate_tools[0]

            logger.info(f"Selected hedging tool: {selected_tool}")
            return selected_tool

        except Exception as e:
            logger.warning(f"Tool selection failed, using rule-based default: {e}")
            return candidate_tools[0] if candidate_tools else "minimum_variance"

    def _analyze_market_conditions(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        分析当前市场环境条件

        Returns:
            市场条件字典，包含各种指标
        """
        import numpy as np

        # 1. VIX 波动率指标
        vix = market_data.get("macro", {}).get("vix", 20.0)
        if vix is None:
            vix = market_data.get("vix", 20.0)

        # 2. 专家观点分析
        views = [r.overall_view for r in expert_reports.values()]
        bullish_count = views.count("bullish")
        bearish_count = views.count("bearish")
        neutral_count = views.count("neutral")

        # 专家分歧度 (0-1, 越高分歧越大)
        view_counts = [bullish_count, bearish_count, neutral_count]
        max_agreement = max(view_counts) / len(views) if views else 0
        expert_disagreement = 1 - max_agreement

        # 3. 计算波动率变化 (如果有历史数据)
        volatility_change = 0.0
        returns_data = market_data.get("returns", {})
        if returns_data:
            import pandas as pd
            try:
                returns_df = pd.DataFrame(returns_data)
                if len(returns_df) >= 20:
                    # 比较近5日vs前15日波动率
                    recent_vol = returns_df.iloc[-5:].std().mean()
                    prev_vol = returns_df.iloc[-20:-5].std().mean()
                    if prev_vol > 0:
                        volatility_change = (recent_vol - prev_vol) / prev_vol
            except Exception as e:
                logger.debug(f"Volatility change calculation failed: {e}")

        # 4. 计算相关性变化 (如果有历史数据)
        correlation_change = 0.0
        if returns_data:
            try:
                returns_df = pd.DataFrame(returns_data)
                if len(returns_df) >= 30:
                    # 比较近10日vs前20日平均相关性
                    recent_corr = returns_df.iloc[-10:].corr().values
                    prev_corr = returns_df.iloc[-30:-10].corr().values
                    # 取上三角矩阵的平均值
                    recent_avg = np.mean(recent_corr[np.triu_indices_from(recent_corr, k=1)])
                    prev_avg = np.mean(prev_corr[np.triu_indices_from(prev_corr, k=1)])
                    correlation_change = abs(recent_avg - prev_avg)
            except Exception as e:
                logger.debug(f"Correlation change calculation failed: {e}")

        # 5. 当前回撤水平 (从 risk_constraints 或估算)
        current_drawdown = risk_constraints.get("current_drawdown", 0.0)

        # 6. 市场状态分类
        if vix > 25:
            market_state = "high_volatility"
        elif vix > 20:
            market_state = "elevated_volatility"
        elif vix < 15:
            market_state = "low_volatility"
        else:
            market_state = "normal"

        return {
            "vix": vix,
            "market_state": market_state,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "expert_disagreement": expert_disagreement,
            "volatility_change": volatility_change,
            "correlation_change": correlation_change,
            "current_drawdown": current_drawdown,
            "bearish_majority": bearish_count > bullish_count and bearish_count >= 3,
            "bullish_majority": bullish_count > bearish_count and bullish_count >= 3,
        }

    def _rule_based_tool_preselection(
        self,
        market_conditions: Dict[str, Any],
    ) -> tuple:
        """
        基于规则的工具预筛选

        根据市场条件确定候选工具列表

        Returns:
            (候选工具列表, 选择理由)
        """
        vix = market_conditions["vix"]
        bearish_majority = market_conditions["bearish_majority"]
        current_drawdown = market_conditions["current_drawdown"]
        volatility_change = market_conditions["volatility_change"]
        correlation_change = market_conditions["correlation_change"]
        expert_disagreement = market_conditions["expert_disagreement"]
        market_state = market_conditions["market_state"]

        candidates = []
        reasons = []
        bullish_majority = market_conditions.get("bullish_majority", False)
        bullish_count = market_conditions.get("bullish_count", 0)
        neutral_count = market_conditions.get("neutral_count", 0)

        # 规则1: 风险环境 -> CVaR优化 (门槛降低)
        if vix > 18 or current_drawdown < -0.03 or bearish_majority:
            candidates.append("cvar_optimization")
            if vix > 18:
                reasons.append(f"VIX={vix:.1f}>18,波动上升")
            if current_drawdown < -0.03:
                reasons.append(f"回撤{current_drawdown:.1%}>3%")
            if bearish_majority:
                reasons.append("多数专家看空")

        # 规则2: 动态相关性变化 -> DCC-GARCH (门槛降低)
        if volatility_change > 0.08 or correlation_change > 0.05 or vix > 15:
            candidates.append("dcc_garch")
            if volatility_change > 0.08:
                reasons.append(f"波动率变化{volatility_change:.1%}>8%")
            if correlation_change > 0.05:
                reasons.append(f"相关性变化{correlation_change:.2f}>0.05")
            if vix > 15 and "dcc_garch" not in candidates:
                reasons.append(f"VIX={vix:.1f}>15")

        # 规则3: 专家分歧 -> 鲁棒优化 (门槛大幅降低)
        if expert_disagreement > 0.3:
            candidates.append("robust_optimization")
            reasons.append(f"专家分歧度{expert_disagreement:.1%}>30%")

        # 规则4: 多数看多 -> Black-Litterman (新增)
        if bullish_majority or bullish_count >= 3:
            candidates.append("black_litterman")
            reasons.append(f"看多专家{bullish_count}/5,使用BL模型")

        # 规则5: 风险平价 (门槛降低)
        if neutral_count >= 2 or (14 < vix < 18):
            candidates.append("risk_parity")
            if neutral_count >= 2:
                reasons.append(f"中性专家{neutral_count}/5")
            else:
                reasons.append(f"VIX适中({vix:.1f})")

        # 规则6: 均值方差 - 低波动高信心 (新增)
        if vix < 14 and bullish_count >= 2:
            candidates.append("mean_variance")
            reasons.append(f"低波动(VIX={vix:.1f})+看多共识")

        # 规则7: 最小方差 - 极低波动时
        if vix < 14 and market_state == "low_volatility":
            candidates.append("minimum_variance")
            if not reasons:
                reasons.append(f"极低波动,VIX={vix:.1f}")

        # 默认：多选择策略轮换
        if not candidates:
            # 根据日期周期轮换默认工具
            import hashlib
            from datetime import datetime
            day_hash = int(hashlib.md5(datetime.now().strftime("%Y-%m-%d").encode()).hexdigest()[:8], 16)
            default_tools = ["minimum_variance", "risk_parity", "dcc_garch", "robust_optimization"]
            candidates = [default_tools[day_hash % len(default_tools)]]
            reasons = [f"周期轮换: {candidates[0]}"]

        # 去重并保持顺序
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        return unique_candidates, "; ".join(reasons)

    def _build_tool_selection_prompt(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        available_tools: List[Dict],
        market_conditions: Dict[str, Any] = None,
        rule_reasoning: str = "",
    ) -> str:
        """
        构建工具选择Prompt

        改进版本：提供更详细的市场环境和明确的选择指南
        """
        # 使用预分析的市场条件或计算新的
        if market_conditions is None:
            volatility = market_data.get("volatility", 0.15)
            correlation = market_data.get("avg_correlation", 0.3)
            views = [r.overall_view for r in expert_reports.values()]
            bullish_count = views.count("bullish")
            bearish_count = views.count("bearish")
            neutral_count = views.count("neutral")
            vix = 20.0
            volatility_change = 0.0
            correlation_change = 0.0
            current_drawdown = 0.0
            market_state = "normal"
        else:
            volatility = market_data.get("volatility", 0.15)
            correlation = market_data.get("avg_correlation", 0.3)
            vix = market_conditions.get("vix", 20.0)
            bullish_count = market_conditions.get("bullish_count", 0)
            bearish_count = market_conditions.get("bearish_count", 0)
            neutral_count = market_conditions.get("neutral_count", 0)
            volatility_change = market_conditions.get("volatility_change", 0.0)
            correlation_change = market_conditions.get("correlation_change", 0.0)
            current_drawdown = market_conditions.get("current_drawdown", 0.0)
            market_state = market_conditions.get("market_state", "normal")

        tools_desc = "\n".join([
            f"- **{t['name']}**: {t['description'][:150]}..."
            for t in available_tools
        ])

        # 构建专家观点详情
        expert_details = ""
        for asset_class, report in expert_reports.items():
            expert_details += f"  - {asset_class}: {report.overall_view}\n"

        return f"""## 对冲工具选择任务

### 当前市场环境分析
- **VIX波动率指数**: {vix:.1f} ({market_state})
- **组合波动率**: {volatility:.2%}
- **资产平均相关性**: {correlation:.2f}
- **波动率变化**: {volatility_change:+.1%} (近5日 vs 前15日)
- **相关性变化**: {correlation_change:.3f}
- **当前回撤**: {current_drawdown:.2%}

### 专家观点分布
- 看多 (Bullish): {bullish_count}/5
- 中性 (Neutral): {neutral_count}/5
- 看空 (Bearish): {bearish_count}/5

专家详情:
{expert_details}

### 规则引擎预筛选结果
候选工具已由规则引擎根据市场条件预选:
**预选理由**: {rule_reasoning}

### 候选对冲工具
{tools_desc}

### 工具选择指南
请根据以下原则选择最优工具:

1. **cvar_optimization** (CVaR优化):
   - 适用: VIX>25, 回撤>5%, 或多数专家看空
   - 特点: 控制尾部风险，防止极端损失

2. **dcc_garch** (DCC-GARCH动态相关):
   - 适用: 波动率变化>15%, 相关性变化>0.10, 或VIX>20
   - 特点: 捕捉动态相关性，适应市场变化

3. **robust_optimization** (鲁棒优化):
   - 适用: 专家意见分歧大(>60%)
   - 特点: 对参数不确定性稳健

4. **minimum_variance** (最小方差):
   - 适用: 稳定市场 (VIX<20)
   - 特点: 最小化组合波动，经典稳健方法

5. **risk_parity** (风险平价):
   - 适用: 多数专家中性，均衡配置需求
   - 特点: 各资产风险贡献相等

### 任务
从上述候选工具中选择**一个**最适合当前市场环境的工具。

输出格式 (严格JSON):
{{"tool_name": "选择的工具名称", "reasoning": "简要说明选择理由(中文)"}}
"""

    def _get_tool_selection_system_prompt(self) -> str:
        """工具选择系统Prompt"""
        return """你是一位专业的量化资产配置专家，精通各种组合优化方法。

你的任务是根据市场环境选择最合适的对冲/配置工具。

关键决策原则:
1. 高波动/高风险环境 (VIX>25或回撤>5%) → 优先选择 cvar_optimization
2. 市场动态变化明显 (波动率或相关性显著变化) → 优先选择 dcc_garch
3. 专家意见分歧大 → 考虑 robust_optimization
4. 稳定市场环境 → minimum_variance 或 risk_parity

重要:
- 必须从提供的候选工具中选择
- 规则引擎已预筛选出适合当前环境的候选工具
- 输出必须是有效的JSON格式: {"tool_name": "xxx", "reasoning": "xxx"}
- 不要输出代码块标记，直接输出JSON"""

    def _compute_target_allocation(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        selected_tool: str,
        risk_constraints: Dict[str, float],
    ) -> Dict[str, float]:
        """
        计算目标资产配置

        策略:
        1. 首先用 LLM 根据专家观点动态调整各资产类别权重
        2. 然后用对冲工具优化
        3. 最后应用硬性约束
        """
        import pandas as pd

        # Step 1: LLM 动态调整资产类别权重 (基于专家观点)
        dynamic_class_weights = self._llm_dynamic_class_allocation(expert_reports, market_data)
        logger.info(f"LLM dynamic class weights: {dynamic_class_weights}")

        # 准备收益率数据
        returns_data = market_data.get("returns", {})
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
        else:
            # 没有收益率数据，直接返回 LLM 动态配置
            return self._apply_allocation_bounds(dynamic_class_weights)

        try:
            # Step 2: 调用对冲工具微调权重
            computed_weights = self.toolkit.call(
                tool_name=selected_tool,
                returns=returns_df,
                expert_views=dynamic_class_weights,  # 使用 LLM 动态权重作为初始输入
                constraints=risk_constraints
            )

            # Step 3: 应用配置约束
            final_allocation = self._apply_allocation_bounds(computed_weights)
            return final_allocation

        except Exception as e:
            logger.warning(f"Hedging tool failed, using LLM allocation: {e}")
            return self._apply_allocation_bounds(dynamic_class_weights)

    def _llm_dynamic_class_allocation(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        使用 LLM 根据专家观点动态调整资产类别权重

        考虑因素:
        - 各专家的整体观点 (bullish/bearish/neutral)
        - 专家信心度
        - 市场波动率 (VIX)
        - 配置约束范围

        Returns:
            动态资产类别权重 {"stocks": 0.45, "bonds": 0.20, ...}
        """
        # 提取专家观点摘要
        expert_summary = []
        for asset_class, report in expert_reports.items():
            view = report.overall_view
            confidence = getattr(report, 'confidence', 0.7)
            top_picks = [f"{r.symbol}({r.action.value if hasattr(r.action, 'value') else r.action})"
                         for r in (report.recommendations or [])[:3]]
            expert_summary.append({
                "asset_class": asset_class,
                "view": view,
                "confidence": confidence,
                "top_picks": top_picks,
                "bounds": self.allocation_bounds.get(asset_class, {"min": 0, "max": 0.5, "default": 0.1}),
            })

        # 获取市场环境数据
        vix = market_data.get("macro", {}).get("vix", 20.0)
        market_volatility = "high" if vix > 25 else "low" if vix < 15 else "moderate"

        # 构建 LLM Prompt
        prompt = self._build_dynamic_allocation_prompt(expert_summary, market_volatility, vix)

        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": self._get_dynamic_allocation_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 较低温度保证稳定性
                max_tokens=800,
            )

            # 解析响应 - 提取 JSON 内容
            json_str = self._extract_json_from_response(response)
            data = json.loads(json_str)
            allocation = data.get("allocation", {})

            # 验证并补全缺失的资产类别
            validated_allocation = self._validate_llm_allocation(allocation)
            logger.info(f"LLM allocation reasoning: {data.get('reasoning', 'N/A')[:100]}...")
            return validated_allocation

        except Exception as e:
            logger.warning(f"LLM dynamic allocation failed: {e}, using expert-weighted defaults")
            return self._expert_weighted_default_allocation(expert_reports)

    def _build_dynamic_allocation_prompt(
        self,
        expert_summary: List[Dict],
        market_volatility: str,
        vix: float,
    ) -> str:
        """构建动态配置 Prompt"""
        experts_text = ""
        for exp in expert_summary:
            experts_text += f"""
- {exp['asset_class'].upper()} 专家:
  观点: {exp['view']} (信心: {exp['confidence']:.0%})
  推荐: {', '.join(exp['top_picks'])}
  权重范围: {exp['bounds']['min']:.0%} - {exp['bounds']['max']:.0%} (默认: {exp['bounds']['default']:.0%})
"""

        return f"""## 动态资产配置任务

### 市场环境
- VIX: {vix:.1f} ({market_volatility} volatility)

### 五位专家观点
{experts_text}

### 任务
根据专家观点和市场环境，决定各资产类别的目标权重。

规则:
1. 看多(bullish)的资产类别应增加权重（接近max）
2. 看空(bearish)的资产类别应减少权重（接近min）
3. 中性(neutral)的资产类别保持默认权重
4. 高波动市场应增加bonds和cash
5. 所有权重之和必须等于1.0
6. 必须遵守各类别的min/max约束

输出格式 (JSON):
{{
  "allocation": {{
    "stocks": 0.40,
    "bonds": 0.25,
    "commodities": 0.15,
    "reits": 0.10,
    "crypto": 0.05,
    "cash": 0.05
  }},
  "reasoning": "简要说明配置理由"
}}
"""

    def _get_dynamic_allocation_system_prompt(self) -> str:
        """动态配置系统 Prompt"""
        return """你是一位专业的资产配置经理。
根据市场环境和多位专家的观点，动态调整资产类别权重。
- 看多专家的领域应增加配置
- 看空专家的领域应减少配置
- 高波动市场应防守性配置
输出必须是有效的JSON格式，权重之和必须等于1.0。"""

    def _validate_llm_allocation(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """验证并修正 LLM 输出的配置"""
        validated = {}

        # 确保所有资产类别都有值
        for asset_class, bounds in self.allocation_bounds.items():
            if asset_class in allocation:
                weight = allocation[asset_class]
                # 应用约束
                validated[asset_class] = max(bounds["min"], min(bounds["max"], weight))
            else:
                # 使用默认值
                validated[asset_class] = bounds["default"]

        # 归一化确保和为1 (使用 epsilon 值比较避免浮点精度问题)
        total = sum(validated.values())
        if abs(total) > 1e-10:
            validated = {k: v / total for k, v in validated.items()}

        return validated

    def _expert_weighted_default_allocation(
        self,
        expert_reports: Dict[str, ExpertReport]
    ) -> Dict[str, float]:
        """
        基于专家观点加权的默认配置 (LLM 失败时的后备方案)
        """
        allocation = {}

        # 观点到权重乘数的映射
        view_multipliers = {
            "bullish": 1.3,
            "neutral": 1.0,
            "bearish": 0.7,
        }

        for asset_class, bounds in self.allocation_bounds.items():
            base_weight = bounds["default"]

            if asset_class in expert_reports:
                view = expert_reports[asset_class].overall_view
                multiplier = view_multipliers.get(view, 1.0)
                adjusted = base_weight * multiplier
                allocation[asset_class] = max(bounds["min"], min(bounds["max"], adjusted))
            else:
                allocation[asset_class] = base_weight

        # 归一化
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}

        return allocation

    def _get_default_allocation(self) -> Dict[str, float]:
        """获取默认配置"""
        return {
            asset: bounds["default"]
            for asset, bounds in self.allocation_bounds.items()
        }

    def _apply_allocation_bounds(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """应用配置约束"""
        bounded = {}
        for asset, weight in weights.items():
            if asset in self.allocation_bounds:
                bounds = self.allocation_bounds[asset]
                bounded[asset] = max(bounds["min"], min(bounds["max"], weight))
            else:
                bounded[asset] = weight

        # 归一化
        total = sum(bounded.values())
        if total > 0:
            bounded = {k: v / total for k, v in bounded.items()}

        return bounded

    def _generate_trades(
        self,
        current_portfolio: Dict[str, float],
        target_allocation: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """生成交易指令"""
        trades = []

        for asset, target_weight in target_allocation.items():
            current_weight = current_portfolio.get(asset, 0.0)
            diff = target_weight - current_weight

            if abs(diff) > self.rebalance_threshold:
                action = "BUY" if diff > 0 else "SELL"
                trades.append({
                    "asset": asset,
                    "action": action,
                    "weight_change": abs(diff),
                    "from_weight": current_weight,
                    "to_weight": target_weight,
                })

        return trades

    def _compute_risk_metrics(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        计算风险指标 - 使用真实市场数据

        Args:
            allocation: 目标资产配置权重
            market_data: 市场数据 (包含 returns, covariance 等)

        Returns:
            风险指标字典
        """
        import numpy as np
        import pandas as pd

        try:
            # 获取收益率数据
            returns_data = market_data.get("returns", {})

            if not returns_data:
                # 首日或数据不可用时使用 VIX 估计（这是正常行为）
                logger.debug("No returns data in market_data, using VIX-based volatility estimate")
                vix = market_data.get("macro", {}).get("vix", 20.0)
                market_vol = vix / 100 * np.sqrt(252 / 365)  # 年化并转换
                return {
                    "expected_volatility": round(market_vol, 4),
                    "diversification_ratio": 1.0,
                    "max_drawdown_estimate": round(-market_vol * 2, 4),
                    "data_source": "vix_estimate",
                }

            # 转换为 DataFrame
            if isinstance(returns_data, dict):
                returns_df = pd.DataFrame(returns_data)
            else:
                returns_df = returns_data

            if returns_df.empty:
                logger.warning("Empty returns DataFrame")
                return self._get_default_risk_metrics(market_data)

            # 计算协方差矩阵 (年化)
            cov_matrix = returns_df.cov() * 252

            # 构建权重向量 (匹配可用资产)
            available_assets = [a for a in allocation.keys() if a in returns_df.columns]
            if not available_assets:
                logger.warning("No matching assets between allocation and returns")
                return self._get_default_risk_metrics(market_data)

            weights = np.array([allocation.get(a, 0) for a in available_assets])
            weights = weights / weights.sum() if weights.sum() > 0 else weights

            # 提取匹配的协方差子矩阵
            cov_sub = cov_matrix.loc[available_assets, available_assets].values

            # 1. 组合波动率: sqrt(w' * Cov * w)
            portfolio_variance = np.dot(weights.T, np.dot(cov_sub, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # 2. 分散化比率: 加权平均波动率 / 组合波动率
            individual_vols = np.sqrt(np.diag(cov_sub))
            weighted_avg_vol = np.dot(weights, individual_vols)
            diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0

            # 3. 最大回撤估计 (基于波动率的经验公式)
            # 使用 Cornish-Fisher 扩展或简单的 2-sigma 估计
            max_drawdown_estimate = -portfolio_volatility * 2.0  # 简化: 2倍年化波动率

            # 4. 计算 VaR (95%) 如果有足够数据
            var_95 = None
            if len(returns_df) >= 20:
                portfolio_returns = returns_df[available_assets].dot(weights)
                var_95 = float(np.percentile(portfolio_returns, 5))

            result = {
                "expected_volatility": round(float(portfolio_volatility), 4),
                "diversification_ratio": round(float(diversification_ratio), 4),
                "max_drawdown_estimate": round(float(max_drawdown_estimate), 4),
                "data_source": "historical_returns",
                "assets_used": len(available_assets),
            }

            if var_95 is not None:
                result["var_95_daily"] = round(var_95, 4)

            logger.debug(f"Computed risk metrics: vol={portfolio_volatility:.2%}, div_ratio={diversification_ratio:.2f}")
            return result

        except Exception as e:
            logger.warning(f"Risk metrics computation failed: {e}, using defaults")
            return self._get_default_risk_metrics(market_data)

    def _get_default_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        获取默认风险指标 (基于市场数据估计)

        Args:
            market_data: 市场数据

        Returns:
            默认风险指标
        """
        # 尝试从 VIX 估计市场波动率
        vix = market_data.get("macro", {}).get("vix")
        if vix is None:
            vix = market_data.get("vix", 20.0)

        # VIX 是年化波动率的市场预期 (以百分比表示)
        market_vol = vix / 100 if vix > 1 else vix

        return {
            "expected_volatility": round(market_vol * 0.8, 4),  # 假设组合比市场稍低
            "diversification_ratio": 1.3,  # 假设适度分散
            "max_drawdown_estimate": round(-market_vol * 1.8, 4),
            "data_source": "default_estimate",
        }

    def _generate_reasoning(
        self,
        expert_summary: Dict[str, str],
        selected_tool: str,
        target_allocation: Dict[str, float],
    ) -> str:
        """生成决策理由"""
        top_allocations = sorted(
            target_allocation.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        reasoning = f"基于5位专家的综合意见，使用{selected_tool}方法进行配置。"
        reasoning += f"重点配置: {', '.join([f'{a}({w:.1%})' for a, w in top_allocations])}。"

        # 添加专家观点
        bullish = [k for k, v in expert_summary.items() if "bullish" in v]
        if bullish:
            reasoning += f"看多领域: {', '.join(bullish)}。"

        return reasoning

    def _extract_json_from_response(self, response: str) -> str:
        """
        从 LLM 响应中提取 JSON 内容

        处理以下情况:
        1. 纯 JSON 字符串
        2. Markdown 代码块 (```json ... ```)
        3. 普通代码块 (``` ... ```)
        4. 带有额外文本的响应

        Args:
            response: LLM 响应字符串

        Returns:
            提取的 JSON 字符串
        """
        import re

        if not response:
            raise ValueError("Empty response from LLM")

        # 去除首尾空白
        response = response.strip()

        # 尝试方法1: 直接解析 (纯 JSON)
        if response.startswith("{") or response.startswith("["):
            # 找到匹配的结束括号
            try:
                # 尝试直接解析
                json.loads(response)
                return response
            except json.JSONDecodeError:
                pass

        # 尝试方法2: 提取 ```json ... ``` 代码块
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_block_match:
            return json_block_match.group(1).strip()

        # 尝试方法3: 提取 ``` ... ``` 代码块
        code_block_match = re.search(r'```\s*([\s\S]*?)\s*```', response)
        if code_block_match:
            content = code_block_match.group(1).strip()
            # 检查是否是有效 JSON
            if content.startswith("{") or content.startswith("["):
                return content

        # 尝试方法4: 使用括号平衡查找 JSON 对象 (避免贪婪匹配问题)
        potential_json = self._find_balanced_json(response, '{', '}')
        if potential_json:
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        # 尝试方法5: 使用括号平衡查找 JSON 数组
        potential_json = self._find_balanced_json(response, '[', ']')
        if potential_json:
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        # 无法提取 JSON，返回原始响应让调用者处理
        logger.warning(f"Could not extract JSON from response: {response[:200]}...")
        raise ValueError(f"Cannot extract JSON from LLM response")

    def _find_balanced_json(
        self,
        text: str,
        open_char: str,
        close_char: str
    ) -> Optional[str]:
        """
        使用括号平衡算法查找有效的 JSON 字符串

        避免贪婪正则表达式匹配问题，正确处理嵌套结构。

        Args:
            text: 要搜索的文本
            open_char: 开始字符 ('{' 或 '[')
            close_char: 结束字符 ('}' 或 ']')

        Returns:
            找到的 JSON 字符串，或 None
        """
        start_idx = text.find(open_char)
        if start_idx == -1:
            return None

        count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            # 处理字符串内的转义
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            # 处理字符串边界
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            # 只在字符串外计数括号
            if not in_string:
                if char == open_char:
                    count += 1
                elif char == close_char:
                    count -= 1
                    if count == 0:
                        # 找到平衡的 JSON
                        return text[start_idx:i + 1]

        return None
