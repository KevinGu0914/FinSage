"""
Position Sizing Agent
仓位规模智能体 - 负责确定每个资产的具体仓位大小
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np

from finsage.agents.base_expert import ExpertReport

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingDecision:
    """仓位决策结果"""
    timestamp: str
    position_sizes: Dict[str, float]  # {asset: position_size}
    sizing_method: str                # 使用的仓位方法
    reasoning: str                    # 决策理由
    risk_contribution: Dict[str, float]  # 每个资产的风险贡献

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "position_sizes": self.position_sizes,
            "sizing_method": self.sizing_method,
            "reasoning": self.reasoning,
            "risk_contribution": self.risk_contribution,
        }


class PositionSizingAgent:
    """
    仓位规模智能体

    核心职责:
    1. 根据风险预算确定仓位大小
    2. 实现多种仓位方法 (等权, 风险平价, Kelly准则等)
    3. 考虑流动性和交易成本约束
    """

    # 仓位方法
    SIZING_METHODS = {
        "equal_weight": "等权配置",
        "risk_parity": "风险平价",
        "kelly": "Kelly准则",
        "volatility_target": "波动率目标",
        "max_sharpe": "最大夏普",
    }

    def __init__(
        self,
        llm_provider: Any,
        config: Optional[Dict] = None
    ):
        """
        初始化仓位规模智能体

        Args:
            llm_provider: LLM服务提供者
            config: 配置参数
        """
        self.llm = llm_provider
        self.config = config or {}

        # 配置参数
        self.max_position_size = self.config.get("max_position_size", 0.15)
        self.min_position_size = self.config.get("min_position_size", 0.01)
        self.target_volatility = self.config.get("target_volatility", 0.12)

        logger.info("Position Sizing Agent initialized")

    def analyze(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
        portfolio_value: float,
    ) -> PositionSizingDecision:
        """
        分析并确定仓位大小

        Args:
            target_allocation: 目标资产类别配置 (来自PM)
            market_data: 市场数据
            risk_constraints: 风控约束
            portfolio_value: 组合总价值

        Returns:
            PositionSizingDecision: 仓位决策
        """
        # Step 1: 选择仓位方法
        sizing_method = self._select_sizing_method(market_data, risk_constraints)

        # Step 2: 计算初步仓位
        initial_sizes = self._compute_position_sizes(
            target_allocation, market_data, sizing_method
        )

        # Step 3: 应用风险约束
        constrained_sizes = self._apply_constraints(
            initial_sizes, risk_constraints, market_data
        )

        # Step 4: 计算风险贡献
        risk_contribution = self._compute_risk_contribution(
            constrained_sizes, market_data
        )

        # Step 5: 生成决策理由
        reasoning = self._generate_reasoning(
            sizing_method, constrained_sizes, risk_contribution
        )

        decision = PositionSizingDecision(
            timestamp=datetime.now().isoformat(),
            position_sizes=constrained_sizes,
            sizing_method=sizing_method,
            reasoning=reasoning,
            risk_contribution=risk_contribution,
        )

        logger.info(f"Position sizing decision: {sizing_method}, {len(constrained_sizes)} positions")
        return decision

    def _select_sizing_method(
        self,
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> str:
        """
        使用 LLM 选择最佳仓位方法
        """
        # 获取市场指标
        vix = market_data.get("macro", {}).get("vix", 20.0)
        market_vol = "high" if vix > 25 else "low" if vix < 15 else "moderate"

        prompt = f"""## 仓位方法选择任务

### 当前市场环境
- VIX: {vix:.1f} ({market_vol} volatility)
- 目标波动率: {risk_constraints.get('target_volatility', 0.12):.1%}
- 最大回撤容忍: {risk_constraints.get('max_drawdown', 0.15):.1%}

### 可用方法
{chr(10).join([f'- {k}: {v}' for k, v in self.SIZING_METHODS.items()])}

### 任务
选择最适合当前市场的仓位方法。

输出格式 (JSON):
{{"method": "方法名", "reasoning": "选择理由"}}
"""

        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": "你是专业的仓位管理专家，请选择最优仓位方法。输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
            )

            data = json.loads(response)
            method = data.get("method", "risk_parity")
            if method in self.SIZING_METHODS:
                return method
            return "risk_parity"

        except Exception as e:
            logger.warning(f"Sizing method selection failed: {e}")
            return "risk_parity"

    def _compute_position_sizes(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        method: str,
    ) -> Dict[str, float]:
        """
        根据选定方法计算仓位大小
        """
        import pandas as pd

        if method == "equal_weight":
            return self._equal_weight_sizing(target_allocation)
        elif method == "risk_parity":
            return self._risk_parity_sizing(target_allocation, market_data)
        elif method == "volatility_target":
            return self._volatility_target_sizing(target_allocation, market_data)
        elif method == "kelly":
            return self._kelly_sizing(target_allocation, market_data)
        else:
            return self._equal_weight_sizing(target_allocation)

    def _equal_weight_sizing(
        self,
        target_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """等权配置"""
        n = len(target_allocation)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {asset: weight for asset in target_allocation}

    def _risk_parity_sizing(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        风险平价配置: 每个资产贡献相等的风险
        """
        returns_data = market_data.get("returns", {})

        if not returns_data:
            # 回退到等权
            return self._equal_weight_sizing(target_allocation)

        import pandas as pd
        returns_df = pd.DataFrame(returns_data)

        # 计算各资产波动率
        volatilities = {}
        for asset in target_allocation:
            if asset in returns_df.columns:
                vol = returns_df[asset].std() * np.sqrt(252)
                volatilities[asset] = max(vol, 0.01)  # 防止除零
            else:
                volatilities[asset] = 0.15  # 默认波动率

        # 风险平价: 权重与波动率成反比
        inv_vols = {asset: 1.0 / vol for asset, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        weights = {asset: inv_vol / total_inv_vol for asset, inv_vol in inv_vols.items()}
        return weights

    def _volatility_target_sizing(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        波动率目标配置: 调整权重使组合波动率达到目标
        """
        returns_data = market_data.get("returns", {})

        if not returns_data:
            return target_allocation

        import pandas as pd
        returns_df = pd.DataFrame(returns_data)

        # 计算当前组合波动率
        available = [a for a in target_allocation if a in returns_df.columns]
        if not available:
            return target_allocation

        weights = np.array([target_allocation.get(a, 0) for a in available])
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        cov_matrix = returns_df[available].cov() * 252
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))

        # 调整比例
        if portfolio_vol > 0:
            scale = self.target_volatility / portfolio_vol
            scale = min(max(scale, 0.5), 2.0)  # 限制调整幅度
        else:
            scale = 1.0

        adjusted = {asset: w * scale for asset, w in target_allocation.items()}

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _kelly_sizing(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Kelly 准则配置: 基于预期收益和风险的最优配置
        """
        returns_data = market_data.get("returns", {})

        if not returns_data:
            return target_allocation

        import pandas as pd
        returns_df = pd.DataFrame(returns_data)

        kelly_weights = {}
        for asset in target_allocation:
            if asset in returns_df.columns:
                ret = returns_df[asset].mean() * 252  # 年化收益
                vol = returns_df[asset].std() * np.sqrt(252)  # 年化波动率

                if vol > 0:
                    # Kelly: f = μ/σ² (简化版)
                    kelly_f = ret / (vol ** 2) if vol > 0 else 0
                    # 使用半 Kelly 更保守
                    kelly_weights[asset] = max(0, kelly_f * 0.5)
                else:
                    kelly_weights[asset] = target_allocation.get(asset, 0)
            else:
                kelly_weights[asset] = target_allocation.get(asset, 0)

        # 归一化
        total = sum(kelly_weights.values())
        if total > 0:
            kelly_weights = {k: v / total for k, v in kelly_weights.items()}
        else:
            return target_allocation

        return kelly_weights

    def _apply_constraints(
        self,
        sizes: Dict[str, float],
        risk_constraints: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """应用风险约束"""
        max_size = risk_constraints.get("max_single_asset", self.max_position_size)
        min_size = self.min_position_size

        constrained = {}
        for asset, size in sizes.items():
            constrained[asset] = max(min_size, min(max_size, size))

        # 归一化
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}

        return constrained

    def _compute_risk_contribution(
        self,
        sizes: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """计算每个资产的风险贡献"""
        returns_data = market_data.get("returns", {})

        if not returns_data:
            # 简单假设等比例风险贡献
            return {asset: size for asset, size in sizes.items()}

        import pandas as pd
        returns_df = pd.DataFrame(returns_data)

        available = [a for a in sizes if a in returns_df.columns]
        if not available:
            return sizes.copy()

        weights = np.array([sizes.get(a, 0) for a in available])
        cov_matrix = returns_df[available].cov() * 252

        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))

        # 边际风险贡献 = (Cov * w) / σ_p * w
        if portfolio_vol > 0:
            marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            risk_contrib = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else weights
        else:
            risk_contrib = weights

        return {asset: float(risk_contrib[i]) for i, asset in enumerate(available)}

    def _generate_reasoning(
        self,
        method: str,
        sizes: Dict[str, float],
        risk_contribution: Dict[str, float],
    ) -> str:
        """生成决策理由"""
        top_positions = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:3]

        reasoning = f"使用{self.SIZING_METHODS.get(method, method)}方法确定仓位。"
        reasoning += f"最大仓位: {', '.join([f'{a}({s:.1%})' for a, s in top_positions])}。"

        # 检查风险集中度
        max_risk = max(risk_contribution.values()) if risk_contribution else 0
        if max_risk > 0.3:
            reasoning += f"注意: 风险集中度较高 ({max_risk:.1%})。"

        return reasoning

    def revise_based_on_feedback(
        self,
        current_decision: PositionSizingDecision,
        feedback: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> PositionSizingDecision:
        """
        根据其他 Agent 的反馈修正决策 (用于并行讨论)

        Args:
            current_decision: 当前决策
            feedback: 来自 PM 和 Hedging Agent 的反馈
            market_data: 市场数据
        """
        prompt = f"""## 仓位修正任务

### 当前仓位决策
- 方法: {current_decision.sizing_method}
- 仓位: {json.dumps(current_decision.position_sizes, indent=2)}

### 其他智能体反馈
{json.dumps(feedback, indent=2, ensure_ascii=False)}

### 任务
根据反馈调整仓位。输出修正后的仓位 (JSON):
{{"position_sizes": {{"asset": weight}}, "reasoning": "修正理由"}}
"""

        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": "你是仓位管理专家，根据其他专家的反馈调整仓位。输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )

            data = json.loads(response)
            revised_sizes = data.get("position_sizes", current_decision.position_sizes)

            # 重新应用约束
            constrained = self._apply_constraints(
                revised_sizes,
                {"max_single_asset": self.max_position_size},
                market_data
            )

            return PositionSizingDecision(
                timestamp=datetime.now().isoformat(),
                position_sizes=constrained,
                sizing_method=current_decision.sizing_method + "_revised",
                reasoning=data.get("reasoning", "根据反馈修正"),
                risk_contribution=self._compute_risk_contribution(constrained, market_data),
            )

        except Exception as e:
            logger.warning(f"Position revision failed: {e}")
            return current_decision
