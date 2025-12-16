"""
Hedging Agent
对冲策略智能体 - 负责制定和管理对冲策略

增强版: 支持动态对冲资产选择，从全市场70+资产中筛选最优对冲工具
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HedgingDecision:
    """对冲决策结果"""
    timestamp: str
    hedging_strategy: str             # 对冲策略名称
    hedge_ratio: float                # 对冲比例
    hedge_instruments: List[Dict]     # 对冲工具列表
    expected_cost: float              # 预期成本
    expected_protection: float        # 预期保护水平
    reasoning: str                    # 决策理由
    tail_risk_metrics: Dict[str, float]  # 尾部风险指标
    dynamic_recommendation: Optional[Dict] = None  # 动态选择推荐详情

    def to_dict(self) -> Dict:
        result = {
            "timestamp": self.timestamp,
            "hedging_strategy": self.hedging_strategy,
            "hedge_ratio": self.hedge_ratio,
            "hedge_instruments": self.hedge_instruments,
            "expected_cost": self.expected_cost,
            "expected_protection": self.expected_protection,
            "reasoning": self.reasoning,
            "tail_risk_metrics": self.tail_risk_metrics,
        }
        if self.dynamic_recommendation:
            result["dynamic_recommendation"] = self.dynamic_recommendation
        return result


class HedgingAgent:
    """
    对冲策略智能体 (增强版)

    核心职责:
    1. 评估组合的尾部风险敞口
    2. 选择合适的对冲策略
    3. 确定对冲比例和工具 (支持动态选择)
    4. 计算对冲成本和效果

    新特性:
    - 动态对冲资产选择: 从全市场70+资产中筛选最优对冲工具
    - 多因子评分: 相关性、流动性、成本、效率综合评估
    - 智能推荐: 根据组合敞口自动选择对冲目标
    """

    # 对冲策略
    HEDGING_STRATEGIES = {
        "put_protection": "买入看跌期权保护",
        "collar": "领口策略 (买看跌卖看涨)",
        "tail_hedge": "尾部风险对冲",
        "dynamic_hedge": "动态对冲",
        "diversification": "分散化对冲",
        "safe_haven": "避险资产对冲",
        "none": "无需对冲",
    }

    # 固定对冲工具 (作为后备)
    HEDGE_INSTRUMENTS = {
        "SPY_PUT": {"symbol": "SPY_PUT", "name": "SPY Put Options", "type": "option", "cost_rate": 0.02},
        "VIX_CALL": {"symbol": "VIX_CALL", "name": "VIX Call Options", "type": "option", "cost_rate": 0.03},
        "TLT": {"symbol": "TLT", "name": "长期国债 ETF", "type": "etf", "cost_rate": 0.001},
        "GLD": {"symbol": "GLD", "name": "黄金 ETF", "type": "etf", "cost_rate": 0.001},
        "TAIL": {"symbol": "TAIL", "name": "尾部风险 ETF", "type": "etf", "cost_rate": 0.002},
        "SH": {"symbol": "SH", "name": "做空标普 ETF", "type": "etf", "cost_rate": 0.001},
        "CASH": {"symbol": "CASH", "name": "现金/短期国债", "type": "cash", "cost_rate": 0.0},
    }

    def __init__(
        self,
        llm_provider: Any,
        config: Optional[Dict] = None,
        use_dynamic_selection: bool = True  # 是否使用动态选择
    ):
        """
        初始化对冲智能体

        Args:
            llm_provider: LLM服务提供者
            config: 配置参数
            use_dynamic_selection: 是否启用动态对冲资产选择
        """
        self.llm = llm_provider
        self.config = config or {}
        self.use_dynamic_selection = use_dynamic_selection

        # 配置参数
        self.max_hedge_cost = self.config.get("max_hedge_cost", 0.03)  # 最大对冲成本 3%
        self.tail_risk_threshold = self.config.get("tail_risk_threshold", 0.10)
        self.vix_spike_threshold = self.config.get("vix_spike_threshold", 25.0)

        # 初始化动态选择器
        self.dynamic_selector = None
        if self.use_dynamic_selection:
            try:
                from finsage.hedging.dynamic_selector import DynamicHedgeSelector
                dynamic_config = self.config.get("dynamic_selector", {})
                self.dynamic_selector = DynamicHedgeSelector(config=dynamic_config)
                logger.info("Dynamic hedge selector enabled")
            except ImportError as e:
                logger.warning(f"Failed to import DynamicHedgeSelector: {e}. Using fixed instruments only.")
                self.use_dynamic_selection = False

        logger.info(f"Hedging Agent initialized (dynamic_selection={self.use_dynamic_selection})")

    def analyze(
        self,
        target_allocation: Dict[str, float],
        position_sizes: Dict[str, float],
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> HedgingDecision:
        """
        分析并制定对冲策略 (增强版)

        Args:
            target_allocation: 目标资产配置
            position_sizes: 仓位大小
            market_data: 市场数据
            risk_constraints: 风控约束

        Returns:
            HedgingDecision: 对冲决策
        """
        # Step 1: 评估尾部风险
        tail_risk = self._assess_tail_risk(target_allocation, market_data)

        # Step 2: 选择对冲策略
        strategy = self._select_hedging_strategy(tail_risk, market_data, risk_constraints)

        # Step 3: 确定基础对冲比例和工具
        hedge_ratio, base_instruments = self._determine_hedge_params(
            strategy, tail_risk, market_data
        )

        # Step 4: 动态资产选择 (如果启用)
        dynamic_recommendation = None
        if self.use_dynamic_selection and self.dynamic_selector and strategy != "none":
            try:
                instruments, dynamic_recommendation = self._apply_dynamic_selection(
                    target_allocation=target_allocation,
                    market_data=market_data,
                    strategy=strategy,
                    hedge_ratio=hedge_ratio,
                    base_instruments=base_instruments,
                    risk_constraints=risk_constraints,
                )
            except Exception as e:
                logger.warning(f"Dynamic selection failed: {e}. Using fixed instruments.")
                instruments = base_instruments
        else:
            instruments = base_instruments

        # Step 5: 计算成本和保护水平
        cost, protection = self._calculate_hedge_economics(
            strategy, hedge_ratio, instruments, market_data
        )

        # 如果使用了动态选择，更新成本和保护估算
        if dynamic_recommendation:
            cost = dynamic_recommendation.get("expected_cost", cost)
            protection = hedge_ratio * 2 + dynamic_recommendation.get("expected_correlation_reduction", 0)

        # Step 6: 生成决策理由
        reasoning = self._generate_reasoning(strategy, tail_risk, hedge_ratio)
        if dynamic_recommendation:
            reasoning += f" {dynamic_recommendation.get('reasoning', '')}"

        decision = HedgingDecision(
            timestamp=datetime.now().isoformat(),
            hedging_strategy=strategy,
            hedge_ratio=hedge_ratio,
            hedge_instruments=instruments,
            expected_cost=cost,
            expected_protection=protection,
            reasoning=reasoning,
            tail_risk_metrics=tail_risk,
            dynamic_recommendation=dynamic_recommendation,
        )

        logger.info(f"Hedging decision: {strategy}, ratio={hedge_ratio:.2%}, cost={cost:.2%}, "
                    f"instruments={len(instruments)}, dynamic={dynamic_recommendation is not None}")
        return decision

    def _apply_dynamic_selection(
        self,
        target_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        strategy: str,
        hedge_ratio: float,
        base_instruments: List[Dict],
        risk_constraints: Dict[str, float],
    ) -> tuple:
        """
        应用动态对冲资产选择

        Returns:
            tuple: (instruments, recommendation_dict)
        """
        # 获取收益率数据
        returns_data = market_data.get("returns", {})
        if isinstance(returns_data, dict):
            returns_df = pd.DataFrame(returns_data)
        elif isinstance(returns_data, pd.DataFrame):
            returns_df = returns_data
        else:
            returns_df = pd.DataFrame()

        # 调用动态选择器
        recommendation = self.dynamic_selector.recommend(
            portfolio_weights=target_allocation,
            returns_data=returns_df,
            hedge_strategy=strategy,
            hedge_ratio=hedge_ratio,
            market_data=market_data,
            risk_constraints=risk_constraints,
        )

        # 转换为工具列表格式
        dynamic_instruments = recommendation.get_instruments_for_agent()

        # 合并固定工具和动态工具
        instruments = self._merge_instruments(base_instruments, dynamic_instruments)

        # 返回推荐详情
        recommendation_dict = recommendation.to_dict()

        return instruments, recommendation_dict

    def _merge_instruments(
        self,
        base_instruments: List[Dict],
        dynamic_instruments: List[Dict]
    ) -> List[Dict]:
        """
        合并固定工具和动态工具

        策略:
        - 动态选择的工具优先
        - 保留现金等固定工具
        - 避免重复
        """
        if not dynamic_instruments:
            return base_instruments

        # 动态工具的符号集合
        dynamic_symbols = {inst.get("symbol") for inst in dynamic_instruments}

        # 从动态工具开始
        merged = list(dynamic_instruments)

        # 添加固定工具中的补充项
        for inst in base_instruments:
            symbol = inst.get("symbol", inst.get("name", ""))

            # 现金始终保留
            if inst.get("type") == "cash" or "CASH" in symbol:
                # 检查动态工具中是否已有现金
                has_cash = any(i.get("type") == "cash" for i in dynamic_instruments)
                if not has_cash:
                    inst["source"] = "fixed"
                    merged.append(inst)

            # 期权类工具保留 (动态选择器通常不包含期权)
            elif inst.get("type") == "option":
                inst["source"] = "fixed"
                merged.append(inst)

        return merged

    def _assess_tail_risk(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        评估组合的尾部风险

        指标:
        - VaR (95%, 99%)
        - Expected Shortfall (CVaR)
        - 偏度和峰度
        - 历史最大回撤
        """
        returns_data = market_data.get("returns", {})
        vix = market_data.get("macro", {}).get("vix", 20.0)

        tail_risk = {
            "vix": vix,
            "vix_level": "high" if vix > 25 else "low" if vix < 15 else "moderate",
            "var_95": -0.02,  # 默认值
            "var_99": -0.04,
            "cvar_95": -0.03,
            "max_drawdown": -0.15,
            "skewness": 0.0,
            "kurtosis": 3.0,
        }

        if not returns_data:
            return tail_risk

        if isinstance(returns_data, dict):
            returns_df = pd.DataFrame(returns_data)
        elif isinstance(returns_data, pd.DataFrame):
            returns_df = returns_data
        else:
            return tail_risk

        # 构建组合收益
        available = [a for a in allocation if a in returns_df.columns]
        if not available:
            return tail_risk

        weights = np.array([allocation.get(a, 0) for a in available])
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        portfolio_returns = returns_df[available].dot(weights)

        if len(portfolio_returns) < 20:
            return tail_risk

        # 计算 VaR
        tail_risk["var_95"] = float(np.percentile(portfolio_returns, 5))
        tail_risk["var_99"] = float(np.percentile(portfolio_returns, 1))

        # 计算 CVaR (Expected Shortfall)
        var_95_mask = portfolio_returns <= tail_risk["var_95"]
        if var_95_mask.sum() > 0:
            tail_risk["cvar_95"] = float(portfolio_returns[var_95_mask].mean())

        # 计算偏度和峰度
        try:
            from scipy import stats
            tail_risk["skewness"] = float(stats.skew(portfolio_returns))
            tail_risk["kurtosis"] = float(stats.kurtosis(portfolio_returns) + 3)  # excess kurtosis + 3
        except ImportError:
            # 手动计算
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            if std > 1e-10:  # 使用 epsilon 阈值避免数值问题
                tail_risk["skewness"] = float(((portfolio_returns - mean) ** 3).mean() / (std ** 3))
                tail_risk["kurtosis"] = float(((portfolio_returns - mean) ** 4).mean() / (std ** 4))
            else:
                # 标准差为 0，无法计算偏度/峰度
                tail_risk["skewness"] = 0.0
                tail_risk["kurtosis"] = 3.0  # 正态分布的峰度

        # 计算历史最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        tail_risk["max_drawdown"] = float(drawdown.min())

        return tail_risk

    def _select_hedging_strategy(
        self,
        tail_risk: Dict[str, float],
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> str:
        """
        使用 LLM 选择对冲策略
        """
        vix = tail_risk.get("vix", 20.0)
        var_95 = tail_risk.get("var_95", -0.02)
        cvar_95 = tail_risk.get("cvar_95", -0.03)
        max_dd = tail_risk.get("max_drawdown", -0.15)
        skewness = tail_risk.get("skewness", 0.0)
        kurtosis = tail_risk.get("kurtosis", 3.0)

        prompt = f"""## 对冲策略选择任务

### 尾部风险评估
- VIX: {vix:.1f} ({tail_risk.get('vix_level', 'moderate')})
- 日VaR (95%): {var_95:.2%}
- 日CVaR (95%): {cvar_95:.2%}
- 历史最大回撤: {max_dd:.2%}
- 偏度: {skewness:.2f} (负值=左偏/下行风险大)
- 峰度: {kurtosis:.2f} (>3=肥尾)

### 风控约束
- 最大回撤容忍: {risk_constraints.get('max_drawdown', 0.15):.1%}
- 目标波动率: {risk_constraints.get('target_volatility', 0.12):.1%}
- 最大对冲成本: {self.max_hedge_cost:.1%}

### 可用策略
{chr(10).join([f'- {k}: {v}' for k, v in self.HEDGING_STRATEGIES.items()])}

### 任务
选择最适合当前风险状况的对冲策略。

考虑因素:
1. VIX高时期权成本高，避免买入期权
2. 负偏度和高峰度表示需要更多尾部保护
3. 如果风险指标在可接受范围内，可以选择"none"

输出格式 (JSON):
{{"strategy": "策略名", "reasoning": "选择理由"}}
"""

        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": "你是专业的风险对冲专家，请选择最优对冲策略。输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
            )

            data = json.loads(response)
            strategy = data.get("strategy", "diversification")
            if strategy in self.HEDGING_STRATEGIES:
                return strategy
            return "diversification"

        except Exception as e:
            logger.warning(f"Hedging strategy selection failed: {e}")
            # 基于 VIX 的简单规则
            if vix > 25:
                return "safe_haven"
            elif var_95 < -0.03:
                return "tail_hedge"
            else:
                return "none"

    def _determine_hedge_params(
        self,
        strategy: str,
        tail_risk: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> tuple:
        """
        确定对冲参数
        """
        vix = tail_risk.get("vix", 20.0)

        # 基础对冲比例 (根据风险水平调整)
        if strategy == "none":
            return 0.0, []

        # 根据 VIX 调整对冲比例
        if vix > 30:
            base_ratio = 0.20
        elif vix > 25:
            base_ratio = 0.15
        elif vix > 20:
            base_ratio = 0.10
        else:
            base_ratio = 0.05

        # 根据策略选择工具
        instruments = []

        if strategy == "put_protection":
            instruments.append({
                **self.HEDGE_INSTRUMENTS["SPY_PUT"],
                "allocation": base_ratio,
                "source": "fixed",
            })
        elif strategy == "collar":
            instruments.append({
                **self.HEDGE_INSTRUMENTS["SPY_PUT"],
                "allocation": base_ratio * 0.6,
                "source": "fixed",
            })
            # Collar 卖出 call 会产生收入，这里简化处理
        elif strategy == "tail_hedge":
            instruments.append({
                **self.HEDGE_INSTRUMENTS["TAIL"],
                "allocation": base_ratio * 0.5,
                "source": "fixed",
            })
            instruments.append({
                **self.HEDGE_INSTRUMENTS["VIX_CALL"],
                "allocation": base_ratio * 0.5,
                "source": "fixed",
            })
        elif strategy == "dynamic_hedge":
            instruments.append({
                **self.HEDGE_INSTRUMENTS["SH"],
                "allocation": base_ratio * 0.5,
                "source": "fixed",
            })
            instruments.append({
                **self.HEDGE_INSTRUMENTS["CASH"],
                "allocation": base_ratio * 0.5,
                "source": "fixed",
            })
        elif strategy == "diversification":
            instruments.append({
                **self.HEDGE_INSTRUMENTS["TLT"],
                "allocation": base_ratio * 0.4,
                "source": "fixed",
            })
            instruments.append({
                **self.HEDGE_INSTRUMENTS["GLD"],
                "allocation": base_ratio * 0.3,
                "source": "fixed",
            })
            instruments.append({
                **self.HEDGE_INSTRUMENTS["CASH"],
                "allocation": base_ratio * 0.3,
                "source": "fixed",
            })
        elif strategy == "safe_haven":
            instruments.append({
                **self.HEDGE_INSTRUMENTS["TLT"],
                "allocation": base_ratio * 0.5,
                "source": "fixed",
            })
            instruments.append({
                **self.HEDGE_INSTRUMENTS["GLD"],
                "allocation": base_ratio * 0.3,
                "source": "fixed",
            })
            instruments.append({
                **self.HEDGE_INSTRUMENTS["CASH"],
                "allocation": base_ratio * 0.2,
                "source": "fixed",
            })

        return base_ratio, instruments

    def _calculate_hedge_economics(
        self,
        strategy: str,
        hedge_ratio: float,
        instruments: List[Dict],
        market_data: Dict[str, Any],
    ) -> tuple:
        """
        计算对冲成本和保护水平
        """
        if strategy == "none" or not instruments:
            return 0.0, 0.0

        # 计算总成本
        total_cost = 0.0
        for inst in instruments:
            alloc = inst.get("allocation", 0)
            cost_rate = inst.get("cost_rate", inst.get("expense_ratio", 0.01))
            total_cost += alloc * cost_rate

        # 限制最大成本
        total_cost = min(total_cost, self.max_hedge_cost)

        # 估算保护水平 (简化计算)
        # 假设对冲工具能提供约 2:1 的下行保护
        protection = hedge_ratio * 2

        return total_cost, protection

    def _generate_reasoning(
        self,
        strategy: str,
        tail_risk: Dict[str, float],
        hedge_ratio: float,
    ) -> str:
        """生成决策理由"""
        if strategy == "none":
            return "当前风险指标在可接受范围内，无需额外对冲。"

        vix = tail_risk.get("vix", 20.0)
        var_95 = tail_risk.get("var_95", -0.02)

        reasoning = f"采用{self.HEDGING_STRATEGIES.get(strategy, strategy)}策略。"
        reasoning += f"当前VIX={vix:.1f}，日VaR(95%)={var_95:.2%}。"
        reasoning += f"对冲比例{hedge_ratio:.1%}以控制尾部风险。"

        if tail_risk.get("skewness", 0) < -0.5:
            reasoning += "组合呈现负偏度，加强下行保护。"

        return reasoning

    def revise_based_on_feedback(
        self,
        current_decision: HedgingDecision,
        feedback: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> HedgingDecision:
        """
        根据其他 Agent 的反馈修正决策 (用于并行讨论)

        Args:
            current_decision: 当前决策
            feedback: 来自 PM 和 Position Sizing Agent 的反馈
            market_data: 市场数据
        """
        prompt = f"""## 对冲策略修正任务

### 当前对冲决策
- 策略: {current_decision.hedging_strategy}
- 对冲比例: {current_decision.hedge_ratio:.1%}
- 预期成本: {current_decision.expected_cost:.2%}
- 工具数量: {len(current_decision.hedge_instruments)}

### 其他智能体反馈
{json.dumps(feedback, indent=2, ensure_ascii=False)}

### 任务
根据反馈调整对冲策略。输出修正后的参数 (JSON):
{{"hedge_ratio": 0.1, "reasoning": "修正理由"}}
"""

        try:
            response = self.llm.create_completion(
                messages=[
                    {"role": "system", "content": "你是对冲策略专家，根据其他专家的反馈调整对冲参数。输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
            )

            data = json.loads(response)
            new_ratio = data.get("hedge_ratio", current_decision.hedge_ratio)

            # 重新计算经济指标
            cost, protection = self._calculate_hedge_economics(
                current_decision.hedging_strategy,
                new_ratio,
                current_decision.hedge_instruments,
                market_data
            )

            return HedgingDecision(
                timestamp=datetime.now().isoformat(),
                hedging_strategy=current_decision.hedging_strategy + "_revised",
                hedge_ratio=new_ratio,
                hedge_instruments=current_decision.hedge_instruments,
                expected_cost=cost,
                expected_protection=protection,
                reasoning=data.get("reasoning", "根据反馈修正"),
                tail_risk_metrics=current_decision.tail_risk_metrics,
                dynamic_recommendation=current_decision.dynamic_recommendation,
            )

        except Exception as e:
            logger.warning(f"Hedging revision failed: {e}")
            return current_decision

    def get_dynamic_selector_summary(self) -> Optional[Dict]:
        """获取动态选择器摘要"""
        if self.dynamic_selector:
            return self.dynamic_selector.get_universe_summary()
        return None
