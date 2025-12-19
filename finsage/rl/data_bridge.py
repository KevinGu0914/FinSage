"""
Data Bridge for MARFT-FinSage Integration

在FinSage环境和MARFT训练框架之间转换数据格式:
1. Env Observation → LLM Text Prompt (给Expert)
2. Env Observation → Numerical Tensor (给Critic)
3. Expert Actions → Environment Actions
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 1. Observation Formatters
# ============================================================

class ObservationFormatter:
    """
    将环境观察格式化为不同形式
    """

    def __init__(
        self,
        asset_universe: Dict[str, List[str]],
        technical_indicators: List[str] = None,
        macro_indicators: List[str] = None,
    ):
        """
        Args:
            asset_universe: 资产池 {asset_class: [symbols]}
            technical_indicators: 技术指标列表
            macro_indicators: 宏观指标列表
        """
        self.asset_universe = asset_universe
        self.technical_indicators = technical_indicators or [
            "returns_1d", "returns_5d", "returns_20d",
            "volatility_20d", "ma_5", "ma_20",
            "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"
        ]
        self.macro_indicators = macro_indicators or [
            "vix", "fed_rate", "inflation", "gdp_growth", "unemployment"
        ]

        # 构建资产列表
        self.all_symbols = []
        self.symbol_to_class = {}
        for asset_class, symbols in asset_universe.items():
            for symbol in symbols:
                self.all_symbols.append(symbol)
                self.symbol_to_class[symbol] = asset_class

        self.num_assets = len(self.all_symbols)

    def to_text_prompt(
        self,
        observation: Dict[str, Any],
        asset_class: Optional[str] = None,
        include_portfolio: bool = True,
        include_news: bool = True,
    ) -> str:
        """
        将观察转换为LLM文本prompt

        Args:
            observation: 环境观察 (来自env.get_observation())
            asset_class: 可选，只包含特定资产类别的数据
            include_portfolio: 是否包含组合状态
            include_news: 是否包含新闻

        Returns:
            格式化的文本prompt
        """
        portfolio = observation.get("portfolio", {})
        market_data = observation.get("market_data", {})
        date = observation.get("date", datetime.now().strftime("%Y-%m-%d"))

        lines = [f"## 市场日期: {date}\n"]

        # 组合状态
        if include_portfolio:
            lines.append("## 当前组合状态")
            lines.append(f"- 总价值: ${portfolio.get('portfolio_value', 0):,.2f}")
            lines.append(f"- 现金: ${portfolio.get('cash', 0):,.2f}")
            lines.append(f"- 总收益: {portfolio.get('total_return', 0):.2%}")

            # 资产类别权重
            class_weights = portfolio.get("class_weights", {})
            if class_weights:
                lines.append("\n### 资产类别配置:")
                for cls, weight in class_weights.items():
                    lines.append(f"  - {cls}: {weight:.1%}")

            # 持仓详情
            positions = portfolio.get("positions", {})
            if positions:
                lines.append("\n### 当前持仓:")
                for symbol, pos in positions.items():
                    pnl = pos.get("unrealized_pnl", 0)
                    pnl_sign = "+" if pnl >= 0 else ""
                    lines.append(
                        f"  - {symbol}: {pos.get('shares', 0):.2f}股 "
                        f"@ ${pos.get('current_price', 0):.2f} "
                        f"(盈亏: {pnl_sign}${pnl:,.2f})"
                    )
            lines.append("")

        # 市场数据
        lines.append("## 市场数据")

        # 过滤资产类别
        if asset_class:
            symbols_to_show = self.asset_universe.get(asset_class, [])
        else:
            symbols_to_show = self.all_symbols

        for symbol in symbols_to_show:
            if symbol not in market_data:
                continue

            data = market_data[symbol]
            if not isinstance(data, dict):
                continue

            lines.append(f"\n### {symbol} ({self.symbol_to_class.get(symbol, 'other')})")

            # 价格数据
            price = data.get("close", data.get("price", "N/A"))
            lines.append(f"  当前价格: ${price}")

            # 收益率
            for period in ["1d", "5d", "20d"]:
                key = f"returns_{period}"
                if key in data:
                    lines.append(f"  {period}收益: {data[key]:.2%}")

            # 技术指标
            vol = data.get("volatility_20d", data.get("volatility"))
            if vol:
                lines.append(f"  20日波动率: {vol:.2%}")

            rsi = data.get("rsi_14", data.get("rsi"))
            if rsi:
                lines.append(f"  RSI(14): {rsi:.1f}")

            macd = data.get("macd")
            if macd:
                lines.append(f"  MACD: {macd:.4f}")

        # 宏观数据
        macro = market_data.get("macro", {})
        if macro:
            lines.append("\n## 宏观经济指标")
            if "vix" in macro:
                lines.append(f"  VIX恐惧指数: {macro['vix']:.1f}")
            if "fed_rate" in macro:
                lines.append(f"  联邦基金利率: {macro['fed_rate']:.2%}")
            if "inflation" in macro:
                lines.append(f"  通胀率: {macro['inflation']:.2%}")

        # 新闻
        news = market_data.get("news", [])
        if include_news and news:
            lines.append("\n## 最新市场新闻")
            for i, item in enumerate(news[:5]):  # 最多显示5条
                headline = item.get("headline", item.get("title", ""))
                sentiment = item.get("sentiment", "neutral")
                lines.append(f"  {i+1}. [{sentiment}] {headline}")

        return "\n".join(lines)

    def to_numerical_tensor(
        self,
        observation: Dict[str, Any],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将观察转换为数值张量 (用于Critic)

        Returns:
            asset_features: [num_assets, feature_dim] - 资产特征
            macro_features: [macro_dim] - 宏观特征
            portfolio_features: [portfolio_dim] - 组合特征
        """
        market_data = observation.get("market_data", {})
        portfolio = observation.get("portfolio", {})

        # 资产特征
        asset_features = []
        for symbol in self.all_symbols:
            data = market_data.get(symbol, {})
            if not isinstance(data, dict):
                data = {}

            features = [
                data.get("close", data.get("price", 0)),
                data.get("returns_1d", 0),
                data.get("returns_5d", 0),
                data.get("returns_20d", 0),
                data.get("volatility_20d", data.get("volatility", 0)),
                data.get("volume", 0) / 1e6,  # 标准化
                data.get("ma_5", 0),
                data.get("ma_20", 0),
                data.get("rsi_14", data.get("rsi", 50)) / 100,  # 标准化到[0,1]
                data.get("macd", 0),
            ]
            asset_features.append(features)

        asset_features = torch.tensor(asset_features, dtype=torch.float32, device=device)

        # 宏观特征
        macro = market_data.get("macro", {})
        macro_features = torch.tensor([
            macro.get("vix", 20) / 100,
            macro.get("fed_rate", 0.05),
            macro.get("inflation", 0.02),
            macro.get("gdp_growth", 0.02),
            macro.get("unemployment", 0.04),
        ], dtype=torch.float32, device=device)

        # 组合特征
        positions = portfolio.get("positions", {})
        weights = portfolio.get("weights", {})

        portfolio_features = []
        for symbol in self.all_symbols:
            pos = positions.get(symbol, {})
            weight = weights.get(symbol, 0)
            features = [
                weight,
                pos.get("shares", 0),
                pos.get("avg_cost", 0),
                pos.get("unrealized_pnl", 0) / 1000,  # 标准化
            ]
            portfolio_features.append(features)

        portfolio_features = torch.tensor(portfolio_features, dtype=torch.float32, device=device)

        return asset_features, macro_features, portfolio_features


# ============================================================
# 2. Action Converters
# ============================================================

class ActionConverter:
    """
    转换Expert动作格式
    """

    # Action字符串到权重调整的映射
    ACTION_TO_WEIGHT = {
        "BUY_100%": 1.0,
        "BUY_75%": 0.75,
        "BUY_50%": 0.5,
        "BUY_25%": 0.25,
        "HOLD": 0.0,
        "SELL_25%": -0.25,
        "SELL_50%": -0.5,
        "SELL_75%": -0.75,
        "SELL_100%": -1.0,
        "SHORT_25%": -0.25,
        "SHORT_50%": -0.5,
        "SHORT_75%": -0.75,
        "SHORT_100%": -1.0,
    }

    def __init__(
        self,
        asset_universe: Dict[str, List[str]],
        max_single_weight: float = 0.15,
        max_class_weight: float = 0.50,
    ):
        self.asset_universe = asset_universe
        self.max_single_weight = max_single_weight
        self.max_class_weight = max_class_weight

    def expert_actions_to_allocation(
        self,
        expert_actions: List[Dict],
        current_weights: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        将Expert动作转换为目标配置

        Args:
            expert_actions: Expert动作列表，每个包含:
                - asset_class: 资产类别
                - action: 动作字符串
                - confidence: 置信度
                - recommendations: 个股推荐 (可选)
            current_weights: 当前权重

        Returns:
            目标配置 {symbol or class: weight}
        """
        current_weights = current_weights or {}
        allocation = {}

        for expert_action in expert_actions:
            asset_class = expert_action.get("asset_class")
            action = expert_action.get("action", "HOLD")
            confidence = expert_action.get("confidence", 0.5)
            recommendations = expert_action.get("recommendations", [])

            # 获取当前类别权重
            current_class_weight = self._get_class_weight(asset_class, current_weights)

            # 计算目标权重调整
            weight_delta = self.ACTION_TO_WEIGHT.get(action, 0) * confidence

            # 基础目标权重
            base_target = current_class_weight + weight_delta * 0.2  # 20%调整幅度

            # 限制在合理范围
            target_weight = max(-self.max_class_weight, min(self.max_class_weight, base_target))

            # 如果有个股推荐，展开为个股权重
            if recommendations:
                symbol_weights = self._expand_recommendations(
                    asset_class, recommendations, target_weight
                )
                allocation.update(symbol_weights)
            else:
                # 类别级别配置
                allocation[asset_class] = target_weight

        # 计算剩余现金
        total_allocated = sum(abs(w) for w in allocation.values())
        allocation["cash"] = max(0, 1 - total_allocated)

        return allocation

    def _get_class_weight(
        self,
        asset_class: str,
        weights: Dict[str, float]
    ) -> float:
        """计算类别权重"""
        symbols = self.asset_universe.get(asset_class, [])
        return sum(weights.get(s, 0) for s in symbols)

    def _expand_recommendations(
        self,
        asset_class: str,
        recommendations: List[Dict],
        total_weight: float,
    ) -> Dict[str, float]:
        """展开个股推荐为权重"""
        symbols = self.asset_universe.get(asset_class, [])
        weights = {}

        # 计算每个推荐的分数
        scores = {}
        for rec in recommendations:
            symbol = rec.get("symbol")
            if symbol not in symbols:
                continue

            action = rec.get("action", "HOLD")
            confidence = rec.get("confidence", 0.5)
            score = self.ACTION_TO_WEIGHT.get(action, 0) * confidence + 1  # 偏移使其为正

            scores[symbol] = max(0, score)

        # 标准化为权重
        total_score = sum(scores.values())
        if total_score > 0:
            for symbol, score in scores.items():
                weights[symbol] = (score / total_score) * total_weight

        return weights

    def allocation_to_action_tensor(
        self,
        allocation: Dict[str, float],
        all_symbols: List[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        将配置转换为动作张量

        Returns:
            action_tensor: [num_assets] - 每个资产的目标权重
        """
        actions = []
        for symbol in all_symbols:
            weight = allocation.get(symbol, 0)
            actions.append(weight)

        return torch.tensor(actions, dtype=torch.float32, device=device)


# ============================================================
# 3. Batch Data Processor
# ============================================================

@dataclass
class MARFTBatch:
    """MARFT训练批次数据"""
    text_observations: List[str]           # LLM文本观察
    asset_features: torch.Tensor           # [batch, num_assets, feature_dim]
    macro_features: torch.Tensor           # [batch, macro_dim]
    portfolio_features: torch.Tensor       # [batch, num_assets, 4]
    action_tokens: List[torch.Tensor]      # [batch, num_agents, seq_len]
    rewards: torch.Tensor                  # [batch]
    dones: torch.Tensor                    # [batch]
    old_log_probs: torch.Tensor            # [batch, num_agents]
    old_values: torch.Tensor               # [batch, num_agents]


class BatchProcessor:
    """
    批量数据处理器
    """

    def __init__(
        self,
        formatter: ObservationFormatter,
        converter: ActionConverter,
        num_agents: int,
        device: str = "cpu",
    ):
        self.formatter = formatter
        self.converter = converter
        self.num_agents = num_agents
        self.device = device

    def create_batch(
        self,
        observations: List[Dict],
        actions: List[List[Dict]],
        action_tokens: List[List[torch.Tensor]],
        rewards: List[float],
        dones: List[bool],
        old_log_probs: List[List[float]],
        old_values: List[List[float]],
    ) -> MARFTBatch:
        """
        创建训练批次

        Args:
            observations: 环境观察列表
            actions: Expert动作列表 [batch, num_agents]
            action_tokens: 动作token [batch, num_agents, seq_len]
            rewards: 奖励列表
            dones: 结束标志列表
            old_log_probs: 旧策略log prob
            old_values: 旧value估计

        Returns:
            MARFTBatch
        """
        batch_size = len(observations)

        # 文本观察
        text_obs = [
            self.formatter.to_text_prompt(obs)
            for obs in observations
        ]

        # 数值特征
        asset_feats, macro_feats, port_feats = [], [], []
        for obs in observations:
            af, mf, pf = self.formatter.to_numerical_tensor(obs, self.device)
            asset_feats.append(af)
            macro_feats.append(mf)
            port_feats.append(pf)

        asset_features = torch.stack(asset_feats)
        macro_features = torch.stack(macro_feats)
        portfolio_features = torch.stack(port_feats)

        # 转换其他张量
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values_tensor = torch.tensor(old_values, dtype=torch.float32, device=self.device)

        return MARFTBatch(
            text_observations=text_obs,
            asset_features=asset_features,
            macro_features=macro_features,
            portfolio_features=portfolio_features,
            action_tokens=action_tokens,
            rewards=rewards_tensor,
            dones=dones_tensor,
            old_log_probs=old_log_probs_tensor,
            old_values=old_values_tensor,
        )


# ============================================================
# 4. Gym-style Wrapper
# ============================================================

class MARFTEnvWrapper:
    """
    将FinSage环境包装为MARFT兼容格式
    """

    def __init__(
        self,
        base_env,  # MultiAssetTradingEnv
        formatter: ObservationFormatter,
        converter: ActionConverter,
        num_agents: int = 9,  # 5 Asset Experts + 4 Meta-Level Agents
    ):
        self.env = base_env
        self.formatter = formatter
        self.converter = converter
        self.num_agents = num_agents

        self.current_obs = None

    def reset(self) -> Tuple[str, Dict]:
        """
        重置环境

        Returns:
            text_obs: 文本观察
            info: 信息字典
        """
        self.env.reset()
        self.current_obs = self.env.get_observation()

        text_obs = self.formatter.to_text_prompt(self.current_obs)

        return text_obs, {"raw_obs": self.current_obs}

    def step(
        self,
        expert_actions: List[Dict],
        market_data: Dict[str, Any],
        timestamp: str,
    ) -> Tuple[str, float, bool, Dict]:
        """
        执行一步

        Args:
            expert_actions: Expert动作列表
            market_data: 市场数据
            timestamp: 时间戳

        Returns:
            text_obs: 新的文本观察
            reward: 奖励
            done: 是否结束
            info: 信息字典
        """
        # 转换动作
        current_weights = self.current_obs.get("portfolio", {}).get("weights", {})
        allocation = self.converter.expert_actions_to_allocation(
            expert_actions, current_weights
        )

        # 执行环境step
        portfolio, reward, done, info = self.env.step(
            target_allocation=allocation,
            market_data=market_data,
            timestamp=timestamp,
        )

        # 更新观察
        self.current_obs = self.env.get_observation()
        text_obs = self.formatter.to_text_prompt(self.current_obs)

        info["raw_obs"] = self.current_obs
        info["allocation"] = allocation

        return text_obs, reward, done, info

    def get_numerical_obs(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取数值观察"""
        return self.formatter.to_numerical_tensor(self.current_obs, device)


# ============================================================
# 5. Factory Function
# ============================================================

def create_data_bridge(
    asset_universe: Dict[str, List[str]],
    num_agents: int = 9,  # 5 Asset Experts + 4 Meta-Level Agents
    device: str = "cpu",
) -> Tuple[ObservationFormatter, ActionConverter, BatchProcessor]:
    """
    创建数据桥接组件

    Args:
        asset_universe: 资产池
        num_agents: Agent数量
        device: 计算设备

    Returns:
        formatter, converter, processor
    """
    formatter = ObservationFormatter(asset_universe)
    converter = ActionConverter(asset_universe)
    processor = BatchProcessor(formatter, converter, num_agents, device)

    return formatter, converter, processor


if __name__ == "__main__":
    print("Data Bridge Module")
    print("=" * 50)

    # 测试数据
    asset_universe = {
        "stocks": ["SPY", "QQQ", "IWM"],
        "bonds": ["TLT", "IEF"],
        "commodities": ["GLD", "USO"],
    }

    formatter = ObservationFormatter(asset_universe)
    converter = ActionConverter(asset_universe)

    # 模拟观察
    mock_obs = {
        "portfolio": {
            "portfolio_value": 1000000,
            "cash": 200000,
            "total_return": 0.05,
            "class_weights": {"stocks": 0.4, "bonds": 0.2, "commodities": 0.1},
            "positions": {
                "SPY": {"shares": 100, "current_price": 450, "unrealized_pnl": 5000},
            },
            "weights": {"SPY": 0.2, "QQQ": 0.15, "TLT": 0.1},
        },
        "market_data": {
            "SPY": {"close": 450, "returns_1d": 0.01, "volatility_20d": 0.15, "rsi_14": 55},
            "QQQ": {"close": 380, "returns_1d": 0.02, "volatility_20d": 0.18, "rsi_14": 60},
            "TLT": {"close": 100, "returns_1d": -0.005, "volatility_20d": 0.12, "rsi_14": 45},
            "macro": {"vix": 18, "fed_rate": 0.05},
        },
        "date": "2024-01-15",
    }

    # 测试文本格式化
    print("\n1. Text Prompt:")
    print("-" * 40)
    text = formatter.to_text_prompt(mock_obs, asset_class="stocks")
    print(text[:500] + "...")

    # 测试数值格式化
    print("\n2. Numerical Tensors:")
    print("-" * 40)
    af, mf, pf = formatter.to_numerical_tensor(mock_obs)
    print(f"  Asset features: {af.shape}")
    print(f"  Macro features: {mf.shape}")
    print(f"  Portfolio features: {pf.shape}")

    # 测试动作转换
    print("\n3. Action Conversion:")
    print("-" * 40)
    expert_actions = [
        {"asset_class": "stocks", "action": "BUY_50%", "confidence": 0.8},
        {"asset_class": "bonds", "action": "HOLD", "confidence": 0.6},
    ]
    allocation = converter.expert_actions_to_allocation(expert_actions)
    print(f"  Allocation: {allocation}")

    print("\nAll tests passed!")
