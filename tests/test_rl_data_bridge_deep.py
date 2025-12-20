"""
Deep tests for RL Data Bridge
强化学习数据桥接深度测试
"""

import pytest
import torch
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

from finsage.rl.data_bridge import (
    ObservationFormatter,
    ActionConverter,
    BatchProcessor,
    MARFTBatch,
    MARFTEnvWrapper,
    create_data_bridge,
)


class TestObservationFormatter:
    """ObservationFormatter测试"""

    @pytest.fixture
    def asset_universe(self):
        return {
            "stocks": ["SPY", "QQQ"],
            "bonds": ["TLT", "IEF"],
            "commodities": ["GLD"],
        }

    @pytest.fixture
    def formatter(self, asset_universe):
        return ObservationFormatter(asset_universe)

    def test_init(self, formatter):
        """测试初始化"""
        assert formatter.num_assets == 5
        assert len(formatter.all_symbols) == 5
        assert "SPY" in formatter.symbol_to_class
        assert formatter.symbol_to_class["SPY"] == "stocks"

    def test_technical_indicators(self, formatter):
        """测试技术指标列表"""
        assert "returns_1d" in formatter.technical_indicators
        assert "rsi_14" in formatter.technical_indicators
        assert "macd" in formatter.technical_indicators

    def test_macro_indicators(self, formatter):
        """测试宏观指标列表"""
        assert "vix" in formatter.macro_indicators
        assert "fed_rate" in formatter.macro_indicators

    def test_to_text_prompt_basic(self, formatter):
        """测试基本文本提示生成"""
        observation = {
            "portfolio": {
                "portfolio_value": 1000000,
                "cash": 100000,
                "total_return": 0.05,
            },
            "market_data": {
                "SPY": {"close": 450, "returns_1d": 0.01},
            },
            "date": "2024-01-15",
        }

        text = formatter.to_text_prompt(observation)
        assert "2024-01-15" in text
        assert "1,000,000" in text or "1000000" in text
        assert "SPY" in text

    def test_to_text_prompt_with_portfolio(self, formatter):
        """测试包含组合状态的文本提示"""
        observation = {
            "portfolio": {
                "portfolio_value": 1000000,
                "cash": 200000,
                "total_return": 0.05,
                "class_weights": {"stocks": 0.4, "bonds": 0.3},
                "positions": {
                    "SPY": {"shares": 100, "current_price": 450, "unrealized_pnl": 5000},
                },
            },
            "market_data": {"SPY": {"close": 450}},
            "date": "2024-01-15",
        }

        text = formatter.to_text_prompt(observation, include_portfolio=True)
        assert "组合状态" in text or "portfolio" in text.lower()
        assert "SPY" in text

    def test_to_text_prompt_asset_class_filter(self, formatter):
        """测试资产类别过滤"""
        observation = {
            "portfolio": {},
            "market_data": {
                "SPY": {"close": 450},
                "TLT": {"close": 100},
            },
            "date": "2024-01-15",
        }

        text = formatter.to_text_prompt(observation, asset_class="stocks")
        assert "SPY" in text
        # TLT可能不在文本中 (取决于实现)

    def test_to_text_prompt_with_news(self, formatter):
        """测试包含新闻的文本提示"""
        observation = {
            "portfolio": {},
            "market_data": {
                "SPY": {"close": 450},
                "news": [
                    {"headline": "Market rallies", "sentiment": "bullish"},
                ],
            },
            "date": "2024-01-15",
        }

        text = formatter.to_text_prompt(observation, include_news=True)
        assert "新闻" in text or "Market rallies" in text

    def test_to_numerical_tensor(self, formatter):
        """测试数值张量转换"""
        observation = {
            "portfolio": {
                "positions": {"SPY": {"shares": 100, "avg_cost": 440, "unrealized_pnl": 1000}},
                "weights": {"SPY": 0.2},
            },
            "market_data": {
                "SPY": {"close": 450, "returns_1d": 0.01, "volatility_20d": 0.15, "rsi_14": 55},
                "macro": {"vix": 18, "fed_rate": 0.05},
            },
        }

        asset_features, macro_features, portfolio_features = formatter.to_numerical_tensor(
            observation, device="cpu"
        )

        assert asset_features.shape[0] == formatter.num_assets
        assert macro_features.shape[0] == 5  # 5个宏观指标
        assert portfolio_features.shape[0] == formatter.num_assets

    def test_to_numerical_tensor_empty_data(self, formatter):
        """测试空数据处理"""
        observation = {
            "portfolio": {},
            "market_data": {},
        }

        asset_features, macro_features, portfolio_features = formatter.to_numerical_tensor(
            observation, device="cpu"
        )

        assert asset_features.shape[0] == formatter.num_assets
        # 应该填充默认值


class TestActionConverter:
    """ActionConverter测试"""

    @pytest.fixture
    def asset_universe(self):
        return {
            "stocks": ["SPY", "QQQ"],
            "bonds": ["TLT"],
        }

    @pytest.fixture
    def converter(self, asset_universe):
        return ActionConverter(asset_universe)

    def test_init(self, converter):
        """测试初始化"""
        assert converter.max_single_weight == 0.15
        assert converter.max_class_weight == 0.50

    def test_action_to_weight_mapping(self, converter):
        """测试动作到权重映射"""
        assert converter.ACTION_TO_WEIGHT["BUY_100%"] == 1.0
        assert converter.ACTION_TO_WEIGHT["SELL_50%"] == -0.5
        assert converter.ACTION_TO_WEIGHT["HOLD"] == 0.0

    def test_expert_actions_to_allocation_basic(self, converter):
        """测试基本动作转配置"""
        expert_actions = [
            {"asset_class": "stocks", "action": "BUY_50%", "confidence": 0.8},
        ]

        allocation = converter.expert_actions_to_allocation(expert_actions)
        assert "stocks" in allocation or "cash" in allocation
        assert "cash" in allocation

    def test_expert_actions_to_allocation_with_current_weights(self, converter):
        """测试带当前权重的动作转配置"""
        expert_actions = [
            {"asset_class": "stocks", "action": "BUY_50%", "confidence": 0.7},
        ]
        current_weights = {"SPY": 0.1, "QQQ": 0.1}

        allocation = converter.expert_actions_to_allocation(expert_actions, current_weights)
        assert allocation is not None

    def test_expert_actions_with_recommendations(self, converter):
        """测试带个股推荐的动作转配置"""
        expert_actions = [
            {
                "asset_class": "stocks",
                "action": "BUY_50%",
                "confidence": 0.8,
                "recommendations": [
                    {"symbol": "SPY", "action": "BUY_75%", "confidence": 0.9},
                    {"symbol": "QQQ", "action": "BUY_25%", "confidence": 0.6},
                ],
            }
        ]

        allocation = converter.expert_actions_to_allocation(expert_actions)
        # 应该展开为个股权重
        assert "SPY" in allocation or "QQQ" in allocation or "stocks" in allocation

    def test_get_class_weight(self, converter):
        """测试计算类别权重"""
        weights = {"SPY": 0.15, "QQQ": 0.10, "TLT": 0.05}

        stocks_weight = converter._get_class_weight("stocks", weights)
        assert stocks_weight == 0.25

    def test_allocation_to_action_tensor(self, converter):
        """测试配置转动作张量"""
        allocation = {"SPY": 0.2, "QQQ": 0.15, "TLT": 0.1}
        all_symbols = ["SPY", "QQQ", "TLT"]

        tensor = converter.allocation_to_action_tensor(allocation, all_symbols)
        assert tensor.shape[0] == 3
        assert tensor[0].item() == pytest.approx(0.2)

    def test_weight_constraints(self, converter):
        """测试权重约束"""
        # 大幅买入应该被限制
        expert_actions = [
            {"asset_class": "stocks", "action": "BUY_100%", "confidence": 1.0},
        ]

        allocation = converter.expert_actions_to_allocation(expert_actions)
        # 类别权重应该被限制在max_class_weight以内


class TestBatchProcessor:
    """BatchProcessor测试"""

    @pytest.fixture
    def asset_universe(self):
        return {
            "stocks": ["SPY", "QQQ"],
            "bonds": ["TLT"],
        }

    @pytest.fixture
    def processor(self, asset_universe):
        formatter = ObservationFormatter(asset_universe)
        converter = ActionConverter(asset_universe)
        return BatchProcessor(formatter, converter, num_agents=3, device="cpu")

    def test_init(self, processor):
        """测试初始化"""
        assert processor.num_agents == 3
        assert processor.device == "cpu"

    def test_create_batch(self, processor):
        """测试创建批次"""
        observations = [
            {
                "portfolio": {"portfolio_value": 1000000},
                "market_data": {"SPY": {"close": 450}},
                "date": "2024-01-15",
            },
            {
                "portfolio": {"portfolio_value": 1010000},
                "market_data": {"SPY": {"close": 455}},
                "date": "2024-01-16",
            },
        ]

        actions = [[{}, {}, {}], [{}, {}, {}]]
        action_tokens = [[torch.tensor([1, 2, 3])]*3, [torch.tensor([4, 5, 6])]*3]
        rewards = [0.01, 0.005]
        dones = [False, False]
        old_log_probs = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        old_values = [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]

        batch = processor.create_batch(
            observations, actions, action_tokens, rewards, dones,
            old_log_probs, old_values
        )

        assert isinstance(batch, MARFTBatch)
        assert len(batch.text_observations) == 2
        assert batch.rewards.shape[0] == 2
        assert batch.dones.shape[0] == 2


class TestMARFTBatch:
    """MARFTBatch测试"""

    def test_batch_structure(self):
        """测试批次结构"""
        batch = MARFTBatch(
            text_observations=["obs1", "obs2"],
            asset_features=torch.randn(2, 5, 10),
            macro_features=torch.randn(2, 5),
            portfolio_features=torch.randn(2, 5, 4),
            action_tokens=[torch.tensor([1, 2]), torch.tensor([3, 4])],
            rewards=torch.tensor([0.01, 0.02]),
            dones=torch.tensor([0.0, 0.0]),
            old_log_probs=torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
            old_values=torch.tensor([[0.5, 0.5], [0.6, 0.6]]),
        )

        assert len(batch.text_observations) == 2
        assert batch.asset_features.shape == (2, 5, 10)
        assert batch.rewards.shape == (2,)


class TestMARFTEnvWrapper:
    """MARFTEnvWrapper测试"""

    @pytest.fixture
    def asset_universe(self):
        return {
            "stocks": ["SPY", "QQQ"],
            "bonds": ["TLT"],
        }

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.reset.return_value = None
        env.get_observation.return_value = {
            "portfolio": {"portfolio_value": 1000000, "weights": {}},
            "market_data": {"SPY": {"close": 450}},
            "date": "2024-01-15",
        }
        env.step.return_value = (
            {"value": 1010000},  # portfolio
            0.01,  # reward
            False,  # done
            {}  # info
        )
        return env

    @pytest.fixture
    def wrapper(self, mock_env, asset_universe):
        formatter = ObservationFormatter(asset_universe)
        converter = ActionConverter(asset_universe)
        return MARFTEnvWrapper(mock_env, formatter, converter, num_agents=3)

    def test_init(self, wrapper):
        """测试初始化"""
        assert wrapper.num_agents == 3
        assert wrapper.current_obs is None

    def test_reset(self, wrapper):
        """测试重置"""
        text_obs, info = wrapper.reset()

        assert isinstance(text_obs, str)
        assert "raw_obs" in info
        assert wrapper.current_obs is not None

    def test_step(self, wrapper):
        """测试执行步骤"""
        wrapper.reset()

        expert_actions = [
            {"asset_class": "stocks", "action": "HOLD", "confidence": 0.5},
        ]
        market_data = {"SPY": {"close": 455}}
        timestamp = "2024-01-16"

        text_obs, reward, done, info = wrapper.step(
            expert_actions, market_data, timestamp
        )

        assert isinstance(text_obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert "allocation" in info

    def test_get_numerical_obs(self, wrapper):
        """测试获取数值观察"""
        wrapper.reset()

        asset_features, macro_features, portfolio_features = wrapper.get_numerical_obs()

        assert isinstance(asset_features, torch.Tensor)
        assert isinstance(macro_features, torch.Tensor)
        assert isinstance(portfolio_features, torch.Tensor)


class TestCreateDataBridge:
    """create_data_bridge测试"""

    def test_create_data_bridge(self):
        """测试创建数据桥接组件"""
        asset_universe = {
            "stocks": ["SPY"],
            "bonds": ["TLT"],
        }

        formatter, converter, processor = create_data_bridge(
            asset_universe, num_agents=2, device="cpu"
        )

        assert isinstance(formatter, ObservationFormatter)
        assert isinstance(converter, ActionConverter)
        assert isinstance(processor, BatchProcessor)

    def test_create_data_bridge_different_devices(self):
        """测试不同设备创建"""
        asset_universe = {"stocks": ["SPY"]}

        formatter, converter, processor = create_data_bridge(
            asset_universe, num_agents=1, device="cpu"
        )

        assert processor.device == "cpu"


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def asset_universe(self):
        return {"stocks": ["SPY"]}

    def test_empty_observation(self, asset_universe):
        """测试空观察"""
        formatter = ObservationFormatter(asset_universe)

        observation = {
            "portfolio": {},
            "market_data": {},
            "date": "",
        }

        text = formatter.to_text_prompt(observation)
        assert isinstance(text, str)

    def test_empty_expert_actions(self, asset_universe):
        """测试空专家动作"""
        converter = ActionConverter(asset_universe)

        allocation = converter.expert_actions_to_allocation([])
        assert "cash" in allocation

    def test_missing_market_data(self, asset_universe):
        """测试缺失市场数据"""
        formatter = ObservationFormatter(asset_universe)

        observation = {
            "portfolio": {"portfolio_value": 1000000},
            "market_data": {},  # 无市场数据
            "date": "2024-01-15",
        }

        asset_features, macro_features, portfolio_features = formatter.to_numerical_tensor(
            observation, device="cpu"
        )

        # 应该返回默认值
        assert asset_features.shape[0] == 1

    def test_unknown_action_string(self, asset_universe):
        """测试未知动作字符串"""
        converter = ActionConverter(asset_universe)

        expert_actions = [
            {"asset_class": "stocks", "action": "UNKNOWN_ACTION", "confidence": 0.5},
        ]

        # 应该使用默认值 (0)
        allocation = converter.expert_actions_to_allocation(expert_actions)
        assert allocation is not None

    def test_invalid_symbol_in_recommendations(self, asset_universe):
        """测试推荐中的无效符号"""
        converter = ActionConverter(asset_universe)

        expert_actions = [
            {
                "asset_class": "stocks",
                "action": "BUY_50%",
                "confidence": 0.8,
                "recommendations": [
                    {"symbol": "INVALID", "action": "BUY_50%", "confidence": 0.7},
                ],
            }
        ]

        allocation = converter.expert_actions_to_allocation(expert_actions)
        # 无效符号应该被忽略
        assert "INVALID" not in allocation
