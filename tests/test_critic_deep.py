#!/usr/bin/env python
"""
Critic Networks Deep Testing - Critic网络深度测试
Coverage: critic.py (FinancialStateEncoder, FinancialCritic, PortfolioValueCritic, create_critic)

注意: ActionCritic 需要实际 LLM 模型,这里仅测试不需要 LLM 的组件
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch


# ============================================================
# Test 1: FinancialStateEncoder
# ============================================================

class TestFinancialStateEncoder:
    """测试金融状态编码器"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.critic import FinancialStateEncoder
        assert FinancialStateEncoder is not None

    def test_initialization_default(self):
        """测试默认初始化"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(num_assets=10)

        assert encoder.num_assets == 10
        assert encoder.hidden_size == 256  # 默认值
        assert encoder.feature_dim == 15  # 5 price + 10 technical

    def test_initialization_custom(self):
        """测试自定义初始化"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(
            num_assets=5,
            price_features=6,
            technical_features=15,
            macro_features=10,
            hidden_size=512,
            num_layers=3,
            num_heads=8,
            dropout=0.2,
        )

        assert encoder.num_assets == 5
        assert encoder.hidden_size == 512
        assert encoder.feature_dim == 21  # 6 + 15

    def test_forward_basic(self):
        """测试基本前向传播"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(
            num_assets=10,
            price_features=5,
            technical_features=10,
            macro_features=5,
            hidden_size=64,  # 使用小尺寸加速测试
            num_layers=1,
        )

        batch_size = 4
        asset_features = torch.randn(batch_size, 10, 15)  # 10 assets, 15 features
        macro_features = torch.randn(batch_size, 5)

        output = encoder(asset_features, macro_features)

        assert output.shape == (batch_size, 64)  # hidden_size

    def test_forward_single_asset(self):
        """测试单个资产"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(
            num_assets=1,
            price_features=5,
            technical_features=10,
            macro_features=5,
            hidden_size=32,
            num_layers=1,
        )

        batch_size = 2
        asset_features = torch.randn(batch_size, 1, 15)
        macro_features = torch.randn(batch_size, 5)

        output = encoder(asset_features, macro_features)

        assert output.shape == (batch_size, 32)

    def test_forward_large_batch(self):
        """测试大批量"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(
            num_assets=5,
            hidden_size=32,
            num_layers=1,
        )

        batch_size = 64
        asset_features = torch.randn(batch_size, 5, 15)
        macro_features = torch.randn(batch_size, 5)

        output = encoder(asset_features, macro_features)

        assert output.shape == (batch_size, 32)

    def test_forward_gradient_flow(self):
        """测试梯度流"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(
            num_assets=5,
            hidden_size=32,
            num_layers=1,
        )

        asset_features = torch.randn(2, 5, 15, requires_grad=True)
        macro_features = torch.randn(2, 5, requires_grad=True)

        output = encoder(asset_features, macro_features)
        loss = output.sum()
        loss.backward()

        assert asset_features.grad is not None
        assert macro_features.grad is not None

    def test_parameters_count(self):
        """测试参数数量"""
        from finsage.rl.critic import FinancialStateEncoder

        encoder = FinancialStateEncoder(
            num_assets=10,
            hidden_size=64,
            num_layers=2,
        )

        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # 所有参数都可训练

    def test_position_embedding_shape(self):
        """测试位置编码形状"""
        from finsage.rl.critic import FinancialStateEncoder

        num_assets = 10
        hidden_size = 64
        encoder = FinancialStateEncoder(
            num_assets=num_assets,
            hidden_size=hidden_size,
            num_layers=1,
        )

        # Position embedding shape should be [1, num_assets + 1, hidden_size]
        # +1 for macro token
        assert encoder.position_embedding.shape == (1, num_assets + 1, hidden_size)


# ============================================================
# Test 2: PortfolioValueCritic
# ============================================================

class TestPortfolioValueCritic:
    """测试组合价值Critic"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.critic import PortfolioValueCritic
        assert PortfolioValueCritic is not None

    def test_initialization_default(self):
        """测试默认初始化"""
        from finsage.rl.critic import PortfolioValueCritic

        critic = PortfolioValueCritic(num_assets=10)

        assert critic is not None

    def test_initialization_custom(self):
        """测试自定义初始化"""
        from finsage.rl.critic import PortfolioValueCritic

        critic = PortfolioValueCritic(
            num_assets=5,
            hidden_size=128,
            num_layers=4,
        )

        assert critic is not None

    def test_forward_basic(self):
        """测试基本前向传播"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 10
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=64,
            num_layers=2,
        )

        batch_size = 4
        portfolio_state = torch.randn(batch_size, num_assets * 4)  # 每个资产4个特征
        market_state = torch.randn(batch_size, num_assets * 10)    # 每个资产10个技术指标
        macro_state = torch.randn(batch_size, 10)

        values = critic(portfolio_state, market_state, macro_state)

        assert values.shape == (batch_size,)

    def test_forward_single_sample(self):
        """测试单个样本"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 5
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=64,
            num_layers=2,
        )

        portfolio_state = torch.randn(1, num_assets * 4)
        market_state = torch.randn(1, num_assets * 10)
        macro_state = torch.randn(1, 10)

        values = critic(portfolio_state, market_state, macro_state)

        assert values.shape == (1,)

    def test_forward_gradient_flow(self):
        """测试梯度流"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 5
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=2,
        )

        portfolio_state = torch.randn(2, num_assets * 4, requires_grad=True)
        market_state = torch.randn(2, num_assets * 10, requires_grad=True)
        macro_state = torch.randn(2, 10, requires_grad=True)

        values = critic(portfolio_state, market_state, macro_state)
        loss = values.sum()
        loss.backward()

        assert portfolio_state.grad is not None
        assert market_state.grad is not None
        assert macro_state.grad is not None

    def test_forward_different_batch_sizes(self):
        """测试不同批量大小"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 10
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=2,
        )

        for batch_size in [1, 2, 8, 16, 32]:
            portfolio_state = torch.randn(batch_size, num_assets * 4)
            market_state = torch.randn(batch_size, num_assets * 10)
            macro_state = torch.randn(batch_size, 10)

            values = critic(portfolio_state, market_state, macro_state)

            assert values.shape == (batch_size,)

    def test_value_range(self):
        """测试输出值范围"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 5
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=2,
        )

        # 多次运行检查输出是否有限
        for _ in range(10):
            portfolio_state = torch.randn(4, num_assets * 4)
            market_state = torch.randn(4, num_assets * 10)
            macro_state = torch.randn(4, 10)

            values = critic(portfolio_state, market_state, macro_state)

            assert torch.isfinite(values).all()

    def test_training_mode(self):
        """测试训练模式"""
        from finsage.rl.critic import PortfolioValueCritic

        critic = PortfolioValueCritic(num_assets=5, hidden_size=32, num_layers=2)

        # 训练模式
        critic.train()
        assert critic.training

        # 评估模式
        critic.eval()
        assert not critic.training


# ============================================================
# Test 3: FinancialCritic (without LLM)
# ============================================================

class TestFinancialCritic:
    """测试金融Critic (不使用LLM)"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.critic import FinancialCritic
        assert FinancialCritic is not None

    def test_initialization_without_llm(self):
        """测试不使用LLM初始化"""
        from finsage.rl.critic import FinancialCritic

        critic = FinancialCritic(
            num_assets=10,
            llm_path=None,  # 不使用LLM
            use_text_encoder=False,
            device="cpu",
        )

        assert critic is not None
        assert not critic.use_text_encoder
        assert critic.text_encoder is None

    def test_initialization_custom(self):
        """测试自定义初始化"""
        from finsage.rl.critic import FinancialCritic

        critic = FinancialCritic(
            num_assets=5,
            llm_path=None,
            price_features=6,
            technical_features=12,
            macro_features=8,
            hidden_size=256,
            num_layers=3,
            num_heads=4,
            use_text_encoder=False,
            device="cpu",
        )

        assert critic is not None

    def test_forward_without_text(self):
        """测试不使用文本的前向传播"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 10
        price_features = 5
        technical_features = 10
        macro_features = 5

        critic = FinancialCritic(
            num_assets=num_assets,
            price_features=price_features,
            technical_features=technical_features,
            macro_features=macro_features,
            hidden_size=64,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        batch_size = 4
        asset_features = torch.randn(batch_size, num_assets, price_features + technical_features)
        macro_features_tensor = torch.randn(batch_size, macro_features)

        values = critic(asset_features, macro_features_tensor, text_obs=None)

        assert values.shape == (batch_size,)

    def test_forward_gradient_flow(self):
        """测试梯度流"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 5
        critic = FinancialCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        asset_features = torch.randn(2, num_assets, 15, requires_grad=True)
        macro_features = torch.randn(2, 5, requires_grad=True)

        values = critic(asset_features, macro_features)
        loss = values.sum()
        loss.backward()

        assert asset_features.grad is not None
        assert macro_features.grad is not None

    def test_get_multi_agent_values(self):
        """测试多智能体价值估计"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 5
        num_agents = 3
        critic = FinancialCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        batch_size = 4
        asset_features = torch.randn(batch_size, num_assets, 15)
        macro_features = torch.randn(batch_size, 5)

        values = critic.get_multi_agent_values(
            asset_features, macro_features, num_agents
        )

        assert values.shape == (batch_size, num_agents)

    def test_forward_single_sample(self):
        """测试单个样本"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 5
        critic = FinancialCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        asset_features = torch.randn(1, num_assets, 15)
        macro_features = torch.randn(1, 5)

        values = critic(asset_features, macro_features)

        assert values.shape == (1,)


# ============================================================
# Test 4: create_critic Factory
# ============================================================

class TestCreateCriticFactory:
    """测试Critic工厂函数"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.critic import create_critic
        assert create_critic is not None

    def test_create_portfolio_critic(self):
        """测试创建Portfolio Critic"""
        from finsage.rl.critic import create_critic

        critic = create_critic(
            critic_type="portfolio",
            num_assets=10,
            hidden_size=64,
        )

        assert critic is not None
        assert hasattr(critic, 'forward')

    def test_create_financial_critic_without_llm(self):
        """测试创建Financial Critic (无LLM)"""
        from finsage.rl.critic import create_critic

        critic = create_critic(
            critic_type="financial",
            num_assets=10,
            llm_path=None,
            use_text_encoder=False,
            device="cpu",
        )

        assert critic is not None

    def test_create_action_critic_requires_llm(self):
        """测试创建Action Critic需要LLM"""
        from finsage.rl.critic import create_critic

        with pytest.raises(ValueError, match="llm_path is required"):
            create_critic(
                critic_type="action",
                num_assets=10,
                llm_path=None,
            )

    def test_create_unknown_critic_type(self):
        """测试创建未知类型Critic"""
        from finsage.rl.critic import create_critic

        with pytest.raises(ValueError, match="Unknown critic type"):
            create_critic(
                critic_type="unknown",
                num_assets=10,
            )

    def test_create_with_kwargs(self):
        """测试使用额外参数创建"""
        from finsage.rl.critic import create_critic

        critic = create_critic(
            critic_type="portfolio",
            num_assets=5,
            hidden_size=128,
            num_layers=4,
        )

        assert critic is not None


# ============================================================
# Test 5: ActionCritic Mock Tests
# ============================================================

class TestActionCriticMock:
    """测试ActionCritic (使用Mock)"""

    def test_action_critic_init_without_transformers(self):
        """测试无transformers时的初始化"""
        from finsage.rl.critic import HAS_TRANSFORMERS

        if not HAS_TRANSFORMERS:
            from finsage.rl.critic import ActionCritic

            with pytest.raises(ImportError, match="transformers is required"):
                ActionCritic(model_path="test", device="cpu")

    @patch('finsage.rl.critic.HAS_TRANSFORMERS', False)
    def test_action_critic_requires_transformers(self):
        """测试ActionCritic需要transformers"""
        # 重新加载模块以应用patch
        import importlib
        import finsage.rl.critic as critic_module
        importlib.reload(critic_module)

        # 应该因为没有transformers而失败
        if not critic_module.HAS_TRANSFORMERS:
            with pytest.raises(ImportError):
                critic_module.ActionCritic(model_path="test", device="cpu")


# ============================================================
# Test 6: Edge Cases
# ============================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_financial_encoder_zero_assets(self):
        """测试零资产情况"""
        from finsage.rl.critic import FinancialStateEncoder

        # 零资产应该仍然可以初始化（位置编码只有macro token）
        encoder = FinancialStateEncoder(
            num_assets=0,
            hidden_size=32,
            num_layers=1,
        )

        # 但前向传播会有问题
        assert encoder is not None

    def test_portfolio_critic_extreme_inputs(self):
        """测试极端输入"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 5
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=2,
        )

        batch_size = 2

        # 极大值
        portfolio_state = torch.ones(batch_size, num_assets * 4) * 1e6
        market_state = torch.ones(batch_size, num_assets * 10) * 1e6
        macro_state = torch.ones(batch_size, 10) * 1e6

        values = critic(portfolio_state, market_state, macro_state)
        assert torch.isfinite(values).all()

        # 极小值
        portfolio_state = torch.ones(batch_size, num_assets * 4) * -1e6
        market_state = torch.ones(batch_size, num_assets * 10) * -1e6
        macro_state = torch.ones(batch_size, 10) * -1e6

        values = critic(portfolio_state, market_state, macro_state)
        assert torch.isfinite(values).all()

    def test_financial_critic_nan_handling(self):
        """测试NaN处理"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 5
        critic = FinancialCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        batch_size = 2
        asset_features = torch.randn(batch_size, num_assets, 15)
        macro_features = torch.randn(batch_size, 5)

        # 输入中包含NaN
        asset_features[0, 0, 0] = float('nan')

        values = critic(asset_features, macro_features)

        # 输出可能包含NaN，但不应该崩溃
        assert values.shape == (batch_size,)

    def test_device_placement(self):
        """测试设备放置"""
        from finsage.rl.critic import FinancialCritic

        critic = FinancialCritic(
            num_assets=5,
            hidden_size=32,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        # 检查模型在正确的设备上
        assert critic.device == torch.device("cpu")

        # 测试前向传播
        asset_features = torch.randn(2, 5, 15)
        macro_features = torch.randn(2, 5)

        values = critic(asset_features, macro_features)
        assert values.device == torch.device("cpu")


# ============================================================
# Test 7: Integration Tests
# ============================================================

class TestIntegration:
    """集成测试"""

    def test_financial_encoder_in_critic(self):
        """测试FinancialStateEncoder在FinancialCritic中的集成"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 10
        critic = FinancialCritic(
            num_assets=num_assets,
            hidden_size=64,
            num_layers=2,
            use_text_encoder=False,
            device="cpu",
        )

        # 验证内部组件存在
        assert critic.numerical_encoder is not None
        assert critic.fusion is not None
        assert critic.value_head is not None

    def test_training_loop(self):
        """测试训练循环"""
        from finsage.rl.critic import PortfolioValueCritic

        num_assets = 5
        critic = PortfolioValueCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=2,
        )

        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        # 模拟训练步骤
        for _ in range(5):
            portfolio_state = torch.randn(4, num_assets * 4)
            market_state = torch.randn(4, num_assets * 10)
            macro_state = torch.randn(4, 10)
            target_values = torch.randn(4)

            optimizer.zero_grad()
            values = critic(portfolio_state, market_state, macro_state)
            loss = ((values - target_values) ** 2).mean()
            loss.backward()
            optimizer.step()

        # 训练应该成功完成
        assert True

    def test_eval_mode_deterministic(self):
        """测试评估模式下的确定性"""
        from finsage.rl.critic import FinancialCritic

        num_assets = 5
        critic = FinancialCritic(
            num_assets=num_assets,
            hidden_size=32,
            num_layers=1,
            use_text_encoder=False,
            device="cpu",
        )

        critic.eval()

        asset_features = torch.randn(2, num_assets, 15)
        macro_features = torch.randn(2, 5)

        # 两次评估应该得到相同结果
        with torch.no_grad():
            values1 = critic(asset_features, macro_features)
            values2 = critic(asset_features, macro_features)

        assert torch.allclose(values1, values2)


# ============================================================
# Test 8: Module Structure Tests
# ============================================================

class TestModuleStructure:
    """测试模块结构"""

    def test_has_transformers_flag(self):
        """测试HAS_TRANSFORMERS标志"""
        from finsage.rl.critic import HAS_TRANSFORMERS

        # 应该是布尔值
        assert isinstance(HAS_TRANSFORMERS, bool)

    def test_module_exports(self):
        """测试模块导出"""
        from finsage.rl import critic

        # 检查主要类是否可用
        assert hasattr(critic, 'FinancialStateEncoder')
        assert hasattr(critic, 'FinancialCritic')
        assert hasattr(critic, 'PortfolioValueCritic')
        assert hasattr(critic, 'create_critic')
        assert hasattr(critic, 'ActionCritic')

    def test_critic_inheritance(self):
        """测试Critic类继承"""
        from finsage.rl.critic import (
            FinancialStateEncoder,
            FinancialCritic,
            PortfolioValueCritic,
        )

        # 所有Critic都应该继承自nn.Module
        assert issubclass(FinancialStateEncoder, nn.Module)
        assert issubclass(FinancialCritic, nn.Module)
        assert issubclass(PortfolioValueCritic, nn.Module)


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Critic Networks Deep Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
