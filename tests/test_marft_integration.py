#!/usr/bin/env python
"""
MARFT-FinSage Integration Tests

测试所有组件的基本功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datetime import datetime

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_ok(msg):
    print(f"  ✓ {msg}")

def print_fail(msg):
    print(f"  ✗ {msg}")


# ============================================================
# Test 1: Configuration Module
# ============================================================

def test_config():
    print_header("Test 1: Configuration Module")

    try:
        from finsage.rl.config import (
            MARFTFinSageConfig,
            LoRAConfig,
            PPOConfig,
            get_debug_config,
        )

        # 默认配置
        config = MARFTFinSageConfig()
        assert config.experiment_name == "marft_finsage"
        assert config.lora.r == 8
        assert config.ppo.clip_param == 0.2
        print_ok(f"Default config: experiment={config.experiment_name}")

        # Debug配置
        debug_config = get_debug_config()
        assert debug_config.training.num_env_steps == 10000
        print_ok(f"Debug config: steps={debug_config.training.num_env_steps}")

        # 保存/加载
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded = MARFTFinSageConfig.load(f.name)
            assert loaded.lora.r == config.lora.r
            os.unlink(f.name)
        print_ok("Config save/load works")

        return True
    except Exception as e:
        print_fail(f"Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 2: Data Bridge Module
# ============================================================

def test_data_bridge():
    print_header("Test 2: Data Bridge Module")

    try:
        from finsage.rl.data_bridge import (
            ObservationFormatter,
            ActionConverter,
            create_data_bridge,
        )

        # 资产池
        asset_universe = {
            "stocks": ["SPY", "QQQ", "IWM"],
            "bonds": ["TLT", "IEF"],
            "commodities": ["GLD", "USO"],
        }

        formatter = ObservationFormatter(asset_universe)
        converter = ActionConverter(asset_universe)
        print_ok(f"Created formatter with {formatter.num_assets} assets")

        # 模拟观察
        mock_obs = {
            "portfolio": {
                "portfolio_value": 1000000,
                "cash": 200000,
                "total_return": 0.05,
                "class_weights": {"stocks": 0.4, "bonds": 0.2},
                "positions": {
                    "SPY": {"shares": 100, "current_price": 450, "unrealized_pnl": 5000},
                },
                "weights": {"SPY": 0.2, "QQQ": 0.15, "TLT": 0.1},
            },
            "market_data": {
                "SPY": {"close": 450, "returns_1d": 0.01, "volatility_20d": 0.15, "rsi_14": 55},
                "QQQ": {"close": 380, "returns_1d": 0.02, "rsi_14": 60},
                "TLT": {"close": 100, "returns_1d": -0.005, "rsi_14": 45},
                "macro": {"vix": 18, "fed_rate": 0.05},
            },
            "date": "2024-01-15",
        }

        # 测试文本格式化
        text_prompt = formatter.to_text_prompt(mock_obs)
        assert "市场日期" in text_prompt
        assert "SPY" in text_prompt
        print_ok(f"Text prompt generated: {len(text_prompt)} chars")

        # 测试数值格式化
        af, mf, pf = formatter.to_numerical_tensor(mock_obs)
        assert af.shape[0] == formatter.num_assets
        assert mf.shape[0] == 5  # macro features
        print_ok(f"Tensors: asset={af.shape}, macro={mf.shape}, portfolio={pf.shape}")

        # 测试动作转换
        expert_actions = [
            {"asset_class": "stocks", "action": "BUY_50%", "confidence": 0.8},
            {"asset_class": "bonds", "action": "HOLD", "confidence": 0.6},
        ]
        allocation = converter.expert_actions_to_allocation(expert_actions)
        assert "cash" in allocation
        print_ok(f"Action conversion: {len(allocation)} allocations")

        return True
    except Exception as e:
        print_fail(f"Data bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 3: Critic Networks (CPU only, no LLM)
# ============================================================

def test_critic_networks():
    print_header("Test 3: Critic Networks (without LLM)")

    try:
        from finsage.rl.critic import (
            FinancialStateEncoder,
            PortfolioValueCritic,
        )

        # 测试金融状态编码器
        encoder = FinancialStateEncoder(
            num_assets=10,
            price_features=5,
            technical_features=10,
            macro_features=5,
            hidden_size=256,
        )

        batch_size = 4
        asset_features = torch.randn(batch_size, 10, 15)
        macro_features = torch.randn(batch_size, 5)

        encoded = encoder(asset_features, macro_features)
        assert encoded.shape == (batch_size, 256)
        print_ok(f"FinancialStateEncoder: {asset_features.shape} -> {encoded.shape}")

        # 测试组合Critic
        critic = PortfolioValueCritic(num_assets=10, hidden_size=256)

        portfolio_state = torch.randn(batch_size, 40)
        market_state = torch.randn(batch_size, 100)
        macro_state = torch.randn(batch_size, 10)

        values = critic(portfolio_state, market_state, macro_state)
        assert values.shape == (batch_size,)
        print_ok(f"PortfolioValueCritic: value shape = {values.shape}")

        # 测试梯度
        values.sum().backward()
        has_grad = any(p.grad is not None for p in critic.parameters())
        assert has_grad
        print_ok("Gradient computation works")

        return True
    except Exception as e:
        print_fail(f"Critic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 4: MARFT Integration Components
# ============================================================

def test_marft_integration():
    print_header("Test 4: MARFT Integration Components")

    try:
        from finsage.rl.marft_integration import (
            FINSAGE_AGENT_PROFILES,
            FinSageActionBuffer,
            FinSageRewardFunction,
        )

        # 测试Agent Profiles
        assert len(FINSAGE_AGENT_PROFILES) == 5
        print_ok(f"Agent profiles: {[p['role'] for p in FINSAGE_AGENT_PROFILES]}")

        # 测试Action Buffer
        buffer = FinSageActionBuffer(
            episode_length=100,
            num_agents=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # 插入数据
        for _ in range(10):
            buffer.insert(
                obs=["obs"] * 5,
                actions=[{"action": "HOLD"}] * 5,
                log_probs=[0.0] * 5,
                reward=0.01,
                value=[0.0] * 5,
                done=False,
            )

        assert buffer.step == 10
        print_ok(f"ActionBuffer: {buffer.step} steps inserted")

        # 测试GAE计算
        next_value = [0.0] * 5
        advantages, returns = buffer.compute_gae_and_returns(next_value)
        assert advantages.shape == (10, 5)
        assert returns.shape == (10, 5)
        print_ok(f"GAE computed: advantages={advantages.shape}, returns={returns.shape}")

        # 测试奖励函数
        reward_fn = FinSageRewardFunction(
            risk_penalty_coef=0.5,
            transaction_cost_rate=0.001,
        )

        total_reward, components = reward_fn.compute_reward(
            portfolio_return=0.02,
            portfolio_volatility=0.15,
            transaction_volume=10000,
            portfolio_weights=np.array([0.2, 0.3, 0.3, 0.1, 0.1]),
            max_drawdown=0.05,
        )

        assert isinstance(total_reward, float)
        assert "return_reward" in components
        print_ok(f"Reward function: total={total_reward:.4f}")

        return True
    except Exception as e:
        print_fail(f"MARFT integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 5: Expert Profiles
# ============================================================

def test_expert_profiles():
    print_header("Test 5: Expert Profiles")

    try:
        from finsage.rl.lora_expert import (
            create_finsage_expert_profiles,
            ExpertProfile,
        )

        profiles = create_finsage_expert_profiles()

        assert len(profiles) == 5
        print_ok(f"Created {len(profiles)} expert profiles")

        # 检查依赖关系
        for p in profiles:
            print_ok(f"  {p.role} ({p.asset_class}) -> deps: {p.dependencies}")

        # 验证拓扑顺序
        stock_expert = profiles[0]
        assert stock_expert.dependencies == []
        print_ok("Dependency topology is valid")

        return True
    except Exception as e:
        print_fail(f"Expert profiles test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 6: Environment Integration (if available)
# ============================================================

def test_environment_integration():
    print_header("Test 6: Environment Integration")

    try:
        from finsage.environment.multi_asset_env import MultiAssetTradingEnv
        from finsage.rl.data_bridge import MARFTEnvWrapper, ObservationFormatter, ActionConverter

        # 创建环境
        env = MultiAssetTradingEnv()
        print_ok(f"MultiAssetTradingEnv created")

        # 重置
        portfolio = env.reset()
        print_ok(f"Environment reset: capital=${portfolio.initial_capital:,.0f}")

        # 获取观察
        obs = env.get_observation()
        assert "portfolio" in obs
        assert "market_data" in obs
        print_ok(f"Observation keys: {list(obs.keys())}")

        # 创建wrapper
        formatter = ObservationFormatter(env.asset_universe)
        converter = ActionConverter(env.asset_universe)
        wrapper = MARFTEnvWrapper(env, formatter, converter)

        text_obs, info = wrapper.reset()
        assert len(text_obs) > 0
        print_ok(f"MARFTEnvWrapper works: text_obs length={len(text_obs)}")

        return True
    except ImportError as e:
        print_fail(f"Environment not available: {e}")
        return False
    except Exception as e:
        print_fail(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 7: LoRA Expert (if transformers available)
# ============================================================

def test_lora_expert():
    print_header("Test 7: LoRA Expert (optional)")

    try:
        from transformers import AutoTokenizer
        print_ok("transformers is available")
    except ImportError:
        print_fail("transformers not installed, skipping LoRA test")
        return True  # Not a failure, just skip

    try:
        from finsage.rl.lora_expert import LoRAExpert, create_finsage_expert_profiles

        # 注意: 实际加载模型需要很长时间和大量显存
        # 这里只测试类是否可以导入
        profiles = create_finsage_expert_profiles()
        print_ok("LoRAExpert class is available")
        print_ok("Note: Full model loading requires GPU and ~16GB VRAM")

        return True
    except Exception as e:
        print_fail(f"LoRA Expert test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 60)
    print(" MARFT-FinSage Integration Tests")
    print("=" * 60)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" PyTorch: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" CUDA device: {torch.cuda.get_device_name(0)}")

    tests = [
        ("Configuration", test_config),
        ("Data Bridge", test_data_bridge),
        ("Critic Networks", test_critic_networks),
        ("MARFT Integration", test_marft_integration),
        ("Expert Profiles", test_expert_profiles),
        ("Environment", test_environment_integration),
        ("LoRA Expert", test_lora_expert),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            results[name] = False
            print(f"  Unexpected error in {name}: {e}")

    # Summary
    print_header("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print("\n  ⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
