"""
Deep tests for RL Configuration
强化学习配置深度测试
"""

import pytest
import json
import tempfile
import os
from dataclasses import asdict

from finsage.rl.config import (
    LoRAConfig,
    PPOConfig,
    TrainingConfig,
    ModelConfig,
    ExpertConfig,
    EnvConfig,
    RewardConfig,
    MARFTFinSageConfig,
    get_debug_config,
    get_small_config,
    get_full_config,
    MODEL_PRESETS,
)


class TestLoRAConfig:
    """LoRA配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = LoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_custom_rank(self):
        """测试自定义rank"""
        config = LoRAConfig(r=16, lora_alpha=32)
        assert config.r == 16
        assert config.lora_alpha == 32

    def test_to_dict(self):
        """测试转字典"""
        config = LoRAConfig()
        d = config.to_dict()
        assert d["r"] == 8
        assert d["lora_alpha"] == 16
        assert "target_modules" in d

    def test_quantization_options(self):
        """测试量化选项"""
        config = LoRAConfig(load_in_4bit=True)
        assert config.load_in_4bit == True
        assert config.load_in_8bit == False

    def test_bias_options(self):
        """测试bias选项"""
        config = LoRAConfig(bias="all")
        assert config.bias == "all"


class TestPPOConfig:
    """PPO配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = PPOConfig()
        assert config.lr == 5e-7
        assert config.critic_lr == 5e-4
        assert config.clip_param == 0.2
        assert config.ppo_epoch == 5

    def test_gae_params(self):
        """测试GAE参数"""
        config = PPOConfig()
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95

    def test_custom_hyperparams(self):
        """测试自定义超参数"""
        config = PPOConfig(
            lr=1e-6,
            clip_param=0.1,
            entropy_coef=0.02
        )
        assert config.lr == 1e-6
        assert config.clip_param == 0.1
        assert config.entropy_coef == 0.02

    def test_gradient_settings(self):
        """测试梯度设置"""
        config = PPOConfig()
        assert config.max_grad_norm == 0.5
        assert config.gradient_cp_steps == 1

    def test_kl_threshold(self):
        """测试KL散度阈值"""
        config = PPOConfig(kl_threshold=0.02)
        assert config.kl_threshold == 0.02


class TestTrainingConfig:
    """训练配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = TrainingConfig()
        assert config.seed == 42
        assert config.cuda == True
        assert config.num_env_steps == 1_000_000

    def test_rollout_settings(self):
        """测试rollout设置"""
        config = TrainingConfig()
        assert config.rollout_length == 256
        assert config.episode_length == 252

    def test_parallel_settings(self):
        """测试并行设置"""
        config = TrainingConfig()
        assert config.n_rollout_threads == 1
        assert config.n_training_threads == 1

    def test_logging_intervals(self):
        """测试日志间隔"""
        config = TrainingConfig()
        assert config.save_interval == 100
        assert config.log_interval == 10
        assert config.eval_interval == 50

    def test_path_settings(self):
        """测试路径设置"""
        config = TrainingConfig()
        assert "checkpoints" in config.save_dir
        assert "logs" in config.log_dir


class TestModelConfig:
    """模型配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = ModelConfig()
        assert "Qwen" in config.model_name_or_path
        assert config.max_new_tokens == 512

    def test_context_window(self):
        """测试上下文窗口"""
        config = ModelConfig()
        assert config.context_window == 4096

    def test_critic_settings(self):
        """测试Critic设置"""
        config = ModelConfig()
        assert config.critic_hidden_size == 1024
        assert config.critic_type == "action"

    def test_custom_model_path(self):
        """测试自定义模型路径"""
        config = ModelConfig(model_name_or_path="custom/model")
        assert config.model_name_or_path == "custom/model"


class TestExpertConfig:
    """Expert配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = ExpertConfig()
        assert config.num_agents == 9  # 5 Asset Experts + 4 Meta-Level Agents
        assert len(config.roles) == 9
        assert "Stock_Expert" in config.roles
        assert "Portfolio_Manager" in config.roles  # Meta-Level Agent

    def test_asset_classes(self):
        """测试资产类别"""
        config = ExpertConfig()
        assert "stocks" in config.asset_classes
        assert "bonds" in config.asset_classes
        assert "crypto" in config.asset_classes

    def test_dependencies(self):
        """测试依赖关系"""
        config = ExpertConfig()
        assert config.dependencies["Stock_Expert"] == []
        assert "Stock_Expert" in config.dependencies["Bond_Expert"]
        assert "Stock_Expert" in config.dependencies["Crypto_Expert"]

    def test_custom_roles(self):
        """测试自定义角色"""
        config = ExpertConfig(
            num_agents=3,
            roles=["Expert_A", "Expert_B", "Expert_C"]
        )
        assert config.num_agents == 3
        assert len(config.roles) == 3


class TestEnvConfig:
    """环境配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = EnvConfig()
        assert config.initial_capital == 1_000_000
        assert config.transaction_cost == 0.001

    def test_weight_constraints(self):
        """测试权重约束"""
        config = EnvConfig()
        assert config.max_single_weight == 0.15
        assert config.max_class_weight == 0.50
        assert config.rebalance_threshold == 0.02

    def test_date_range(self):
        """测试日期范围"""
        config = EnvConfig()
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2023-12-31"

    def test_trading_costs(self):
        """测试交易成本"""
        config = EnvConfig()
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005


class TestRewardConfig:
    """奖励配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = RewardConfig()
        assert config.risk_penalty_coef == 0.5
        assert config.transaction_cost_rate == 0.001

    def test_bonus_coefficients(self):
        """测试奖励系数"""
        config = RewardConfig()
        assert config.diversification_bonus_coef == 0.1

    def test_drawdown_settings(self):
        """测试回撤设置"""
        config = RewardConfig()
        assert config.max_drawdown_penalty == 1.0
        assert config.max_drawdown_threshold == 0.1


class TestMARFTFinSageConfig:
    """完整配置测试"""

    def test_default_init(self):
        """测试默认初始化"""
        config = MARFTFinSageConfig()
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.ppo, PPOConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.expert, ExpertConfig)
        assert isinstance(config.env, EnvConfig)
        assert isinstance(config.reward, RewardConfig)

    def test_experiment_name(self):
        """测试实验名称"""
        config = MARFTFinSageConfig()
        assert config.experiment_name == "marft_finsage"

    def test_save_and_load(self):
        """测试保存和加载"""
        config = MARFTFinSageConfig()
        config.experiment_name = "test_experiment"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            loaded = MARFTFinSageConfig.load(f.name)

            assert loaded.experiment_name == "test_experiment"
            assert loaded.lora.r == config.lora.r
            assert loaded.ppo.lr == config.ppo.lr

            os.unlink(f.name)

    def test_nested_config_modification(self):
        """测试嵌套配置修改"""
        config = MARFTFinSageConfig()
        config.lora.r = 32
        config.ppo.lr = 1e-5

        assert config.lora.r == 32
        assert config.ppo.lr == 1e-5


class TestPresetConfigs:
    """预设配置测试"""

    def test_debug_config(self):
        """测试调试配置"""
        config = get_debug_config()
        assert config.training.num_env_steps == 10000
        assert config.training.rollout_length == 32
        assert config.ppo.ppo_epoch == 2
        assert config.experiment_name == "debug"

    def test_small_config(self):
        """测试小规模配置"""
        config = get_small_config()
        assert config.training.num_env_steps == 100000
        assert "3B" in config.model.model_name_or_path
        assert config.lora.load_in_4bit == True
        assert config.experiment_name == "small"

    def test_full_config(self):
        """测试完整配置"""
        config = get_full_config()
        assert config.training.num_env_steps == 1_000_000
        assert config.training.rollout_length == 512
        assert config.lora.r == 16
        assert "32B" in config.model.model_name_or_path


class TestModelPresets:
    """模型预设测试"""

    def test_qwen_models(self):
        """测试Qwen模型"""
        assert "qwen2.5-3b" in MODEL_PRESETS
        assert "qwen2.5-7b" in MODEL_PRESETS
        assert "qwen2.5-14b" in MODEL_PRESETS
        assert "qwen2.5-32b" in MODEL_PRESETS

    def test_llama_models(self):
        """测试LLaMA模型"""
        assert "llama3-8b" in MODEL_PRESETS
        assert "llama3.1-8b" in MODEL_PRESETS

    def test_mistral_models(self):
        """测试Mistral模型"""
        assert "mistral-7b" in MODEL_PRESETS

    def test_model_paths_valid(self):
        """测试模型路径格式"""
        for name, path in MODEL_PRESETS.items():
            assert "/" in path  # HuggingFace格式


class TestConfigSerialization:
    """配置序列化测试"""

    def test_asdict_conversion(self):
        """测试asdict转换"""
        config = MARFTFinSageConfig()
        d = asdict(config)

        assert "lora" in d
        assert "ppo" in d
        assert "training" in d
        assert d["lora"]["r"] == 8

    def test_json_serialization(self):
        """测试JSON序列化"""
        config = MARFTFinSageConfig()
        d = asdict(config)

        # 应该可以序列化为JSON
        json_str = json.dumps(d)
        loaded = json.loads(json_str)

        assert loaded["lora"]["r"] == 8
        assert loaded["ppo"]["lr"] == 5e-7

    def test_load_partial_config(self):
        """测试加载部分配置"""
        partial_config = {
            "lora": {"r": 16},
            "experiment_name": "partial_test"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            f.flush()

            loaded = MARFTFinSageConfig.load(f.name)
            assert loaded.lora.r == 16
            assert loaded.experiment_name == "partial_test"
            # 其他字段应该使用默认值
            assert loaded.ppo.lr == 5e-7

            os.unlink(f.name)


class TestConfigValidation:
    """配置验证测试"""

    def test_lora_rank_positive(self):
        """测试LoRA rank必须为正"""
        config = LoRAConfig(r=16)
        assert config.r > 0

    def test_learning_rate_range(self):
        """测试学习率范围"""
        config = PPOConfig()
        assert 0 < config.lr < 1
        assert 0 < config.critic_lr < 1

    def test_clip_param_range(self):
        """测试clip参数范围"""
        config = PPOConfig()
        assert 0 < config.clip_param < 1

    def test_gamma_range(self):
        """测试gamma范围"""
        config = PPOConfig()
        assert 0 < config.gamma <= 1

    def test_weight_constraints_consistency(self):
        """测试权重约束一致性"""
        config = EnvConfig()
        # 单资产最大权重应小于类别最大权重
        assert config.max_single_weight <= config.max_class_weight
