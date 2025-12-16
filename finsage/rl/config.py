"""
MARFT-FinSage Configuration

完整的配置文件，包括:
1. LoRA配置
2. PPO训练配置
3. 环境配置
4. Expert配置
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os


# ============================================================
# 1. LoRA Configuration
# ============================================================

@dataclass
class LoRAConfig:
    """LoRA微调配置"""

    # LoRA基本参数
    r: int = 8                          # LoRA rank
    lora_alpha: int = 16                # LoRA alpha (通常设为2*r)
    lora_dropout: float = 0.05          # Dropout rate

    # 目标模块 (不同模型可能不同)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        # "gate_proj", "up_proj", "down_proj",   # MLP (可选)
    ])

    # 其他参数
    bias: str = "none"                  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"

    # 量化选项
    load_in_4bit: bool = False          # 4bit量化
    load_in_8bit: bool = False          # 8bit量化
    bf16: bool = True                   # 使用bfloat16

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }


# ============================================================
# 2. PPO Training Configuration
# ============================================================

@dataclass
class PPOConfig:
    """PPO训练配置"""

    # 学习率
    lr: float = 5e-7                    # Policy学习率 (LLM微调需要很小的lr)
    critic_lr: float = 5e-4             # Critic学习率

    # PPO超参数
    clip_param: float = 0.2             # PPO clip参数
    ppo_epoch: int = 5                  # PPO更新轮数
    num_mini_batch: int = 4             # Mini-batch数量

    # 损失系数
    entropy_coef: float = 0.01          # Entropy正则化系数
    value_loss_coef: float = 0.5        # Value loss系数

    # 梯度
    max_grad_norm: float = 0.5          # 梯度裁剪
    gradient_cp_steps: int = 1          # 梯度累积步数

    # GAE参数
    gamma: float = 0.99                 # 折扣因子
    gae_lambda: float = 0.95            # GAE lambda

    # 信任区域
    kl_threshold: float = 0.01          # KL散度阈值

    # Agent训练策略
    agent_iteration_interval: int = 0   # 0=同时训练, >0=轮流训练间隔

    # 正则化
    weight_decay: float = 0.0


# ============================================================
# 3. Training Configuration
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""

    # 基本参数
    seed: int = 42
    cuda: bool = True
    cuda_deterministic: bool = True

    # 环境参数
    num_env_steps: int = 1_000_000      # 总训练步数
    rollout_length: int = 256           # Rollout长度
    episode_length: int = 252           # 一年交易日

    # 并行
    n_rollout_threads: int = 1          # 并行环境数 (金融场景通常为1)
    n_training_threads: int = 1         # 训练线程数

    # 保存和日志
    save_interval: int = 100            # 保存间隔 (updates)
    log_interval: int = 10              # 日志间隔 (updates)
    eval_interval: int = 50             # 评估间隔 (updates)

    # 路径
    save_dir: str = "./checkpoints/marft"
    log_dir: str = "./logs/marft"

    # 评估
    use_eval: bool = True
    eval_episodes: int = 5


# ============================================================
# 4. Model Configuration
# ============================================================

@dataclass
class ModelConfig:
    """模型配置"""

    # 基础模型 (96GB显存使用32B全精度)
    model_name_or_path: str = "Qwen/Qwen2.5-32B-Instruct"

    # 加载检查点
    load_path: Optional[str] = None

    # 生成参数
    max_new_tokens: int = 512
    context_window: int = 4096

    # Critic
    critic_hidden_size: int = 1024
    critic_type: str = "action"         # "action" or "financial"


# ============================================================
# 5. Expert Configuration
# ============================================================

@dataclass
class ExpertConfig:
    """Expert配置"""

    # Expert数量
    num_agents: int = 5

    # Expert角色
    roles: List[str] = field(default_factory=lambda: [
        "Stock_Expert",
        "Bond_Expert",
        "Commodity_Expert",
        "REITs_Expert",
        "Crypto_Expert",
    ])

    # 资产类别映射
    asset_classes: List[str] = field(default_factory=lambda: [
        "stocks",
        "bonds",
        "commodities",
        "reits",
        "crypto",
    ])

    # 依赖关系
    dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        "Stock_Expert": [],
        "Bond_Expert": ["Stock_Expert"],
        "Commodity_Expert": ["Stock_Expert", "Bond_Expert"],
        "REITs_Expert": ["Stock_Expert", "Bond_Expert"],
        "Crypto_Expert": ["Stock_Expert"],
    })


# ============================================================
# 6. Environment Configuration
# ============================================================

@dataclass
class EnvConfig:
    """环境配置"""

    # 初始资金
    initial_capital: float = 1_000_000

    # 交易成本
    transaction_cost: float = 0.001     # 0.1%
    slippage: float = 0.0005            # 0.05%

    # 限制
    max_single_weight: float = 0.15     # 单资产最大权重
    max_class_weight: float = 0.50      # 单类别最大权重
    rebalance_threshold: float = 0.02   # 再平衡阈值

    # 数据
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"


# ============================================================
# 7. Reward Configuration
# ============================================================

@dataclass
class RewardConfig:
    """奖励函数配置"""

    # 风险惩罚
    risk_penalty_coef: float = 0.5

    # 交易成本
    transaction_cost_rate: float = 0.001

    # 多样化奖励
    diversification_bonus_coef: float = 0.1

    # 最大回撤惩罚
    max_drawdown_penalty: float = 1.0
    max_drawdown_threshold: float = 0.1


# ============================================================
# 8. Complete Configuration
# ============================================================

@dataclass
class MARFTFinSageConfig:
    """MARFT-FinSage完整配置"""

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # 实验名称
    experiment_name: str = "marft_finsage"

    def save(self, path: str):
        """保存配置到JSON"""
        import json
        from dataclasses import asdict

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MARFTFinSageConfig":
        """从JSON加载配置"""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            lora=LoRAConfig(**data.get("lora", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            training=TrainingConfig(**data.get("training", {})),
            model=ModelConfig(**data.get("model", {})),
            expert=ExpertConfig(**data.get("expert", {})),
            env=EnvConfig(**data.get("env", {})),
            reward=RewardConfig(**data.get("reward", {})),
            experiment_name=data.get("experiment_name", "marft_finsage"),
        )


# ============================================================
# 9. Preset Configurations
# ============================================================

def get_debug_config() -> MARFTFinSageConfig:
    """调试配置 (快速迭代)"""
    config = MARFTFinSageConfig()
    config.training.num_env_steps = 10000
    config.training.rollout_length = 32
    config.training.save_interval = 10
    config.ppo.ppo_epoch = 2
    config.experiment_name = "debug"
    return config


def get_small_config() -> MARFTFinSageConfig:
    """小规模配置 (测试)"""
    config = MARFTFinSageConfig()
    config.training.num_env_steps = 100000
    config.training.rollout_length = 128
    config.model.model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    config.lora.load_in_4bit = True
    config.experiment_name = "small"
    return config


def get_full_config() -> MARFTFinSageConfig:
    """完整配置 (正式训练 - 96GB显存优化)"""
    config = MARFTFinSageConfig()
    config.training.num_env_steps = 1_000_000
    config.training.rollout_length = 512  # 大rollout
    config.model.model_name_or_path = "Qwen/Qwen2.5-32B-Instruct"
    config.lora.r = 16  # 更高rank
    config.lora.lora_alpha = 32
    config.lora.load_in_4bit = False  # 全精度BF16
    config.experiment_name = "full_32b"
    return config


# ============================================================
# 10. Model Path Presets
# ============================================================

MODEL_PRESETS = {
    # Qwen系列
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",

    # LLaMA系列
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",

    # Mistral系列
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",

    # 其他
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat",
}


if __name__ == "__main__":
    print("MARFT-FinSage Configuration")
    print("=" * 50)

    # 默认配置
    config = MARFTFinSageConfig()
    print("\nDefault Configuration:")
    print(f"  Model: {config.model.model_name_or_path}")
    print(f"  LoRA rank: {config.lora.r}")
    print(f"  PPO lr: {config.ppo.lr}")
    print(f"  Training steps: {config.training.num_env_steps}")
    print(f"  Experts: {config.expert.roles}")

    # 保存示例
    config.save("/tmp/marft_config.json")
    print("\nConfig saved to /tmp/marft_config.json")

    # 加载示例
    loaded = MARFTFinSageConfig.load("/tmp/marft_config.json")
    print(f"Config loaded: {loaded.experiment_name}")
