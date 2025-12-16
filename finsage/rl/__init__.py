"""
MARFT-FinSage RL Module

多智能体强化学习微调模块，整合MARFT框架与FinSage金融投资系统

模块结构:
- config.py: 配置定义 (LoRA, PPO, Training, Expert等)
- lora_expert.py: LoRA微调的Expert Agent实现
- critic.py: Critic网络 (ActionCritic, FinancialCritic)
- data_bridge.py: 数据格式转换 (Env Obs <-> LLM Prompt <-> Tensor)
- marft_integration.py: 主整合类

使用示例:
```python
from finsage.rl import MARFTFinSageConfig, LoRAExpert, ActionCritic
from finsage.rl.data_bridge import create_data_bridge

# 加载配置
config = MARFTFinSageConfig()

# 创建Expert
expert = LoRAExpert(
    model_path=config.model.model_name_or_path,
    profile=expert_profile,
    device="cuda:0",
)

# 创建Critic
critic = ActionCritic(
    model_path=config.model.model_name_or_path,
    device="cuda:0",
)

# 创建数据桥接
formatter, converter, processor = create_data_bridge(asset_universe)
```
"""

# Configuration
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

# LoRA Expert
from finsage.rl.lora_expert import (
    LoRAExpert,
    ExpertProfile,
    FinSageMAS,
    create_finsage_expert_profiles,
    FINSAGE_LORA_CONFIG,
)

# Critic Networks
from finsage.rl.critic import (
    ActionCritic,
    FinancialStateEncoder,
    FinancialCritic,
    PortfolioValueCritic,
    create_critic,
)

# Data Bridge
from finsage.rl.data_bridge import (
    ObservationFormatter,
    ActionConverter,
    BatchProcessor,
    MARFTBatch,
    MARFTEnvWrapper,
    create_data_bridge,
)

# Main Integration
from finsage.rl.marft_integration import (
    FINSAGE_AGENT_PROFILES,
    FlexMGState,
    FinSageFlexMGEnv,
    FinSageActionBuffer,
    FinSageAPPOTrainer,
    FinSageRewardFunction,
    MARFTFinSageIntegration,
    DEFAULT_MARFT_FINSAGE_CONFIG,
)


__all__ = [
    # Config
    "LoRAConfig",
    "PPOConfig",
    "TrainingConfig",
    "ModelConfig",
    "ExpertConfig",
    "EnvConfig",
    "RewardConfig",
    "MARFTFinSageConfig",
    "get_debug_config",
    "get_small_config",
    "get_full_config",
    "MODEL_PRESETS",

    # LoRA Expert
    "LoRAExpert",
    "ExpertProfile",
    "FinSageMAS",
    "create_finsage_expert_profiles",
    "FINSAGE_LORA_CONFIG",

    # Critic
    "ActionCritic",
    "FinancialStateEncoder",
    "FinancialCritic",
    "PortfolioValueCritic",
    "create_critic",

    # Data Bridge
    "ObservationFormatter",
    "ActionConverter",
    "BatchProcessor",
    "MARFTBatch",
    "MARFTEnvWrapper",
    "create_data_bridge",

    # Integration
    "FINSAGE_AGENT_PROFILES",
    "FlexMGState",
    "FinSageFlexMGEnv",
    "FinSageActionBuffer",
    "FinSageAPPOTrainer",
    "FinSageRewardFunction",
    "MARFTFinSageIntegration",
    "DEFAULT_MARFT_FINSAGE_CONFIG",
]
