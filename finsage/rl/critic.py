"""
Critic Networks for MARFT-FinSage Integration

实现两种Critic架构:
1. ActionCritic: 基于LLM hidden states的value head (参考MARFT)
2. FinancialCritic: 结合数值特征和文本特征的混合Critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


# ============================================================
# 1. Action-Level Critic (基于MARFT的实现)
# ============================================================

class ActionCritic(nn.Module):
    """
    Action-Level Critic

    使用LLM的hidden states来估计state value
    LLM参数冻结，只训练value head
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        bf16: bool = True,
        hidden_size: int = 1024,
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for ActionCritic")

        self.device = torch.device(device)

        # 加载LLM作为encoder (冻结参数)
        logger.info(f"Loading critic base model from {model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
        ).to(self.device)

        # 冻结LLM参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 获取hidden size
        config = self.base_model.config
        if hasattr(config, "hidden_size"):
            embed_dim = config.hidden_size
        elif hasattr(config, "n_embd"):
            embed_dim = config.n_embd
        else:
            embed_dim = 4096  # 默认值

        # Value Head (MLP)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # 启用gradient checkpointing以节省显存
        self.base_model.gradient_checkpointing_enable()

        logger.info(f"ActionCritic initialized with embed_dim={embed_dim}, hidden_size={hidden_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            values: [batch_size]
        """
        with torch.no_grad():
            outputs = self.base_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # 获取最后一层hidden states的最后一个token
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        last_hidden = hidden_states[:, -1, :].float()  # [batch, hidden_dim]

        # Value head
        values = self.value_head(last_hidden).squeeze(-1)  # [batch]

        return values

    def get_value(self, obs: List[str]) -> torch.Tensor:
        """
        从文本观察获取value估计

        Args:
            obs: 观察文本列表

        Returns:
            values: [batch_size]
        """
        inputs = self.tokenizer(
            obs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        return self.forward(input_ids, attention_mask)

    def get_hidden_representation(self, obs: List[str]) -> torch.Tensor:
        """
        获取文本观察的hidden representation (用于特征融合)

        Args:
            obs: 观察文本列表

        Returns:
            hidden: [batch_size, hidden_size] - value head第一层后的表示
        """
        inputs = self.tokenizer(
            obs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.base_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # 获取最后一层hidden states的最后一个token
        hidden_states = outputs.hidden_states[-1]
        last_hidden = hidden_states[:, -1, :].float()

        # 通过value head的第一层获取中间表示
        hidden = self.value_head[0](last_hidden)  # Linear
        hidden = self.value_head[1](hidden)       # ReLU

        return hidden

    def save_value_head(self, path: str):
        """保存value head参数"""
        torch.save(self.value_head.state_dict(), path)
        logger.info(f"Saved value head to {path}")

    def load_value_head(self, path: str, map_location: str = "cpu"):
        """加载value head参数"""
        state_dict = torch.load(path, map_location=map_location)
        self.value_head.load_state_dict(state_dict)
        logger.info(f"Loaded value head from {path}")


# ============================================================
# 2. Financial State Encoder (数值特征编码器)
# ============================================================

class FinancialStateEncoder(nn.Module):
    """
    金融状态编码器

    将数值型金融特征编码为dense representation
    """

    def __init__(
        self,
        num_assets: int,
        price_features: int = 5,    # OHLCV
        technical_features: int = 10,  # 技术指标数量
        macro_features: int = 5,    # 宏观指标数量
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.feature_dim = price_features + technical_features

        # 资产特征投影
        self.asset_embedding = nn.Linear(self.feature_dim, hidden_size)

        # 宏观特征投影
        self.macro_embedding = nn.Linear(macro_features, hidden_size)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, num_assets + 1, hidden_size))

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.hidden_size = hidden_size

    def forward(
        self,
        asset_features: torch.Tensor,   # [batch, num_assets, feature_dim]
        macro_features: torch.Tensor,   # [batch, macro_dim]
    ) -> torch.Tensor:
        """
        编码金融状态

        Returns:
            encoded: [batch, hidden_size] - 聚合后的状态表示
        """
        batch_size = asset_features.shape[0]

        # 投影资产特征
        asset_embeds = self.asset_embedding(asset_features)  # [batch, num_assets, hidden]

        # 投影宏观特征
        macro_embeds = self.macro_embedding(macro_features).unsqueeze(1)  # [batch, 1, hidden]

        # 拼接 (宏观作为CLS token)
        x = torch.cat([macro_embeds, asset_embeds], dim=1)  # [batch, num_assets+1, hidden]

        # 添加位置编码
        x = x + self.position_embedding[:, :x.shape[1], :]

        # Transformer编码
        x = self.transformer(x)

        # 取CLS token作为输出
        encoded = self.layer_norm(x[:, 0, :])  # [batch, hidden]

        return encoded


# ============================================================
# 3. Financial Critic (混合Critic)
# ============================================================

class FinancialCritic(nn.Module):
    """
    金融场景专用Critic

    结合:
    1. 数值型金融特征 (价格、技术指标、宏观数据)
    2. 文本型观察 (Expert分析、新闻摘要)
    """

    def __init__(
        self,
        num_assets: int,
        llm_path: Optional[str] = None,
        price_features: int = 5,
        technical_features: int = 10,
        macro_features: int = 5,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        use_text_encoder: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.use_text_encoder = use_text_encoder and llm_path is not None

        # 数值特征编码器
        self.numerical_encoder = FinancialStateEncoder(
            num_assets=num_assets,
            price_features=price_features,
            technical_features=technical_features,
            macro_features=macro_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # 文本特征编码器 (可选)
        if self.use_text_encoder:
            self.text_encoder = ActionCritic(
                model_path=llm_path,
                device=device,
                hidden_size=hidden_size,
            )
            combined_size = hidden_size * 2
        else:
            self.text_encoder = None
            combined_size = hidden_size

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        # Value Head (per-agent values)
        self.value_head = nn.Linear(hidden_size // 2, 1)

        # 可选: 多Agent分别估值
        self.multi_agent_head = None

        logger.info(f"FinancialCritic initialized: use_text_encoder={use_text_encoder}")

    def forward(
        self,
        asset_features: torch.Tensor,
        macro_features: torch.Tensor,
        text_obs: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            asset_features: [batch, num_assets, feature_dim]
            macro_features: [batch, macro_dim]
            text_obs: 文本观察列表 (可选)

        Returns:
            values: [batch]
        """
        # 编码数值特征
        numerical_embed = self.numerical_encoder(asset_features, macro_features)  # [batch, hidden_size]

        # 编码文本特征 (如果启用)
        if self.use_text_encoder and text_obs is not None:
            # 获取hidden representation而非value
            text_embed = self.text_encoder.get_hidden_representation(text_obs)  # [batch, hidden_size]
            # 拼接数值和文本特征
            combined = torch.cat([numerical_embed, text_embed], dim=-1)  # [batch, hidden_size * 2]
        else:
            combined = numerical_embed

        # 融合
        fused = self.fusion(combined)

        # Value输出
        values = self.value_head(fused).squeeze(-1)

        return values

    def get_multi_agent_values(
        self,
        asset_features: torch.Tensor,
        macro_features: torch.Tensor,
        num_agents: int,
        text_obs: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        获取多Agent的value估计

        Returns:
            values: [batch, num_agents]
        """
        # 基础value
        base_value = self.forward(asset_features, macro_features, text_obs)

        # 扩展到所有agents (可以根据需要实现不同agent的差异化value)
        values = base_value.unsqueeze(-1).expand(-1, num_agents)

        return values


# ============================================================
# 4. Portfolio Value Critic (组合级别Critic)
# ============================================================

class PortfolioValueCritic(nn.Module):
    """
    组合级别Value Critic

    输入: 当前组合状态 + 市场状态
    输出: 预期组合价值
    """

    def __init__(
        self,
        num_assets: int,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        # 组合状态编码
        # 输入: [权重, 持仓成本, 当前价格, 未实现盈亏, ...]
        portfolio_features = num_assets * 4  # 每个资产4个特征

        # 市场状态编码
        market_features = num_assets * 10  # 每个资产10个技术指标

        total_input = portfolio_features + market_features + 10  # +10 for macro

        # MLP网络
        layers = []
        in_features = total_input
        for i in range(num_layers):
            out_features = hidden_size if i < num_layers - 1 else hidden_size // 2
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_features = out_features

        self.encoder = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_size // 2, 1)

    def forward(
        self,
        portfolio_state: torch.Tensor,  # [batch, portfolio_features]
        market_state: torch.Tensor,     # [batch, market_features]
        macro_state: torch.Tensor,      # [batch, macro_features]
    ) -> torch.Tensor:
        """
        前向传播

        Returns:
            values: [batch]
        """
        # 拼接所有特征
        x = torch.cat([portfolio_state, market_state, macro_state], dim=-1)

        # 编码
        x = self.encoder(x)

        # Value输出
        values = self.value_head(x).squeeze(-1)

        return values


# ============================================================
# 5. Critic Factory
# ============================================================

def create_critic(
    critic_type: str,
    num_assets: int,
    llm_path: Optional[str] = None,
    device: str = "cuda:0",
    **kwargs,
) -> nn.Module:
    """
    创建Critic实例

    Args:
        critic_type: "action" | "financial" | "portfolio"
        num_assets: 资产数量
        llm_path: LLM模型路径 (action和financial类型需要)
        device: 计算设备
        **kwargs: 其他参数

    Returns:
        critic: Critic实例
    """
    if critic_type == "action":
        if llm_path is None:
            raise ValueError("llm_path is required for ActionCritic")
        return ActionCritic(model_path=llm_path, device=device, **kwargs)

    elif critic_type == "financial":
        return FinancialCritic(
            num_assets=num_assets,
            llm_path=llm_path,
            device=device,
            **kwargs,
        )

    elif critic_type == "portfolio":
        return PortfolioValueCritic(num_assets=num_assets, **kwargs)

    else:
        raise ValueError(f"Unknown critic type: {critic_type}")


if __name__ == "__main__":
    print("Critic Networks Module")
    print("=" * 50)

    # 测试数值编码器
    print("\nTesting FinancialStateEncoder...")
    encoder = FinancialStateEncoder(
        num_assets=10,
        price_features=5,
        technical_features=10,
        macro_features=5,
        hidden_size=256,
    )

    batch_size = 4
    asset_features = torch.randn(batch_size, 10, 15)  # 10 assets, 15 features each
    macro_features = torch.randn(batch_size, 5)

    encoded = encoder(asset_features, macro_features)
    print(f"  Input: asset_features {asset_features.shape}, macro_features {macro_features.shape}")
    print(f"  Output: {encoded.shape}")

    # 测试Portfolio Critic
    print("\nTesting PortfolioValueCritic...")
    critic = PortfolioValueCritic(num_assets=10, hidden_size=256)

    portfolio_state = torch.randn(batch_size, 40)  # 10 * 4
    market_state = torch.randn(batch_size, 100)    # 10 * 10
    macro_state = torch.randn(batch_size, 10)

    values = critic(portfolio_state, market_state, macro_state)
    print(f"  Input: portfolio {portfolio_state.shape}, market {market_state.shape}, macro {macro_state.shape}")
    print(f"  Output values: {values.shape}")

    print("\nAll tests passed!")
