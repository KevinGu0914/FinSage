"""
LoRA-Enabled Expert Agent for MARFT-FinSage Integration

使用PEFT库实现LoRA微调的Expert Agent，支持:
1. 基于LLM的金融分析
2. 动作log_prob计算 (用于PPO训练)
3. 高效的参数更新 (仅训练LoRA适配器)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
import logging

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/peft not installed. LoRA features disabled.")

logger = logging.getLogger(__name__)


# ============================================================
# 1. LoRA Configuration for Financial Experts
# ============================================================

FINSAGE_LORA_CONFIG = {
    "r": 8,                           # LoRA rank
    "lora_alpha": 16,                 # LoRA alpha
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],  # 更多投影层
    "lora_dropout": 0.05,             # Dropout for regularization
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


@dataclass
class ExpertProfile:
    """Expert Agent Profile"""
    role: str
    asset_class: str
    expertise: str
    system_prompt: str
    dependencies: List[str]
    device: str = "cuda:0"


# ============================================================
# 2. LoRA-Enabled Expert Agent
# ============================================================

class LoRAExpert:
    """
    LoRA微调的Expert Agent

    实现:
    1. 基于LLM的金融分析和建议生成
    2. 动作log_prob计算 (用于PPO)
    3. LoRA参数的高效训练
    """

    def __init__(
        self,
        model_path: str,
        profile: ExpertProfile,
        device: str = "cuda:0",
        load_path: Optional[str] = None,
        load_in_4bit: bool = False,
        bf16: bool = True,
        lora_config: Optional[Dict] = None,
        max_new_tokens: int = 512,
        context_window: int = 4096,
    ):
        """
        初始化LoRA Expert

        Args:
            model_path: 基础LLM路径 (如 "Qwen/Qwen2.5-7B-Instruct")
            profile: Expert配置
            device: 计算设备
            load_path: LoRA检查点路径 (可选)
            load_in_4bit: 是否使用4bit量化
            bf16: 是否使用bfloat16
            lora_config: LoRA配置 (可选)
            max_new_tokens: 最大生成token数
            context_window: 上下文窗口大小
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and peft are required for LoRAExpert")

        self.profile = profile
        self.role = profile.role
        self.asset_class = profile.asset_class
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window

        # 量化配置
        if load_in_4bit:
            assert bf16, "4bit quantization requires bf16"
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            quant_config = None

        # 加载基础模型
        logger.info(f"Loading base model from {model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            quantization_config=quant_config,
            device_map="auto" if load_in_4bit else None,
        )

        if not load_in_4bit:
            self.base_model = self.base_model.to(self.device)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # 启用输入梯度
        self.base_model.enable_input_require_grads()

        # 配置LoRA
        lora_cfg = lora_config or FINSAGE_LORA_CONFIG
        if load_path is None:
            # 新建LoRA适配器
            peft_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg.get("lora_dropout", 0),
                bias=lora_cfg.get("bias", "none"),
                task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            )
            self.model = get_peft_model(self.base_model, peft_config)
            logger.info(f"Created new LoRA adapter for {self.role}")
        else:
            # 加载已有LoRA适配器
            adapter_path = os.path.join(load_path, self.role)
            self.model = PeftModel.from_pretrained(
                self.base_model, adapter_path, adapter_name=self.role
            )
            self.model.set_adapter(self.role)
            logger.info(f"Loaded LoRA adapter from {adapter_path}")

        self.model.print_trainable_parameters()
        self.model.train()

    def _build_prompt(self, market_obs: str, predecessor_actions: List[Dict] = None) -> str:
        """
        构建完整prompt

        Args:
            market_obs: 市场观察 (格式化的市场数据)
            predecessor_actions: 前序Agent的动作

        Returns:
            完整的prompt字符串
        """
        # System prompt
        prompt = f"""<|im_start|>system
{self.profile.system_prompt}
<|im_end|>
"""

        # 添加前序Agent的建议
        if predecessor_actions:
            prompt += "<|im_start|>context\n## 其他专家的分析建议:\n"
            for action in predecessor_actions:
                role = action.get("role", "Expert")
                analysis = action.get("analysis", "")
                prompt += f"\n### {role}:\n{analysis}\n"
            prompt += "<|im_end|>\n"

        # User prompt (市场观察)
        prompt += f"""<|im_start|>user
{market_obs}

请分析上述市场数据，给出你对{self.asset_class}资产的投资建议。
输出格式要求JSON:
{{
    "action": "BUY_25%/BUY_50%/BUY_75%/BUY_100%/HOLD/SELL_25%/SELL_50%/SELL_75%/SELL_100%/SHORT_25%/SHORT_50%/SHORT_75%/SHORT_100%",
    "confidence": 0.0-1.0,
    "reasoning": "决策理由",
    "risk_assessment": {{"volatility": 0.0-1.0, "downside_risk": 0.0-1.0}}
}}
<|im_end|>
<|im_start|>{self.role}
"""
        return prompt

    @torch.no_grad()
    def generate_action(
        self,
        market_obs: str,
        predecessor_actions: List[Dict] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> Tuple[Dict, torch.Tensor, str]:
        """
        生成投资动作

        Args:
            market_obs: 市场观察
            predecessor_actions: 前序Agent动作
            temperature: 采样温度
            top_k: Top-K采样
            do_sample: 是否采样

        Returns:
            action_dict: 动作字典
            action_tokens: 动作token序列
            raw_response: 原始响应文本
        """
        prompt = self._build_prompt(market_obs, predecessor_actions)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_window,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_length = input_ids.shape[1]

        # Generate
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

        # 提取生成的token
        action_tokens = outputs.sequences[0, input_length:]
        raw_response = self.tokenizer.decode(action_tokens, skip_special_tokens=True)

        # 解析JSON
        action_dict = self._parse_action_response(raw_response)
        action_dict["role"] = self.role
        action_dict["analysis"] = raw_response

        return action_dict, action_tokens, raw_response

    def _parse_action_response(self, response: str) -> Dict:
        """解析LLM响应为动作字典"""
        try:
            # 找到第一个{和最后一个}来提取完整JSON (支持嵌套)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # 默认返回HOLD
        return {
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": "Unable to parse response",
            "risk_assessment": {"volatility": 0.5, "downside_risk": 0.5}
        }

    def get_action_log_prob(
        self,
        obs: str,
        action_tokens: torch.Tensor,
        predecessor_actions: List[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算动作的log probability和entropy

        这是PPO训练的核心方法

        Args:
            obs: 市场观察
            action_tokens: 动作token序列 (shape: [seq_len])
            predecessor_actions: 前序Agent动作

        Returns:
            log_prob: 动作的log probability (scalar)
            entropy: 动作分布的entropy (scalar)
        """
        prompt = self._build_prompt(obs, predecessor_actions)

        # Tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_window,
        )
        prompt_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_mask = prompt_inputs["attention_mask"].to(self.device)
        prompt_length = prompt_ids.shape[1]

        # 确保action_tokens在正确设备上
        if action_tokens.dim() == 1:
            action_tokens = action_tokens.unsqueeze(0)
        action_tokens = action_tokens.to(self.device)

        # 拼接prompt和action
        full_ids = torch.cat([prompt_ids, action_tokens], dim=1)
        action_mask = (action_tokens != self.tokenizer.pad_token_id).long()
        full_mask = torch.cat([prompt_mask, action_mask], dim=1)

        # 前向传播获取logits
        outputs = self.model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # 提取action部分的logits (需要shift by 1因为是预测下一个token)
        action_logits = logits[:, prompt_length - 1:-1, :]  # [batch, action_len, vocab_size]

        # 计算log softmax
        log_softmax = torch.log_softmax(action_logits, dim=-1)

        # 获取实际action token的log prob
        action_len = (action_tokens != self.tokenizer.pad_token_id).sum().item()
        token_log_probs = torch.gather(
            log_softmax[:, :action_len, :],
            dim=-1,
            index=action_tokens[:, :action_len].unsqueeze(-1)
        ).squeeze(-1)

        # 汇总为整个动作的log prob
        action_log_prob = token_log_probs.sum()

        # 计算entropy
        probs = torch.softmax(action_logits[:, :action_len, :], dim=-1)
        entropy = Categorical(probs=probs).entropy().mean()

        return action_log_prob, entropy

    def get_batch_action_log_probs(
        self,
        obs_list: List[str],
        action_tokens_list: List[torch.Tensor],
        predecessor_actions_list: List[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量计算动作log probability

        Args:
            obs_list: 观察列表
            action_tokens_list: 动作token列表
            predecessor_actions_list: 前序动作列表

        Returns:
            log_probs: [batch_size]
            entropies: [batch_size]
        """
        batch_size = len(obs_list)
        log_probs = []
        entropies = []

        predecessor_actions_list = predecessor_actions_list or [None] * batch_size

        for obs, action_tokens, pred_actions in zip(
            obs_list, action_tokens_list, predecessor_actions_list
        ):
            log_prob, entropy = self.get_action_log_prob(obs, action_tokens, pred_actions)
            log_probs.append(log_prob)
            entropies.append(entropy)

        return torch.stack(log_probs), torch.stack(entropies)

    def save(self, save_dir: str):
        """保存LoRA适配器"""
        adapter_path = os.path.join(save_dir, self.role)
        os.makedirs(adapter_path, exist_ok=True)
        self.model.save_pretrained(adapter_path)
        logger.info(f"Saved LoRA adapter to {adapter_path}")

    def train(self):
        """设置为训练模式"""
        self.model.train()

    def eval(self):
        """设置为评估模式"""
        self.model.eval()

    def parameters(self):
        """返回可训练参数"""
        return self.model.parameters()


# ============================================================
# 3. Expert Profile Factory
# ============================================================

def create_finsage_expert_profiles() -> List[ExpertProfile]:
    """创建FinSage的5个Expert Profile"""

    profiles = [
        ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票投资分析",
            system_prompt="""你是一位专业的股票投资专家，擅长:
- 基本面分析：财务报表、盈利能力、估值模型 (P/E, P/B, DCF)
- 技术面分析：价格趋势、成交量、技术指标 (MA, RSI, MACD)
- 行业分析：行业周期、竞争格局、政策影响
- 市场情绪分析：新闻事件、投资者情绪指标

你负责分析股票类资产 (SPY, QQQ, IWM, VTI等ETF及个股)。
基于提供的数据给出明确的交易建议和置信度。""",
            dependencies=[],
        ),
        ExpertProfile(
            role="Bond_Expert",
            asset_class="bonds",
            expertise="债券投资分析",
            system_prompt="""你是一位专业的债券投资专家，擅长:
- 利率分析：收益率曲线、久期、凸性计算
- 信用分析：信用评级变化、违约风险、信用利差
- 宏观分析：央行政策、通胀预期、经济周期
- 债券估值：现金流折现、相对价值分析

你负责分析债券类资产 (TLT, IEF, LQD, HYG等)。
需要参考股票专家的观点来判断风险偏好环境。""",
            dependencies=["Stock_Expert"],
        ),
        ExpertProfile(
            role="Commodity_Expert",
            asset_class="commodities",
            expertise="大宗商品投资分析",
            system_prompt="""你是一位专业的大宗商品投资专家，擅长:
- 供需分析：产量、库存、消费趋势
- 宏观因素：美元走势、地缘政治、季节性规律
- 期货市场：期限结构 (contango/backwardation)、持仓分析
- 跨品种分析：能源、贵金属、农产品关联

你负责分析商品类资产 (GLD, SLV, USO, DBA等)。
需要综合考虑股票和债券专家的宏观判断。""",
            dependencies=["Stock_Expert", "Bond_Expert"],
        ),
        ExpertProfile(
            role="REITs_Expert",
            asset_class="reits",
            expertise="房地产投资信托分析",
            system_prompt="""你是一位专业的房地产投资信托(REITs)专家，擅长:
- 物业分析：租金收益率、出租率、物业估值
- 行业细分：办公、零售、工业、住宅、数据中心REITs
- 利率敏感性：REITs与利率的关系分析
- 财务分析：FFO、AFFO、NAV、派息率

你负责分析REITs类资产 (VNQ, IYR, XLRE等)。
需要参考股票和债券专家的观点 (REITs兼具股债特性)。""",
            dependencies=["Stock_Expert", "Bond_Expert"],
        ),
        ExpertProfile(
            role="Crypto_Expert",
            asset_class="crypto",
            expertise="加密货币投资分析",
            system_prompt="""你是一位专业的加密货币投资专家，擅长:
- 链上分析：交易量、活跃地址、持仓分布、whale动向
- 技术发展：协议升级、生态系统发展、DeFi/NFT趋势
- 市场情绪：社交媒体情绪、恐惧贪婪指数
- 监管动态：各国政策变化、合规发展

你负责分析加密货币资产 (BTC, ETH等)。
加密货币与风险资产相关性较高，需参考股票专家观点。""",
            dependencies=["Stock_Expert"],
        ),
    ]

    return profiles


# ============================================================
# 4. Multi-Expert System (MAS for FinSage)
# ============================================================

class FinSageMAS:
    """
    FinSage Multi-Agent System

    管理多个LoRA Expert的协同决策
    """

    def __init__(
        self,
        model_path: str,
        profiles: List[ExpertProfile] = None,
        load_path: Optional[str] = None,
        load_in_4bit: bool = False,
        bf16: bool = True,
        max_new_tokens: int = 512,
        context_window: int = 4096,
    ):
        self.profiles = profiles or create_finsage_expert_profiles()
        self.num_agents = len(self.profiles)
        self.max_new_tokens = max_new_tokens

        # 分配设备
        available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not available_devices:
            available_devices = ["cpu"]

        # 初始化Experts
        self.experts: Dict[str, LoRAExpert] = {}
        for i, profile in enumerate(self.profiles):
            device = available_devices[i % len(available_devices)]
            profile.device = device

            expert = LoRAExpert(
                model_path=model_path,
                profile=profile,
                device=device,
                load_path=load_path,
                load_in_4bit=load_in_4bit,
                bf16=bf16,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            self.experts[profile.role] = expert

        self.tokenizer = list(self.experts.values())[0].tokenizer
        logger.info(f"Initialized FinSageMAS with {self.num_agents} experts")

    def get_dependency_order(self) -> List[str]:
        """获取Expert执行顺序 (基于依赖关系的拓扑排序)"""
        # 简化版: 按定义顺序 (已经考虑了依赖)
        return [p.role for p in self.profiles]

    @torch.no_grad()
    def generate_joint_action(
        self,
        market_obs: str,
        temperature: float = 0.7,
    ) -> Tuple[List[Dict], List[torch.Tensor], List[str]]:
        """
        Sequential action generation

        按依赖顺序生成所有Expert的动作

        Returns:
            actions: 所有Expert的动作字典
            action_tokens: 所有Expert的动作token
            prompts: 所有Expert的完整prompt
        """
        execution_order = self.get_dependency_order()
        predecessor_actions = []

        all_actions = []
        all_tokens = []
        all_prompts = []

        for role in execution_order:
            expert = self.experts[role]

            # 获取该Expert依赖的前序动作
            profile = next(p for p in self.profiles if p.role == role)
            relevant_predecessors = [
                a for a in predecessor_actions
                if a["role"] in profile.dependencies
            ]

            # 生成动作
            action, tokens, response = expert.generate_action(
                market_obs=market_obs,
                predecessor_actions=relevant_predecessors,
                temperature=temperature,
            )

            all_actions.append(action)
            all_tokens.append(tokens)
            all_prompts.append(expert._build_prompt(market_obs, relevant_predecessors))

            # 添加到前序动作
            predecessor_actions.append(action)

        return all_actions, all_tokens, all_prompts

    def get_joint_action_log_probs(
        self,
        obs_list: List[str],
        action_tokens_list: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算联合动作的log probability

        Args:
            obs_list: 观察列表 [batch_size]
            action_tokens_list: 动作token [batch_size, num_agents, seq_len]

        Returns:
            log_probs: [batch_size, num_agents]
            entropies: [batch_size, num_agents]
        """
        batch_size = len(obs_list)
        execution_order = self.get_dependency_order()

        all_log_probs = torch.zeros(batch_size, self.num_agents)
        all_entropies = torch.zeros(batch_size, self.num_agents)

        for batch_idx in range(batch_size):
            predecessor_actions = []

            for agent_idx, role in enumerate(execution_order):
                expert = self.experts[role]
                profile = next(p for p in self.profiles if p.role == role)

                # 获取相关前序动作
                relevant_predecessors = [
                    a for a in predecessor_actions
                    if a["role"] in profile.dependencies
                ]

                # 计算log prob
                action_tokens = action_tokens_list[batch_idx][agent_idx]
                log_prob, entropy = expert.get_action_log_prob(
                    obs_list[batch_idx],
                    action_tokens,
                    relevant_predecessors,
                )

                all_log_probs[batch_idx, agent_idx] = log_prob
                all_entropies[batch_idx, agent_idx] = entropy

                # 更新前序动作 (需要解码token来获取action)
                action_text = self.tokenizer.decode(action_tokens, skip_special_tokens=True)
                predecessor_actions.append({
                    "role": role,
                    "analysis": action_text,
                })

        return all_log_probs, all_entropies

    def save(self, save_dir: str):
        """保存所有Expert"""
        os.makedirs(save_dir, exist_ok=True)
        for role, expert in self.experts.items():
            expert.save(save_dir)
        logger.info(f"Saved all experts to {save_dir}")

    def train(self):
        for expert in self.experts.values():
            expert.train()

    def eval(self):
        for expert in self.experts.values():
            expert.eval()


if __name__ == "__main__":
    # 测试代码
    print("LoRA Expert Module")
    print("=" * 50)

    profiles = create_finsage_expert_profiles()
    print(f"\nCreated {len(profiles)} expert profiles:")
    for p in profiles:
        print(f"  - {p.role} ({p.asset_class})")
        print(f"    Dependencies: {p.dependencies if p.dependencies else 'None'}")
