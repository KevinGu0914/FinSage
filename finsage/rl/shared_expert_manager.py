"""
Shared Base Model Expert Manager

使用单个基础模型 + 多个LoRA适配器实现多Expert训练
避免加载多个完整模型导致的显存不足问题

Architecture:
- 1个Base Model (~65GB for 32B model)
- 5个LoRA Adapters (~16MB each)
- Total: ~65GB + 80MB = ~65GB

Inference Acceleration Features:
- Flash Attention 2: 高效注意力实现，减少显存占用
- torch.compile: PyTorch 2.0 编译优化
- Static KV Cache: 静态缓存预分配，减少动态内存分配
- Batch Inference: 批量推理，提高吞吐量
- Prompt Caching: 系统提示缓存，避免重复编码
"""

import os
import json
import time
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import StaticCache
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    StaticCache = None

# 检查 Flash Attention 2 可用性
try:
    from transformers.utils import is_flash_attn_2_available
    HAS_FLASH_ATTN = is_flash_attn_2_available()
except ImportError:
    HAS_FLASH_ATTN = False

# 检查 vLLM 可用性
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None
    SamplingParams = None
    LoRARequest = None

logger = logging.getLogger(__name__)


# ============================================================
# vLLM Inference Engine (Optional, 3-5x faster)
# ============================================================

class VLLMInferenceEngine:
    """
    vLLM 推理引擎封装

    使用 vLLM 进行高性能推理:
    - Continuous Batching: 连续批处理
    - PagedAttention: 高效 KV Cache 管理
    - 支持 LoRA 动态加载

    安装: pip install vllm

    性能对比 (A100 80GB, 32B model):
    - HuggingFace: ~10 tokens/s
    - vLLM: ~50 tokens/s (5x faster)
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        enable_lora: bool = True,
        max_lora_rank: int = 16,
        dtype: str = "bfloat16",
    ):
        """
        初始化 vLLM 推理引擎

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行 GPU 数量
            gpu_memory_utilization: GPU 显存使用率
            max_model_len: 最大上下文长度
            enable_lora: 是否启用 LoRA 支持
            max_lora_rank: 最大 LoRA rank
            dtype: 数据类型
        """
        if not HAS_VLLM:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        self.model_path = model_path
        self.enable_lora = enable_lora

        logger.info(f"Initializing vLLM engine with {model_path}")
        logger.info(f"  tensor_parallel_size={tensor_parallel_size}")
        logger.info(f"  gpu_memory_utilization={gpu_memory_utilization}")
        logger.info(f"  enable_lora={enable_lora}")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            trust_remote_code=True,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.lora_requests: Dict[str, LoRARequest] = {}
        self.inference_times: List[float] = []

        logger.info("vLLM engine initialized successfully")

    def register_lora(self, lora_name: str, lora_path: str):
        """注册 LoRA 适配器"""
        if not self.enable_lora:
            logger.warning("LoRA not enabled in vLLM engine")
            return

        lora_id = len(self.lora_requests) + 1
        self.lora_requests[lora_name] = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_id,
            lora_local_path=lora_path,
        )
        logger.info(f"Registered LoRA adapter: {lora_name} (id={lora_id})")

    def generate(
        self,
        prompts: List[str],
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        max_tokens: int = 512,
    ) -> List[str]:
        """
        批量生成

        Args:
            prompts: 提示列表
            lora_name: LoRA 适配器名称 (可选)
            temperature: 采样温度
            top_k: Top-K 采样
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本列表
        """
        start_time = time.time()

        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        # LoRA 请求
        lora_request = None
        if lora_name and lora_name in self.lora_requests:
            lora_request = self.lora_requests[lora_name]

        # 批量生成
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )

        # 提取结果
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        tokens_generated = sum(len(self.tokenizer.encode(r)) for r in results)
        throughput = tokens_generated / elapsed if elapsed > 0 else 0

        logger.debug(f"vLLM generated {len(prompts)} prompts in {elapsed:.2f}s "
                     f"({throughput:.1f} tokens/s)")

        return results

    def generate_single(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        max_tokens: int = 512,
    ) -> str:
        """单个提示生成"""
        results = self.generate([prompt], lora_name, temperature, top_k, max_tokens)
        return results[0] if results else ""

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.inference_times:
            return {"count": 0, "avg_time": 0}

        return {
            "engine": "vLLM",
            "count": len(self.inference_times),
            "avg_time": f"{np.mean(self.inference_times):.3f}s",
            "total_time": f"{sum(self.inference_times):.1f}s",
            "registered_loras": list(self.lora_requests.keys()),
        }


# ============================================================
# Prompt Cache for System Prompts
# ============================================================

@dataclass
class CachedPrompt:
    """缓存的提示信息"""
    prompt_hash: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    timestamp: float = field(default_factory=time.time)


class PromptCache:
    """
    Prompt 缓存管理器
    缓存系统提示的编码结果，避免重复tokenization
    """

    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict[str, CachedPrompt] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_prompt(self, prompt: str) -> str:
        """计算提示的哈希值"""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def get(self, prompt: str) -> Optional[CachedPrompt]:
        """获取缓存的提示"""
        prompt_hash = self._hash_prompt(prompt)
        if prompt_hash in self.cache:
            self.hits += 1
            # 移动到末尾 (LRU)
            self.cache.move_to_end(prompt_hash)
            return self.cache[prompt_hash]
        self.misses += 1
        return None

    def put(
        self,
        prompt: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """存入缓存"""
        prompt_hash = self._hash_prompt(prompt)

        # 检查容量
        if len(self.cache) >= self.max_size:
            # 移除最旧的条目
            self.cache.popitem(last=False)

        self.cache[prompt_hash] = CachedPrompt(
            prompt_hash=prompt_hash,
            input_ids=input_ids.cpu().clone(),
            attention_mask=attention_mask.cpu().clone(),
        )

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        """缓存统计"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ============================================================
# Expert Profile Definitions
# ============================================================

EXPERT_CONFIGS = [
    {
        "role": "Stock_Expert",
        "asset_class": "stocks",
        "assets": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
        "dependencies": [],
        "system_prompt": """你是一位专业的股票市场分析师。
你需要分析股票市场数据，包括价格走势、技术指标(RSI、MACD、均线)和宏观环境。
基于分析给出投资建议：BUY/SELL/HOLD，以及具体仓位建议。
输出格式：JSON格式，包含action、confidence、reasoning字段。"""
    },
    {
        "role": "Bond_Expert",
        "asset_class": "bonds",
        "assets": ["TLT", "IEF", "LQD", "HYG"],
        "dependencies": ["Stock_Expert"],
        "system_prompt": """你是一位专业的债券市场分析师。
你需要分析债券市场数据，包括收益率曲线、信用利差、利率预期等。
参考股票专家的分析结论，给出债券配置建议。
输出格式：JSON格式，包含action、confidence、reasoning字段。"""
    },
    {
        "role": "Commodity_Expert",
        "asset_class": "commodities",
        "assets": ["GLD", "SLV", "USO", "DBA"],
        "dependencies": ["Stock_Expert", "Bond_Expert"],
        "system_prompt": """你是一位专业的大宗商品分析师。
你需要分析商品市场数据，包括供需基本面、美元走势、通胀预期等。
参考股票和债券专家的分析，给出商品配置建议。
输出格式：JSON格式，包含action、confidence、reasoning字段。"""
    },
    {
        "role": "REITs_Expert",
        "asset_class": "reits",
        "assets": ["VNQ", "IYR", "XLRE"],
        "dependencies": ["Stock_Expert", "Bond_Expert"],
        "system_prompt": """你是一位专业的房地产投资信托(REITs)分析师。
你需要分析REITs市场数据，包括租金收益率、资本化率、利率敏感度等。
参考股票和债券专家的分析，给出REITs配置建议。
输出格式：JSON格式，包含action、confidence、reasoning字段。"""
    },
    {
        "role": "Crypto_Expert",
        "asset_class": "crypto",
        "assets": ["BTC-USD", "ETH-USD"],
        "dependencies": ["Stock_Expert"],
        "system_prompt": """你是一位专业的加密货币分析师。
你需要分析加密货币市场数据，包括链上数据、市场情绪、技术指标等。
参考股票专家的风险偏好分析，给出加密货币配置建议。
输出格式：JSON格式，包含action、confidence、reasoning字段。"""
    },
    # ============================================================
    # Meta-Level Agents (Coordinators)
    # ============================================================
    {
        "role": "Portfolio_Manager",
        "asset_class": "portfolio",
        "assets": [],  # 管理所有资产类别
        "dependencies": ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"],
        "system_prompt": """你是一位高级投资组合管理专家。
你的职责是综合各资产类别专家的分析，制定整体投资组合配置策略。
需要考虑：资产配置比例、风险预算、流动性需求、投资期限等。
参考所有资产专家的建议，给出最优投资组合权重分配。
输出格式：JSON格式，包含target_allocation(各资产权重)、rebalance_action、confidence、reasoning字段。"""
    },
    {
        "role": "Hedging_Agent",
        "asset_class": "hedging",
        "assets": [],  # 使用对冲工具
        "dependencies": ["Stock_Expert", "Bond_Expert", "Portfolio_Manager"],
        "system_prompt": """你是一位专业的对冲策略专家。
你的职责是识别投资组合的风险敞口，设计相应的对冲策略。
需要分析：尾部风险、系统性风险、行业集中度风险、利率风险等。
参考股票、债券专家和投资组合管理器的分析，给出对冲建议。
输出格式：JSON格式，包含hedge_strategy、hedge_ratio、hedge_instruments、expected_cost、reasoning字段。"""
    },
    {
        "role": "Position_Sizing_Agent",
        "asset_class": "position",
        "assets": [],  # 管理仓位大小
        "dependencies": ["Stock_Expert", "Bond_Expert", "Portfolio_Manager", "Risk_Controller"],
        "system_prompt": """你是一位专业的仓位管理专家。
你的职责是根据风险评估和市场状况，确定每个头寸的最优规模。
需要考虑：波动率调整、凯利公式、风险预算、最大回撤限制等。
参考各专家分析和风控建议，给出具体仓位大小建议。
输出格式：JSON格式，包含position_sizes(各资产仓位)、sizing_method、risk_budget、reasoning字段。"""
    },
    {
        "role": "Risk_Controller",
        "asset_class": "risk",
        "assets": [],  # 监控整体风险
        "dependencies": ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"],
        "system_prompt": """你是一位专业的风险控制专家。
你的职责是监控投资组合的整体风险，提供风险预警和控制建议。
需要评估：VaR、CVaR、最大回撤、波动率、相关性风险、流动性风险等。
参考所有资产专家的分析，给出风险评估和控制建议。
输出格式：JSON格式，包含risk_assessment(风险指标)、violations、warnings、veto(是否否决交易)、recommendations字段。"""
    },
]

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


class SharedModelExpertManager:
    """
    共享基础模型的Expert管理器

    核心思路:
    1. 加载一个基础模型到GPU (~65GB)
    2. 为每个Expert创建独立的LoRA适配器 (~16MB each)
    3. 通过set_adapter()切换不同的Expert

    优势:
    - 5个Expert共享一个基础模型
    - 显存占用: 65GB + 5*16MB ≈ 65GB (而非5*65GB = 325GB)
    - 可以在单卡96GB显存上运行5个32B Expert

    推理加速功能:
    - Flash Attention 2: 高效注意力实现
    - torch.compile: PyTorch 2.0 编译优化
    - Static KV Cache: 静态缓存预分配
    - Prompt Caching: 系统提示缓存
    - Batch Inference: 批量推理支持
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        bf16: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 512,
        context_window: int = 8192,  # 增加到8K以支持更详细的市场分析
        lora_config: Optional[Dict] = None,
        use_gradient_checkpointing: bool = True,
        # ============ 推理加速参数 ============
        use_flash_attention: bool = True,
        use_torch_compile: bool = False,
        use_static_cache: bool = False,
        use_prompt_cache: bool = True,
        torch_compile_mode: str = "reduce-overhead",
        prompt_cache_size: int = 100,
    ):
        """
        初始化共享模型管理器

        Args:
            model_path: 基础模型路径
            device: 计算设备
            bf16: 是否使用bfloat16
            load_in_4bit: 是否使用4bit量化
            load_in_8bit: 是否使用8bit量化 (INT8, 适合40GB显存运行32B模型)
            max_new_tokens: 最大生成token数
            context_window: 上下文窗口大小
            lora_config: LoRA配置
            use_gradient_checkpointing: 是否启用梯度检查点 (省显存但会减慢速度)

            推理加速参数:
            use_flash_attention: 是否使用 Flash Attention 2 (需要安装 flash-attn)
            use_torch_compile: 是否使用 torch.compile 优化 (PyTorch 2.0+)
            use_static_cache: 是否使用静态 KV Cache (减少动态内存分配)
            use_prompt_cache: 是否启用 Prompt 缓存 (避免重复编码系统提示)
            torch_compile_mode: torch.compile 模式 ("default", "reduce-overhead", "max-autotune")
            prompt_cache_size: Prompt 缓存大小 (LRU)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and peft required")

        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.lora_config = lora_config or LORA_CONFIG
        self.expert_configs = {cfg["role"]: cfg for cfg in EXPERT_CONFIGS}
        self.current_adapter = None
        self.quantized = load_in_4bit or load_in_8bit

        # 推理加速配置
        self.use_flash_attention = use_flash_attention and HAS_FLASH_ATTN
        self.use_torch_compile = use_torch_compile
        self.use_static_cache = use_static_cache
        self.use_prompt_cache = use_prompt_cache
        self.torch_compile_mode = torch_compile_mode
        self._compiled = False

        # 初始化 Prompt 缓存
        if use_prompt_cache:
            self.prompt_cache = PromptCache(max_size=prompt_cache_size)
            logger.info(f"Prompt cache enabled (size: {prompt_cache_size})")
        else:
            self.prompt_cache = None

        # 推理计时统计
        self.inference_times: List[float] = []

        # 量化配置
        if load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            logger.info("Using 8-bit quantization (INT8) - ~33GB for 32B model")
        elif load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Using 4-bit quantization (NF4) - ~18GB for 32B model")
        else:
            quant_config = None

        # 确定注意力实现方式
        attn_implementation = None
        if self.use_flash_attention:
            attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 enabled - faster attention computation")
        else:
            # 使用 SDPA (Scaled Dot Product Attention) 作为后备
            attn_implementation = "sdpa"
            logger.info("Using SDPA (Scaled Dot Product Attention)")

        # 加载基础模型
        logger.info(f"Loading base model from {model_path}")
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if bf16 else "auto",
            "quantization_config": quant_config,
            "device_map": "auto" if self.quantized else None,
        }

        # 添加注意力实现 (transformers >= 4.36.0)
        if attn_implementation:
            try:
                load_kwargs["attn_implementation"] = attn_implementation
            except Exception as e:
                logger.warning(f"attn_implementation not supported: {e}")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )

        if not self.quantized:
            self.base_model = self.base_model.to(self.device)

        # 可选: 启用gradient checkpointing减少显存 (会减慢速度)
        if use_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (saves memory, slower)")
        else:
            logger.info("Gradient checkpointing disabled (uses more memory, faster)")

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

        # 创建第一个LoRA适配器
        peft_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config.get("lora_dropout", 0),
            bias=self.lora_config.get("bias", "none"),
            task_type=self.lora_config.get("task_type", "CAUSAL_LM"),
        )

        first_expert = EXPERT_CONFIGS[0]["role"]
        self.model = get_peft_model(self.base_model, peft_config, adapter_name=first_expert)
        self.current_adapter = first_expert
        logger.info(f"Created first adapter: {first_expert}")

        # 为其他Expert添加LoRA适配器
        for cfg in EXPERT_CONFIGS[1:]:
            role = cfg["role"]
            self.model.add_adapter(role, peft_config)
            logger.info(f"Added adapter: {role}")

        self.model.print_trainable_parameters()
        self.model.train()

        # 应用 torch.compile 优化
        if self.use_torch_compile and hasattr(torch, "compile"):
            self._apply_torch_compile()

        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"All adapters created. GPU Memory: {mem_gb:.1f} GB")
        logger.info(f"Acceleration: FlashAttn={self.use_flash_attention}, "
                    f"Compile={self.use_torch_compile}, "
                    f"StaticCache={self.use_static_cache}, "
                    f"PromptCache={self.use_prompt_cache}")

    def _apply_torch_compile(self):
        """应用 torch.compile 优化"""
        if self._compiled:
            return

        try:
            logger.info(f"Applying torch.compile with mode='{self.torch_compile_mode}'...")

            # 编译 forward 函数
            self.model.forward = torch.compile(
                self.model.forward,
                mode=self.torch_compile_mode,
                fullgraph=False,  # 允许图中断
            )

            self._compiled = True
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed (falling back to eager mode): {e}")
            self._compiled = False

    def _create_static_cache(self, batch_size: int = 1) -> Optional[Any]:
        """创建静态 KV Cache"""
        if not self.use_static_cache or StaticCache is None:
            return None

        try:
            # 获取模型配置
            config = self.base_model.config
            num_layers = getattr(config, "num_hidden_layers", 32)
            num_heads = getattr(config, "num_key_value_heads", 8)
            head_dim = getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)

            cache = StaticCache(
                config=config,
                max_batch_size=batch_size,
                max_cache_len=self.context_window,
                device=self.device,
                dtype=torch.bfloat16,
            )
            return cache
        except Exception as e:
            logger.warning(f"Failed to create static cache: {e}")
            return None

    def switch_expert(self, role: str):
        """切换到指定Expert的LoRA适配器"""
        if role != self.current_adapter:
            self.model.set_adapter(role)
            self.current_adapter = role

    def _build_prompt(self, role: str, market_obs: str, predecessor_actions: Dict[str, Dict] = None) -> str:
        """构建Expert的prompt"""
        cfg = self.expert_configs[role]
        system_prompt = cfg["system_prompt"]
        asset_class = cfg["asset_class"]

        prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
"""

        # 添加前序Expert的建议
        if predecessor_actions:
            deps = cfg.get("dependencies", [])
            if deps:
                prompt += "<|im_start|>context\n## 其他专家的分析建议:\n"
                for dep_role in deps:
                    if dep_role in predecessor_actions:
                        action = predecessor_actions[dep_role]
                        prompt += f"\n### {dep_role}:\n"
                        prompt += f"- 动作: {action.get('action', 'N/A')}\n"
                        prompt += f"- 信心度: {action.get('confidence', 'N/A')}\n"
                        prompt += f"- 理由: {action.get('reasoning', 'N/A')[:200]}\n"
                prompt += "<|im_end|>\n"

        # User prompt
        prompt += f"""<|im_start|>user
{market_obs}

请分析上述市场数据，给出你对{asset_class}资产的投资建议。
输出格式要求JSON:
{{
    "action": "BUY_25%/BUY_50%/BUY_75%/BUY_100%/HOLD/SELL_25%/SELL_50%/SELL_75%/SELL_100%",
    "confidence": 0.0-1.0,
    "reasoning": "决策理由"
}}
<|im_end|>
<|im_start|>{role}
"""
        return prompt

    def _parse_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {"action": "HOLD", "confidence": 0.5, "reasoning": "Parse failed"}

    @torch.no_grad()
    def generate_action(
        self,
        role: str,
        market_obs: str,
        predecessor_actions: Dict[str, Dict] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        use_cache: bool = True,
    ) -> Tuple[Dict, torch.Tensor, str]:
        """
        为指定Expert生成投资动作

        Args:
            role: Expert角色名
            market_obs: 市场观察
            predecessor_actions: 前序Expert的动作
            temperature: 采样温度
            top_k: Top-K采样
            use_cache: 是否使用 Prompt 缓存

        Returns:
            action_dict: 解析后的动作字典
            action_tokens: 生成的token序列
            raw_response: 原始响应文本
        """
        start_time = time.time()

        # 切换到对应Expert的LoRA适配器
        self.switch_expert(role)

        # 构建prompt
        prompt = self._build_prompt(role, market_obs, predecessor_actions)

        # 尝试使用 Prompt 缓存
        cached = None
        if use_cache and self.prompt_cache is not None:
            cached = self.prompt_cache.get(prompt)

        if cached is not None:
            # 使用缓存的编码
            input_ids = cached.input_ids.to(self.device)
            attention_mask = cached.attention_mask.to(self.device)
        else:
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

            # 存入缓存
            if self.prompt_cache is not None:
                self.prompt_cache.put(prompt, input_ids, attention_mask)

        input_length = input_ids.shape[1]

        # Generate with optimized settings
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "use_cache": True,  # 启用 KV Cache
        }

        # 静态缓存 (如果启用)
        if self.use_static_cache:
            static_cache = self._create_static_cache(batch_size=1)
            if static_cache is not None:
                generate_kwargs["past_key_values"] = static_cache

        outputs = self.model.generate(**generate_kwargs)

        # 提取生成的token
        action_tokens = outputs.sequences[0, input_length:]
        raw_response = self.tokenizer.decode(action_tokens, skip_special_tokens=True)

        # 解析JSON
        action_dict = self._parse_response(raw_response)
        action_dict["role"] = role
        action_dict["analysis"] = raw_response

        # 记录推理时间
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)

        return action_dict, action_tokens, raw_response

    @torch.no_grad()
    def generate_actions_batch(
        self,
        roles: List[str],
        market_obs: str,
        predecessor_actions_list: List[Optional[Dict[str, Dict]]] = None,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> List[Tuple[Dict, torch.Tensor, str]]:
        """
        批量生成多个Expert的投资动作 (同一个 LoRA 适配器)

        注意: 由于不同 Expert 使用不同的 LoRA 适配器，
        真正的批量推理需要使用 merge_and_unload 或 vLLM。
        此方法用于同一 Expert 的多个不同输入的批量推理。

        Args:
            roles: Expert角色名列表 (必须相同)
            market_obs: 市场观察 (共享)
            predecessor_actions_list: 每个请求的前序Expert动作列表
            temperature: 采样温度
            top_k: Top-K采样

        Returns:
            结果列表: [(action_dict, action_tokens, raw_response), ...]
        """
        if not roles:
            return []

        # 检查所有角色是否相同 (批量推理只能用于同一适配器)
        if len(set(roles)) > 1:
            logger.warning("Batch inference requires same adapter. Falling back to sequential.")
            results = []
            for i, role in enumerate(roles):
                pred = predecessor_actions_list[i] if predecessor_actions_list else None
                results.append(self.generate_action(role, market_obs, pred, temperature, top_k))
            return results

        role = roles[0]
        self.switch_expert(role)

        # 构建所有 prompts
        prompts = []
        if predecessor_actions_list is None:
            predecessor_actions_list = [None] * len(roles)

        for pred in predecessor_actions_list:
            prompts.append(self._build_prompt(role, market_obs, pred))

        # 批量 tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_window,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_length = input_ids.shape[1]

        # 批量生成
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )

        # 解析所有结果
        results = []
        for i in range(len(roles)):
            action_tokens = outputs.sequences[i, input_length:]
            raw_response = self.tokenizer.decode(action_tokens, skip_special_tokens=True)
            action_dict = self._parse_response(raw_response)
            action_dict["role"] = role
            action_dict["analysis"] = raw_response
            results.append((action_dict, action_tokens, raw_response))

        return results

    def get_inference_stats(self) -> Dict:
        """获取推理统计信息"""
        if not self.inference_times:
            return {"count": 0, "avg_time": 0, "total_time": 0}

        return {
            "count": len(self.inference_times),
            "avg_time": f"{np.mean(self.inference_times):.3f}s",
            "total_time": f"{sum(self.inference_times):.1f}s",
            "min_time": f"{min(self.inference_times):.3f}s",
            "max_time": f"{max(self.inference_times):.3f}s",
            "prompt_cache": self.prompt_cache.stats() if self.prompt_cache else None,
        }

    def reset_stats(self):
        """重置统计信息"""
        self.inference_times.clear()
        if self.prompt_cache:
            self.prompt_cache.clear()

    def run_expert_chain(
        self,
        market_obs: str,
        expert_order: List[str] = None,
    ) -> Dict[str, Dict]:
        """
        按依赖顺序运行Expert链

        Args:
            market_obs: 市场观察
            expert_order: Expert执行顺序 (默认按依赖拓扑排序)

        Returns:
            所有Expert的动作字典
        """
        if expert_order is None:
            # 默认顺序: Stock -> Bond -> Commodity/REITs -> Crypto
            expert_order = [
                "Stock_Expert",
                "Bond_Expert",
                "Commodity_Expert",
                "REITs_Expert",
                "Crypto_Expert",
            ]

        all_actions = {}

        for role in expert_order:
            if role not in self.expert_configs:
                continue

            # 收集前序Expert的动作
            deps = self.expert_configs[role].get("dependencies", [])
            predecessor_actions = {d: all_actions[d] for d in deps if d in all_actions}

            # 生成动作
            action_dict, _, _ = self.generate_action(
                role=role,
                market_obs=market_obs,
                predecessor_actions=predecessor_actions if predecessor_actions else None,
            )

            all_actions[role] = action_dict
            logger.info(f"{role}: {action_dict.get('action', 'N/A')} (conf: {action_dict.get('confidence', 'N/A')})")

        return all_actions

    def get_action_log_prob(
        self,
        role: str,
        obs: str,
        action_tokens: torch.Tensor,
        predecessor_actions: Dict[str, Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算动作的log probability (用于PPO训练)

        Args:
            role: Expert角色名
            obs: 市场观察
            action_tokens: 动作token序列
            predecessor_actions: 前序Expert动作

        Returns:
            log_prob: 动作的log probability
            entropy: 动作分布的entropy
        """
        from torch.distributions.categorical import Categorical

        self.switch_expert(role)

        prompt = self._build_prompt(role, obs, predecessor_actions)

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

        if action_tokens.dim() == 1:
            action_tokens = action_tokens.unsqueeze(0)
        action_tokens = action_tokens.to(self.device)

        full_ids = torch.cat([prompt_ids, action_tokens], dim=1)
        action_mask = (action_tokens != self.tokenizer.pad_token_id).long()
        full_mask = torch.cat([prompt_mask, action_mask], dim=1)

        outputs = self.model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits

        action_logits = logits[:, prompt_length - 1:-1, :]
        log_softmax = torch.log_softmax(action_logits, dim=-1)

        action_len = (action_tokens != self.tokenizer.pad_token_id).sum().item()
        token_log_probs = torch.gather(
            log_softmax[:, :action_len, :],
            dim=-1,
            index=action_tokens[:, :action_len].unsqueeze(-1)
        ).squeeze(-1)

        # 使用mean而不是sum，避免长序列导致log_prob过大引起ratio爆炸
        action_log_prob = token_log_probs.mean()

        # Use logits directly to avoid numerical precision issues with bfloat16
        entropy = Categorical(logits=action_logits[:, :action_len, :]).entropy().mean()

        return action_log_prob, entropy

    def save_all_adapters(self, save_dir: str):
        """保存所有LoRA适配器"""
        os.makedirs(save_dir, exist_ok=True)

        for cfg in EXPERT_CONFIGS:
            role = cfg["role"]
            adapter_path = os.path.join(save_dir, role)
            self.model.save_pretrained(adapter_path, selected_adapters=[role])
            logger.info(f"Saved adapter {role} to {adapter_path}")

    def load_adapters(self, load_dir: str):
        """从目录加载所有LoRA适配器"""
        from peft import PeftModel

        for cfg in EXPERT_CONFIGS:
            role = cfg["role"]
            adapter_path = os.path.join(load_dir, role)

            # 检查嵌套目录结构 (role/role/)
            nested_path = os.path.join(adapter_path, role)
            if os.path.exists(nested_path) and os.path.isdir(nested_path):
                adapter_path = nested_path

            if os.path.exists(adapter_path):
                # 使用 is_local=True 来加载本地适配器
                try:
                    self.model.load_adapter(adapter_path, adapter_name=role, is_local=True)
                except TypeError:
                    # 旧版peft不支持is_local参数
                    self.model.load_adapter(adapter_path, adapter_name=role)
                logger.info(f"Loaded adapter {role} from {adapter_path}")

    def parameters(self, role: str = None):
        """获取可训练参数"""
        if role:
            self.switch_expert(role)
        return (p for p in self.model.parameters() if p.requires_grad)

    def train(self):
        """设置为训练模式"""
        self.model.train()

    def eval(self):
        """设置为评估模式"""
        self.model.eval()


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 80)
    print(" Shared Model Expert Manager Test")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 创建管理器
    print("\nInitializing SharedModelExpertManager...")
    manager = SharedModelExpertManager(
        model_path="Qwen/Qwen2.5-32B-Instruct",
        device="cuda:0",
        bf16=True,
    )

    print(f"\nGPU Memory after init: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 创建测试市场数据
    market_obs = f"""## 市场日期: 2024-01-15
## 资产类别: multi-asset

### 股票市场
- SPY: $450.00, 日涨跌: +1.2%, RSI: 55
- QQQ: $380.00, 日涨跌: +1.5%, RSI: 58

### 债券市场
- TLT: $95.00, 日涨跌: -0.3%, 10Y Yield: 4.2%
- LQD: $112.00, 日涨跌: -0.1%, Credit Spread: 1.2%

### 商品市场
- GLD: $185.00, 日涨跌: +0.5%
- USO: $72.00, 日涨跌: +2.1%

### 加密货币
- BTC-USD: $42000, 日涨跌: +3.5%
- ETH-USD: $2500, 日涨跌: +4.2%

### 宏观环境
- VIX: 15.5
- 美元指数: 102.5
- 市场情绪: 偏乐观
"""

    # 运行Expert链
    print("\n" + "=" * 80)
    print(" Running Expert Chain")
    print("=" * 80)

    all_actions = manager.run_expert_chain(market_obs)

    # 汇总结果
    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)
    for role, action in all_actions.items():
        print(f"{role}:")
        print(f"  Action: {action.get('action', 'N/A')}")
        print(f"  Confidence: {action.get('confidence', 'N/A')}")
        print(f"  Reasoning: {str(action.get('reasoning', 'N/A'))[:100]}...")

    # 保存adapters
    print("\nSaving all adapters...")
    manager.save_all_adapters("/root/checkpoints/shared_experts")

    print("\n" + "=" * 80)
    print(" Test Complete!")
    print("=" * 80)
