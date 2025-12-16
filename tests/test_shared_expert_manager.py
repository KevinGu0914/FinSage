#!/usr/bin/env python
"""
Shared Expert Manager Tests - 共享专家管理器模块测试
覆盖: shared_expert_manager.py (VLLMInferenceEngine, PromptCache, SharedModelExpertManager)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: Module Constants and Config
# ============================================================

class TestModuleConstants:
    """测试模块常量"""

    def test_expert_configs_defined(self):
        """测试专家配置定义"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        assert EXPERT_CONFIGS is not None
        assert len(EXPERT_CONFIGS) == 9  # 5个资产专家 + 4个元级代理

        # 验证资产专家角色
        roles = [cfg["role"] for cfg in EXPERT_CONFIGS]
        assert "Stock_Expert" in roles
        assert "Bond_Expert" in roles
        assert "Commodity_Expert" in roles
        assert "REITs_Expert" in roles
        assert "Crypto_Expert" in roles
        # 验证元级代理角色
        assert "Portfolio_Manager" in roles
        assert "Hedging_Agent" in roles
        assert "Position_Sizing_Agent" in roles
        assert "Risk_Controller" in roles

    def test_expert_configs_structure(self):
        """测试专家配置结构"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        for cfg in EXPERT_CONFIGS:
            assert "role" in cfg
            assert "asset_class" in cfg
            assert "assets" in cfg
            assert "system_prompt" in cfg

    def test_lora_config_defined(self):
        """测试LoRA配置定义"""
        from finsage.rl.shared_expert_manager import LORA_CONFIG

        assert LORA_CONFIG is not None
        assert "r" in LORA_CONFIG
        assert "lora_alpha" in LORA_CONFIG
        assert "target_modules" in LORA_CONFIG


# ============================================================
# Test 2: PromptCache
# ============================================================

class TestPromptCache:
    """测试提示缓存"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.shared_expert_manager import PromptCache
        assert PromptCache is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.shared_expert_manager import PromptCache

        cache = PromptCache(max_size=10)
        assert cache.max_size == 10

    def test_put_and_get(self):
        """测试存取操作"""
        import torch
        from finsage.rl.shared_expert_manager import PromptCache

        cache = PromptCache(max_size=10)

        # 存入 - 使用正确的参数签名
        input_ids1 = torch.tensor([1, 2, 3])
        attention_mask1 = torch.tensor([1, 1, 1])
        cache.put("prompt1", input_ids1, attention_mask1)

        input_ids2 = torch.tensor([4, 5, 6])
        attention_mask2 = torch.tensor([1, 1, 1])
        cache.put("prompt2", input_ids2, attention_mask2)

        # 取出 - 返回CachedPrompt对象
        result1 = cache.get("prompt1")
        result2 = cache.get("prompt2")

        assert result1 is not None
        assert result2 is not None
        assert torch.equal(result1.input_ids, input_ids1)
        assert torch.equal(result2.input_ids, input_ids2)

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        from finsage.rl.shared_expert_manager import PromptCache

        cache = PromptCache(max_size=10)

        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        import torch
        from finsage.rl.shared_expert_manager import PromptCache

        cache = PromptCache(max_size=3)

        # 使用正确的参数格式
        cache.put("prompt1", torch.tensor([1]), torch.tensor([1]))
        cache.put("prompt2", torch.tensor([2]), torch.tensor([1]))
        cache.put("prompt3", torch.tensor([3]), torch.tensor([1]))

        # 访问prompt1使其变为最近使用
        _ = cache.get("prompt1")

        # 添加prompt4应该淘汰最久未使用的prompt2
        cache.put("prompt4", torch.tensor([4]), torch.tensor([1]))

        assert cache.get("prompt1") is not None
        assert cache.get("prompt3") is not None
        assert cache.get("prompt4") is not None

    def test_cache_stats(self):
        """测试缓存统计"""
        import torch
        from finsage.rl.shared_expert_manager import PromptCache

        cache = PromptCache(max_size=10)

        cache.put("prompt1", torch.tensor([1]), torch.tensor([1]))
        _ = cache.get("prompt1")  # hit
        _ = cache.get("prompt1")  # hit
        _ = cache.get("prompt2")  # miss

        stats = cache.stats()  # 正确的方法名

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1


# ============================================================
# Test 3: VLLMInferenceEngine (Mock Tests)
# ============================================================

class TestVLLMInferenceEngine:
    """测试vLLM推理引擎"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.shared_expert_manager import VLLMInferenceEngine
        assert VLLMInferenceEngine is not None

    def test_vllm_availability_check(self):
        """测试vLLM可用性检查"""
        from finsage.rl.shared_expert_manager import HAS_VLLM

        # 测试常量是否定义
        assert isinstance(HAS_VLLM, bool)


# ============================================================
# Test 4: SharedModelExpertManager (Mock Tests)
# ============================================================

class TestSharedModelExpertManager:
    """测试共享模型专家管理器"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.shared_expert_manager import SharedModelExpertManager
        assert SharedModelExpertManager is not None

    def test_transformers_availability_check(self):
        """测试transformers可用性检查"""
        from finsage.rl.shared_expert_manager import HAS_TRANSFORMERS

        assert isinstance(HAS_TRANSFORMERS, bool)

    def test_flash_attention_availability_check(self):
        """测试Flash Attention可用性检查"""
        from finsage.rl.shared_expert_manager import HAS_FLASH_ATTN

        assert isinstance(HAS_FLASH_ATTN, bool)


# ============================================================
# Test 5: Expert Configuration
# ============================================================

class TestExpertConfiguration:
    """测试专家配置"""

    def test_stock_expert_config(self):
        """测试股票专家配置"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        stock_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "Stock_Expert"
        )

        assert stock_config["asset_class"] == "stocks"
        assert len(stock_config["assets"]) > 0
        assert "SPY" in stock_config["assets"] or "QQQ" in stock_config["assets"]

    def test_bond_expert_config(self):
        """测试债券专家配置"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        bond_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "Bond_Expert"
        )

        assert bond_config["asset_class"] == "bonds"
        assert "TLT" in bond_config["assets"] or len(bond_config["assets"]) > 0

    def test_commodity_expert_config(self):
        """测试商品专家配置"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        commodity_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "Commodity_Expert"
        )

        assert commodity_config["asset_class"] == "commodities"

    def test_reits_expert_config(self):
        """测试REITs专家配置"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        reits_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "REITs_Expert"
        )

        assert reits_config["asset_class"] == "reits"

    def test_crypto_expert_config(self):
        """测试加密货币专家配置"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        crypto_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "Crypto_Expert"
        )

        assert crypto_config["asset_class"] == "crypto"

    def test_expert_dependencies(self):
        """测试专家依赖关系"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        # 股票专家应该没有依赖
        stock_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "Stock_Expert"
        )
        assert stock_config.get("dependencies", []) == []

        # REITs专家应该依赖股票和债券专家
        reits_config = next(
            cfg for cfg in EXPERT_CONFIGS if cfg["role"] == "REITs_Expert"
        )
        assert "Stock_Expert" in reits_config.get("dependencies", [])
        assert "Bond_Expert" in reits_config.get("dependencies", [])


# ============================================================
# Test 6: LoRA Configuration
# ============================================================

class TestLoRAConfiguration:
    """测试LoRA配置"""

    def test_lora_rank(self):
        """测试LoRA秩"""
        from finsage.rl.shared_expert_manager import LORA_CONFIG

        assert LORA_CONFIG["r"] >= 4
        assert LORA_CONFIG["r"] <= 64

    def test_lora_alpha(self):
        """测试LoRA alpha"""
        from finsage.rl.shared_expert_manager import LORA_CONFIG

        assert LORA_CONFIG["lora_alpha"] >= LORA_CONFIG["r"]

    def test_lora_target_modules(self):
        """测试LoRA目标模块"""
        from finsage.rl.shared_expert_manager import LORA_CONFIG

        target_modules = LORA_CONFIG["target_modules"]

        # 应该包含注意力相关的模块
        assert any("proj" in m for m in target_modules)

    def test_lora_dropout(self):
        """测试LoRA dropout"""
        from finsage.rl.shared_expert_manager import LORA_CONFIG

        dropout = LORA_CONFIG.get("lora_dropout", 0)
        assert 0 <= dropout <= 0.5


# ============================================================
# Test 7: Action Space
# ============================================================

class TestActionSpace:
    """测试动作空间"""

    def test_actions_defined(self):
        """测试动作定义"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        # 每个专家的系统提示应该有相关内容（中英文都可以）
        for cfg in EXPERT_CONFIGS:
            prompt_lower = cfg["system_prompt"].lower()
            # 检查是否包含行动相关的关键词（中文或英文）
            has_action_keywords = (
                "action" in prompt_lower or
                "策略" in cfg["system_prompt"] or
                "建议" in cfg["system_prompt"] or
                "操作" in cfg["system_prompt"]
            )
            assert has_action_keywords, f"System prompt missing action keywords: {cfg['system_prompt'][:50]}..."

    def test_output_format_json(self):
        """测试输出格式为JSON"""
        from finsage.rl.shared_expert_manager import EXPERT_CONFIGS

        for cfg in EXPERT_CONFIGS:
            assert "JSON" in cfg["system_prompt"] or "json" in cfg["system_prompt"]


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Shared Expert Manager Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
