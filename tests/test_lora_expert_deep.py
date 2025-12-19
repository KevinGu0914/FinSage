#!/usr/bin/env python
"""
Deep Tests for LoRA Expert Module - LoRA专家模块深度测试
覆盖: ExpertProfile, LoRAExpert, FinSageMAS 的所有方法和代码路径
目标覆盖率: 从23%提升到90%+
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import torch
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from dataclasses import dataclass


# ============================================================
# Test 1: ExpertProfile Dataclass
# ============================================================

class TestExpertProfile:
    """测试ExpertProfile数据类"""

    def test_expert_profile_creation(self):
        """测试ExpertProfile创建"""
        from finsage.rl.lora_expert import ExpertProfile

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票投资分析",
            system_prompt="You are a stock expert",
            dependencies=["Bond_Expert"],
            device="cuda:0"
        )

        assert profile.role == "Stock_Expert"
        assert profile.asset_class == "stocks"
        assert profile.expertise == "股票投资分析"
        assert profile.system_prompt == "You are a stock expert"
        assert profile.dependencies == ["Bond_Expert"]
        assert profile.device == "cuda:0"

    def test_expert_profile_default_device(self):
        """测试ExpertProfile默认设备"""
        from finsage.rl.lora_expert import ExpertProfile

        profile = ExpertProfile(
            role="Bond_Expert",
            asset_class="bonds",
            expertise="债券分析",
            system_prompt="You are a bond expert",
            dependencies=[]
        )

        # 默认设备应为cuda:0
        assert profile.device == "cuda:0"

    def test_expert_profile_empty_dependencies(self):
        """测试空依赖列表"""
        from finsage.rl.lora_expert import ExpertProfile

        profile = ExpertProfile(
            role="Independent_Expert",
            asset_class="general",
            expertise="通用分析",
            system_prompt="General expert",
            dependencies=[]
        )

        assert profile.dependencies == []
        assert isinstance(profile.dependencies, list)


# ============================================================
# Test 2: create_finsage_expert_profiles Factory
# ============================================================

class TestCreateFinSageExpertProfiles:
    """测试专家配置工厂函数"""

    def test_creates_five_profiles(self):
        """测试创建5个专家配置"""
        from finsage.rl.lora_expert import create_finsage_expert_profiles

        profiles = create_finsage_expert_profiles()

        assert len(profiles) == 5
        assert all(hasattr(p, 'role') for p in profiles)

    def test_profile_roles(self):
        """测试专家角色名称"""
        from finsage.rl.lora_expert import create_finsage_expert_profiles

        profiles = create_finsage_expert_profiles()
        roles = [p.role for p in profiles]

        expected_roles = [
            "Stock_Expert",
            "Bond_Expert",
            "Commodity_Expert",
            "REITs_Expert",
            "Crypto_Expert"
        ]

        assert roles == expected_roles

    def test_profile_asset_classes(self):
        """测试资产类别"""
        from finsage.rl.lora_expert import create_finsage_expert_profiles

        profiles = create_finsage_expert_profiles()
        asset_classes = [p.asset_class for p in profiles]

        expected_classes = ["stocks", "bonds", "commodities", "reits", "crypto"]
        assert asset_classes == expected_classes

    def test_profile_dependencies(self):
        """测试依赖关系"""
        from finsage.rl.lora_expert import create_finsage_expert_profiles

        profiles = create_finsage_expert_profiles()
        profile_dict = {p.role: p for p in profiles}

        # Stock_Expert 无依赖
        assert profile_dict["Stock_Expert"].dependencies == []

        # Bond_Expert 依赖 Stock_Expert
        assert profile_dict["Bond_Expert"].dependencies == ["Stock_Expert"]

        # Commodity_Expert 依赖 Stock_Expert 和 Bond_Expert
        assert "Stock_Expert" in profile_dict["Commodity_Expert"].dependencies
        assert "Bond_Expert" in profile_dict["Commodity_Expert"].dependencies

        # REITs_Expert 依赖 Stock_Expert 和 Bond_Expert
        assert "Stock_Expert" in profile_dict["REITs_Expert"].dependencies
        assert "Bond_Expert" in profile_dict["REITs_Expert"].dependencies

        # Crypto_Expert 依赖 Stock_Expert
        assert profile_dict["Crypto_Expert"].dependencies == ["Stock_Expert"]

    def test_profile_system_prompts(self):
        """测试系统提示词"""
        from finsage.rl.lora_expert import create_finsage_expert_profiles

        profiles = create_finsage_expert_profiles()

        for profile in profiles:
            assert len(profile.system_prompt) > 0
            assert isinstance(profile.system_prompt, str)

    def test_profile_expertise(self):
        """测试专业领域描述"""
        from finsage.rl.lora_expert import create_finsage_expert_profiles

        profiles = create_finsage_expert_profiles()
        profile_dict = {p.role: p for p in profiles}

        assert "股票投资分析" in profile_dict["Stock_Expert"].expertise
        assert "债券投资分析" in profile_dict["Bond_Expert"].expertise
        assert "大宗商品投资分析" in profile_dict["Commodity_Expert"].expertise


# ============================================================
# Test 3: LoRAExpert Initialization (Mocked)
# ============================================================

class TestLoRAExpertInit:
    """测试LoRAExpert初始化"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', False)
    def test_init_without_transformers(self):
        """测试没有transformers时的错误"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        profile = ExpertProfile(
            role="Test_Expert",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        with pytest.raises(ImportError, match="transformers and peft are required"):
            expert = LoRAExpert(
                model_path="test/model",
                profile=profile
            )

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_init_basic(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试基本初始化"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Mock model
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock peft model
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票分析",
            system_prompt="You are a stock expert",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        assert expert.role == "Stock_Expert"
        assert expert.asset_class == "stocks"
        assert expert.max_new_tokens == 512
        assert expert.context_window == 4096

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.BitsAndBytesConfig')
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_init_with_4bit_quantization(self, mock_get_peft, mock_tokenizer, mock_model, mock_bnb_config):
        """测试4bit量化初始化"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Mock BitsAndBytesConfig
        mock_bnb_config.return_value = MagicMock()

        # Mock model
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_model.from_pretrained.return_value = mock_base_model

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock peft model
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Bond_Expert",
            asset_class="bonds",
            expertise="债券分析",
            system_prompt="Bond expert",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cuda:0",
            load_in_4bit=True,
            bf16=True
        )

        # 验证量化配置被使用
        mock_bnb_config.assert_called_once()
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert 'quantization_config' in call_kwargs
        assert call_kwargs['device_map'] == "auto"

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.PeftModel')
    def test_init_with_load_path(self, mock_peft_model, mock_tokenizer, mock_model):
        """测试从检查点加载"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Mock model
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock loaded peft model
        mock_loaded_model = MagicMock()
        mock_peft_model.from_pretrained.return_value = mock_loaded_model

        profile = ExpertProfile(
            role="Crypto_Expert",
            asset_class="crypto",
            expertise="加密货币",
            system_prompt="Crypto expert",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_path="/path/to/checkpoint",
            load_in_4bit=False,
            bf16=False
        )

        # 验证从检查点加载
        mock_peft_model.from_pretrained.assert_called_once()
        mock_loaded_model.set_adapter.assert_called_once_with("Crypto_Expert")

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_init_with_custom_lora_config(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试自定义LoRA配置"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Mock model
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock peft model
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Test_Expert",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        custom_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "all",
            "task_type": "CAUSAL_LM"
        }

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            lora_config=custom_lora_config,
            load_in_4bit=False,
            bf16=False
        )

        # 验证LoRA配置被使用
        assert mock_get_peft.called

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_init_with_custom_context_window(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试自定义上下文窗口"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Mock setup
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Test_Expert",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            max_new_tokens=1024,
            context_window=8192,
            load_in_4bit=False,
            bf16=False
        )

        assert expert.max_new_tokens == 1024
        assert expert.context_window == 8192


# ============================================================
# Test 4: LoRAExpert Prompt Building
# ============================================================

class TestLoRAExpertPromptBuilding:
    """测试LoRAExpert提示词构建"""

    def _create_mock_expert(self):
        """创建模拟expert"""
        from finsage.rl.lora_expert import ExpertProfile

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票分析",
            system_prompt="You are a professional stock analyst.",
            dependencies=[]
        )

        # 创建一个简化的mock expert
        expert = MagicMock()
        expert.profile = profile
        expert.role = profile.role
        expert.asset_class = profile.asset_class

        # 导入真实的_build_prompt方法
        from finsage.rl.lora_expert import LoRAExpert
        expert._build_prompt = LoRAExpert._build_prompt.__get__(expert, type(expert))

        return expert

    def test_build_prompt_basic(self):
        """测试基本提示词构建"""
        expert = self._create_mock_expert()

        market_obs = "SPY: $450.00, Volume: 100M"
        prompt = expert._build_prompt(market_obs, None)

        assert "<|im_start|>system" in prompt
        assert "You are a professional stock analyst." in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>user" in prompt
        assert market_obs in prompt
        assert "stocks" in prompt
        assert "Stock_Expert" in prompt

    def test_build_prompt_with_predecessors(self):
        """测试包含前序动作的提示词"""
        expert = self._create_mock_expert()

        market_obs = "SPY: $450.00"
        predecessor_actions = [
            {
                "role": "Bond_Expert",
                "analysis": "Interest rates are rising, bonds under pressure"
            }
        ]

        prompt = expert._build_prompt(market_obs, predecessor_actions)

        assert "<|im_start|>context" in prompt
        assert "其他专家的分析建议" in prompt
        assert "Bond_Expert" in prompt
        assert "Interest rates are rising" in prompt

    def test_build_prompt_with_multiple_predecessors(self):
        """测试多个前序专家"""
        expert = self._create_mock_expert()

        market_obs = "SPY: $450.00"
        predecessor_actions = [
            {"role": "Expert1", "analysis": "Analysis 1"},
            {"role": "Expert2", "analysis": "Analysis 2"},
            {"role": "Expert3", "analysis": "Analysis 3"}
        ]

        prompt = expert._build_prompt(market_obs, predecessor_actions)

        assert "Expert1" in prompt
        assert "Expert2" in prompt
        assert "Expert3" in prompt
        assert "Analysis 1" in prompt
        assert "Analysis 2" in prompt
        assert "Analysis 3" in prompt

    def test_build_prompt_json_format_requirement(self):
        """测试JSON格式要求"""
        expert = self._create_mock_expert()

        market_obs = "SPY: $450.00"
        prompt = expert._build_prompt(market_obs, None)

        # 验证JSON格式说明
        assert "JSON" in prompt
        assert "action" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt
        assert "risk_assessment" in prompt

    def test_build_prompt_action_options(self):
        """测试动作选项说明"""
        expert = self._create_mock_expert()

        market_obs = "SPY: $450.00"
        prompt = expert._build_prompt(market_obs, None)

        # 验证包含各种动作选项
        assert "BUY" in prompt
        assert "SELL" in prompt
        assert "SHORT" in prompt
        assert "HOLD" in prompt


# ============================================================
# Test 5: LoRAExpert Action Parsing
# ============================================================

class TestLoRAExpertActionParsing:
    """测试动作解析"""

    def _create_mock_expert(self):
        """创建模拟expert"""
        from finsage.rl.lora_expert import ExpertProfile

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票分析",
            system_prompt="Test",
            dependencies=[]
        )

        expert = MagicMock()
        expert.profile = profile
        expert.role = profile.role

        from finsage.rl.lora_expert import LoRAExpert
        expert._parse_action_response = LoRAExpert._parse_action_response.__get__(expert, type(expert))

        return expert

    def test_parse_valid_json(self):
        """测试解析有效JSON"""
        expert = self._create_mock_expert()

        response = '''
        Based on the analysis:
        {
            "action": "BUY_50%",
            "confidence": 0.85,
            "reasoning": "Strong momentum signals",
            "risk_assessment": {"volatility": 0.15, "downside_risk": 0.10}
        }
        '''

        action = expert._parse_action_response(response)

        assert action["action"] == "BUY_50%"
        assert action["confidence"] == 0.85
        assert action["reasoning"] == "Strong momentum signals"
        assert action["risk_assessment"]["volatility"] == 0.15

    def test_parse_json_with_extra_text(self):
        """测试带额外文本的JSON"""
        expert = self._create_mock_expert()

        response = '''
        Here is my analysis. After careful consideration, I recommend:
        {
            "action": "SELL_25%",
            "confidence": 0.65,
            "reasoning": "Profit taking recommended",
            "risk_assessment": {"volatility": 0.20, "downside_risk": 0.15}
        }
        This is my final recommendation.
        '''

        action = expert._parse_action_response(response)

        assert action["action"] == "SELL_25%"
        assert action["confidence"] == 0.65

    def test_parse_nested_json(self):
        """测试嵌套JSON"""
        expert = self._create_mock_expert()

        response = '''
        {
            "action": "HOLD",
            "confidence": 0.75,
            "reasoning": "Wait for better entry",
            "risk_assessment": {
                "volatility": 0.18,
                "downside_risk": 0.12,
                "max_drawdown": 0.05
            }
        }
        '''

        action = expert._parse_action_response(response)

        assert action["action"] == "HOLD"
        assert "max_drawdown" in action["risk_assessment"]

    def test_parse_invalid_json_returns_default(self):
        """测试无效JSON返回默认值"""
        expert = self._create_mock_expert()

        response = "This is not valid JSON at all"
        action = expert._parse_action_response(response)

        # 应该返回默认HOLD动作
        assert action["action"] == "HOLD"
        assert action["confidence"] == 0.5
        assert "Unable to parse" in action["reasoning"]

    def test_parse_malformed_json_returns_default(self):
        """测试格式错误的JSON"""
        expert = self._create_mock_expert()

        response = '{"action": "BUY_50%", "confidence": 0.8'  # 缺少闭合括号
        action = expert._parse_action_response(response)

        assert action["action"] == "HOLD"

    def test_parse_empty_response(self):
        """测试空响应"""
        expert = self._create_mock_expert()

        response = ""
        action = expert._parse_action_response(response)

        assert action["action"] == "HOLD"

    def test_parse_short_action(self):
        """测试做空动作"""
        expert = self._create_mock_expert()

        response = '''
        {
            "action": "SHORT_75%",
            "confidence": 0.90,
            "reasoning": "Bearish reversal pattern",
            "risk_assessment": {"volatility": 0.25, "downside_risk": 0.30}
        }
        '''

        action = expert._parse_action_response(response)

        assert action["action"] == "SHORT_75%"
        assert action["confidence"] == 0.90


# ============================================================
# Test 6: LoRAExpert Action Generation (Mocked)
# ============================================================

class TestLoRAExpertActionGeneration:
    """测试动作生成"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_generate_action_basic(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试基本动作生成"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock tokenize
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Mock generation output
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6]])

        mock_peft_model = MagicMock()
        mock_peft_model.generate.return_value = mock_output
        mock_get_peft.return_value = mock_peft_model

        # Mock decode
        mock_tok.decode.return_value = '{"action": "BUY_50%", "confidence": 0.8, "reasoning": "Test", "risk_assessment": {"volatility": 0.1, "downside_risk": 0.05}}'

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票分析",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        market_obs = "SPY: $450.00"
        action_dict, action_tokens, raw_response = expert.generate_action(market_obs)

        assert action_dict["role"] == "Stock_Expert"
        assert "action" in action_dict
        assert action_tokens is not None
        assert isinstance(raw_response, str)

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_generate_action_with_temperature(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试带温度参数的生成"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])

        mock_peft_model = MagicMock()
        mock_peft_model.generate.return_value = mock_output
        mock_get_peft.return_value = mock_peft_model

        mock_tok.decode.return_value = '{"action": "HOLD", "confidence": 0.6, "reasoning": "Neutral", "risk_assessment": {"volatility": 0.15, "downside_risk": 0.1}}'

        profile = ExpertProfile(
            role="Bond_Expert",
            asset_class="bonds",
            expertise="债券分析",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        market_obs = "TLT: $95.00"
        action_dict, action_tokens, raw_response = expert.generate_action(
            market_obs,
            temperature=0.3,
            top_k=20
        )

        # 验证generate被调用时使用了正确的参数
        call_kwargs = mock_peft_model.generate.call_args[1]
        assert call_kwargs['temperature'] == 0.3
        assert call_kwargs['top_k'] == 20

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_generate_action_no_sampling(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试贪婪解码（不采样）"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4]])

        mock_peft_model = MagicMock()
        mock_peft_model.generate.return_value = mock_output
        mock_get_peft.return_value = mock_peft_model

        mock_tok.decode.return_value = '{"action": "BUY_25%", "confidence": 0.7, "reasoning": "Test", "risk_assessment": {"volatility": 0.12, "downside_risk": 0.08}}'

        profile = ExpertProfile(
            role="Crypto_Expert",
            asset_class="crypto",
            expertise="加密货币",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        market_obs = "BTC: $50000"
        action_dict, action_tokens, raw_response = expert.generate_action(
            market_obs,
            do_sample=False
        )

        # 验证do_sample=False
        call_kwargs = mock_peft_model.generate.call_args[1]
        assert call_kwargs['do_sample'] == False


# ============================================================
# Test 7: LoRAExpert Log Probability Calculation
# ============================================================

class TestLoRAExpertLogProb:
    """测试log probability计算"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_get_action_log_prob(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试单个动作log probability"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock tokenize to return specific tensors
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }

        # Mock model output
        mock_output = MagicMock()
        vocab_size = 1000
        seq_len = 10
        mock_output.logits = torch.randn(1, seq_len, vocab_size)

        mock_peft_model = MagicMock()
        mock_peft_model.return_value = mock_output
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票分析",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        obs = "SPY: $450.00"
        action_tokens = torch.tensor([10, 20, 30])

        log_prob, entropy = expert.get_action_log_prob(obs, action_tokens)

        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert log_prob.dim() == 0  # scalar
        assert entropy.dim() == 0  # scalar

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_get_batch_action_log_probs(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试批量log probability计算"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 1000)

        mock_peft_model = MagicMock()
        mock_peft_model.return_value = mock_output
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Bond_Expert",
            asset_class="bonds",
            expertise="债券分析",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        obs_list = ["TLT: $95.00", "IEF: $100.00", "LQD: $110.00"]
        action_tokens_list = [
            torch.tensor([10, 20]),
            torch.tensor([30, 40]),
            torch.tensor([50, 60])
        ]

        log_probs, entropies = expert.get_batch_action_log_probs(
            obs_list, action_tokens_list
        )

        assert log_probs.shape[0] == 3
        assert entropies.shape[0] == 3

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_get_batch_log_probs_with_predecessors(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试带前序动作的批量计算"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 1000)

        mock_peft_model = MagicMock()
        mock_peft_model.return_value = mock_output
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Commodity_Expert",
            asset_class="commodities",
            expertise="商品分析",
            system_prompt="Test",
            dependencies=["Stock_Expert"]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        obs_list = ["GLD: $180.00", "SLV: $23.00"]
        action_tokens_list = [
            torch.tensor([10, 20]),
            torch.tensor([30, 40])
        ]
        predecessor_actions_list = [
            [{"role": "Stock_Expert", "analysis": "Bullish"}],
            [{"role": "Stock_Expert", "analysis": "Bearish"}]
        ]

        log_probs, entropies = expert.get_batch_action_log_probs(
            obs_list, action_tokens_list, predecessor_actions_list
        )

        assert log_probs.shape[0] == 2
        assert entropies.shape[0] == 2


# ============================================================
# Test 8: LoRAExpert Model Management
# ============================================================

class TestLoRAExpertModelManagement:
    """测试模型管理方法"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_train_mode(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试训练模式切换"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Test_Expert",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        expert.train()
        mock_peft_model.train.assert_called()

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_eval_mode(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试评估模式切换"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Test_Expert",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        expert.eval()
        mock_peft_model.eval.assert_called()

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    def test_parameters(self, mock_get_peft, mock_tokenizer, mock_model):
        """测试获取参数"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Test_Expert",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        params = expert.parameters()
        assert params is not None

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.AutoModelForCausalLM')
    @patch('finsage.rl.lora_expert.AutoTokenizer')
    @patch('finsage.rl.lora_expert.get_peft_model')
    @patch('os.makedirs')
    def test_save(self, mock_makedirs, mock_get_peft, mock_tokenizer, mock_model):
        """测试保存模型"""
        from finsage.rl.lora_expert import LoRAExpert, ExpertProfile

        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_model.from_pretrained.return_value = mock_base_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        profile = ExpertProfile(
            role="Stock_Expert",
            asset_class="stocks",
            expertise="股票分析",
            system_prompt="Test",
            dependencies=[]
        )

        expert = LoRAExpert(
            model_path="test/model",
            profile=profile,
            device="cpu",
            load_in_4bit=False,
            bf16=False
        )

        save_dir = "/path/to/save"
        expert.save(save_dir)

        # 验证创建目录
        expected_path = os.path.join(save_dir, "Stock_Expert")
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

        # 验证调用save_pretrained
        mock_peft_model.save_pretrained.assert_called_once_with(expected_path)


# ============================================================
# Test 9: FinSageMAS Initialization
# ============================================================

class TestFinSageMASInit:
    """测试FinSageMAS初始化"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_init_with_default_profiles(self, mock_device_count, mock_lora_expert):
        """测试使用默认配置初始化"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 2

        # Mock LoRAExpert instances
        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        assert mas.num_agents == 9  # 5 Asset Experts + 4 Meta-Level Agents
        assert len(mas.experts) == 9
        # Asset Experts
        assert "Stock_Expert" in mas.experts
        assert "Bond_Expert" in mas.experts
        assert "Commodity_Expert" in mas.experts
        assert "REITs_Expert" in mas.experts
        assert "Crypto_Expert" in mas.experts
        # Meta-Level Agents
        assert "Portfolio_Manager" in mas.experts
        assert "Hedging_Agent" in mas.experts
        assert "Position_Sizing_Agent" in mas.experts
        assert "Risk_Controller" in mas.experts

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_init_with_custom_profiles(self, mock_device_count, mock_lora_expert):
        """测试使用自定义配置"""
        from finsage.rl.lora_expert import FinSageMAS, ExpertProfile

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        custom_profiles = [
            ExpertProfile(
                role="Custom_Expert_1",
                asset_class="custom1",
                expertise="Test1",
                system_prompt="Prompt1",
                dependencies=[]
            ),
            ExpertProfile(
                role="Custom_Expert_2",
                asset_class="custom2",
                expertise="Test2",
                system_prompt="Prompt2",
                dependencies=["Custom_Expert_1"]
            )
        ]

        mas = FinSageMAS(
            model_path="test/model",
            profiles=custom_profiles,
            load_in_4bit=False,
            bf16=False
        )

        assert mas.num_agents == 2
        assert "Custom_Expert_1" in mas.experts
        assert "Custom_Expert_2" in mas.experts

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_device_allocation_single_gpu(self, mock_device_count, mock_lora_expert):
        """测试单GPU设备分配"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        # 所有专家应该在同一个GPU上
        for profile in mas.profiles:
            assert profile.device == "cuda:0"

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_device_allocation_multi_gpu(self, mock_device_count, mock_lora_expert):
        """测试多GPU设备分配"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 3

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        # 专家应该分配到不同GPU
        devices = [profile.device for profile in mas.profiles]
        # 5个专家，3个GPU: cuda:0, cuda:1, cuda:2, cuda:0, cuda:1
        expected = ["cuda:0", "cuda:1", "cuda:2", "cuda:0", "cuda:1"]
        assert devices == expected

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_device_allocation_no_gpu(self, mock_device_count, mock_lora_expert):
        """测试无GPU时的CPU分配"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 0

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        # 所有专家应该在CPU上
        for profile in mas.profiles:
            assert profile.device == "cpu"


# ============================================================
# Test 10: FinSageMAS Dependency Order
# ============================================================

class TestFinSageMASDepend:
    """测试依赖顺序"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_get_dependency_order(self, mock_device_count, mock_lora_expert):
        """测试获取依赖执行顺序"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        order = mas.get_dependency_order()

        # 应该返回所有5个专家的顺序
        assert len(order) == 5

        # Stock_Expert应该在前面（无依赖）
        assert order[0] == "Stock_Expert"

        # Commodity_Expert依赖Stock和Bond，应该在它们之后
        stock_idx = order.index("Stock_Expert")
        bond_idx = order.index("Bond_Expert")
        commodity_idx = order.index("Commodity_Expert")
        assert commodity_idx > stock_idx
        assert commodity_idx > bond_idx


# ============================================================
# Test 11: FinSageMAS Joint Action Generation
# ============================================================

class TestFinSageMASJointAction:
    """测试联合动作生成"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_generate_joint_action(self, mock_device_count, mock_lora_expert):
        """测试联合动作生成"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        # Mock expert behavior
        def create_mock_expert(role):
            mock_expert = MagicMock()
            mock_expert.tokenizer = MagicMock()
            mock_expert.generate_action.return_value = (
                {"role": role, "action": "BUY_50%", "confidence": 0.8},
                torch.tensor([1, 2, 3]),
                "Response text"
            )
            mock_expert._build_prompt.return_value = f"Prompt for {role}"
            return mock_expert

        # 为每个调用返回不同的expert (5 Asset Experts + 4 Meta-Level Agents)
        roles = [
            "Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert",
            "Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"
        ]
        mock_lora_expert.side_effect = [create_mock_expert(role) for role in roles]

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        market_obs = "Market observation data"
        actions, tokens, prompts = mas.generate_joint_action(market_obs)

        # 验证返回9个动作 (5 Asset Experts + 4 Meta-Level Agents)
        assert len(actions) == 9
        assert len(tokens) == 9
        assert len(prompts) == 9

        # 验证每个动作都有role
        for action in actions:
            assert "role" in action
            assert action["role"] in roles

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_generate_joint_action_with_temperature(self, mock_device_count, mock_lora_expert):
        """测试带温度参数的联合动作生成"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_expert.generate_action.return_value = (
            {"role": "Test", "action": "HOLD", "confidence": 0.5},
            torch.tensor([1, 2]),
            "Response"
        )
        mock_expert._build_prompt.return_value = "Prompt"
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        market_obs = "Market data"
        actions, tokens, prompts = mas.generate_joint_action(
            market_obs,
            temperature=0.3
        )

        # 验证温度参数传递给每个expert
        for call in mock_expert.generate_action.call_args_list:
            assert call[1]['temperature'] == 0.3


# ============================================================
# Test 12: FinSageMAS Joint Log Probability
# ============================================================

class TestFinSageMASJointLogProb:
    """测试联合log probability计算"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_get_joint_action_log_probs(self, mock_device_count, mock_lora_expert):
        """测试联合log probability"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_expert.tokenizer.decode.return_value = "Action text"
        mock_expert.get_action_log_prob.return_value = (
            torch.tensor(-2.5),
            torch.tensor(1.2)
        )
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        batch_size = 3
        num_agents = 9  # 5 Asset Experts + 4 Meta-Level Agents

        obs_list = ["Obs1", "Obs2", "Obs3"]
        action_tokens_list = [
            [torch.tensor([1, 2]) for _ in range(num_agents)]
            for _ in range(batch_size)
        ]

        log_probs, entropies = mas.get_joint_action_log_probs(
            obs_list,
            action_tokens_list
        )

        # 验证形状
        assert log_probs.shape == (batch_size, num_agents)
        assert entropies.shape == (batch_size, num_agents)

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_joint_log_probs_single_batch(self, mock_device_count, mock_lora_expert):
        """测试单个批次的log probability"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_expert.tokenizer.decode.return_value = "Action"
        mock_expert.get_action_log_prob.return_value = (
            torch.tensor(-1.5),
            torch.tensor(0.8)
        )
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        obs_list = ["Market observation"]
        action_tokens_list = [
            [torch.tensor([10, 20, 30]) for _ in range(9)]  # 9 experts
        ]

        log_probs, entropies = mas.get_joint_action_log_probs(
            obs_list,
            action_tokens_list
        )

        assert log_probs.shape == (1, 9)  # 9 experts
        assert entropies.shape == (1, 9)  # 9 experts


# ============================================================
# Test 13: FinSageMAS Model Management
# ============================================================

class TestFinSageMASManagement:
    """测试FinSageMAS模型管理"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_train_mode(self, mock_device_count, mock_lora_expert):
        """测试训练模式"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        mas.train()

        # 验证所有expert都调用了train
        for expert in mas.experts.values():
            expert.train.assert_called()

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_eval_mode(self, mock_device_count, mock_lora_expert):
        """测试评估模式"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        mas.eval()

        # 验证所有expert都调用了eval
        for expert in mas.experts.values():
            expert.eval.assert_called()

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    @patch('os.makedirs')
    def test_mas_save(self, mock_makedirs, mock_device_count, mock_lora_expert):
        """测试保存所有专家"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        save_dir = "/path/to/save"
        mas.save(save_dir)

        # 验证创建目录（使用assert_any_call因为其他库也可能调用makedirs）
        mock_makedirs.assert_any_call(save_dir, exist_ok=True)

        # 验证所有expert都调用了save（由于所有expert共享同一个mock，验证总调用次数）
        assert mock_expert.save.call_count == len(mas.experts)
        mock_expert.save.assert_called_with(save_dir)


# ============================================================
# Test 14: LoRA Config Constants
# ============================================================

class TestLoRAConfig:
    """测试LoRA配置常量"""

    def test_finsage_lora_config_exists(self):
        """测试配置常量存在"""
        from finsage.rl.lora_expert import FINSAGE_LORA_CONFIG

        assert FINSAGE_LORA_CONFIG is not None
        assert isinstance(FINSAGE_LORA_CONFIG, dict)

    def test_finsage_lora_config_values(self):
        """测试配置值"""
        from finsage.rl.lora_expert import FINSAGE_LORA_CONFIG

        assert "r" in FINSAGE_LORA_CONFIG
        assert "lora_alpha" in FINSAGE_LORA_CONFIG
        assert "target_modules" in FINSAGE_LORA_CONFIG
        assert "lora_dropout" in FINSAGE_LORA_CONFIG
        assert "bias" in FINSAGE_LORA_CONFIG
        assert "task_type" in FINSAGE_LORA_CONFIG

    def test_finsage_lora_config_target_modules(self):
        """测试目标模块配置"""
        from finsage.rl.lora_expert import FINSAGE_LORA_CONFIG

        target_modules = FINSAGE_LORA_CONFIG["target_modules"]

        assert isinstance(target_modules, list)
        assert len(target_modules) > 0
        assert "q_proj" in target_modules
        assert "v_proj" in target_modules


# ============================================================
# Test 15: Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_parse_action_with_nested_braces(self):
        """测试嵌套大括号的JSON"""
        from finsage.rl.lora_expert import ExpertProfile, LoRAExpert

        profile = ExpertProfile(
            role="Test",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        expert = MagicMock()
        expert.profile = profile
        expert.role = profile.role

        expert._parse_action_response = LoRAExpert._parse_action_response.__get__(expert, type(expert))

        response = '''
        Some text before
        {
            "action": "BUY_50%",
            "confidence": 0.75,
            "reasoning": "Complex reasoning with {nested: braces}",
            "risk_assessment": {
                "volatility": 0.15,
                "metrics": {"var": 0.02, "cvar": 0.03}
            }
        }
        Text after
        '''

        action = expert._parse_action_response(response)

        assert action["action"] == "BUY_50%"
        assert "nested: braces" in action["reasoning"]

    def test_generate_action_with_empty_obs(self):
        """测试空观察"""
        from finsage.rl.lora_expert import ExpertProfile

        profile = ExpertProfile(
            role="Test",
            asset_class="test",
            expertise="测试",
            system_prompt="Test",
            dependencies=[]
        )

        expert = MagicMock()
        expert.profile = profile
        expert.role = profile.role
        expert.asset_class = profile.asset_class

        from finsage.rl.lora_expert import LoRAExpert
        expert._build_prompt = LoRAExpert._build_prompt.__get__(expert, type(expert))

        # 空观察应该仍然生成prompt
        prompt = expert._build_prompt("", None)
        assert len(prompt) > 0

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_mas_with_empty_profiles(self, mock_device_count, mock_lora_expert):
        """测试空配置列表"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_lora_expert.return_value = mock_expert

        # 空列表会触发默认profiles创建（因为Python中空列表是falsy）
        mas = FinSageMAS(
            model_path="test/model",
            profiles=[],  # 空列表 - 将使用默认profiles
            load_in_4bit=False,
            bf16=False
        )

        # 验证使用了默认profiles（9个专家: 5 Asset + 4 Meta-Level）
        assert mas.num_agents == 9
        assert len(mas.experts) == 9


# ============================================================
# Test 16: Integration Scenarios
# ============================================================

class TestIntegrationScenarios:
    """测试集成场景"""

    @patch('finsage.rl.lora_expert.HAS_TRANSFORMERS', True)
    @patch('finsage.rl.lora_expert.LoRAExpert')
    @patch('torch.cuda.device_count')
    def test_sequential_decision_flow(self, mock_device_count, mock_lora_expert):
        """测试顺序决策流程"""
        from finsage.rl.lora_expert import FinSageMAS

        mock_device_count.return_value = 1

        # 模拟专家依次决策
        call_count = [0]

        def mock_generate_action(market_obs, predecessor_actions=None, temperature=0.7):
            role = f"Expert_{call_count[0]}"
            call_count[0] += 1

            return (
                {
                    "role": role,
                    "action": "BUY_50%",
                    "confidence": 0.8,
                    "analysis": f"Analysis from {role}"
                },
                torch.tensor([1, 2, 3]),
                f"Response from {role}"
            )

        mock_expert = MagicMock()
        mock_expert.tokenizer = MagicMock()
        mock_expert.generate_action.side_effect = mock_generate_action
        mock_expert._build_prompt.return_value = "Prompt"
        mock_lora_expert.return_value = mock_expert

        mas = FinSageMAS(
            model_path="test/model",
            load_in_4bit=False,
            bf16=False
        )

        market_obs = "SPY: $450, TLT: $95, GLD: $180"
        actions, tokens, prompts = mas.generate_joint_action(market_obs)

        # 验证5个专家都生成了动作
        assert len(actions) == 5
        assert call_count[0] == 5


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Deep Tests for LoRA Expert Module")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short", "-x"])


if __name__ == "__main__":
    run_tests()
