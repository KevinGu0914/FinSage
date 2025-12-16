#!/usr/bin/env python
"""
LLM Module Tests - LLM模块测试
覆盖: llm_provider
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: LLM Provider
# ============================================================

class TestLLMProvider:
    """测试LLM提供者"""

    def test_import(self):
        """测试导入"""
        from finsage.llm.llm_provider import LLMProvider
        assert LLMProvider is not None

    def test_initialization_openai(self):
        """测试OpenAI初始化"""
        from finsage.llm.llm_provider import LLMProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = LLMProvider(
                provider="openai",
                model="gpt-4o-mini",
                api_key="test-key"
            )

            assert provider.provider == "openai"
            assert provider.model == "gpt-4o-mini"

    def test_initialization_anthropic(self):
        """测试Anthropic初始化"""
        from finsage.llm.llm_provider import LLMProvider

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = LLMProvider(
                provider="anthropic",
                model="claude-3-sonnet-20240229",
                api_key="test-key"
            )

            assert provider.provider == "anthropic"

    def test_initialization_with_base_url(self):
        """测试带base_url初始化"""
        from finsage.llm.llm_provider import LLMProvider

        provider = LLMProvider(
            provider="openai",
            model="local-model",
            api_key="test-key",
            base_url="http://localhost:8000/v1"
        )

        assert provider.base_url == "http://localhost:8000/v1"

    def test_create_completion_mocked(self):
        """测试文本补全 (mocked)"""
        from finsage.llm.llm_provider import LLMProvider

        # 创建provider并直接替换_client
        provider = LLMProvider(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key"
        )

        # 直接mock _client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        response = provider.create_completion(messages)

        assert response == "Test response"

    def test_create_completion_with_temperature(self):
        """测试带温度参数的补全"""
        from finsage.llm.llm_provider import LLMProvider

        provider = LLMProvider(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key"
        )

        # 直接mock _client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        response = provider.create_completion(
            messages,
            temperature=0.5,
            max_tokens=1000
        )

        assert isinstance(response, str)

    def test_provider_attributes(self):
        """测试提供者属性"""
        from finsage.llm.llm_provider import LLMProvider

        provider = LLMProvider(
            provider="openai",
            model="test-model",
            api_key="test-key"
        )

        assert hasattr(provider, 'provider')
        assert hasattr(provider, 'model')
        assert hasattr(provider, 'api_key')
        assert hasattr(provider, '_client')

    def test_json_mode(self):
        """测试JSON模式"""
        from finsage.llm.llm_provider import LLMProvider

        provider = LLMProvider(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key"
        )

        # 直接mock _client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        messages = [{"role": "user", "content": "Return JSON"}]
        response = provider.create_completion(messages, json_mode=True)

        assert isinstance(response, str)


# ============================================================
# Test 2: LLM Provider Error Handling
# ============================================================

class TestLLMProviderErrorHandling:
    """测试LLM提供者错误处理"""

    def test_missing_api_key_uses_env(self):
        """测试缺少显式API密钥时从环境变量获取"""
        from finsage.llm.llm_provider import LLMProvider

        # 设置环境变量
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
            provider = LLMProvider(
                provider="openai",
                model="gpt-4o-mini",
                api_key=None  # 不显式传入，从环境变量获取
            )
            # 应该从环境变量获取api_key
            assert provider.api_key == "env-test-key"

    def test_unsupported_provider(self):
        """测试不支持的提供者"""
        from finsage.llm.llm_provider import LLMProvider

        provider = LLMProvider(
            provider="unsupported",
            model="test",
            api_key="test"
        )

        # 不支持的提供者应该能初始化，但客户端为None
        assert provider._client is None


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" LLM Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
