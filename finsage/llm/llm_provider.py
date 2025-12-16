"""
LLM Provider
LLM服务提供者 - 统一接口支持多种LLM
"""

from typing import Dict, List, Any, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    LLM服务提供者

    支持:
    - OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
    - Anthropic (Claude)
    - 本地模型 (通过兼容API)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        初始化LLM提供者

        Args:
            provider: 提供者名称 ("openai", "anthropic", "local")
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL (用于本地模型)
            **kwargs: 其他参数
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.config = kwargs

        self._client = None
        self._setup_client()

        # 资源管理状态
        self._closed = False

        # Log initialization status (never log any part of API key)
        if self.api_key:
            logger.info(f"LLMProvider initialized: {provider}/{model} (API key configured)")
        else:
            logger.warning(f"LLMProvider initialized: {provider}/{model} (NO API KEY!)")

    def _setup_client(self):
        """设置客户端"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                logger.error("openai package not installed")

        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                )
            except ImportError:
                logger.error("anthropic package not installed")

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        创建文本补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            json_mode: 是否强制JSON输出
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        if self.provider == "openai":
            return self._openai_completion(
                messages, temperature, max_tokens, json_mode, **kwargs
            )
        elif self.provider == "anthropic":
            return self._anthropic_completion(
                messages, temperature, max_tokens, **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _openai_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        **kwargs
    ) -> str:
        """OpenAI补全"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if json_mode:
                params["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**params)

            # 验证响应结构
            if not response or not hasattr(response, 'choices'):
                raise ValueError("Invalid response structure from OpenAI")

            if not response.choices or len(response.choices) == 0:
                raise ValueError("No choices in OpenAI response")

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Message content is None")

            return content

        except Exception as e:
            # 不记录完整的异常以避免敏感信息泄露
            error_type = type(e).__name__
            logger.error(f"OpenAI completion failed ({error_type}): {str(e)[:200]}")
            raise

    def _anthropic_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Anthropic补全"""
        try:
            # 验证 temperature 范围 (Anthropic: 0-1)
            if not (0 <= temperature <= 1):
                logger.warning(f"Temperature {temperature} out of range [0,1], clamping")
                temperature = max(0, min(1, temperature))

            # 转换消息格式
            system_msg = ""
            converted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    converted_messages.append(msg)

            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg if system_msg else None,
                messages=converted_messages,
            )

            # 验证响应
            if not response.content or len(response.content) == 0:
                raise ValueError("No content in Anthropic response")

            content = response.content[0].text
            if content is None:
                raise ValueError("Text content is None")

            return content

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Anthropic completion failed ({error_type}): {str(e)[:200]}")
            raise

    def create_analysis(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        创建分析 (简化接口)

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            分析结果
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.create_completion(
            messages, temperature, max_tokens, json_mode=True
        )

    def create_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.5,
    ) -> Dict:
        """
        创建JSON格式响应

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数

        Returns:
            解析后的JSON字典
        """
        response = self.create_analysis(
            system_prompt, user_prompt, temperature
        )

        try:
            # 尝试解析JSON
            json_str = response
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    inner_parts = parts[1].split("```")
                    if len(inner_parts) > 0:
                        json_str = inner_parts[0]
            elif "```" in response:
                parts = response.split("```")
                if len(parts) > 1:
                    json_str = parts[1]

            return json.loads(json_str.strip())

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"error": "JSON parsing failed", "raw": response}

    def close(self):
        """显式关闭资源"""
        if self._closed:
            return
        try:
            if self._client is not None:
                # 关闭HTTP客户端连接
                if hasattr(self._client, 'close'):
                    self._client.close()
                self._client = None
            self._closed = True
            logger.debug("LLMProvider resources closed")
        except Exception as e:
            logger.warning(f"Error closing LLMProvider: {e}")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理资源"""
        self.close()

    def __del__(self):
        """析构时清理"""
        try:
            self.close()
        except Exception:
            pass
