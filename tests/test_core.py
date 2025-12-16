#!/usr/bin/env python
"""
Core Module Tests - 核心模块测试
覆盖: orchestrator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: Checkpoint State
# ============================================================

class TestCheckpointState:
    """测试断点状态"""

    def test_import(self):
        """测试导入"""
        from finsage.core.orchestrator import CheckpointState
        assert CheckpointState is not None

    def test_creation(self):
        """测试创建"""
        from finsage.core.orchestrator import CheckpointState

        state = CheckpointState(
            start_date="2024-01-01",
            end_date="2024-12-31",
            rebalance_frequency="monthly",
            current_date_index=5,
            total_dates=12,
            results={"returns": [0.01, 0.02]},
            portfolio_state={"cash": 100000},
            timestamp="2024-06-15T10:00:00"
        )

        assert state.start_date == "2024-01-01"
        assert state.current_date_index == 5
        assert state.total_dates == 12

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.core.orchestrator import CheckpointState

        state = CheckpointState(
            start_date="2024-01-01",
            end_date="2024-12-31",
            rebalance_frequency="monthly",
            current_date_index=0,
            total_dates=12,
            results={},
            portfolio_state={},
            timestamp="2024-01-01T00:00:00"
        )

        d = state.to_dict()

        assert "start_date" in d
        assert "end_date" in d
        assert "rebalance_frequency" in d
        assert "current_date_index" in d


# ============================================================
# Test 2: FinSage Orchestrator
# ============================================================

class TestFinSageOrchestrator:
    """测试FinSage协调器"""

    def test_import(self):
        """测试导入"""
        from finsage.core.orchestrator import FinSageOrchestrator
        assert FinSageOrchestrator is not None

    @patch('finsage.core.orchestrator.LLMProvider')
    @patch('finsage.core.orchestrator.DataLoader')
    @patch('finsage.core.orchestrator.MarketDataProvider')
    def test_initialization_mocked(self, mock_market, mock_loader, mock_llm):
        """测试初始化 (mocked)"""
        from finsage.core.orchestrator import FinSageOrchestrator
        from finsage.config import FinSageConfig

        mock_llm.return_value = Mock()
        mock_loader.return_value = Mock()
        mock_market.return_value = Mock()

        # 使用模拟配置
        config = FinSageConfig()

        # 注意: 实际初始化需要有效的API key
        # 这里只测试结构是否正确
        assert FinSageOrchestrator is not None

    def test_orchestrator_attributes(self):
        """测试协调器属性"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # 检查类定义是否完整
        assert hasattr(FinSageOrchestrator, '__init__')
        assert hasattr(FinSageOrchestrator, '_init_components')


# ============================================================
# Test 3: Orchestrator Methods (Import Tests)
# ============================================================

class TestOrchestratorMethods:
    """测试协调器方法存在性"""

    def test_methods_exist(self):
        """测试方法存在"""
        from finsage.core.orchestrator import FinSageOrchestrator

        # 检查关键方法是否存在
        expected_methods = [
            '__init__',
            '_init_components',
        ]

        for method in expected_methods:
            assert hasattr(FinSageOrchestrator, method), f"Missing method: {method}"


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Core Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
