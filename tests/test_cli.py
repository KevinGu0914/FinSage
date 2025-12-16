#!/usr/bin/env python
"""
CLI Module Tests - 命令行模块测试
覆盖: cli.py (setup_logging, validate_date, parse_args, main)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import argparse
import logging
import tempfile
from unittest.mock import Mock, MagicMock, patch


# ============================================================
# Test 1: setup_logging
# ============================================================

class TestSetupLogging:
    """测试日志设置"""

    def test_import(self):
        """测试导入"""
        from finsage.cli import setup_logging
        assert setup_logging is not None

    def test_default_logging(self):
        """测试默认日志设置"""
        from finsage.cli import setup_logging

        # 应该不抛出异常
        setup_logging()
        logger = logging.getLogger("test_default")
        logger.info("Test message")

    def test_debug_level(self):
        """测试DEBUG级别"""
        from finsage.cli import setup_logging

        # 使用临时日志文件来测试
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "debug.log")
            setup_logging(log_level="DEBUG", log_file=log_file)

            # 验证可以写入DEBUG级别日志
            logger = logging.getLogger("debug_test_logger")
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug message")

            # 日志文件应该存在
            assert os.path.exists(log_file)

    def test_error_level(self):
        """测试ERROR级别"""
        from finsage.cli import setup_logging

        setup_logging(log_level="ERROR")

    def test_with_log_file(self):
        """测试带日志文件"""
        from finsage.cli import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "subdir", "test.log")
            setup_logging(log_level="INFO", log_file=log_file)

            # 验证目录被创建
            assert os.path.exists(os.path.dirname(log_file))


# ============================================================
# Test 2: validate_date
# ============================================================

class TestValidateDate:
    """测试日期验证"""

    def test_import(self):
        """测试导入"""
        from finsage.cli import validate_date
        assert validate_date is not None

    def test_valid_date(self):
        """测试有效日期"""
        from finsage.cli import validate_date

        result = validate_date("2024-01-15")
        assert result == "2024-01-15"

    def test_valid_date_leap_year(self):
        """测试闰年日期"""
        from finsage.cli import validate_date

        result = validate_date("2024-02-29")
        assert result == "2024-02-29"

    def test_invalid_format(self):
        """测试无效格式"""
        from finsage.cli import validate_date

        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            validate_date("01-15-2024")

        assert "Invalid date format" in str(excinfo.value)

    def test_invalid_date(self):
        """测试无效日期"""
        from finsage.cli import validate_date

        with pytest.raises(argparse.ArgumentTypeError):
            validate_date("2024-13-01")  # 13月不存在

    def test_invalid_day(self):
        """测试无效日期"""
        from finsage.cli import validate_date

        with pytest.raises(argparse.ArgumentTypeError):
            validate_date("2024-02-30")  # 2月没有30号


# ============================================================
# Test 3: parse_args
# ============================================================

class TestParseArgs:
    """测试参数解析"""

    def test_import(self):
        """测试导入"""
        from finsage.cli import parse_args
        assert parse_args is not None

    def test_required_args(self):
        """测试必需参数"""
        from finsage.cli import parse_args

        args = parse_args(["--start", "2024-01-01", "--end", "2024-12-31"])

        assert args.start == "2024-01-01"
        assert args.end == "2024-12-31"

    def test_default_values(self):
        """测试默认值"""
        from finsage.cli import parse_args

        args = parse_args(["--start", "2024-01-01", "--end", "2024-12-31"])

        assert args.frequency == "daily"
        assert args.config == "default"
        assert args.capital == 1_000_000.0
        assert args.model == "gpt-4o-mini"
        assert args.output == "./results"
        assert args.log_level == "INFO"
        assert args.resume == False

    def test_short_args(self):
        """测试短参数"""
        from finsage.cli import parse_args

        args = parse_args(["-s", "2024-01-01", "-e", "2024-12-31"])

        assert args.start == "2024-01-01"
        assert args.end == "2024-12-31"

    def test_frequency_weekly(self):
        """测试周频率"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--frequency", "weekly"
        ])

        assert args.frequency == "weekly"

    def test_frequency_monthly(self):
        """测试月频率"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "-f", "monthly"
        ])

        assert args.frequency == "monthly"

    def test_config_conservative(self):
        """测试保守配置"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--config", "conservative"
        ])

        assert args.config == "conservative"

    def test_config_aggressive(self):
        """测试激进配置"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "-c", "aggressive"
        ])

        assert args.config == "aggressive"

    def test_custom_capital(self):
        """测试自定义资金"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--capital", "500000"
        ])

        assert args.capital == 500000.0

    def test_custom_model(self):
        """测试自定义模型"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--model", "gpt-4-turbo"
        ])

        assert args.model == "gpt-4-turbo"

    def test_resume_flag(self):
        """测试恢复标志"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--resume"
        ])

        assert args.resume == True

    def test_checkpoint_dir(self):
        """测试检查点目录"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--checkpoint-dir", "/custom/checkpoint"
        ])

        assert args.checkpoint_dir == "/custom/checkpoint"

    def test_log_level_debug(self):
        """测试日志级别"""
        from finsage.cli import parse_args

        args = parse_args([
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--log-level", "DEBUG"
        ])

        assert args.log_level == "DEBUG"

    def test_missing_required_args(self):
        """测试缺少必需参数"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

    def test_missing_end_date(self):
        """测试缺少结束日期"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--start", "2024-01-01"])

    def test_invalid_date_order(self):
        """测试无效日期顺序"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--start", "2024-12-31", "--end", "2024-01-01"])

    def test_negative_capital(self):
        """测试负资金"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--start", "2024-01-01",
                "--end", "2024-12-31",
                "--capital", "-1000"
            ])

    def test_zero_capital(self):
        """测试零资金"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--start", "2024-01-01",
                "--end", "2024-12-31",
                "--capital", "0"
            ])

    def test_invalid_frequency(self):
        """测试无效频率"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--start", "2024-01-01",
                "--end", "2024-12-31",
                "--frequency", "hourly"
            ])

    def test_invalid_config(self):
        """测试无效配置"""
        from finsage.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--start", "2024-01-01",
                "--end", "2024-12-31",
                "--config", "invalid"
            ])


# ============================================================
# Test 4: main function
# ============================================================

class TestMainFunction:
    """测试主函数"""

    def test_import(self):
        """测试导入"""
        from finsage.cli import main
        assert main is not None

    def test_main_missing_args(self):
        """测试缺少参数时的退出"""
        from finsage.cli import main

        # 没有参数应该返回非零退出码
        result = main([])
        assert result != 0

    def test_main_help(self):
        """测试帮助信息"""
        from finsage.cli import parse_args
        import io
        import contextlib

        # 帮助信息会导致SystemExit
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["--help"])

        assert excinfo.value.code == 0

    def test_main_version(self):
        """测试版本信息"""
        from finsage.cli import parse_args

        # 版本信息会导致SystemExit
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["--version"])

        assert excinfo.value.code == 0


# ============================================================
# Test 5: run function
# ============================================================

class TestRunFunction:
    """测试运行函数"""

    def test_import(self):
        """测试导入"""
        from finsage.cli import run
        assert run is not None

    @patch('finsage.core.orchestrator.FinSageOrchestrator')
    def test_run_with_default_config(self, mock_orchestrator):
        """测试默认配置运行"""
        from finsage.cli import run, parse_args

        # 设置mock
        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            "final_metrics": {
                "cumulative_return": 0.15,
                "annual_return": 0.12,
                "volatility": 0.10,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.05,
                "win_rate": 0.55,
                "n_trades": 100
            },
            "final_portfolio": {"portfolio_value": 1150000}
        }
        mock_orchestrator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args([
                "--start", "2024-01-01",
                "--end", "2024-06-30",
                "--output", tmpdir
            ])
            result = run(args)

            assert result == 0

    @patch('finsage.core.orchestrator.FinSageOrchestrator')
    def test_run_with_conservative_config(self, mock_orchestrator):
        """测试保守配置运行"""
        from finsage.cli import run, parse_args

        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            "final_metrics": {},
            "final_portfolio": {}
        }
        mock_orchestrator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args([
                "--start", "2024-01-01",
                "--end", "2024-06-30",
                "--config", "conservative",
                "--output", tmpdir
            ])
            result = run(args)

            assert result == 0

    @patch('finsage.core.orchestrator.FinSageOrchestrator')
    def test_run_with_aggressive_config(self, mock_orchestrator):
        """测试激进配置运行"""
        from finsage.cli import run, parse_args

        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            "final_metrics": {},
            "final_portfolio": {}
        }
        mock_orchestrator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args([
                "--start", "2024-01-01",
                "--end", "2024-06-30",
                "--config", "aggressive",
                "--output", tmpdir
            ])
            result = run(args)

            assert result == 0


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" CLI Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
