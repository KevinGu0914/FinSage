#!/usr/bin/env python3
"""
FinSage CLI Entry Point
命令行入口 - 支持 pip install 后通过 finsage 命令运行
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """设置日志"""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def validate_date(date_string: str) -> str:
    """验证日期格式"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return date_string
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_string}. Expected YYYY-MM-DD"
        )


def parse_args(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="finsage",
        description="FinSage - AI-Powered Multi-Asset Portfolio Management System"
    )

    parser.add_argument(
        "--start", "-s",
        type=validate_date,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-e",
        type=validate_date,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=str,
        default="daily",
        choices=["daily", "weekly", "monthly"],
        help="Rebalance frequency (default: daily)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="default",
        choices=["default", "conservative", "aggressive"],
        help="Configuration template (default: default)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital (default: 1,000,000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/finsage",
        help="Checkpoint directory (default: ./checkpoints/finsage)"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 2.0.0"
    )

    parsed = parser.parse_args(args)

    # 验证日期范围
    start = datetime.strptime(parsed.start, '%Y-%m-%d')
    end = datetime.strptime(parsed.end, '%Y-%m-%d')
    if start > end:
        parser.error(f"Start date {parsed.start} cannot be after end date {parsed.end}")

    # 验证初始资金
    if parsed.capital <= 0:
        parser.error(f"Initial capital must be positive, got {parsed.capital}")

    return parsed


def run(args) -> int:
    """运行 FinSage"""
    from finsage.config import FinSageConfig, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
    from finsage.core.orchestrator import FinSageOrchestrator

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("FinSage - AI-Powered Multi-Asset Portfolio Management")
    logger.info("=" * 60)

    # 获取配置
    if args.config == "conservative":
        config = CONSERVATIVE_CONFIG
    elif args.config == "aggressive":
        config = AGGRESSIVE_CONFIG
    else:
        config = FinSageConfig()

    # 覆盖参数
    config.trading.initial_capital = args.capital
    config.llm.model = args.model

    logger.info(f"Configuration: {args.config}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Rebalance frequency: {args.frequency}")
    logger.info(f"Initial capital: ${args.capital:,.2f}")
    logger.info(f"LLM model: {args.model}")

    # 创建协调器
    orchestrator = FinSageOrchestrator(
        config=config,
        checkpoint_dir=args.checkpoint_dir
    )

    # 运行
    if args.resume:
        logger.info(f"Attempting to resume from checkpoint: {args.checkpoint_dir}")

    results = orchestrator.run(
        start_date=args.start,
        end_date=args.end,
        rebalance_frequency=args.frequency,
        resume=args.resume,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("FinSage Results Summary")
    print("=" * 60)

    metrics = results.get("final_metrics", {})
    print(f"Cumulative Return: {metrics.get('cumulative_return', 0):.2%}")
    print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Total Trades: {metrics.get('n_trades', 0)}")

    final_portfolio = results.get("final_portfolio", {})
    print(f"\nFinal Portfolio Value: ${final_portfolio.get('portfolio_value', 0):,.2f}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output,
        f"finsage_results_{args.start}_{args.end}_{timestamp}.json"
    )
    orchestrator.save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")

    return 0


def main(args=None) -> int:
    """主入口函数"""
    # 加载环境变量
    load_dotenv(override=True)

    try:
        parsed_args = parse_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(parsed_args.output, "logs", f"finsage_{timestamp}.log")
    setup_logging(parsed_args.log_level, log_file)

    try:
        return run(parsed_args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        logging.getLogger(__name__).error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
