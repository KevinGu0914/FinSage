#!/usr/bin/env python3
"""
FinSage - Multi-Agent Financial Portfolio Management System
金融智者 - 多智能体金融组合管理系统

Usage:
    python main.py --start 2023-01-01 --end 2023-12-31 --frequency weekly

    # Resume from checkpoint:
    python main.py --start 2023-01-01 --end 2023-12-31 --frequency weekly --resume
"""

import argparse
import logging
import os
from datetime import datetime

# Load environment variables from .env file (override=True to fix pre-existing shell env vars)
from dotenv import load_dotenv
load_dotenv(override=True)

from finsage.config import FinSageConfig, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
from finsage.core.orchestrator import FinSageOrchestrator


def setup_logging(log_level: str = "INFO", log_file: str = None):
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FinSage - Multi-Agent Financial Portfolio Management"
    )

    parser.add_argument(
        "--start", "-s",
        type=validate_date,  # 使用自定义验证函数
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-e",
        type=validate_date,  # 使用自定义验证函数
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

    args = parser.parse_args()

    # 验证日期范围
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    if start > end:
        parser.error(f"Start date {args.start} cannot be after end date {args.end}")

    # 验证初始资金
    if args.capital <= 0:
        parser.error(f"Initial capital must be positive, got {args.capital}")

    return args


def get_config(config_name: str, capital: float, model: str) -> FinSageConfig:
    """获取配置"""
    if config_name == "conservative":
        config = CONSERVATIVE_CONFIG
    elif config_name == "aggressive":
        config = AGGRESSIVE_CONFIG
    else:
        config = FinSageConfig()

    # 覆盖参数
    config.trading.initial_capital = capital
    config.llm.model = model

    return config


def main():
    """主函数"""
    try:
        args = parse_args()
    except SystemExit:
        return 1

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output, "logs", f"finsage_{timestamp}.log")
    setup_logging(args.log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("FinSage - Multi-Agent Financial Portfolio Management")
    logger.info("=" * 60)

    # 获取配置
    config = get_config(args.config, args.capital, args.model)

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
    output_file = os.path.join(
        args.output,
        f"finsage_results_{args.start}_{args.end}_{timestamp}.json"
    )
    orchestrator.save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(f"Fatal error: {e}", exc_info=True)
        exit(1)
