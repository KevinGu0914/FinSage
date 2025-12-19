#!/usr/bin/env python3
"""
MARFT V4 训练好的 LoRA 权重评估脚本

在测试集上评估 Aggressive 和 Balanced 策略的性能，
并生成详细的监控日志。

Usage:
    python scripts/evaluate_trained_lora.py --checkpoint aggressive_final --test-start 2024-10-01 --test-end 2024-11-30
    python scripts/evaluate_trained_lora.py --checkpoint balanced_final --test-start 2024-10-01 --test-end 2024-11-30
"""

import os
import sys

# 确保 finsage 模块可导入 (兼容本地和远程服务器)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

# 远程服务器备用路径
if not os.path.exists(os.path.join(_project_root, 'finsage')):
    for fallback in ['/root/finsage', '/root/FinSage', '/home/finsage']:
        if os.path.exists(os.path.join(fallback, 'finsage')):
            sys.path.insert(0, fallback)
            break

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import traceback
import numpy as np
import pandas as pd
import torch

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# Bug 检测与追踪系统
# ============================================================

class BugTracker:
    """追踪评估过程中发现的所有潜在问题"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.issues = defaultdict(list)  # category -> [issues]
        self.warnings = defaultdict(int)  # warning_type -> count
        self.expert_call_counts = defaultdict(int)
        self.meta_agent_outputs = defaultdict(list)
        self.error_samples = []  # 保存前10个错误样本

    def log_issue(self, category: str, message: str, severity: str = "WARNING",
                  date: str = None, extra: dict = None):
        """记录一个问题"""
        issue = {
            "message": message,
            "severity": severity,
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "extra": extra or {}
        }
        self.issues[category].append(issue)

        if severity == "ERROR":
            self.logger.error(f"[BUG-{category}] {message}")
            if len(self.error_samples) < 10:
                self.error_samples.append(issue)
        elif severity == "WARNING":
            self.logger.warning(f"[BUG-{category}] {message}")
            self.warnings[category] += 1
        else:
            self.logger.debug(f"[BUG-{category}] {message}")

    def track_expert_call(self, role: str, success: bool, output: dict = None, error: str = None):
        """追踪Expert调用"""
        self.expert_call_counts[role] += 1
        if not success:
            self.log_issue("EXPERT_CALL_FAILED", f"{role} 调用失败: {error}",
                          severity="ERROR", extra={"role": role, "error": error})

        # 追踪Meta-Level Agent输出
        if role in ["Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"]:
            self.meta_agent_outputs[role].append(output or {})

    def validate_expert_chain(self, all_actions: dict, expected_experts: list):
        """验证Expert Chain是否完整"""
        missing = []
        for expert in expected_experts:
            if expert not in all_actions:
                missing.append(expert)
                self.log_issue("MISSING_EXPERT", f"Expert链缺失: {expert}",
                              severity="ERROR")

        # 检查是否有意外的Expert
        for expert in all_actions:
            if expert not in expected_experts:
                self.log_issue("UNEXPECTED_EXPERT", f"意外的Expert: {expert}",
                              severity="WARNING")

        return len(missing) == 0

    def validate_meta_agent_output(self, role: str, output: dict, date: str):
        """验证Meta-Level Agent输出格式"""
        if role == "Risk_Controller":
            # 必须有 veto 或 risk_level
            if "veto" not in output and "risk_level" not in output and "action" not in output:
                self.log_issue("INVALID_OUTPUT",
                              f"Risk_Controller 输出缺少 veto/risk_level/action 字段",
                              severity="WARNING", date=date, extra={"output": str(output)[:200]})
                return False

        elif role == "Position_Sizing_Agent":
            # 必须有 risk_budget 或 position_scale
            if "risk_budget" not in output and "position_scale" not in output and "action" not in output:
                self.log_issue("INVALID_OUTPUT",
                              f"Position_Sizing_Agent 输出缺少 risk_budget/position_scale 字段",
                              severity="WARNING", date=date, extra={"output": str(output)[:200]})
                return False

        elif role == "Portfolio_Manager":
            # 检查是否有配置建议
            if "target_allocation" not in output and "allocation" not in output and "action" not in output:
                self.log_issue("INVALID_OUTPUT",
                              f"Portfolio_Manager 输出缺少配置建议",
                              severity="WARNING", date=date, extra={"output": str(output)[:200]})
                return False

        elif role == "Hedging_Agent":
            # 检查是否有对冲建议
            if "hedge_strategy" not in output and "hedge_ratio" not in output and "action" not in output:
                self.log_issue("INVALID_OUTPUT",
                              f"Hedging_Agent 输出缺少对冲建议",
                              severity="WARNING", date=date, extra={"output": str(output)[:200]})
                return False

        return True

    def check_data_issues(self, obs, date: str, history: pd.DataFrame = None):
        """检查输入数据问题"""
        # obs 可能是 string (prompt) 或 dict
        if isinstance(obs, str):
            # 如果是 string，从 history 检查数据问题
            if history is not None and isinstance(history, pd.DataFrame):
                if history.empty:
                    self.log_issue("DATA_MISSING", "history DataFrame为空",
                                  severity="WARNING", date=date)
                elif len(history) < 5:
                    self.log_issue("DATA_INSUFFICIENT", f"history数据不足: {len(history)}行",
                                  severity="WARNING", date=date)
            return

        market_data = obs.get("market_data", {}) if isinstance(obs, dict) else {}

        # 检查价格数据
        prices = market_data.get("prices", {})
        if not prices:
            self.log_issue("DATA_MISSING", "市场数据缺少价格",
                          severity="ERROR", date=date)

        # 检查收益率数据
        returns = market_data.get("returns", {})
        if isinstance(returns, pd.DataFrame):
            if returns.empty:
                self.log_issue("DATA_MISSING", "收益率数据为空DataFrame",
                              severity="WARNING", date=date)
        elif isinstance(returns, dict):
            if len(returns) == 0:
                self.log_issue("DATA_MISSING", "收益率数据为空dict",
                              severity="WARNING", date=date)

        # 检查returns_window
        returns_window = market_data.get("returns_window")
        if returns_window is None:
            self.log_issue("DATA_MISSING", "缺少 returns_window 字段",
                          severity="WARNING", date=date)
        elif isinstance(returns_window, pd.DataFrame) and returns_window.empty:
            self.log_issue("DATA_MISSING", "returns_window 为空",
                          severity="WARNING", date=date)

    def generate_summary(self) -> dict:
        """生成问题摘要"""
        summary = {
            "total_issues": sum(len(v) for v in self.issues.values()),
            "issues_by_category": {k: len(v) for k, v in self.issues.items()},
            "warnings_by_type": dict(self.warnings),
            "expert_call_counts": dict(self.expert_call_counts),
            "error_samples": self.error_samples[:5],
            "meta_agent_analysis": {}
        }

        # 分析Meta-Level Agent
        expected_meta = ["Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"]
        for agent in expected_meta:
            outputs = self.meta_agent_outputs.get(agent, [])
            summary["meta_agent_analysis"][agent] = {
                "total_calls": len(outputs),
                "non_empty_outputs": sum(1 for o in outputs if o),
                "sample_output": outputs[0] if outputs else None
            }

        return summary

    def print_summary(self):
        """打印问题摘要"""
        summary = self.generate_summary()

        self.logger.info("\n" + "=" * 80)
        self.logger.info(" 🐛 BUG 检测报告")
        self.logger.info("=" * 80)

        self.logger.info(f"\n📊 问题统计:")
        self.logger.info(f"  总问题数: {summary['total_issues']}")
        for cat, count in summary['issues_by_category'].items():
            self.logger.info(f"  - {cat}: {count}")

        self.logger.info(f"\n📞 Expert 调用统计:")
        for expert, count in summary['expert_call_counts'].items():
            status = "✅" if count > 0 else "❌"
            self.logger.info(f"  {status} {expert}: {count} 次")

        # 检查Meta-Level Agents
        self.logger.info(f"\n🎯 Meta-Level Agent 分析:")
        for agent, analysis in summary['meta_agent_analysis'].items():
            status = "✅" if analysis['total_calls'] > 0 else "❌ [BUG]"
            self.logger.info(f"  {status} {agent}:")
            self.logger.info(f"      调用次数: {analysis['total_calls']}")
            self.logger.info(f"      有效输出: {analysis['non_empty_outputs']}")
            if analysis['sample_output']:
                self.logger.info(f"      样例输出: {str(analysis['sample_output'])[:100]}...")

        # 检测关键Bug
        self.logger.info(f"\n🚨 关键问题检测:")
        critical_bugs = []

        # Bug 1: Meta-Level Agents 未调用
        for agent in ["Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"]:
            if summary['expert_call_counts'].get(agent, 0) == 0:
                critical_bugs.append(f"{agent} 从未被调用!")

        # Bug 2: Expert Chain 不完整
        expected_all = ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert",
                       "Crypto_Expert", "Portfolio_Manager", "Hedging_Agent",
                       "Position_Sizing_Agent", "Risk_Controller"]
        missing = [e for e in expected_all if summary['expert_call_counts'].get(e, 0) == 0]
        if missing:
            critical_bugs.append(f"缺失的Experts: {missing}")

        if critical_bugs:
            for bug in critical_bugs:
                self.logger.error(f"  ❌ {bug}")
        else:
            self.logger.info("  ✅ 未发现关键Bug")

        return summary

# ============================================================
# 详细监控日志配置
# ============================================================

def setup_detailed_logging(log_file: str) -> logging.Logger:
    """设置详细监控日志"""
    logger = logging.getLogger("LoRAEval")
    logger.setLevel(logging.DEBUG)

    # 清除现有handlers
    logger.handlers = []

    # 文件handler - 详细日志
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台handler - INFO级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# 资产类别和符号配置
# ============================================================

ASSETS = {
    "stocks": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "bonds": ["TLT", "LQD", "HYG", "AGG"],
    "commodities": ["GLD", "SLV", "USO", "UNG"],
    "reits": ["VNQ", "IYR", "XLRE"],
    "crypto": ["BTC-USD", "ETH-USD"],
}

# 资产类别权重配置 - 控制各类资产的仓位上限
# 股票权重更高，债券权重降低
ASSET_CLASS_WEIGHTS = {
    "stocks": 1.0,      # 股票: 100% 权重 (最高优先)
    "bonds": 0.3,       # 债券: 30% 权重 (降低)
    "commodities": 0.6, # 商品: 60% 权重
    "reits": 0.5,       # REITs: 50% 权重
    "crypto": 0.4,      # 加密货币: 40% 权重 (风险高)
}

ALL_SYMBOLS = []
for symbols in ASSETS.values():
    ALL_SYMBOLS.extend(symbols)


# ============================================================
# Portfolio Manager
# ============================================================

class Portfolio:
    """投资组合管理 - 带详细日志"""

    def __init__(self, initial_capital: float, logger: logging.Logger):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.position_costs: Dict[str, float] = {}  # symbol -> avg cost
        self.value_history: List[float] = []
        self.date_history: List[str] = []
        self.trade_log: List[Dict] = []
        self.trade_count = 0
        self.transaction_cost_rate = 0.001  # 0.1%
        self.logger = logger

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总价值"""
        total = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices and qty > 0:
                total += qty * prices[symbol]
        return total

    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        total_value: float,
        date: str,
        expert_role: str
    ) -> Dict:
        """执行交易并记录详细日志"""
        result = {
            "date": date,
            "symbol": symbol,
            "action": action,
            "price": price,
            "expert": expert_role,
            "executed": False
        }

        if "HOLD" in action or action == "":
            return result

        # 解析比例
        pct = 0.25  # 默认25%
        if "_" in action:
            parts = action.split("_")
            if len(parts) >= 2:
                pct_str = parts[1].replace("%", "")
                try:
                    pct = int(pct_str) / 100
                except:
                    pct = 0.25

        current_qty = self.positions.get(symbol, 0)

        if "BUY" in action:
            buy_amount = total_value * pct
            shares_to_buy = int(buy_amount / price)
            cost = shares_to_buy * price * (1 + self.transaction_cost_rate)

            if cost <= self.cash and shares_to_buy > 0:
                self.cash -= cost
                old_qty = self.positions.get(symbol, 0)
                old_cost = self.position_costs.get(symbol, 0)

                new_qty = old_qty + shares_to_buy
                new_avg_cost = ((old_qty * old_cost) + (shares_to_buy * price)) / new_qty if new_qty > 0 else 0

                self.positions[symbol] = new_qty
                self.position_costs[symbol] = new_avg_cost
                self.trade_count += 1

                result["executed"] = True
                result["shares"] = shares_to_buy
                result["cost"] = cost
                result["new_position"] = new_qty

                self.logger.debug(
                    f"BUY {symbol}: {shares_to_buy} shares @ ${price:.2f} = ${cost:.2f}, "
                    f"new position: {new_qty} shares"
                )

        elif "SELL" in action:
            if current_qty > 0:
                shares_to_sell = int(current_qty * pct)
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price * (1 - self.transaction_cost_rate)
                    self.cash += proceeds
                    self.positions[symbol] = current_qty - shares_to_sell
                    self.trade_count += 1

                    # 计算盈亏
                    avg_cost = self.position_costs.get(symbol, price)
                    pnl = (price - avg_cost) * shares_to_sell

                    result["executed"] = True
                    result["shares"] = shares_to_sell
                    result["proceeds"] = proceeds
                    result["pnl"] = pnl
                    result["new_position"] = current_qty - shares_to_sell

                    self.logger.debug(
                        f"SELL {symbol}: {shares_to_sell} shares @ ${price:.2f} = ${proceeds:.2f}, "
                        f"PnL: ${pnl:.2f}, remaining: {current_qty - shares_to_sell} shares"
                    )

        if result["executed"]:
            self.trade_log.append(result)

        return result

    def record_value(self, prices: Dict[str, float], date: str):
        """记录当前组合价值"""
        value = self.get_total_value(prices)
        self.value_history.append(value)
        self.date_history.append(date)

    def get_metrics(self) -> Dict:
        """计算绩效指标"""
        if len(self.value_history) < 2:
            return {}

        values = np.array(self.value_history)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1

        # 年化收益率 (假设252交易日)
        days = len(values)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # 波动率
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # 夏普比率 (假设无风险利率4%)
        rf = 0.04 / 252
        excess_returns = returns - rf
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Sortino比率
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = np.mean(excess_returns) * 252 / downside_std if downside_std > 0 else 0

        # Calmar比率
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        winning_trades = sum(1 for t in self.trade_log if t.get("pnl", 0) > 0)
        total_trades_with_pnl = sum(1 for t in self.trade_log if "pnl" in t)
        win_rate = winning_trades / total_trades_with_pnl if total_trades_with_pnl > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "total_trades": self.trade_count,
            "win_rate": win_rate,
            "final_value": values[-1],
            "initial_value": values[0],
        }


# ============================================================
# 数据获取
# ============================================================

def fetch_market_data(start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    """获取市场数据 (使用 FMP API)"""
    from finsage.data.fmp_client import FMPClient

    logger.info(f"Fetching data for {len(ALL_SYMBOLS)} symbols from {start_date} to {end_date}")

    client = FMPClient()

    all_data = {}
    for symbol in ALL_SYMBOLS:
        try:
            df = client.get_historical_price(symbol, start_date, end_date)
            if df is not None and not df.empty:
                if 'close' in df.columns:
                    all_data[symbol] = df['close']
                elif 'Close' in df.columns:
                    all_data[symbol] = df['Close']
                logger.debug(f"  {symbol}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  {symbol}: failed - {e}")

    if not all_data:
        logger.error("No data fetched!")
        return pd.DataFrame()

    prices = pd.DataFrame(all_data)
    prices = prices.ffill().bfill()
    logger.info(f"Got {len(prices)} trading days of data for {len(all_data)} symbols")

    return prices


def create_market_observation(
    date: str,
    prices: pd.Series,
    history: pd.DataFrame,
    portfolio: Portfolio,
    logger: logging.Logger
) -> str:
    """创建详细的市场观察"""

    obs_parts = [
        f"## 市场日期: {date}",
        f"## 资产类别: multi-asset",
        f"## 组合价值: ${portfolio.get_total_value(prices.to_dict()):,.2f}",
        ""
    ]

    for asset_class, symbols in ASSETS.items():
        obs_parts.append(f"### {asset_class.upper()}")
        for symbol in symbols:
            if symbol in prices.index:
                price = prices[symbol]

                if symbol in history.columns:
                    hist = history[symbol].dropna()
                    if len(hist) > 14:
                        # 日涨跌
                        change = (hist.iloc[-1] / hist.iloc[-2] - 1) * 100 if len(hist) > 1 else 0

                        # RSI
                        delta = hist.diff()
                        gain = delta.where(delta > 0, 0).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50

                        # SMA
                        sma_20 = hist.rolling(20).mean().iloc[-1] if len(hist) >= 20 else price
                        sma_trend = "above" if price > sma_20 else "below"

                        # 持仓信息
                        position = portfolio.positions.get(symbol, 0)
                        position_str = f", 持仓: {position}" if position > 0 else ""

                        obs_parts.append(
                            f"- {symbol}: ${price:.2f}, 日涨跌: {change:+.2f}%, RSI: {rsi:.1f}, "
                            f"SMA20: {sma_trend}{position_str}"
                        )
                    else:
                        obs_parts.append(f"- {symbol}: ${price:.2f}")
                else:
                    obs_parts.append(f"- {symbol}: ${price:.2f}")

    # 宏观环境
    spy_trend = "乐观"
    if "SPY" in history.columns and len(history["SPY"]) >= 5:
        spy_trend = "乐观" if prices.get("SPY", 0) > history["SPY"].iloc[-5] else "谨慎"

    obs_parts.extend([
        "",
        "### 宏观环境",
        f"- 市场情绪: {spy_trend}",
    ])

    return "\n".join(obs_parts)


# ============================================================
# Main Evaluation
# ============================================================

def run_evaluation(
    checkpoint_name: str,
    test_start: str,
    test_end: str,
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 5,
    model_path: str = "Qwen/Qwen2.5-14B-Instruct",
    log_dir: str = "results",
    load_in_4bit: bool = False,
):
    """运行评估

    Args:
        load_in_4bit: 使用4-bit量化，适合24GB显存GPU (RTX 3090/4090)
    """

    # 设置日志
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用checkpoint目录的最后两级作为名称 (如 marft_v4_aggressive/epoch_2)
    checkpoint_basename = os.path.basename(checkpoint_name.rstrip('/'))
    if os.path.isabs(checkpoint_name):
        parent = os.path.basename(os.path.dirname(checkpoint_name.rstrip('/')))
        checkpoint_basename = f"{parent}_{checkpoint_basename}"
    log_file = os.path.join(log_dir, f"eval_{checkpoint_basename}_{timestamp}.log")
    logger = setup_detailed_logging(log_file)

    logger.info("=" * 80)
    logger.info(" MARFT V4 LoRA 权重评估")
    logger.info("=" * 80)
    logger.info(f" Checkpoint: {checkpoint_name}")
    logger.info(f" 测试时间: {test_start} ~ {test_end}")
    logger.info(f" 初始资金: ${initial_capital:,.2f}")
    logger.info(f" 再平衡频率: 每{rebalance_freq}天")
    logger.info(f" 模型: {model_path}")
    logger.info(f" 日志文件: {log_file}")
    logger.info("=" * 80)

    # Checkpoint路径 - 支持绝对路径或相对于checkpoints_trained的名称
    if os.path.isabs(checkpoint_name) or checkpoint_name.startswith("/"):
        checkpoint_dir = checkpoint_name
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dir = os.path.join(project_root, "checkpoints_trained", checkpoint_name)

    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint目录不存在: {checkpoint_dir}")
        return None

    logger.info(f"Checkpoint目录: {checkpoint_dir}")

    # 检查adapter文件
    adapters_found = []
    for role in ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert",
                 "Crypto_Expert", "Portfolio_Manager", "Hedging_Agent",
                 "Position_Sizing_Agent", "Risk_Controller"]:
        adapter_path = os.path.join(checkpoint_dir, role, role, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            size_mb = os.path.getsize(adapter_path) / 1024 / 1024
            adapters_found.append(role)
            logger.debug(f"  Found {role}: {size_mb:.1f} MB")

    logger.info(f"找到 {len(adapters_found)} 个 LoRA 适配器")

    # 获取市场数据
    logger.info("\n获取市场数据...")
    prices_df = fetch_market_data(test_start, test_end, logger)
    trading_days = prices_df.index.tolist()

    if len(trading_days) < 2:
        logger.error("数据不足，无法评估")
        return None

    logger.info(f"共 {len(trading_days)} 个交易日")

    # 检查GPU
    if not torch.cuda.is_available():
        logger.error("CUDA不可用!")
        return None

    logger.info(f"\nGPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载模型
    logger.info("\n加载LLM Expert Manager...")
    from finsage.rl.shared_expert_manager import SharedModelExpertManager

    manager = SharedModelExpertManager(
        model_path=model_path,
        device="cuda:0",
        bf16=not load_in_4bit,  # 4bit时不用bf16
        load_in_4bit=load_in_4bit,  # 支持24GB GPU
        use_gradient_checkpointing=False,  # 评估时不需要
    )

    logger.info(f"基础模型加载完成, GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 加载训练好的LoRA权重
    logger.info(f"\n加载训练好的LoRA权重: {checkpoint_dir}")
    manager.load_adapters(checkpoint_dir)
    logger.info(f"LoRA权重加载完成, GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 设置为评估模式
    manager.eval()

    # 创建组合
    portfolio = Portfolio(initial_capital, logger)

    # 创建Bug追踪器
    bug_tracker = BugTracker(logger)

    # 期望的完整Expert列表 (9个)
    EXPECTED_EXPERTS = [
        "Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert",
        "Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"
    ]

    # 评估循环
    logger.info("\n" + "=" * 80)
    logger.info(" 开始评估 (增强Bug检测)")
    logger.info("=" * 80)
    logger.info(f" 期望的Experts: {EXPECTED_EXPERTS}")

    decision_log = []
    # 初始化所有9个Expert的统计 (不仅仅是找到的adapters)
    expert_stats = {role: {"calls": 0, "actions": {}, "avg_confidence": []} for role in EXPECTED_EXPERTS}

    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d")
        prices = prices_df.loc[date]

        # 记录组合价值
        price_dict = {s: prices[s] for s in ALL_SYMBOLS if s in prices.index}
        portfolio.record_value(price_dict, date_str)

        # 是否需要rebalance
        if i % rebalance_freq != 0 and i > 0:
            continue

        # 获取历史数据
        lookback = min(i + 1, 30)
        history = prices_df.iloc[max(0, i-lookback):i+1]

        # 创建市场观察
        obs = create_market_observation(date_str, prices, history, portfolio, logger)

        # 检查输入数据问题
        bug_tracker.check_data_issues(obs, date_str, history=history)

        current_value = portfolio.get_total_value(price_dict)
        logger.info(f"\n[{date_str}] 组合价值: ${current_value:,.2f} (收益: {(current_value/initial_capital-1)*100:+.2f}%)")

        # 获取Expert决策
        start_time = time.time()
        try:
            all_actions = manager.run_expert_chain(obs)
            chain_success = True
        except Exception as e:
            bug_tracker.log_issue("EXPERT_CHAIN_ERROR",
                                 f"Expert Chain 执行失败: {str(e)}",
                                 severity="ERROR", date=date_str,
                                 extra={"traceback": traceback.format_exc()})
            all_actions = {}
            chain_success = False

        inference_time = time.time() - start_time

        logger.debug(f"推理时间: {inference_time:.2f}s")

        # 验证Expert Chain完整性
        if chain_success:
            bug_tracker.validate_expert_chain(all_actions, EXPECTED_EXPERTS)
            logger.info(f"  Expert Chain 返回 {len(all_actions)} 个Experts: {list(all_actions.keys())}")

        # 记录Expert决策
        daily_decisions = {}

        # ============================================================
        # 第一步: 收集所有Expert决策和Meta-Level Agent建议
        # ============================================================
        asset_expert_actions = {}  # 5个资产类别Expert的动作
        meta_agent_outputs = {}    # 4个Meta-Level Agent的输出

        for role, action_dict in all_actions.items():
            action = action_dict.get("action", "HOLD")
            confidence = action_dict.get("confidence", 0.5)
            reasoning = action_dict.get("reasoning", "")[:100]

            daily_decisions[role] = action

            # Bug追踪: 记录每个Expert调用
            bug_tracker.track_expert_call(role, success=True, output=action_dict)

            # 更新统计 (所有9个Expert)
            if role in expert_stats:
                expert_stats[role]["calls"] += 1
                expert_stats[role]["actions"][action] = expert_stats[role]["actions"].get(action, 0) + 1
                expert_stats[role]["avg_confidence"].append(confidence)

            logger.debug(f"  {role}: {action} (conf={confidence:.2f}) - {reasoning}...")

            # 区分Asset Experts和Meta-Level Agents
            if role in ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"]:
                asset_expert_actions[role] = action_dict
            else:
                meta_agent_outputs[role] = action_dict
                # 验证Meta-Level Agent输出格式
                bug_tracker.validate_meta_agent_output(role, action_dict, date_str)

        # ============================================================
        # 第二步: 处理Meta-Level Agent的输出
        # ============================================================

        # Risk_Controller: 检查是否有veto (否决交易)
        risk_veto = False
        if "Risk_Controller" in meta_agent_outputs:
            risk_output = meta_agent_outputs["Risk_Controller"]
            risk_veto = risk_output.get("veto", False)
            if risk_veto:
                logger.warning(f"  [RISK] Risk_Controller 否决本轮交易!")

        # Position_Sizing_Agent: 获取仓位调整建议
        position_scale = 0.5  # 默认使用 50% 资金
        if "Position_Sizing_Agent" in meta_agent_outputs:
            pos_output = meta_agent_outputs["Position_Sizing_Agent"]
            # 根据risk_budget调整仓位
            risk_budget = pos_output.get("risk_budget", 0.5)
            if isinstance(risk_budget, (int, float)) and 0 < risk_budget <= 1:
                # 保护: 如果 risk_budget 太小 (< 0.1)，使用默认值 0.5
                # 避免因为模型输出过于保守导致仓位过小
                if risk_budget < 0.1:
                    logger.warning(f"  [POSITION] risk_budget={risk_budget:.2f} 太小，使用默认值 0.5")
                    position_scale = 0.5
                else:
                    position_scale = risk_budget
                logger.info(f"  [POSITION] 仓位缩放系数: {position_scale:.2f}")

        # Portfolio_Manager: 记录整体配置建议
        if "Portfolio_Manager" in meta_agent_outputs:
            pm_output = meta_agent_outputs["Portfolio_Manager"]
            target_alloc = pm_output.get("target_allocation", {})
            if target_alloc:
                logger.info(f"  [PM] 目标配置: {target_alloc}")

        # Hedging_Agent: 记录对冲建议
        if "Hedging_Agent" in meta_agent_outputs:
            hedge_output = meta_agent_outputs["Hedging_Agent"]
            hedge_strategy = hedge_output.get("hedge_strategy", "none")
            logger.info(f"  [HEDGE] 对冲策略: {hedge_strategy}")

        # ============================================================
        # 第三步: 执行交易 (如果未被Risk_Controller否决)
        # ============================================================
        if risk_veto:
            logger.info("  [SKIP] 本轮交易被风控否决，跳过执行")
        else:
            for role, action_dict in asset_expert_actions.items():
                action = action_dict.get("action", "HOLD")

                # 根据Expert类型决定交易哪些资产
                asset_class = role.split("_")[0].lower()
                if asset_class == "stock":
                    symbols = ASSETS.get("stocks", [])
                    class_key = "stocks"
                elif asset_class == "bond":
                    symbols = ASSETS.get("bonds", [])
                    class_key = "bonds"
                elif asset_class == "commodity":
                    symbols = ASSETS.get("commodities", [])
                    class_key = "commodities"
                elif asset_class == "reits":
                    symbols = ASSETS.get("reits", [])
                    class_key = "reits"
                elif asset_class == "crypto":
                    symbols = ASSETS.get("crypto", [])
                    class_key = "crypto"
                else:
                    continue

                # 获取资产类别权重 (控制该类资产的仓位比例)
                class_weight = ASSET_CLASS_WEIGHTS.get(class_key, 1.0)

                # 对每个资产执行相同动作 (应用仓位缩放 + 资产类别权重)
                for symbol in symbols:
                    if symbol in price_dict:
                        # 根据Position_Sizing和资产类别权重调整交易规模
                        scaled_value = current_value * position_scale * class_weight
                        result = portfolio.execute_trade(
                            symbol=symbol,
                            action=action,
                            price=price_dict[symbol],
                            total_value=scaled_value,
                            date=date_str,
                            expert_role=role
                        )
                        if result["executed"]:
                            logger.info(f"  {role} -> {symbol}: {action} (class_weight={class_weight:.1f})")

        decision_log.append({
            "date": date_str,
            "portfolio_value": current_value,
            "decisions": daily_decisions,
            "inference_time": inference_time
        })

    # 计算最终指标
    metrics = portfolio.get_metrics()

    # Expert统计摘要
    logger.info("\n" + "=" * 80)
    logger.info(" Expert 决策统计")
    logger.info("=" * 80)
    for role, stats in expert_stats.items():
        if stats["calls"] > 0:
            avg_conf = np.mean(stats["avg_confidence"]) if stats["avg_confidence"] else 0
            logger.info(f"\n{role}:")
            logger.info(f"  调用次数: {stats['calls']}")
            logger.info(f"  平均置信度: {avg_conf:.3f}")
            logger.info(f"  动作分布: {stats['actions']}")

    # 输出最终结果
    logger.info("\n" + "=" * 80)
    logger.info(" 评估结果")
    logger.info("=" * 80)
    logger.info(f" Checkpoint: {checkpoint_name}")
    logger.info(f" 测试期间: {test_start} ~ {test_end}")
    logger.info(f" 初始资金: ${initial_capital:,.2f}")
    logger.info(f" 最终价值: ${metrics.get('final_value', 0):,.2f}")
    logger.info(f" 总收益率: {metrics.get('total_return', 0)*100:.2f}%")
    logger.info(f" 年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
    logger.info(f" 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f" Sortino比率: {metrics.get('sortino_ratio', 0):.2f}")
    logger.info(f" Calmar比率: {metrics.get('calmar_ratio', 0):.2f}")
    logger.info(f" 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f" 波动率: {metrics.get('volatility', 0)*100:.2f}%")
    logger.info(f" 交易次数: {metrics.get('total_trades', 0)}")
    logger.info(f" 胜率: {metrics.get('win_rate', 0)*100:.1f}%")
    logger.info("=" * 80)

    # 打印Bug检测报告
    bug_summary = bug_tracker.print_summary()

    # 保存结果
    result = {
        "checkpoint": checkpoint_name,
        "test_start": test_start,
        "test_end": test_end,
        "initial_capital": initial_capital,
        "model_path": model_path,
        "metrics": metrics,
        "portfolio_values": portfolio.value_history,
        "dates": portfolio.date_history,
        "decisions": decision_log,
        "expert_stats": {k: {**v, "avg_confidence": float(np.mean(v["avg_confidence"])) if v["avg_confidence"] else 0}
                        for k, v in expert_stats.items()},
        "trade_log": portfolio.trade_log,
        "bug_detection": {
            "total_issues": bug_summary["total_issues"],
            "issues_by_category": bug_summary["issues_by_category"],
            "expert_call_counts": bug_summary["expert_call_counts"],
            "meta_agent_analysis": bug_summary["meta_agent_analysis"],
            "critical_bugs_found": any(
                bug_summary["expert_call_counts"].get(agent, 0) == 0
                for agent in ["Portfolio_Manager", "Hedging_Agent", "Position_Sizing_Agent", "Risk_Controller"]
            )
        }
    }

    result_file = os.path.join(log_dir, f"eval_{checkpoint_basename}_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\n结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估训练好的MARFT V4 LoRA权重")
    parser.add_argument("--checkpoint", default="aggressive_final",
                       help="Checkpoint名称 (aggressive_final 或 balanced_final)")
    parser.add_argument("--test-start", default="2024-07-01", help="测试开始日期")
    parser.add_argument("--test-end", default="2024-12-31", help="测试结束日期")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--freq", type=int, default=1, help="再平衡频率(天), 1=每天决策")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="基础模型")
    parser.add_argument("--log-dir", default="results", help="日志输出目录")
    parser.add_argument("--4bit", dest="load_4bit", action="store_true",
                       help="使用4-bit量化 (适合24GB显存GPU)")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_name=args.checkpoint,
        test_start=args.test_start,
        test_end=args.test_end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        model_path=args.model,
        log_dir=args.log_dir,
        load_in_4bit=args.load_4bit,
    )


if __name__ == "__main__":
    main()
