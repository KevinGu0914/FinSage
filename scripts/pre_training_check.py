#!/usr/bin/env python3
"""
FinSage 训练前完整性检查脚本
Pre-Training Integrity Check Script

在服务器上运行训练前，执行此脚本确保所有组件正常工作。
运行方式: python scripts/pre_training_check.py

检查项目:
1. 核心模块导入
2. 配置系统
3. 数据加载
4. 环境初始化
5. 5个专家Agent
6. 奖励函数
7. 11个对冲工具
8. 风控系统
9. RL组件 (MARFT)
10. 端到端模拟
"""

import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd


class CheckResult:
    """检查结果"""
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details


class PreTrainingChecker:
    """训练前检查器"""

    def __init__(self):
        self.results: List[CheckResult] = []
        self.total_checks = 0
        self.passed_checks = 0

    def log(self, message: str, level: str = "INFO"):
        """日志输出"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "  ", "PASS": "✓ ", "FAIL": "✗ ", "WARN": "⚠ "}
        print(f"[{timestamp}] {prefix.get(level, '  ')}{message}")

    def check(self, name: str, check_func, *args, **kwargs) -> bool:
        """执行检查"""
        self.total_checks += 1
        try:
            result = check_func(*args, **kwargs)
            if result is True or result is None:
                self.passed_checks += 1
                self.results.append(CheckResult(name, True))
                self.log(f"{name}", "PASS")
                return True
            else:
                self.results.append(CheckResult(name, False, str(result)))
                self.log(f"{name}: {result}", "FAIL")
                return False
        except Exception as e:
            self.results.append(CheckResult(name, False, str(e), traceback.format_exc()))
            self.log(f"{name}: {e}", "FAIL")
            return False

    # ========================================
    # 1. 核心模块导入检查
    # ========================================

    def check_core_imports(self):
        """检查核心模块导入"""
        print("\n" + "=" * 60)
        print("1. 核心模块导入检查")
        print("=" * 60)

        imports = [
            ("finsage.config", "FinSageConfig"),
            ("finsage.core.orchestrator", "FinSageOrchestrator"),
            ("finsage.environment.portfolio_state", "PortfolioState"),
            ("finsage.environment.multi_asset_env", "MultiAssetTradingEnv"),
            ("finsage.data.data_loader", "DataLoader"),
            ("finsage.llm.llm_provider", "LLMProvider"),
        ]

        for module, cls in imports:
            def check_import(m=module, c=cls):
                mod = __import__(m, fromlist=[c])
                assert hasattr(mod, c), f"Missing {c}"
                return True
            self.check(f"Import {module}.{cls}", check_import)

    # ========================================
    # 2. 配置系统检查
    # ========================================

    def check_config_system(self):
        """检查配置系统"""
        print("\n" + "=" * 60)
        print("2. 配置系统检查")
        print("=" * 60)

        def check_finsage_config():
            from finsage.config import FinSageConfig
            config = FinSageConfig()
            assert config.llm is not None
            assert config.trading is not None
            assert config.risk is not None
            assert config.assets is not None
            return True

        def check_asset_config():
            from finsage.config import AssetConfig
            config = AssetConfig()
            universe = config.default_universe
            assert "stocks" in universe
            assert "bonds" in universe
            assert "commodities" in universe
            assert "reits" in universe
            assert "crypto" in universe
            return True

        def check_risk_config():
            from finsage.config import RiskConfig
            config = RiskConfig()
            assert config.max_single_asset > 0
            assert config.max_drawdown_trigger > 0
            assert config.target_volatility > 0
            return True

        self.check("FinSageConfig初始化", check_finsage_config)
        self.check("AssetConfig资产池", check_asset_config)
        self.check("RiskConfig风险约束", check_risk_config)

    # ========================================
    # 3. 数据模块检查
    # ========================================

    def check_data_modules(self):
        """检查数据模块"""
        print("\n" + "=" * 60)
        print("3. 数据模块检查")
        print("=" * 60)

        def check_data_loader():
            from finsage.data.data_loader import DataLoader
            loader = DataLoader(data_source="yfinance")
            assert loader.data_source == "yfinance"
            return True

        def check_fmp_client():
            from finsage.data.fmp_client import FMPClient
            # 只检查导入，不测试API调用
            assert FMPClient is not None
            return True

        def check_macro_loader():
            from finsage.data.macro_loader import MacroDataLoader
            assert MacroDataLoader is not None
            return True

        def check_intraday_loader():
            from finsage.data.intraday_loader import IntradayDataLoader
            assert IntradayDataLoader is not None
            return True

        self.check("DataLoader初始化", check_data_loader)
        self.check("FMPClient导入", check_fmp_client)
        self.check("MacroDataLoader导入", check_macro_loader)
        self.check("IntradayDataLoader导入", check_intraday_loader)

    # ========================================
    # 4. 环境模块检查
    # ========================================

    def check_environment(self):
        """检查环境模块"""
        print("\n" + "=" * 60)
        print("4. 环境模块检查")
        print("=" * 60)

        def check_portfolio_state():
            from finsage.environment.portfolio_state import PortfolioState, Position

            # 创建组合
            portfolio = PortfolioState(initial_capital=1_000_000, cash=1_000_000)
            assert portfolio.portfolio_value == 1_000_000

            # 添加持仓
            portfolio.positions["SPY"] = Position("SPY", 100, 450, 460, "stocks")
            assert portfolio.long_market_value == 46000

            # 执行交易
            trade = portfolio.execute_trade("QQQ", 50, 380, "stocks")
            assert trade["action"] == "BUY"

            return True

        def check_multi_asset_env():
            from finsage.environment.multi_asset_env import MultiAssetTradingEnv, EnvConfig

            config = EnvConfig(initial_capital=500_000)
            env = MultiAssetTradingEnv(config=config)

            portfolio = env.reset()
            assert portfolio.initial_capital == 500_000

            obs = env.get_observation()
            assert "portfolio" in obs

            return True

        self.check("PortfolioState功能", check_portfolio_state)
        self.check("MultiAssetTradingEnv功能", check_multi_asset_env)

    # ========================================
    # 5. 专家Agent检查
    # ========================================

    def check_expert_agents(self):
        """检查5个专家Agent"""
        print("\n" + "=" * 60)
        print("5. 专家Agent检查 (5个)")
        print("=" * 60)

        experts = [
            ("finsage.agents.experts.stock_expert", "StockExpert", "stocks"),
            ("finsage.agents.experts.bond_expert", "BondExpert", "bonds"),
            ("finsage.agents.experts.commodity_expert", "CommodityExpert", "commodities"),
            ("finsage.agents.experts.reits_expert", "REITsExpert", "reits"),
            ("finsage.agents.experts.crypto_expert", "CryptoExpert", "crypto"),
        ]

        for module, cls, asset_class in experts:
            def check_expert(m=module, c=cls, ac=asset_class):
                mod = __import__(m, fromlist=[c])
                ExpertClass = getattr(mod, c)
                assert hasattr(ExpertClass, '__init__')
                assert hasattr(ExpertClass, 'analyze')
                return True
            self.check(f"{cls} ({asset_class})", check_expert)

        # 检查BaseExpert
        def check_base_expert():
            from finsage.agents.base_expert import BaseExpert, ExpertRecommendation, ExpertReport

            rec = ExpertRecommendation(
                asset_class="stocks",
                symbol="SPY",
                action="BUY",
                confidence=0.8,
                target_weight=0.1,
                reasoning="Test",
                market_view="bullish",
                risk_assessment="low"
            )
            assert rec.confidence == 0.8
            return True

        self.check("BaseExpert数据类", check_base_expert)

    # ========================================
    # 6. 奖励函数检查
    # ========================================

    def check_reward_functions(self):
        """检查奖励函数"""
        print("\n" + "=" * 60)
        print("6. 奖励函数检查")
        print("=" * 60)

        reward_classes = [
            ("PortfolioManagerReward", {}),
            ("PositionSizingReward", {}),
            ("HedgingReward", {}),
            ("ExpertReward", {"expert_type": "stock"}),
            ("CoordinationReward", {}),
        ]

        for cls_name, init_kwargs in reward_classes:
            def check_reward(name=cls_name, kwargs=init_kwargs):
                from finsage.rl import reward_functions
                RewardClass = getattr(reward_functions, name)
                reward = RewardClass(**kwargs)
                assert hasattr(reward, 'compute')
                return True
            self.check(f"{cls_name}", check_reward)

        # 测试奖励计算
        def check_portfolio_reward_compute():
            from finsage.rl.reward_functions import PortfolioManagerReward

            reward = PortfolioManagerReward()
            result = reward.compute(
                portfolio_return=0.02,
                portfolio_volatility=0.15,
                expert_recommendations={},
                actual_allocation={"SPY": 0.5, "TLT": 0.5},
                asset_returns={"SPY": 0.03, "TLT": 0.01},
                market_regime="bull"
            )

            assert result.total is not None
            assert "return_reward" in result.components
            return True

        self.check("PortfolioManagerReward计算", check_portfolio_reward_compute)

    # ========================================
    # 7. 对冲工具检查
    # ========================================

    def check_hedging_tools(self):
        """检查11个对冲工具"""
        print("\n" + "=" * 60)
        print("7. 对冲工具检查 (11个)")
        print("=" * 60)

        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        tools = [
            "minimum_variance",
            "risk_parity",
            "black_litterman",
            "mean_variance",
            "cvar_optimization",
            "dcc_garch",
            "hrp",
            "robust_optimization",
            "factor_hedging",
            "regime_switching",
            "copula_hedging",
        ]

        from finsage.hedging.toolkit import HedgingToolkit
        toolkit = HedgingToolkit()

        for tool_name in tools:
            def check_tool(name=tool_name, ret=returns):
                tool = toolkit.get(name)
                if tool is None:
                    return f"Tool {name} not registered"

                weights = tool.compute_weights(ret)
                total = sum(weights.values())
                if abs(total - 1.0) > 0.05:
                    return f"Weights sum to {total}, expected 1.0"

                return True

            self.check(f"{tool_name}", check_tool)

    # ========================================
    # 8. 风控系统检查
    # ========================================

    def check_risk_system(self):
        """检查风控系统"""
        print("\n" + "=" * 60)
        print("8. 风控系统检查")
        print("=" * 60)

        def check_risk_controller():
            from finsage.agents.risk_controller import RiskController, RiskAssessment

            assessment = RiskAssessment(
                timestamp="2024-01-15T10:00:00",
                portfolio_var_95=0.02,
                portfolio_cvar_99=0.03,
                current_drawdown=0.05,
                max_drawdown=0.15,
                volatility=0.12,
                sharpe_ratio=1.2,
                concentration_risk=0.25,
                violations=[],
                warnings=[],
                veto=False,
                recommendations=[],
                intraday_alerts=[],
                emergency_rebalance=False,
                defensive_allocation=None
            )

            assert assessment.volatility == 0.12
            return True

        def check_intraday_monitor():
            from finsage.risk.intraday_monitor import IntradayRiskMonitor, AlertLevel, AlertType

            monitor = IntradayRiskMonitor()
            assert hasattr(monitor, 'monitor')
            assert hasattr(monitor, 'thresholds')
            assert len(monitor.thresholds) > 0

            # 检查警报级别枚举
            assert AlertLevel.NORMAL is not None
            assert AlertLevel.WARNING is not None
            assert AlertLevel.CRITICAL is not None

            # 检查警报类型
            assert AlertType.PRICE_SPIKE is not None
            assert AlertType.FLASH_CRASH is not None

            return True

        self.check("RiskController数据类", check_risk_controller)
        self.check("IntradayRiskMonitor", check_intraday_monitor)

    # ========================================
    # 9. RL组件检查 (MARFT)
    # ========================================

    def check_rl_components(self):
        """检查RL组件"""
        print("\n" + "=" * 60)
        print("9. RL组件检查 (MARFT)")
        print("=" * 60)

        def check_rl_config():
            from finsage.rl.config import MARFTFinSageConfig, TrainingConfig, PPOConfig, LoRAConfig
            config = MARFTFinSageConfig()
            # MARFTFinSageConfig uses 'expert' and 'training' attributes
            assert config.expert is not None
            assert config.training is not None
            assert config.lora is not None
            assert config.ppo is not None
            return True

        def check_data_bridge():
            from finsage.rl.data_bridge import MARFTEnvWrapper, ObservationFormatter, ActionConverter
            assert MARFTEnvWrapper is not None
            assert ObservationFormatter is not None
            assert ActionConverter is not None
            return True

        def check_critic():
            from finsage.rl.critic import ActionCritic, FinancialCritic, PortfolioValueCritic
            assert ActionCritic is not None
            assert FinancialCritic is not None
            assert PortfolioValueCritic is not None
            return True

        def check_shared_expert_manager():
            from finsage.rl.shared_expert_manager import (
                EXPERT_CONFIGS, LORA_CONFIG, PromptCache
            )
            assert len(EXPERT_CONFIGS) == 5
            assert "r" in LORA_CONFIG
            assert "lora_alpha" in LORA_CONFIG

            # 测试PromptCache
            import torch
            cache = PromptCache(max_size=10)
            cache.put("test", torch.tensor([1,2,3]), torch.tensor([1,1,1]))
            result = cache.get("test")
            assert result is not None

            return True

        def check_marft_integration():
            from finsage.rl.marft_integration import MARFTFinSageIntegration, FinSageFlexMGEnv, FinSageAPPOTrainer
            assert MARFTFinSageIntegration is not None
            assert FinSageFlexMGEnv is not None
            assert FinSageAPPOTrainer is not None
            return True

        self.check("MARFTFinSageConfig", check_rl_config)
        self.check("DataBridge组件", check_data_bridge)
        self.check("Critic Networks", check_critic)
        self.check("SharedExpertManager组件", check_shared_expert_manager)
        self.check("MARFTFinSageIntegration", check_marft_integration)

    # ========================================
    # 10. 策略模块检查
    # ========================================

    def check_strategies(self):
        """检查策略模块"""
        print("\n" + "=" * 60)
        print("10. 策略模块检查")
        print("=" * 60)

        def check_strategic_allocation():
            from finsage.strategies.strategic_allocation import StrategicAllocationStrategy
            strategy = StrategicAllocationStrategy()
            assert strategy.name is not None
            assert hasattr(strategy, 'compute_allocation')
            return True

        def check_tactical_allocation():
            from finsage.strategies.tactical_allocation import TacticalAllocationStrategy
            strategy = TacticalAllocationStrategy()
            assert strategy.name is not None
            assert hasattr(strategy, 'compute_allocation')
            return True

        def check_core_satellite():
            from finsage.strategies.core_satellite import CoreSatelliteStrategy
            strategy = CoreSatelliteStrategy()
            assert strategy.name is not None
            assert hasattr(strategy, 'compute_allocation')
            return True

        def check_dynamic_rebalancing():
            from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy
            strategy = DynamicRebalancingStrategy()
            assert strategy.name is not None
            assert hasattr(strategy, 'compute_allocation')
            return True

        def check_strategy_toolkit():
            from finsage.strategies.strategy_toolkit import StrategyToolkit
            toolkit = StrategyToolkit()
            assert hasattr(toolkit, 'get')
            return True

        self.check("StrategicAllocationStrategy", check_strategic_allocation)
        self.check("TacticalAllocationStrategy", check_tactical_allocation)
        self.check("CoreSatelliteStrategy", check_core_satellite)
        self.check("DynamicRebalancingStrategy", check_dynamic_rebalancing)
        self.check("StrategyToolkit", check_strategy_toolkit)

    # ========================================
    # 11. 因子评分检查
    # ========================================

    def check_factor_scorers(self):
        """检查因子评分器"""
        print("\n" + "=" * 60)
        print("11. 因子评分检查")
        print("=" * 60)

        scorers = [
            ("finsage.factors.stock_factors", "StockFactorScorer"),
            ("finsage.factors.bond_factors", "BondFactorScorer"),
            ("finsage.factors.commodity_factors", "CommodityFactorScorer"),
            ("finsage.factors.reits_factors", "REITsFactorScorer"),
            ("finsage.factors.crypto_factors", "CryptoFactorScorer"),
        ]

        for module, cls in scorers:
            def check_scorer(m=module, c=cls):
                mod = __import__(m, fromlist=[c])
                ScorerClass = getattr(mod, c)
                scorer = ScorerClass()
                assert hasattr(scorer, 'score')
                return True

            self.check(f"{cls}", check_scorer)

    # ========================================
    # 12. 端到端模拟检查
    # ========================================

    def check_end_to_end(self):
        """端到端模拟检查"""
        print("\n" + "=" * 60)
        print("12. 端到端模拟检查")
        print("=" * 60)

        def check_simple_backtest_flow():
            """模拟简单回测流程"""
            from finsage.environment.multi_asset_env import MultiAssetTradingEnv, EnvConfig
            from finsage.environment.portfolio_state import Position
            from finsage.hedging.toolkit import HedgingToolkit

            # 1. 创建环境
            config = EnvConfig(initial_capital=1_000_000)
            env = MultiAssetTradingEnv(config=config)
            portfolio = env.reset()

            # 2. 模拟市场数据
            np.random.seed(42)
            returns = pd.DataFrame({
                "SPY": np.random.normal(0.001, 0.02, 50),
                "TLT": np.random.normal(0.0005, 0.01, 50),
            })

            # 3. 使用对冲工具计算权重
            toolkit = HedgingToolkit()
            weights = toolkit.get("risk_parity").compute_weights(returns)

            # 4. 执行交易
            for symbol, weight in weights.items():
                target_value = portfolio.portfolio_value * weight
                price = 450 if symbol == "SPY" else 100
                shares = int(target_value / price)
                if shares > 0:
                    portfolio.execute_trade(symbol, shares, price, "stocks" if symbol == "SPY" else "bonds")

            # 5. 验证
            assert len(portfolio.positions) > 0
            assert portfolio.portfolio_value > 0

            return True

        def check_reward_calculation_flow():
            """模拟奖励计算流程"""
            from finsage.rl.reward_functions import (
                PortfolioManagerReward, ExpertReward, CombinedRewardCalculator
            )

            # 创建奖励计算器
            pm_reward = PortfolioManagerReward()
            expert_reward = ExpertReward(expert_type="stock")

            # 计算组合奖励
            pm_result = pm_reward.compute(
                portfolio_return=0.015,
                portfolio_volatility=0.12,
                expert_recommendations={"stock": {"SPY": 0.4}},
                actual_allocation={"SPY": 0.4, "TLT": 0.3, "cash": 0.3},
                asset_returns={"SPY": 0.02, "TLT": 0.005},
                market_regime="bull"
            )

            assert pm_result.total is not None
            assert -10 <= pm_result.total <= 10  # 奖励在合理范围内

            return True

        self.check("回测流程模拟", check_simple_backtest_flow)
        self.check("奖励计算流程", check_reward_calculation_flow)

    # ========================================
    # 主运行函数
    # ========================================

    def run_all_checks(self):
        """运行所有检查"""
        print("\n" + "=" * 60)
        print(" FinSage 训练前完整性检查")
        print(" Pre-Training Integrity Check")
        print("=" * 60)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"项目路径: {PROJECT_ROOT}")

        # 运行所有检查
        self.check_core_imports()
        self.check_config_system()
        self.check_data_modules()
        self.check_environment()
        self.check_expert_agents()
        self.check_reward_functions()
        self.check_hedging_tools()
        self.check_risk_system()
        self.check_rl_components()
        self.check_strategies()
        self.check_factor_scorers()
        self.check_end_to_end()

        # 输出总结
        print("\n" + "=" * 60)
        print(" 检查结果总结")
        print("=" * 60)

        pass_rate = self.passed_checks / self.total_checks * 100 if self.total_checks > 0 else 0

        print(f"\n总检查项: {self.total_checks}")
        print(f"通过: {self.passed_checks}")
        print(f"失败: {self.total_checks - self.passed_checks}")
        print(f"通过率: {pass_rate:.1f}%")

        if self.passed_checks == self.total_checks:
            print("\n" + "=" * 60)
            print(" ✓ 所有检查通过！可以开始训练")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print(" ✗ 存在失败项，请修复后再训练")
            print("=" * 60)

            # 列出失败项
            print("\n失败详情:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.message}")
                    if result.details:
                        print(f"    {result.details[:200]}...")

            return 1


def main():
    """主函数"""
    checker = PreTrainingChecker()
    exit_code = checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
