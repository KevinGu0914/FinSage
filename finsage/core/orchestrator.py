"""
FinSage Orchestrator
核心协调器 - 协调所有Agent的工作流程

支持断点保存和恢复功能
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import os
import pickle
from dataclasses import dataclass, asdict

from finsage.config import FinSageConfig
from finsage.llm.llm_provider import LLMProvider
from finsage.data.data_loader import DataLoader
from finsage.data.market_data import MarketDataProvider
from finsage.data.dynamic_screener import DynamicStockScreener
from finsage.hedging.toolkit import HedgingToolkit
from finsage.environment.multi_asset_env import MultiAssetTradingEnv, EnvConfig
from finsage.agents.portfolio_manager import PortfolioManager
from finsage.agents.risk_controller import RiskController
from finsage.agents.experts.stock_expert import StockExpert
from finsage.agents.experts.bond_expert import BondExpert
from finsage.agents.experts.commodity_expert import CommodityExpert
from finsage.agents.experts.reits_expert import REITsExpert
from finsage.agents.experts.crypto_expert import CryptoExpert

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """断点状态数据"""
    start_date: str
    end_date: str
    rebalance_frequency: str
    current_date_index: int
    total_dates: int
    results: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    timestamp: str
    # Risk Controller 状态
    risk_controller_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class FinSageOrchestrator:
    """
    FinSage核心协调器

    协调多个Agent的工作流程:
    1. 数据收集和预处理
    2. 调用5位专家分析
    3. Portfolio Manager综合决策
    4. Risk Controller风险评估
    5. 执行交易
    """

    def __init__(self, config: Optional[FinSageConfig] = None, checkpoint_dir: str = "./checkpoints/finsage"):
        """
        初始化协调器

        Args:
            config: 配置对象
            checkpoint_dir: 断点保存目录
        """
        self.config = config or FinSageConfig()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 初始化组件
        self._init_components()

        logger.info("FinSageOrchestrator initialized")

    def _init_components(self) -> None:
        """初始化所有组件"""
        # LLM Provider
        self.llm = LLMProvider(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
        )

        # 数据加载
        self.data_loader = DataLoader(
            data_source=self.config.data.data_source,
            cache_dir=self.config.data.cache_dir,
            api_key=self.config.data.fmp_api_key,
        )
        self.market_data = MarketDataProvider(self.data_loader)

        # 对冲工具箱
        self.hedging_toolkit = HedgingToolkit()

        # 动态股票筛选器
        self.dynamic_screener = DynamicStockScreener(
            api_key=self.config.data.fmp_api_key,
            cache_hours=168,  # 1周缓存
        )
        self._last_universe_refresh = None  # 上次刷新日期
        # 资产池刷新频率 (可通过config配置)
        # 1=每日, 7=每周, 30=每月
        # 从 data 配置获取，默认为1天 (每日刷新)
        self._refresh_interval_days = getattr(self.config.data, "universe_refresh_days", 1)

        # 交易环境
        env_config = EnvConfig(
            initial_capital=self.config.trading.initial_capital,
            transaction_cost=self.config.trading.transaction_cost,
            slippage=self.config.trading.slippage,
            max_single_asset=self.config.risk.max_single_asset,
            max_asset_class=self.config.risk.max_asset_class,
        )
        self.env = MultiAssetTradingEnv(
            config=env_config,
            asset_universe=self.config.assets.default_universe,
        )

        # 专家Agents
        self.experts = {
            "stocks": StockExpert(self.llm),
            "bonds": BondExpert(self.llm),
            "commodities": CommodityExpert(self.llm),
            "reits": REITsExpert(self.llm),
            "crypto": CryptoExpert(self.llm),
        }

        # Portfolio Manager
        self.portfolio_manager = PortfolioManager(
            llm_provider=self.llm,
            hedging_toolkit=self.hedging_toolkit,
            config={
                "allocation_bounds": self.config.assets.allocation_bounds,
                "rebalance_threshold": self.config.trading.rebalance_threshold,
            }
        )

        # Risk Controller
        self.risk_controller = RiskController(
            hard_limits={
                "max_single_asset": self.config.risk.max_single_asset,
                "max_asset_class": self.config.risk.max_asset_class,
                "max_drawdown_trigger": self.config.risk.max_drawdown_trigger,
                "max_portfolio_var_95": self.config.risk.max_portfolio_var_95,
            },
            soft_limits={
                "target_volatility": self.config.risk.target_volatility,
                "max_correlation_cluster": self.config.risk.max_correlation_cluster,
                "min_diversification_ratio": self.config.risk.min_diversification_ratio,
            }
        )

    def _refresh_dynamic_universe(self, current_date: str, force: bool = False) -> Dict[str, List[str]]:
        """
        刷新动态资产候选池

        根据市场状态从全市场筛选最优标的
        每周自动刷新一次，或强制刷新

        Args:
            current_date: 当前日期
            force: 是否强制刷新

        Returns:
            Dict[asset_class, List[symbols]]
        """
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")

        # 检查是否需要刷新
        need_refresh = force or self._last_universe_refresh is None
        if not need_refresh and self._last_universe_refresh:
            days_since_refresh = (current_dt - self._last_universe_refresh).days
            need_refresh = days_since_refresh >= self._refresh_interval_days

        if not need_refresh:
            return {}

        logger.info(f"Refreshing dynamic universe on {current_date}...")

        # 获取各资产类别的动态候选池
        dynamic_universe = {}
        for asset_class in ["stocks", "bonds", "commodities", "reits", "crypto"]:
            try:
                symbols = self.dynamic_screener.get_dynamic_universe(
                    asset_class=asset_class,
                    date=current_date,
                )
                dynamic_universe[asset_class] = symbols

                # 更新对应专家的候选池
                if asset_class in self.experts and symbols:
                    self.experts[asset_class].update_symbols(symbols)
                    logger.info(f"Updated {asset_class} expert with {len(symbols)} symbols: {symbols[:5]}...")

            except Exception as e:
                logger.warning(f"Failed to refresh {asset_class} universe: {e}")

        self._last_universe_refresh = current_dt
        logger.info(f"Dynamic universe refresh completed: {len(dynamic_universe)} asset classes updated")

        return dynamic_universe

    def run(
        self,
        start_date: str,
        end_date: str,
        rebalance_frequency: str = "weekly",
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        运行回测/模拟

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            rebalance_frequency: 再平衡频率
            resume: 是否从断点恢复

        Returns:
            运行结果
        """
        logger.info(f"Starting FinSage run: {start_date} to {end_date}")

        # 生成再平衡日期
        rebalance_dates = self._generate_rebalance_dates(
            start_date, end_date, rebalance_frequency
        )

        # 尝试恢复断点
        start_index = 0
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_index = checkpoint.current_date_index
                results = checkpoint.results
                # 恢复环境状态
                self.env.restore_state(checkpoint.portfolio_state)
                # 恢复 Risk Controller 状态
                if checkpoint.risk_controller_state:
                    self.risk_controller.peak_value = checkpoint.risk_controller_state.get(
                        "peak_value", 1.0
                    )
                    self.risk_controller.max_drawdown_history = checkpoint.risk_controller_state.get(
                        "max_drawdown_history", 0.0
                    )
                    logger.info(f"Restored Risk Controller: peak={self.risk_controller.peak_value}, max_dd={self.risk_controller.max_drawdown_history}")
                logger.info(f"Resumed from checkpoint: {start_index}/{len(rebalance_dates)} dates completed")
            else:
                # 没有断点，从头开始
                self.env.reset(start_date=start_date)
                results = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "rebalance_frequency": rebalance_frequency,
                    "decisions": [],
                    "risk_assessments": [],
                }
        else:
            # 重置环境
            self.env.reset(start_date=start_date)
            results = {
                "start_date": start_date,
                "end_date": end_date,
                "rebalance_frequency": rebalance_frequency,
                "decisions": [],
                "risk_assessments": [],
            }

        # 初始获取所有资产 (后续会动态更新)
        all_symbols = []
        for symbols in self.config.assets.default_universe.values():
            all_symbols.extend(symbols)

        # 当前动态候选池
        current_universe = {}

        for i, date in enumerate(rebalance_dates[start_index:], start=start_index):
            logger.info(f"Processing date: {date}")

            try:
                # Step 0: 刷新动态候选池 (每周自动刷新)
                refreshed_universe = self._refresh_dynamic_universe(date)
                if refreshed_universe:
                    current_universe = refreshed_universe
                    # 更新 all_symbols 以包含动态筛选的资产
                    all_symbols = []
                    for asset_class, symbols in current_universe.items():
                        all_symbols.extend(symbols)
                    logger.info(f"Dynamic universe updated: {len(all_symbols)} total symbols")

                # Step 1: 获取市场数据
                market_snapshot = self.market_data.get_market_snapshot(
                    symbols=all_symbols,
                    date=date,
                    lookback_days=self.config.data.lookback_days,
                )

                # Step 2: 收集专家意见
                expert_reports = self._collect_expert_reports(market_snapshot, date)

                # Step 3: 获取风控约束
                risk_constraints = self.risk_controller.get_constraints()

                # Step 4: Portfolio Manager决策
                decision = self.portfolio_manager.decide(
                    expert_reports=expert_reports,
                    market_data=market_snapshot,
                    current_portfolio=self.env.portfolio.weights,
                    risk_constraints=risk_constraints,
                )

                # Step 5: 风控评估 (带日内监控)
                risk_assessment = self.risk_controller.assess_with_intraday(
                    current_allocation=self.env.portfolio.weights,
                    proposed_allocation=decision.target_allocation,
                    market_data=market_snapshot,
                    portfolio_value=self.env.portfolio.portfolio_value,
                )

                # 记录日内警报
                if risk_assessment.intraday_alerts:
                    for alert in risk_assessment.intraday_alerts:
                        logger.warning(f"[{alert.severity.upper()}] {alert.message}")

                # Step 6: 根据风控结果执行
                if risk_assessment.emergency_rebalance and risk_assessment.defensive_allocation:
                    # 紧急情况：使用防御性配置
                    logger.warning(f"Emergency rebalance triggered on {date} - switching to defensive allocation")
                    target = risk_assessment.defensive_allocation
                elif risk_assessment.veto:
                    logger.warning(f"Risk Controller VETO on {date}")
                    # 使用调整后的配置
                    if risk_assessment.recommendations:
                        target = self._apply_risk_adjustments(
                            decision.target_allocation,
                            risk_assessment.recommendations
                        )
                    else:
                        target = self.env.portfolio.weights  # 维持现状
                else:
                    target = decision.target_allocation

                # Step 7: 执行交易 (传递专家报告以支持专家驱动的个股权重分配)
                portfolio_state, reward, done, info = self.env.step(
                    target_allocation=target,
                    market_data=market_snapshot,
                    timestamp=date,
                    expert_reports=expert_reports,
                )

                # 记录结果
                results["decisions"].append({
                    "date": date,
                    "decision": decision.to_dict(),
                    "executed_allocation": target,
                    "trades": info.get("trades", []),
                })
                results["risk_assessments"].append({
                    "date": date,
                    "assessment": risk_assessment.to_dict(),
                })

                logger.info(f"Date {date}: Portfolio value = ${portfolio_state.portfolio_value:,.2f}")

                # 保存断点 (每次迭代后保存)
                self._save_checkpoint(
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_frequency=rebalance_frequency,
                    current_date_index=i + 1,
                    total_dates=len(rebalance_dates),
                    results=results,
                )
                logger.debug(f"Checkpoint saved: {i + 1}/{len(rebalance_dates)}")

            except (KeyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Data processing error on {date}: {type(e).__name__}: {e}", exc_info=True)
                # 保存断点后继续
                self._save_checkpoint(
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_frequency=rebalance_frequency,
                    current_date_index=i,
                    total_dates=len(rebalance_dates),
                    results=results,
                )
                continue

        # 最终指标
        results["final_metrics"] = self.env.get_metrics()
        results["final_portfolio"] = self.env.portfolio.to_dict()

        # 清理断点文件 (实验完成)
        self._clear_checkpoint()

        logger.info("FinSage run completed")
        return results

    def _collect_expert_reports(
        self,
        market_data: Dict[str, Any],
        date: str
    ) -> Dict[str, Any]:
        """收集所有专家的报告"""
        expert_reports = {}

        for asset_class, expert in self.experts.items():
            try:
                # 提取该资产类别的数据
                class_data = self._filter_market_data(market_data, asset_class)
                news_data = market_data.get("news", [])

                # 从 market_data 中提取技术指标
                # 技术指标现在嵌入在每个资产的数据中
                technical_indicators = self._extract_technical_indicators(class_data, expert.symbols)

                # 调用专家分析
                report = expert.analyze(
                    market_data=class_data,
                    news_data=news_data,
                    technical_indicators=technical_indicators,
                )
                expert_reports[asset_class] = report

            except Exception as e:
                logger.warning(f"Expert {asset_class} analysis failed: {e}")

        return expert_reports

    def _extract_technical_indicators(
        self,
        market_data: Dict[str, Any],
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        从 market_data 中提取技术指标

        Args:
            market_data: 市场数据 (每个资产包含技术指标)
            symbols: 需要提取的资产符号列表

        Returns:
            技术指标字典: {symbol: {rsi, macd, ma_20, ma_50, ...}}
        """
        indicators = {}
        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                # 提取技术指标字段
                indicators[symbol] = {
                    "rsi": data.get("rsi", data.get("rsi_14", 50)),
                    "rsi_14": data.get("rsi_14", data.get("rsi", 50)),
                    "macd": data.get("macd", 0),
                    "macd_signal": data.get("macd_signal", 0),
                    "macd_hist": data.get("macd_hist", 0),
                    "macd_cross": data.get("macd_cross", "neutral"),
                    "ma_20": data.get("ma_20", data.get("sma_20", 0)),
                    "ma_50": data.get("ma_50", data.get("sma_50", 0)),
                    "sma_20": data.get("sma_20", data.get("ma_20", 0)),
                    "sma_50": data.get("sma_50", data.get("ma_50", 0)),
                    "bb_upper": data.get("bb_upper", 0),
                    "bb_middle": data.get("bb_middle", 0),
                    "bb_lower": data.get("bb_lower", 0),
                    "bb_position": data.get("bb_position", "neutral"),
                    "trend": data.get("trend", "sideways"),
                    "price": data.get("close", data.get("price", 0)),
                }
        return indicators

    def _filter_market_data(
        self,
        market_data: Dict[str, Any],
        asset_class: str
    ) -> Dict[str, Any]:
        """过滤特定资产类别的数据"""
        # 优先使用专家的动态 symbols，如果没有则回退到 config
        if asset_class in self.experts:
            symbols = self.experts[asset_class].symbols
        else:
            symbols = self.config.assets.default_universe.get(asset_class, [])
        filtered = {}
        for symbol in symbols:
            if symbol in market_data:
                filtered[symbol] = market_data[symbol]
        return filtered

    def _apply_risk_adjustments(
        self,
        allocation: Dict[str, float],
        recommendations: Dict[str, Any]
    ) -> Dict[str, float]:
        """应用风险调整"""
        adjusted = allocation.copy()

        for key, rec in recommendations.items():
            if key.startswith("reduce_"):
                asset = rec.get("asset")
                target = rec.get("to", adjusted.get(asset, 0) * 0.8)
                if asset in adjusted:
                    adjusted[asset] = target

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _generate_rebalance_dates(
        self,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> List[str]:
        """生成再平衡日期 (自动跳过周末和美国假日)"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # 美国股市假日 (固定日期，需要根据年份更新)
        # 格式: (月, 日) 或 动态计算
        us_holidays = self._get_us_market_holidays(start.year, end.year)

        dates = []
        current = start

        if frequency == "daily":
            delta = timedelta(days=1)
        elif frequency == "weekly":
            delta = timedelta(weeks=1)
        elif frequency == "monthly":
            delta = timedelta(days=30)
        else:
            delta = timedelta(weeks=1)

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            # 跳过周末 (周六=5, 周日=6)
            if current.weekday() >= 5:
                logger.debug(f"Skipping weekend date: {date_str} (weekday={current.weekday()})")
            # 跳过美国假日
            elif date_str in us_holidays:
                logger.info(f"Skipping US market holiday: {date_str} ({us_holidays[date_str]})")
            else:
                dates.append(date_str)
            current += delta

        return dates

    def _get_us_market_holidays(self, start_year: int, end_year: int) -> Dict[str, str]:
        """
        获取美国股市假日列表

        Returns:
            Dict: {日期字符串: 假日名称}
        """
        holidays = {}

        for year in range(start_year, end_year + 1):
            # 新年 (1月1日，如果是周末则顺延到周一)
            new_year = datetime(year, 1, 1)
            if new_year.weekday() == 5:  # 周六
                new_year = datetime(year, 1, 3)  # 顺延到周一
            elif new_year.weekday() == 6:  # 周日
                new_year = datetime(year, 1, 2)  # 顺延到周一
            holidays[new_year.strftime("%Y-%m-%d")] = "New Year's Day"

            # 马丁·路德·金日 (1月第三个周一)
            mlk_day = self._get_nth_weekday(year, 1, 0, 3)  # 0=周一, 第3个
            holidays[mlk_day.strftime("%Y-%m-%d")] = "MLK Day"

            # 总统日 (2月第三个周一)
            presidents_day = self._get_nth_weekday(year, 2, 0, 3)
            holidays[presidents_day.strftime("%Y-%m-%d")] = "Presidents Day"

            # 耶稣受难日 (复活节前的周五) - 需要复杂计算，使用常见日期
            # 2024: 3月29日, 2023: 4月7日, 2025: 4月18日
            good_friday_dates = {
                2023: (4, 7),
                2024: (3, 29),
                2025: (4, 18),
                2026: (4, 3),
            }
            if year in good_friday_dates:
                gf_month, gf_day = good_friday_dates[year]
                holidays[f"{year}-{gf_month:02d}-{gf_day:02d}"] = "Good Friday"

            # 阵亡将士纪念日 (5月最后一个周一)
            memorial_day = self._get_last_weekday(year, 5, 0)
            holidays[memorial_day.strftime("%Y-%m-%d")] = "Memorial Day"

            # 六月节 (6月19日)
            juneteenth = datetime(year, 6, 19)
            if juneteenth.weekday() == 5:
                juneteenth = datetime(year, 6, 18)
            elif juneteenth.weekday() == 6:
                juneteenth = datetime(year, 6, 20)
            holidays[juneteenth.strftime("%Y-%m-%d")] = "Juneteenth"

            # 独立日 (7月4日)
            july_4th = datetime(year, 7, 4)
            if july_4th.weekday() == 5:
                july_4th = datetime(year, 7, 3)
            elif july_4th.weekday() == 6:
                july_4th = datetime(year, 7, 5)
            holidays[july_4th.strftime("%Y-%m-%d")] = "Independence Day"

            # 劳动节 (9月第一个周一)
            labor_day = self._get_nth_weekday(year, 9, 0, 1)
            holidays[labor_day.strftime("%Y-%m-%d")] = "Labor Day"

            # 感恩节 (11月第四个周四)
            thanksgiving = self._get_nth_weekday(year, 11, 3, 4)  # 3=周四
            holidays[thanksgiving.strftime("%Y-%m-%d")] = "Thanksgiving"

            # 圣诞节 (12月25日)
            christmas = datetime(year, 12, 25)
            if christmas.weekday() == 5:
                christmas = datetime(year, 12, 24)
            elif christmas.weekday() == 6:
                christmas = datetime(year, 12, 26)
            holidays[christmas.strftime("%Y-%m-%d")] = "Christmas"

        return holidays

    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> datetime:
        """获取某月第n个指定星期几的日期"""
        first_day = datetime(year, month, 1)
        first_weekday = first_day.weekday()

        # 计算第一个指定星期几的日期
        days_until_weekday = (weekday - first_weekday) % 7
        first_target = first_day + timedelta(days=days_until_weekday)

        # 加上 (n-1) 周
        return first_target + timedelta(weeks=n-1)

    def _get_last_weekday(self, year: int, month: int, weekday: int) -> datetime:
        """获取某月最后一个指定星期几的日期"""
        # 获取下个月第一天
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        # 回退到上个月最后一天
        last_day = next_month - timedelta(days=1)

        # 找到最后一个指定星期几
        days_back = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=days_back)

    def save_results(self, results: Dict, output_path: str):
        """保存结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    # ==================== Checkpoint Methods ====================

    def _get_checkpoint_path(self) -> str:
        """获取断点文件路径"""
        return os.path.join(self.checkpoint_dir, "finsage_checkpoint.pkl")

    def _save_checkpoint(
        self,
        start_date: str,
        end_date: str,
        rebalance_frequency: str,
        current_date_index: int,
        total_dates: int,
        results: Dict[str, Any],
    ):
        """
        保存断点

        Args:
            start_date: 开始日期
            end_date: 结束日期
            rebalance_frequency: 再平衡频率
            current_date_index: 当前处理的日期索引
            total_dates: 总日期数
            results: 当前结果
        """
        try:
            checkpoint = CheckpointState(
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency=rebalance_frequency,
                current_date_index=current_date_index,
                total_dates=total_dates,
                results=results,
                portfolio_state=self.env.get_state(),
                timestamp=datetime.now().isoformat(),
                risk_controller_state={
                    "peak_value": self.risk_controller.peak_value,
                    "max_drawdown_history": self.risk_controller.max_drawdown_history,
                },
            )

            checkpoint_path = self._get_checkpoint_path()
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)

            # 同时保存 JSON 版本用于查看进度
            json_path = checkpoint_path.replace(".pkl", "_progress.json")
            progress_info = {
                "start_date": start_date,
                "end_date": end_date,
                "current_date_index": current_date_index,
                "total_dates": total_dates,
                "progress_pct": f"{current_date_index / total_dates * 100:.1f}%",
                "timestamp": checkpoint.timestamp,
            }
            with open(json_path, "w") as f:
                json.dump(progress_info, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> Optional[CheckpointState]:
        """
        加载断点

        Returns:
            CheckpointState 或 None
        """
        checkpoint_path = self._get_checkpoint_path()

        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint found, starting from scratch")
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)

            logger.info(f"Loaded checkpoint: {checkpoint.current_date_index}/{checkpoint.total_dates} dates completed")
            logger.info(f"Checkpoint timestamp: {checkpoint.timestamp}")
            return checkpoint

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _clear_checkpoint(self):
        """清理断点文件"""
        try:
            checkpoint_path = self._get_checkpoint_path()
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info("Checkpoint cleared (experiment completed)")

            json_path = checkpoint_path.replace(".pkl", "_progress.json")
            if os.path.exists(json_path):
                os.remove(json_path)

        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")
