"""
Strategic Asset Allocation (SAA)
战略资产配置策略

基于长期资本市场假设的资产配置，着重于长期风险收益特征。
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from finsage.strategies.base_strategy import AllocationStrategy

logger = logging.getLogger(__name__)


class StrategicAllocationStrategy(AllocationStrategy):
    """
    战略资产配置策略 (Strategic Asset Allocation, SAA)

    基于长期资本市场假设（Capital Market Assumptions），
    使用均值-方差优化或风险平价构建长期战略配置。

    参考文献:
    - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
    - Ibbotson, R.G. & Kaplan, P.D. (2000). Does Asset Allocation Policy Explain
      40, 90, or 100 Percent of Performance? Financial Analysts Journal.
    - Brinson, G.P., Hood, L.R., & Beebower, G.L. (1986). Determinants of
      Portfolio Performance. Financial Analysts Journal.

    关键特点:
    1. 长期视角（5-10年以上）
    2. 基于资产类别的长期预期收益和风险
    3. 低换手率（通常年度再平衡）
    4. 注重分散化和风险控制
    """

    def __init__(
        self,
        method: str = "mean_variance",
        use_black_litterman: bool = False,
    ):
        """
        初始化战略配置策略

        Args:
            method: 优化方法 (mean_variance, risk_parity, equal_weight)
            use_black_litterman: 是否使用Black-Litterman模型整合观点
        """
        self.method = method
        self.use_black_litterman = use_black_litterman

        # 长期资本市场假设 (基于历史数据和经济预期)
        # 这些是年化的长期预期
        self.capital_market_assumptions = {
            "stocks": {"expected_return": 0.08, "volatility": 0.18},
            "bonds": {"expected_return": 0.03, "volatility": 0.05},
            "commodities": {"expected_return": 0.04, "volatility": 0.15},
            "reits": {"expected_return": 0.06, "volatility": 0.14},
            "crypto": {"expected_return": 0.15, "volatility": 0.60},
            "cash": {"expected_return": 0.02, "volatility": 0.01},
        }

        # 资产类别相关性矩阵 (长期)
        self.correlation_matrix = {
            ("stocks", "bonds"): 0.1,
            ("stocks", "commodities"): 0.3,
            ("stocks", "reits"): 0.6,
            ("stocks", "crypto"): 0.4,
            ("stocks", "cash"): 0.0,
            ("bonds", "commodities"): -0.1,
            ("bonds", "reits"): 0.2,
            ("bonds", "crypto"): 0.0,
            ("bonds", "cash"): 0.3,
            ("commodities", "reits"): 0.2,
            ("commodities", "crypto"): 0.3,
            ("commodities", "cash"): 0.0,
            ("reits", "crypto"): 0.2,
            ("reits", "cash"): 0.1,
            ("crypto", "cash"): 0.0,
        }

    @property
    def name(self) -> str:
        return "strategic_allocation"

    @property
    def description(self) -> str:
        return """战略资产配置策略 (Strategic Asset Allocation, SAA)
基于长期资本市场假设，构建战略性资产配置。
着眼于5-10年以上的投资视角，强调分散化和长期风险收益平衡。
适用场景：长期投资规划、养老金配置、基金会资产配置。
优点：理论基础扎实、低换手率、降低择时风险。
缺点：对市场变化反应较慢、依赖长期假设的准确性。"""

    @property
    def rebalance_frequency(self) -> str:
        return "annually"

    def compute_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算战略资产配置

        Args:
            market_data: 各资产类别的市场数据
            expert_views: 专家观点
            risk_profile: 风险偏好
            constraints: 配置约束
            **kwargs: 其他参数

        Returns:
            资产类别配置
        """
        asset_classes = list(market_data.keys())
        if not asset_classes:
            # 使用默认资产类别
            asset_classes = ["stocks", "bonds", "commodities", "reits", "cash"]

        n_assets = len(asset_classes)

        # 获取风险偏好参数
        risk_params = self.get_risk_profile_params(risk_profile)

        # 获取或估计预期收益和波动率
        mu, sigma = self._get_expected_returns_and_cov(
            asset_classes, market_data, expert_views
        )

        # 如果使用Black-Litterman且有专家观点
        if self.use_black_litterman and expert_views:
            mu = self._apply_black_litterman(mu, sigma, expert_views, asset_classes)

        # 根据方法进行优化
        if self.method == "mean_variance":
            weights = self._mean_variance_optimization(
                mu, sigma, risk_params, constraints, asset_classes
            )
        elif self.method == "risk_parity":
            weights = self._risk_parity_optimization(sigma, asset_classes)
        else:  # equal_weight
            weights = {ac: 1.0 / n_assets for ac in asset_classes}

        # 应用风险偏好约束
        weights = self._apply_risk_constraints(weights, risk_params, asset_classes)

        return self.validate_allocation(weights)

    def _get_expected_returns_and_cov(
        self,
        asset_classes: list,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None
    ) -> tuple:
        """
        获取预期收益向量和协方差矩阵

        结合长期资本市场假设和历史数据
        """
        n = len(asset_classes)

        # 预期收益（使用CMA，根据历史数据微调）
        mu = np.zeros(n)
        for i, ac in enumerate(asset_classes):
            cma = self.capital_market_assumptions.get(ac, {"expected_return": 0.05})
            base_return = cma["expected_return"]

            # 如果有历史数据，用来微调
            if ac in market_data and not market_data[ac].empty:
                hist_return = market_data[ac].mean().mean() * 252
                # 混合：70% CMA + 30% 历史
                mu[i] = 0.7 * base_return + 0.3 * hist_return
            else:
                mu[i] = base_return

            # 整合专家观点
            if expert_views and ac in expert_views:
                view = expert_views[ac].get("return_adjustment", 0)
                mu[i] += view * 0.02  # 微调

        # 协方差矩阵
        sigma = np.zeros((n, n))
        for i, ac_i in enumerate(asset_classes):
            cma_i = self.capital_market_assumptions.get(ac_i, {"volatility": 0.15})
            vol_i = cma_i["volatility"]

            for j, ac_j in enumerate(asset_classes):
                cma_j = self.capital_market_assumptions.get(ac_j, {"volatility": 0.15})
                vol_j = cma_j["volatility"]

                if i == j:
                    # 使用历史波动率（如果有）
                    if ac_i in market_data and not market_data[ac_i].empty:
                        hist_vol = market_data[ac_i].std().mean() * np.sqrt(252)
                        vol = 0.7 * vol_i + 0.3 * hist_vol
                    else:
                        vol = vol_i
                    sigma[i, j] = vol ** 2
                else:
                    # 相关性
                    key = (ac_i, ac_j) if (ac_i, ac_j) in self.correlation_matrix else (ac_j, ac_i)
                    corr = self.correlation_matrix.get(key, 0.2)
                    sigma[i, j] = corr * vol_i * vol_j

        return mu, sigma

    def _mean_variance_optimization(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_params: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
        asset_classes: list
    ) -> Dict[str, float]:
        """
        均值-方差优化

        max { μ'w - λ/2 * w'Σw }
        """
        n = len(asset_classes)
        risk_aversion = risk_params.get("risk_aversion", 1.5)

        def objective(w):
            ret = mu @ w
            risk = w @ sigma @ w
            return -(ret - risk_aversion / 2 * risk)

        # 约束
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # 边界
        min_w = constraints.get("min_weight", 0.0) if constraints else 0.0
        max_w = constraints.get("max_weight", 0.4) if constraints else 0.4
        bounds = [(min_w, max_w) for _ in range(n)]

        # 初始权重
        init_w = np.array([1.0 / n] * n)

        try:
            result = minimize(
                objective, init_w, method="SLSQP",
                bounds=bounds, constraints=constraints_list,
                options={"maxiter": 500}
            )
            weights = result.x if result.success else init_w
        except Exception as e:
            logger.warning(f"Mean-variance optimization failed: {e}")
            weights = init_w

        return dict(zip(asset_classes, weights))

    def _risk_parity_optimization(
        self,
        sigma: np.ndarray,
        asset_classes: list
    ) -> Dict[str, float]:
        """
        风险平价优化

        使每个资产的风险贡献相等
        """
        n = len(asset_classes)

        def risk_contribution(w):
            """计算各资产风险贡献"""
            portfolio_vol = np.sqrt(w @ sigma @ w)
            marginal_risk = sigma @ w / portfolio_vol
            return w * marginal_risk

        def objective(w):
            rc = risk_contribution(w)
            target_rc = np.sum(rc) / n
            return np.sum((rc - target_rc) ** 2)

        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.5) for _ in range(n)]
        init_w = np.array([1.0 / n] * n)

        try:
            result = minimize(
                objective, init_w, method="SLSQP",
                bounds=bounds, constraints=constraints_list,
                options={"maxiter": 500}
            )
            weights = result.x if result.success else init_w
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")
            weights = init_w

        return dict(zip(asset_classes, weights))

    def _apply_black_litterman(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        expert_views: Dict[str, Dict[str, float]],
        asset_classes: list
    ) -> np.ndarray:
        """
        应用Black-Litterman模型整合专家观点

        简化版本：对有观点的资产进行收益调整
        """
        tau = 0.05  # 缩放因子
        adjusted_mu = mu.copy()

        for asset, views in expert_views.items():
            if asset in asset_classes:
                idx = asset_classes.index(asset)
                view_return = views.get("expected_return", None)
                confidence = views.get("confidence", 0.5)

                if view_return is not None:
                    # 混合先验和观点
                    adjusted_mu[idx] = (
                        (1 - confidence * tau) * mu[idx] +
                        confidence * tau * view_return
                    )

        return adjusted_mu

    def _apply_risk_constraints(
        self,
        weights: Dict[str, float],
        risk_params: Dict[str, float],
        asset_classes: list
    ) -> Dict[str, float]:
        """
        应用风险偏好约束
        """
        max_equity = risk_params.get("max_equity", 0.60)
        min_fixed_income = risk_params.get("min_fixed_income", 0.20)
        max_crypto = risk_params.get("max_crypto", 0.05)

        # 股票类（stocks + reits）
        equity_weight = weights.get("stocks", 0) + weights.get("reits", 0)
        if equity_weight > max_equity:
            scale = max_equity / equity_weight
            if "stocks" in weights:
                weights["stocks"] *= scale
            if "reits" in weights:
                weights["reits"] *= scale

        # 固定收益（bonds + cash）
        fixed_income = weights.get("bonds", 0) + weights.get("cash", 0)
        if fixed_income < min_fixed_income:
            shortage = min_fixed_income - fixed_income
            if "bonds" in weights:
                weights["bonds"] += shortage * 0.7
            if "cash" in weights:
                weights["cash"] += shortage * 0.3

        # 加密货币
        if "crypto" in weights and weights["crypto"] > max_crypto:
            excess = weights["crypto"] - max_crypto
            weights["crypto"] = max_crypto
            # 将多余的分配给其他资产
            if "bonds" in weights:
                weights["bonds"] += excess

        return weights

    def get_policy_portfolio(self, risk_profile: str = "moderate") -> Dict[str, float]:
        """
        获取政策组合（Policy Portfolio）

        这是基于风险偏好的默认战略配置，
        可作为SAA的起点或基准。

        Args:
            risk_profile: 风险偏好

        Returns:
            政策组合配置
        """
        portfolios = {
            "conservative": {
                "stocks": 0.20,
                "bonds": 0.50,
                "commodities": 0.05,
                "reits": 0.10,
                "crypto": 0.00,
                "cash": 0.15,
            },
            "moderate": {
                "stocks": 0.40,
                "bonds": 0.30,
                "commodities": 0.10,
                "reits": 0.10,
                "crypto": 0.02,
                "cash": 0.08,
            },
            "aggressive": {
                "stocks": 0.55,
                "bonds": 0.15,
                "commodities": 0.10,
                "reits": 0.10,
                "crypto": 0.05,
                "cash": 0.05,
            },
        }
        return portfolios.get(risk_profile, portfolios["moderate"])
