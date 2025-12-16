"""
Hierarchical Risk Parity (HRP)
层次风险平价 - López de Prado (2016)
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from finsage.hedging.base_tool import HedgingTool


class HierarchicalRiskParityTool(HedgingTool):
    """
    层次风险平价(Hierarchical Risk Parity, HRP)

    结合图论和机器学习技术的现代资产配置方法，
    通过层次聚类考虑资产间的相关性结构。

    参考文献:
    López de Prado, M. (2016). Building Diversified Portfolios that
    Outperform Out-of-Sample. Journal of Portfolio Management.
    """

    @property
    def name(self) -> str:
        return "hrp"

    @property
    def description(self) -> str:
        return """层次风险平价(Hierarchical Risk Parity)
结合层次聚类的现代资产配置方法，考虑资产间相关性结构。
适用场景：资产数量较多、需要稳健配置的场景。
优点：不需要矩阵求逆，数值稳定性好，样本外表现优异。
缺点：结果依赖聚类方法选择。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "linkage_method": "聚类方法 (默认'single')",
            "distance_metric": "距离度量 (默认相关性距离)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算HRP组合权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点
            constraints: 约束条件
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重
        """
        if returns.empty or len(returns) < 10:
            assets = returns.columns.tolist()
            n = len(assets)
            return {a: 1.0/n for a in assets} if n > 0 else {}

        assets = returns.columns.tolist()
        n_assets = len(assets)

        # Step 1: 计算相关性矩阵
        corr_matrix = returns.corr().values

        # Step 2: 计算协方差矩阵
        cov_matrix = returns.cov().values

        # Step 3: 层次聚类
        linkage_method = kwargs.get("linkage_method", "single")
        order = self._get_quasi_diag(corr_matrix, linkage_method)

        # Step 4: 递归二分配置
        weights = self._recursive_bisection(cov_matrix, order)

        # Step 5: 应用约束
        if constraints:
            weights = self._apply_constraints(weights, constraints)

        return dict(zip([assets[i] for i in order], weights))

    def _get_quasi_diag(
        self,
        corr_matrix: np.ndarray,
        linkage_method: str = "single"
    ) -> List[int]:
        """
        通过层次聚类获取准对角化排序

        Args:
            corr_matrix: 相关性矩阵
            linkage_method: 聚类方法

        Returns:
            排序后的资产索引
        """
        # 计算相关性距离
        dist = np.sqrt(0.5 * (1 - corr_matrix))
        np.fill_diagonal(dist, 0)

        # 确保对称性
        dist = (dist + dist.T) / 2

        # 转换为压缩格式
        dist_condensed = squareform(dist)

        # 层次聚类
        link = linkage(dist_condensed, method=linkage_method)

        # 获取叶子节点顺序
        order = leaves_list(link)

        return order.tolist()

    def _recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        order: List[int]
    ) -> np.ndarray:
        """
        递归二分配置

        Args:
            cov_matrix: 协方差矩阵
            order: 资产排序

        Returns:
            权重数组
        """
        n_assets = len(order)
        weights = np.ones(n_assets)

        # 递归分配
        clusters = [order]
        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # 分成两半
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]

                    # 计算各半的方差
                    left_var = self._get_cluster_var(cov_matrix, left)
                    right_var = self._get_cluster_var(cov_matrix, right)

                    # 按方差倒数分配
                    alpha = 1 - left_var / (left_var + right_var)

                    # 更新权重
                    for i in left:
                        weights[order.index(i)] *= alpha
                    for i in right:
                        weights[order.index(i)] *= (1 - alpha)

                    # 添加到下一轮
                    if len(left) > 1:
                        new_clusters.append(left)
                    if len(right) > 1:
                        new_clusters.append(right)

            clusters = new_clusters

        return weights

    def _get_cluster_var(
        self,
        cov_matrix: np.ndarray,
        cluster_indices: List[int]
    ) -> float:
        """
        计算聚类内的方差

        Args:
            cov_matrix: 协方差矩阵
            cluster_indices: 聚类内资产索引

        Returns:
            聚类方差
        """
        # 提取子协方差矩阵
        sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]

        # 计算逆波动率权重
        inv_diag = 1 / np.diag(sub_cov)
        inv_vol_weights = inv_diag / inv_diag.sum()

        # 计算组合方差
        cluster_var = np.dot(inv_vol_weights, np.dot(sub_cov, inv_vol_weights))

        return cluster_var

    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict[str, float]
    ) -> np.ndarray:
        """应用权重约束"""
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_single_asset", 0.25)

        # 裁剪权重
        weights = np.clip(weights, min_weight, max_weight)

        # 归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights
