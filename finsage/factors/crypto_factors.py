"""
Crypto Factor Scorer - Network, Adoption & Risk Factors
加密货币基本面因子评分器

学术基础:
- Biais, B. et al. (2023). "Equilibrium Bitcoin Pricing"
  Journal of Finance.

  核心观点: 加密货币的价值来源于:
  1. 网络效应 (Network Effects): 用户越多，价值越高
  2. 交易效用 (Transaction Utility T_t): 净交易效用驱动需求
  3. 崩盘风险 (Crash Risk): 需要溢价补偿

- Liu, Y. & Tsyvinski, A. (2021). "Risks and Returns of Cryptocurrency"
  Review of Financial Studies.

  核心发现:
  - 加密货币与传统资产低相关
  - 动量效应显著
  - 网络因子解释大部分收益

- Makarov, I. & Schoar, A. (2020). "Trading and Arbitrage in Cryptocurrency Markets"
  Journal of Financial Economics.

加密货币因子框架:
1. Network Factor (网络因子): 活跃地址、交易量、算力
2. Adoption Factor (采纳因子): T_t净交易效用
3. Momentum Factor (动量因子): 价格趋势
4. Crash Risk Factor (崩盘风险): 尾部风险度量
5. Sentiment Factor (情绪因子): 市场情绪指标
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

from finsage.factors.base_factor import (
    BaseFactorScorer,
    FactorType,
    FactorExposure,
    FactorScore,
)

logger = logging.getLogger(__name__)


class CryptoFactorScorer(BaseFactorScorer):
    """
    加密货币基本面因子评分器

    基于网络效应、采纳度和崩盘风险评估加密货币的投资价值。

    核心理论 (Biais et al. 2023):
    - 加密货币价格由预期未来交易效用的贴现值决定
    - 网络效应创造正反馈循环
    - 崩盘风险需要风险溢价补偿

    使用方法:
    ```python
    scorer = CryptoFactorScorer()
    score = scorer.score("BTC", crypto_data, returns)
    print(score.factor_exposures["network"])  # 网络效应信号
    ```
    """

    # 加密货币分类
    CRYPTO_CATEGORIES = {
        "store_of_value": ["BTC", "WBTC"],  # 价值存储
        "smart_contract": ["ETH", "SOL", "ADA", "AVAX", "DOT"],  # 智能合约平台
        "defi": ["UNI", "AAVE", "LINK", "MKR", "CRV"],  # DeFi协议
        "layer2": ["MATIC", "ARB", "OP"],  # Layer 2扩展
        "stablecoin": ["USDT", "USDC", "DAI"],  # 稳定币
        "meme": ["DOGE", "SHIB"],  # Meme币
    }

    # 各类别的基本面权重调整
    CATEGORY_ADJUSTMENTS = {
        "store_of_value": {"network": 1.2, "adoption": 1.0, "crash_risk": 0.8},
        "smart_contract": {"network": 1.0, "adoption": 1.2, "crash_risk": 1.0},
        "defi": {"network": 0.8, "adoption": 1.3, "crash_risk": 1.2},
        "layer2": {"network": 0.9, "adoption": 1.2, "crash_risk": 1.1},
        "stablecoin": {"network": 0.5, "adoption": 0.5, "crash_risk": 2.0},  # 稳定币主要看脱锚风险
        "meme": {"network": 0.6, "adoption": 0.4, "crash_risk": 1.5},  # Meme币高风险
    }

    @property
    def asset_class(self) -> str:
        return "crypto"

    @property
    def supported_factors(self) -> List[FactorType]:
        return [
            FactorType.NETWORK,
            FactorType.ADOPTION,
            FactorType.MOMENTUM,
            FactorType.CRASH_RISK,
        ]

    def _default_weights(self) -> Dict[str, float]:
        """
        默认因子权重

        基于学术文献:
        - Network: 35% - 核心价值驱动
        - Adoption: 25% - 交易效用 T_t
        - Momentum: 25% - 加密货币动量效应强
        - Crash Risk: 15% - 风险调整
        """
        return {
            "network": 0.35,
            "adoption": 0.25,
            "momentum": 0.25,
            "crash_risk": 0.15,
        }

    def _compute_factor_exposures(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, FactorExposure]:
        """
        计算加密货币因子暴露

        Args:
            symbol: 加密货币代码
            data: 加密货币数据，应包含:
                - active_addresses: 活跃地址数
                - transaction_volume: 交易量
                - hash_rate: 算力 (PoW链)
                - tvl: 总锁仓价值 (DeFi)
                - market_cap: 市值
                - volume_24h: 24小时交易量
                - developer_activity: 开发者活动
                - social_volume: 社交媒体热度
            returns: 历史收益率序列

        Returns:
            Dict[factor_name, FactorExposure]
        """
        exposures = {}

        # 获取币种类别
        category = self._get_category(symbol)

        # 1. Network Factor (网络因子) - 核心!
        exposures["network"] = self._compute_network_exposure(symbol, data, category)

        # 2. Adoption Factor (采纳因子)
        exposures["adoption"] = self._compute_adoption_exposure(data, category)

        # 3. Momentum Factor (动量因子)
        exposures["momentum"] = self._compute_momentum_exposure(data, returns)

        # 4. Crash Risk Factor (崩盘风险因子)
        exposures["crash_risk"] = self._compute_crash_risk_exposure(data, returns)

        return exposures

    def _get_category(self, symbol: str) -> str:
        """获取加密货币类别"""
        for category, symbols in self.CRYPTO_CATEGORIES.items():
            if symbol.upper() in symbols:
                return category
        return "smart_contract"  # 默认分类

    def _compute_network_exposure(
        self,
        symbol: str,
        data: Dict[str, Any],
        category: str
    ) -> FactorExposure:
        """
        计算网络因子暴露

        网络效应是加密货币价值的核心驱动力!

        Metcalfe's Law: V ∝ n²
        - 用户越多，网络价值呈指数增长

        指标:
        - Active Addresses: 活跃地址数
        - Transaction Volume: 链上交易量
        - Hash Rate: 算力 (PoW)
        - TVL: 总锁仓价值 (DeFi)
        """
        # 活跃地址 (相对于历史)
        active_addresses = data.get("active_addresses", 0)
        active_addresses_percentile = data.get("active_addresses_percentile", 50)

        # 交易量 (相对于市值)
        tx_volume = data.get("transaction_volume", 0)
        market_cap = data.get("market_cap", 1)
        volume_to_mcap = tx_volume / market_cap if market_cap > 0 else 0

        # 算力变化 (PoW链)
        hash_rate_growth = data.get("hash_rate_growth_30d", 0)

        # TVL (DeFi)
        tvl = data.get("tvl", 0)
        tvl_growth = data.get("tvl_growth_30d", 0)

        # 开发者活动
        dev_activity = data.get("developer_activity_percentile", 50)

        # 综合网络评分
        network_score = 0.0

        # 活跃地址贡献 (0-40分)
        network_score += (active_addresses_percentile / 100) * 0.4

        # 交易量/市值比 (0-20分)
        # 高周转率表示高使用率
        volume_score = min(volume_to_mcap / 0.1, 1.0) * 0.2
        network_score += volume_score

        # 算力增长 (0-15分) - 仅对PoW链
        if symbol.upper() in ["BTC", "LTC", "DOGE"]:
            hash_score = self.clip_exposure(hash_rate_growth / 0.2, 0, 1) * 0.15
            network_score += hash_score
        else:
            # 非PoW链用TVL增长替代
            tvl_score = self.clip_exposure(tvl_growth / 0.3, 0, 1) * 0.15
            network_score += tvl_score

        # 开发者活动 (0-25分)
        network_score += (dev_activity / 100) * 0.25

        # 转换为exposure [-1, 1]
        # 0.5分为中性点
        exposure = self.clip_exposure((network_score - 0.5) * 2, -1, 1)

        # Z-score
        z_score = self.normalize_to_zscore(network_score, mean=0.5, std=0.15)

        # 信号
        if network_score > 0.7:
            signal = "STRONG_NETWORK"
        elif network_score > 0.5:
            signal = "LONG"
        elif network_score > 0.3:
            signal = "NEUTRAL"
        else:
            signal = "WEAK_NETWORK"

        return FactorExposure(
            factor_type=FactorType.NETWORK,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.70,
        )

    def _compute_adoption_exposure(
        self,
        data: Dict[str, Any],
        category: str
    ) -> FactorExposure:
        """
        计算采纳因子暴露

        T_t (Net Transaction Utility) - Biais et al. 2023核心概念

        T_t = 交易便利性收益 - 机会成本 - 交易成本

        指标:
        - 新地址增长率
        - 持币地址分布
        - 交易所流入流出
        - 机构持仓
        """
        # 新地址增长 (采纳率)
        new_addresses_growth = data.get("new_addresses_growth_30d", 0)

        # 持币分布 (去中心化程度)
        # 高集中度 = 风险
        whale_concentration = data.get("whale_concentration", 0.5)
        decentralization_score = 1 - whale_concentration

        # 交易所净流出 (正值 = 持有意愿强)
        exchange_netflow = data.get("exchange_netflow_7d", 0)
        # 负流出 = 流入交易所 = 卖压
        holding_intention = self.clip_exposure(-exchange_netflow / 10000, -1, 1)

        # 机构采纳
        institutional_holdings = data.get("institutional_holdings_pct", 0)

        # 综合采纳评分
        adoption_score = 0.0

        # 新地址增长 (0-30分)
        # 10%月增长 = 满分
        new_addr_score = self.clip_exposure(new_addresses_growth / 0.10, 0, 1) * 0.30
        adoption_score += new_addr_score

        # 去中心化程度 (0-25分)
        adoption_score += decentralization_score * 0.25

        # 持有意愿 (0-25分)
        adoption_score += (holding_intention + 1) / 2 * 0.25

        # 机构采纳 (0-20分)
        # 10%机构持仓 = 满分
        inst_score = min(institutional_holdings / 0.10, 1.0) * 0.20
        adoption_score += inst_score

        # 转换
        exposure = self.clip_exposure((adoption_score - 0.5) * 2, -1, 1)

        z_score = self.normalize_to_zscore(adoption_score, mean=0.5, std=0.15)

        # 信号
        if adoption_score > 0.7:
            signal = "HIGH_ADOPTION"
        elif adoption_score > 0.5:
            signal = "GROWING"
        elif adoption_score > 0.3:
            signal = "NEUTRAL"
        else:
            signal = "LOW_ADOPTION"

        return FactorExposure(
            factor_type=FactorType.ADOPTION,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.65,
        )

    def _compute_momentum_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算动量因子暴露

        加密货币动量效应比传统资产更强!
        - Liu & Tsyvinski (2021): 1周动量因子R² > 10%

        时间窗口:
        - 短期: 7天
        - 中期: 30天
        - 长期: 90天
        """
        # 计算多期动量
        if returns is not None and len(returns) >= 7:
            mom_7d = returns.iloc[-7:].sum() if len(returns) >= 7 else 0
            mom_30d = returns.iloc[-30:].sum() if len(returns) >= 30 else 0
            mom_90d = returns.iloc[-90:].sum() if len(returns) >= 90 else 0
        else:
            mom_7d = data.get("return_7d", 0)
            mom_30d = data.get("return_30d", 0)
            mom_90d = data.get("return_90d", 0)

        # 综合动量 (短期权重高)
        # 加密货币短期动量更重要
        combined_momentum = mom_7d * 0.5 + mom_30d * 0.3 + mom_90d * 0.2

        # 转换: -50% -> -1, 0% -> 0, +50% -> +1
        exposure = self.clip_exposure(combined_momentum / 0.50, -1, 1)

        z_score = self.normalize_to_zscore(combined_momentum, mean=0.05, std=0.30)

        # 信号
        if combined_momentum > 0.30:
            signal = "STRONG_MOMENTUM"
        elif combined_momentum > 0.10:
            signal = "LONG"
        elif combined_momentum > -0.10:
            signal = "NEUTRAL"
        elif combined_momentum > -0.30:
            signal = "SHORT"
        else:
            signal = "STRONG_REVERSAL"

        return FactorExposure(
            factor_type=FactorType.MOMENTUM,
            exposure=exposure,
            z_score=z_score,
            percentile=self.zscore_to_percentile(z_score),
            signal=signal,
            confidence=0.75,  # 动量因子置信度较高
        )

    def _compute_crash_risk_exposure(
        self,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> FactorExposure:
        """
        计算崩盘风险因子暴露

        加密货币崩盘风险需要特别关注!

        指标:
        - 历史波动率
        - 最大回撤
        - 尾部风险 (VaR, CVaR)
        - 杠杆水平 (衍生品市场)
        - 清算风险
        """
        # 波动率
        if returns is not None and len(returns) >= 30:
            volatility = returns.iloc[-30:].std() * np.sqrt(365)
        else:
            volatility = data.get("volatility_30d", 0.80)  # 默认80%年化波动

        # 最大回撤
        max_drawdown = data.get("max_drawdown_90d", -0.30)

        # 期货未平仓合约变化 (高杠杆 = 高风险)
        open_interest_change = data.get("open_interest_change_7d", 0)

        # 资金费率 (高资金费率 = 过度杠杆)
        funding_rate = data.get("funding_rate", 0)

        # 清算金额 (高清算 = 高风险)
        liquidation_volume = data.get("liquidation_volume_24h", 0)
        market_cap = data.get("market_cap", 1e9)
        liquidation_ratio = liquidation_volume / market_cap if market_cap > 0 else 0

        # 崩盘风险评分 (越高越危险)
        crash_risk_score = 0.0

        # 波动率贡献 (0-35分)
        # 100%年化波动 = 满分
        vol_score = min(volatility / 1.0, 1.0) * 0.35
        crash_risk_score += vol_score

        # 回撤贡献 (0-25分)
        # -50%回撤 = 满分
        dd_score = min(abs(max_drawdown) / 0.50, 1.0) * 0.25
        crash_risk_score += dd_score

        # 杠杆贡献 (0-20分)
        leverage_score = min(abs(funding_rate) / 0.001, 1.0) * 0.10
        leverage_score += min(abs(open_interest_change) / 0.30, 1.0) * 0.10
        crash_risk_score += leverage_score

        # 清算贡献 (0-20分)
        liq_score = min(liquidation_ratio / 0.01, 1.0) * 0.20
        crash_risk_score += liq_score

        # 转换 (注意: 高风险 = 负暴露 = 避免)
        # 风险分数0.5为中性
        exposure = self.clip_exposure((0.5 - crash_risk_score) * 2, -1, 1)

        z_score = self.normalize_to_zscore(crash_risk_score, mean=0.5, std=0.15)

        # 信号 (反向)
        if crash_risk_score > 0.7:
            signal = "HIGH_RISK_AVOID"
        elif crash_risk_score > 0.5:
            signal = "ELEVATED_RISK"
        elif crash_risk_score > 0.3:
            signal = "MODERATE_RISK"
        else:
            signal = "LOW_RISK"

        return FactorExposure(
            factor_type=FactorType.CRASH_RISK,
            exposure=exposure,
            z_score=-z_score,  # 反向
            percentile=100 - self.zscore_to_percentile(z_score),  # 反向百分位
            signal=signal,
            confidence=0.70,
        )

    def _get_factor_premiums(self) -> Dict[str, float]:
        """
        获取历史因子溢价 (年化)

        基于学术文献:
        - Network: ~8% (网络效应溢价)
        - Adoption: ~5% (采纳增长溢价)
        - Momentum: ~12% (加密货币动量溢价很高!)
        - Crash Risk: ~4% (风险补偿)
        """
        return {
            "network": 0.08,
            "adoption": 0.05,
            "momentum": 0.12,
            "crash_risk": 0.04,
        }

    def get_on_chain_summary(self, symbol: str, data: Dict[str, Any]) -> str:
        """
        生成链上数据分析摘要

        这是加密货币基本面分析的核心信息
        """
        active_addr = data.get("active_addresses", 0)
        tx_volume = data.get("transaction_volume", 0)
        new_addr_growth = data.get("new_addresses_growth_30d", 0)
        whale_conc = data.get("whale_concentration", 0)
        exchange_flow = data.get("exchange_netflow_7d", 0)

        # 网络健康评估
        if active_addr > 1000000:
            network_health = "非常健康 (>100万活跃地址)"
        elif active_addr > 100000:
            network_health = "健康 (>10万活跃地址)"
        elif active_addr > 10000:
            network_health = "一般 (>1万活跃地址)"
        else:
            network_health = "较弱 (<1万活跃地址)"

        # 采纳趋势
        if new_addr_growth > 0.1:
            adoption_trend = "强劲增长 (>10%/月)"
        elif new_addr_growth > 0.05:
            adoption_trend = "稳定增长 (5-10%/月)"
        elif new_addr_growth > 0:
            adoption_trend = "缓慢增长 (<5%/月)"
        else:
            adoption_trend = "负增长 (用户流失)"

        # 持有意愿
        if exchange_flow < -1000:
            holding = "强持有意愿 (大量流出交易所)"
        elif exchange_flow < 0:
            holding = "持有倾向 (净流出交易所)"
        elif exchange_flow > 1000:
            holding = "卖出压力 (大量流入交易所)"
        else:
            holding = "中性 (流入流出平衡)"

        summary = f"""
=== {symbol} 链上数据分析 ===
活跃地址数: {active_addr:,}
交易量: {tx_volume:,.0f}
新地址月增长: {new_addr_growth:.1%}
巨鲸集中度: {whale_conc:.1%}
交易所7日净流量: {exchange_flow:,.0f}

网络健康: {network_health}
采纳趋势: {adoption_trend}
持有意愿: {holding}

注意: 链上数据是加密货币真正的基本面!
      "Price follows on-chain fundamentals."
"""
        return summary

    def get_risk_assessment(
        self,
        symbol: str,
        data: Dict[str, Any],
        returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        获取风险评估报告

        包含:
        - 波动率分析
        - 尾部风险
        - 杠杆风险
        - 流动性风险
        """
        # 波动率
        if returns is not None and len(returns) >= 30:
            vol_30d = returns.iloc[-30:].std() * np.sqrt(365)
            vol_90d = returns.iloc[-90:].std() * np.sqrt(365) if len(returns) >= 90 else vol_30d
        else:
            vol_30d = data.get("volatility_30d", 0.80)
            vol_90d = data.get("volatility_90d", 0.80)

        # VaR (95%)
        if returns is not None and len(returns) >= 30:
            var_95 = np.percentile(returns.iloc[-30:], 5)
        else:
            var_95 = data.get("var_95", -0.10)

        # 最大回撤
        max_dd = data.get("max_drawdown_90d", -0.30)

        # 杠杆指标
        funding_rate = data.get("funding_rate", 0)
        oi_change = data.get("open_interest_change_7d", 0)

        # 风险等级
        risk_score = vol_30d * 0.4 + abs(max_dd) * 0.3 + abs(funding_rate) * 100 * 0.3
        if risk_score > 0.8:
            risk_level = "极高"
        elif risk_score > 0.6:
            risk_level = "高"
        elif risk_score > 0.4:
            risk_level = "中等"
        else:
            risk_level = "较低"

        return {
            "symbol": symbol,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "volatility_30d": vol_30d,
            "volatility_90d": vol_90d,
            "var_95": var_95,
            "max_drawdown_90d": max_dd,
            "funding_rate": funding_rate,
            "open_interest_change_7d": oi_change,
            "recommendation": self._get_risk_recommendation(risk_level),
        }

    def _get_risk_recommendation(self, risk_level: str) -> str:
        """根据风险等级给出建议"""
        recommendations = {
            "极高": "建议极度谨慎，仅适合风险承受能力极强的投资者。考虑减仓或使用期权对冲。",
            "高": "建议控制仓位，设置止损。考虑分批建仓。",
            "中等": "风险可控，建议正常仓位配置。",
            "较低": "相对低风险期，可考虑适度增加配置。",
        }
        return recommendations.get(risk_level, "请根据自身风险承受能力决定。")

    def get_category_analysis(
        self,
        scores: Dict[str, FactorScore]
    ) -> Dict[str, Dict[str, float]]:
        """
        按类别分析加密货币评分

        Args:
            scores: 各币种的因子评分

        Returns:
            各类别的平均分数和建议配置
        """
        category_scores = {}

        for category, symbols in self.CRYPTO_CATEGORIES.items():
            category_sum = 0.0
            count = 0
            for symbol in symbols:
                if symbol in scores:
                    category_sum += scores[symbol].composite_score
                    count += 1
            if count > 0:
                avg_score = category_sum / count
                category_scores[category] = {
                    "average_score": avg_score,
                    "count": count,
                    "signal": self._generate_signal(avg_score),
                }

        return category_scores

    def generate_investment_thesis(
        self,
        symbol: str,
        score: FactorScore,
        data: Dict[str, Any]
    ) -> str:
        """
        生成投资论点

        基于因子分析生成完整的投资论点
        """
        network_exp = score.factor_exposures.get("network")
        adoption_exp = score.factor_exposures.get("adoption")
        momentum_exp = score.factor_exposures.get("momentum")
        crash_risk_exp = score.factor_exposures.get("crash_risk")

        thesis = f"""
=== {symbol} 投资论点 ===

综合评分: {score.composite_score:.2f} ({score.signal})
预期Alpha: {score.expected_alpha:.1%}

因子分析:
1. 网络效应 ({network_exp.signal if network_exp else 'N/A'})
   - 暴露度: {network_exp.exposure:+.2f if network_exp else 'N/A'}
   - Metcalfe定律支持: {'是' if network_exp and network_exp.exposure > 0 else '否'}

2. 采纳趋势 ({adoption_exp.signal if adoption_exp else 'N/A'})
   - 暴露度: {adoption_exp.exposure:+.2f if adoption_exp else 'N/A'}
   - T_t净交易效用: {'正向' if adoption_exp and adoption_exp.exposure > 0 else '负向'}

3. 动量信号 ({momentum_exp.signal if momentum_exp else 'N/A'})
   - 暴露度: {momentum_exp.exposure:+.2f if momentum_exp else 'N/A'}
   - 趋势跟踪: {'适合' if momentum_exp and momentum_exp.exposure > 0.3 else '谨慎'}

4. 崩盘风险 ({crash_risk_exp.signal if crash_risk_exp else 'N/A'})
   - 暴露度: {crash_risk_exp.exposure:+.2f if crash_risk_exp else 'N/A'}
   - 风险调整: {'需要' if crash_risk_exp and crash_risk_exp.exposure < -0.3 else '适中'}

投资建议:
{score.reasoning}

学术参考:
- Biais et al. (2023) JF: 网络效应与T_t决定均衡价格
- Liu & Tsyvinski (2021) RFS: 动量因子显著
"""
        return thesis
