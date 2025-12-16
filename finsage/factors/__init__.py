"""
FinSage Factor Modules
因子评分模块

基于学术文献的资产类别内因子选择:
- stock_factors: Fama-French五因子 (JFE 2015)
- bond_factors: 债券四因子 (FAJ 2017, JPM 2018)
- commodity_factors: 商品三因子 (JPM 2013)
- reits_factors: REITs特质风险 (RFS 2021)
- crypto_factors: 加密货币基本面 (JF 2023)
"""

from finsage.factors.base_factor import BaseFactorScorer, FactorScore
from finsage.factors.stock_factors import StockFactorScorer
from finsage.factors.bond_factors import BondFactorScorer
from finsage.factors.commodity_factors import CommodityFactorScorer
from finsage.factors.reits_factors import REITsFactorScorer
from finsage.factors.crypto_factors import CryptoFactorScorer

__all__ = [
    "BaseFactorScorer",
    "FactorScore",
    "StockFactorScorer",
    "BondFactorScorer",
    "CommodityFactorScorer",
    "REITsFactorScorer",
    "CryptoFactorScorer",
]
