"""
测试 FMP 宏观数据加载器
Test FMP Macro Data Loader

验证:
1. VIX 获取
2. DXY 美元指数
3. 国债收益率曲线
4. Fear & Greed Index
5. 板块表现
6. 商品数据
7. 加密货币数据
8. MarketDataProvider 集成
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# 同时检查 FinCon 的 .env
fincon_env = Path("/Users/guboyang/Desktop/Project/FinCon/.env")
if fincon_env.exists():
    load_dotenv(fincon_env)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_connection():
    """测试 API 连接"""
    print("\n" + "="*60)
    print("1. 测试 API 连接")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()

    # 检查 API Key
    if not loader.api_key:
        print("❌ API Key 未配置")
        return False

    print(f"✅ API Key 已配置: {loader.api_key[:8]}...")

    # 测试连接
    is_connected = loader.check_api_connection()
    if is_connected:
        print("✅ API 连接成功")
        return True
    else:
        print("❌ API 连接失败")
        return False


def test_vix():
    """测试 VIX 获取"""
    print("\n" + "="*60)
    print("2. 测试 VIX 获取")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    vix = loader.get_vix()

    if vix:
        print(f"✅ VIX: {vix:.2f}")

        # VIX 水平解读
        if vix < 15:
            print("   市场情绪: 🟢 低波动 (市场平静)")
        elif vix < 20:
            print("   市场情绪: 🟡 正常波动")
        elif vix < 30:
            print("   市场情绪: 🟠 较高波动 (需关注)")
        elif vix < 40:
            print("   市场情绪: 🔴 高波动 (恐慌)")
        else:
            print("   市场情绪: ⚫ 极端恐慌")

        return True
    else:
        print("❌ VIX 获取失败")
        return False


def test_dxy():
    """测试美元指数获取"""
    print("\n" + "="*60)
    print("3. 测试美元指数 (DXY)")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    dxy = loader.get_dxy()

    if dxy:
        print(f"✅ DXY: {dxy:.2f}")

        # DXY 水平解读
        if dxy > 105:
            print("   美元: 🔴 强势 (利空新兴市场/商品)")
        elif dxy > 100:
            print("   美元: 🟡 中性偏强")
        elif dxy > 95:
            print("   美元: 🟢 中性偏弱")
        else:
            print("   美元: 🟢 弱势 (利好新兴市场/商品)")

        return True
    else:
        print("❌ DXY 获取失败")
        return False


def test_treasury_rates():
    """测试国债收益率"""
    print("\n" + "="*60)
    print("4. 测试国债收益率曲线")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    rates = loader.get_treasury_rates()

    if rates:
        print("✅ 国债收益率曲线:")
        print(f"   1M:  {rates.get('treasury_1m', 0):.2f}%")
        print(f"   3M:  {rates.get('treasury_3m', 0):.2f}%")
        print(f"   6M:  {rates.get('treasury_6m', 0):.2f}%")
        print(f"   1Y:  {rates.get('treasury_1y', 0):.2f}%")
        print(f"   2Y:  {rates.get('treasury_2y', 0):.2f}%")
        print(f"   5Y:  {rates.get('treasury_5y', 0):.2f}%")
        print(f"   10Y: {rates.get('treasury_10y', 0):.2f}%")
        print(f"   30Y: {rates.get('treasury_30y', 0):.2f}%")

        # 收益率曲线分析
        spread_10_2 = rates.get('treasury_10y', 0) - rates.get('treasury_2y', 0)
        print(f"\n   10Y-2Y 利差: {spread_10_2:.2f}%")
        if spread_10_2 < 0:
            print("   ⚠️ 收益率曲线倒挂 (可能预示衰退)")
        elif spread_10_2 < 0.5:
            print("   🟡 收益率曲线平坦")
        else:
            print("   🟢 收益率曲线正常")

        return True
    else:
        print("❌ 国债收益率获取失败")
        return False


def test_fear_greed():
    """测试 Fear & Greed Index"""
    print("\n" + "="*60)
    print("5. 测试 Fear & Greed Index")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    fg = loader.get_fear_greed_index()

    if fg:
        value = fg.get('value', 50)
        classification = fg.get('classification', 'Neutral')
        print(f"✅ Fear & Greed Index: {value:.0f} ({classification})")

        # 解读
        if value < 25:
            print("   🔴 极度恐惧 (可能是买入机会)")
        elif value < 45:
            print("   🟡 恐惧")
        elif value < 55:
            print("   ⚪ 中性")
        elif value < 75:
            print("   🟢 贪婪")
        else:
            print("   🔴 极度贪婪 (可能是卖出信号)")

        return True
    else:
        print("⚠️ Fear & Greed Index 获取失败 (可能需要特定订阅)")
        return False


def test_sector_performance():
    """测试板块表现"""
    print("\n" + "="*60)
    print("6. 测试板块表现")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    sectors = loader.get_sector_performance()

    if sectors:
        print("✅ 板块表现:")
        # 按涨跌幅排序
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
        for sector, change in sorted_sectors:
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            print(f"   {emoji} {sector}: {change:+.2f}%")
        return True
    else:
        print("❌ 板块表现获取失败")
        return False


def test_commodities():
    """测试商品数据"""
    print("\n" + "="*60)
    print("7. 测试商品数据")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    commodities = loader.get_commodities()

    if commodities:
        print("✅ 商品数据:")
        for name, data in commodities.items():
            price = data.get('price', 0)
            change_pct = data.get('change_percent', 0)
            emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
            print(f"   {emoji} {name}: ${price:.2f} ({change_pct:+.2f}%)")
        return True
    else:
        print("❌ 商品数据获取失败")
        return False


def test_crypto():
    """测试加密货币数据"""
    print("\n" + "="*60)
    print("8. 测试加密货币数据")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    crypto = loader.get_crypto_data()

    if crypto:
        print("✅ 加密货币数据:")
        for name, data in crypto.items():
            price = data.get('price', 0)
            change_pct = data.get('change_percent', 0)
            market_cap = data.get('market_cap', 0)
            emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
            print(f"   {emoji} {name.upper()}: ${price:,.2f} ({change_pct:+.2f}%) | MCap: ${market_cap/1e9:.1f}B")
        return True
    else:
        print("❌ 加密货币数据获取失败")
        return False


def test_full_macro_snapshot():
    """测试完整宏观快照"""
    print("\n" + "="*60)
    print("9. 测试完整宏观数据快照")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()
    snapshot = loader.get_full_macro_snapshot()

    if snapshot:
        print("✅ 完整宏观快照:")
        print(f"   VIX: {snapshot.get('vix')}")
        print(f"   DXY: {snapshot.get('dxy')}")
        print(f"   10Y Treasury: {snapshot.get('treasury_10y')}%")
        print(f"   Yield Curve Spread: {snapshot.get('yield_curve_spread')}%")
        print(f"   Fear & Greed: {snapshot.get('fear_greed_value')} ({snapshot.get('fear_greed_class')})")
        print(f"   时间戳: {snapshot.get('timestamp')}")
        return True
    else:
        print("❌ 完整宏观快照获取失败")
        return False


def test_reits_cap_rate():
    """测试 REITs Cap Rate 计算"""
    print("\n" + "="*60)
    print("10. 测试 REITs Cap Rate 计算")
    print("="*60)

    from finsage.data.macro_loader import MacroDataLoader

    loader = MacroDataLoader()

    # 测试单个 REIT Cap Rate
    print("\n测试单个 REIT Cap Rate:")
    test_symbols = ["VNQ", "DLR", "PLD", "O"]
    for symbol in test_symbols:
        cap_rate = loader.get_reit_cap_rate(symbol)
        if cap_rate:
            print(f"   ✅ {symbol}: {cap_rate:.2f}%")
        else:
            print(f"   ⚠️ {symbol}: 获取失败")

    # 测试平均 Cap Rate
    print("\n测试平均 Cap Rate:")
    avg_data = loader.get_reits_average_cap_rate()
    if avg_data:
        print(f"   ✅ 平均 Cap Rate: {avg_data.get('avg_cap_rate'):.2f}%")
        print(f"   样本数量: {avg_data.get('sample_size')}")
        print(f"   各 REIT Cap Rate:")
        for symbol, rate in avg_data.get('individual_rates', {}).items():
            print(f"      {symbol}: {rate:.2f}%")
    else:
        print("   ❌ 平均 Cap Rate 获取失败")
        return False

    # 测试 Cap Rate Spread
    print("\n测试 Cap Rate Spread:")
    spread = loader.get_cap_rate_spread()
    if spread is not None:
        print(f"   ✅ Cap Rate Spread: {spread:.1f} bps")
        if spread > 150:
            print("   🟢 REITs 相对国债有吸引力")
        elif spread > 50:
            print("   🟡 REITs 估值中性")
        else:
            print("   🔴 REITs 估值偏高")
    else:
        print("   ❌ Cap Rate Spread 获取失败")

    # 测试 REITs Expert 数据
    print("\n测试 REITs Expert 数据:")
    reits_data = loader.get_reits_expert_data()
    if reits_data:
        print(f"   ✅ 10Y Treasury: {reits_data.get('treasury_10y')}%")
        print(f"   平均 Cap Rate: {reits_data.get('avg_cap_rate')}%")
        print(f"   Cap Rate Spread: {reits_data.get('cap_rate_spread')} bps")
        print(f"   利率预期: {reits_data.get('rate_expectation')}")
        return True
    else:
        print("   ❌ REITs Expert 数据获取失败")
        return False


def test_market_data_provider_integration():
    """测试 MarketDataProvider 集成"""
    print("\n" + "="*60)
    print("11. 测试 MarketDataProvider 集成")
    print("="*60)

    from finsage.data.market_data import MarketDataProvider

    provider = MarketDataProvider()

    # 获取市场快照
    symbols = ["AAPL", "MSFT", "GOOGL"]
    date = datetime.now().strftime("%Y-%m-%d")

    try:
        snapshot = provider.get_market_snapshot(
            symbols=symbols,
            date=date,
            lookback_days=30,
            include_news=False,
            include_technicals=True,
        )

        print("✅ MarketDataProvider 集成成功:")
        print(f"\n   价格数据:")
        for symbol in symbols:
            if symbol in snapshot:
                data = snapshot[symbol]
                print(f"   {symbol}: ${data.get('close', 0):.2f} ({data.get('change_pct', 0)*100:+.2f}%)")

        print(f"\n   宏观数据 (真实 FMP 数据):")
        macro = snapshot.get('macro', {})
        print(f"   VIX: {macro.get('vix')}")
        print(f"   DXY: {macro.get('dxy')}")
        print(f"   10Y: {macro.get('treasury_10y')}%")
        print(f"   2Y:  {macro.get('treasury_2y')}%")
        print(f"   利差: {macro.get('yield_curve_spread')}%")

        return True

    except Exception as e:
        print(f"❌ MarketDataProvider 集成失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("FinSage 宏观数据模块测试 (FMP API)")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = {}

    # 运行测试
    results['API连接'] = test_api_connection()
    results['VIX'] = test_vix()
    results['DXY美元指数'] = test_dxy()
    results['国债收益率'] = test_treasury_rates()
    results['Fear & Greed'] = test_fear_greed()
    results['板块表现'] = test_sector_performance()
    results['商品数据'] = test_commodities()
    results['加密货币'] = test_crypto()
    results['完整快照'] = test_full_macro_snapshot()
    results['REITs Cap Rate'] = test_reits_cap_rate()
    results['Provider集成'] = test_market_data_provider_integration()

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
