"""
测试 FMP 日内数据获取
Test FMP Intraday Data Loading

验证:
1. API 连接
2. 小时级数据获取
3. 实时报价获取
4. VIX 获取
5. 与 IntradayRiskMonitor 集成
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

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_connection():
    """测试 API 连接"""
    print("\n" + "="*60)
    print("1. 测试 API 连接")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader

    loader = IntradayDataLoader()

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


def test_hourly_data():
    """测试小时级数据获取"""
    print("\n" + "="*60)
    print("2. 测试小时级数据获取")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader

    loader = IntradayDataLoader()

    symbols = ["AAPL", "TSLA", "MSFT"]
    lookback_hours = 48

    print(f"获取 {symbols} 最近 {lookback_hours} 小时的数据...")

    hourly_data = loader.load_hourly_data(symbols, lookback_hours=lookback_hours)

    for symbol, df in hourly_data.items():
        print(f"\n{symbol}:")
        print(f"  数据行数: {len(df)}")
        if not df.empty:
            print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")
            print(f"  列: {list(df.columns)}")
            print(f"  最新价格: ${df['Close'].iloc[-1]:.2f}")
            print(f"  最近5行:")
            print(df.tail().to_string())

    return len(hourly_data) > 0


def test_minute_data():
    """测试分钟级数据获取"""
    print("\n" + "="*60)
    print("3. 测试分钟级数据获取 (5分钟)")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader

    loader = IntradayDataLoader()

    symbols = ["AAPL"]

    print(f"获取 {symbols} 5分钟级数据...")

    minute_data = loader.load_minute_data(symbols, interval="5m", lookback_periods=50)

    for symbol, df in minute_data.items():
        print(f"\n{symbol}:")
        print(f"  数据行数: {len(df)}")
        if not df.empty:
            print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")
            print(f"  最近5行:")
            print(df.tail().to_string())

    return len(minute_data) > 0


def test_realtime_quote():
    """测试实时报价获取"""
    print("\n" + "="*60)
    print("4. 测试实时报价获取")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader

    loader = IntradayDataLoader()

    # 单个报价
    print("\n获取 AAPL 实时报价...")
    quote = loader.get_realtime_quote("AAPL")

    if quote:
        print(f"  股票: {quote.get('symbol')}")
        price = quote.get('price') or 0
        change = quote.get('change') or 0
        change_pct = quote.get('change_percent') or 0
        open_price = quote.get('open') or 0
        high = quote.get('high') or 0
        low = quote.get('low') or 0
        volume = quote.get('volume') or 0
        print(f"  价格: ${price:.2f}")
        print(f"  涨跌: {change:+.2f} ({change_pct:+.2f}%)")
        print(f"  今日开盘: ${open_price:.2f}")
        print(f"  今日最高: ${high:.2f}")
        print(f"  今日最低: ${low:.2f}")
        print(f"  成交量: {int(volume):,}")
    else:
        print("  ❌ 获取失败")

    # 批量报价
    print("\n获取批量报价 (AAPL, TSLA, MSFT, NVDA)...")
    symbols = ["AAPL", "TSLA", "MSFT", "NVDA"]
    snapshots = loader.get_realtime_snapshot(symbols)

    for symbol, data in snapshots.items():
        price = data.get('price') or 0
        change_pct = data.get('change_percent') or 0
        print(f"  {symbol}: ${price:.2f} ({change_pct:+.2f}%)")

    return quote is not None


def test_vix_level():
    """测试 VIX 获取"""
    print("\n" + "="*60)
    print("5. 测试 VIX 水平获取")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader

    loader = IntradayDataLoader()

    vix = loader.get_vix_level()

    if vix:
        print(f"  当前 VIX: {vix:.2f}")

        # VIX 水平解读
        if vix < 15:
            print("  市场情绪: 🟢 低波动 (市场平静)")
        elif vix < 20:
            print("  市场情绪: 🟡 正常波动")
        elif vix < 30:
            print("  市场情绪: 🟠 较高波动 (需关注)")
        elif vix < 40:
            print("  市场情绪: 🔴 高波动 (恐慌)")
        else:
            print("  市场情绪: ⚫ 极端恐慌")

        return True
    else:
        print("  ❌ VIX 获取失败 (可能是周末或非交易时段)")
        return False


def test_market_status():
    """测试市场状态获取"""
    print("\n" + "="*60)
    print("6. 测试市场状态获取")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader

    loader = IntradayDataLoader()

    status = loader.get_market_hours_status()

    if status:
        print(f"  市场状态: {status}")
        return True
    else:
        print("  ❌ 获取市场状态失败")
        return False


def test_risk_monitor_integration():
    """测试与 IntradayRiskMonitor 集成"""
    print("\n" + "="*60)
    print("7. 测试 IntradayRiskMonitor 集成")
    print("="*60)

    from finsage.data.intraday_loader import IntradayDataLoader
    from finsage.risk.intraday_monitor import IntradayRiskMonitor, AlertLevel

    loader = IntradayDataLoader()
    monitor = IntradayRiskMonitor()

    # 获取数据
    symbols = ["AAPL", "TSLA", "MSFT"]
    hourly_data = loader.load_hourly_data(symbols, lookback_hours=48)
    vix = loader.get_vix_level() or 20.0

    # 模拟持仓
    current_holdings = {"AAPL": 0.4, "TSLA": 0.3, "MSFT": 0.3}
    portfolio_value = 100000

    print(f"当前持仓: {current_holdings}")
    print(f"组合价值: ${portfolio_value:,}")
    print(f"VIX: {vix:.2f}")

    # 运行监控
    report = monitor.monitor(
        hourly_data=hourly_data,
        current_holdings=current_holdings,
        portfolio_value=portfolio_value,
        vix_level=vix
    )

    print(f"\n风险监控报告:")
    print(f"  监控时间: {report.timestamp}")
    print(f"  最高警报级别: {report.overall_level.value}")
    print(f"  警报数量: {len(report.alerts)}")

    if report.alerts:
        print(f"\n  警报详情:")
        for alert in report.alerts[:5]:  # 只显示前5个
            print(f"    [{alert.alert_level.value}] {alert.alert_type.value}: {alert.description}")

    print(f"\n  建议动作:")
    for action in report.recommended_actions:
        if isinstance(action, dict):
            print(f"    - {action.get('action', 'N/A')}: {action.get('reason', '')}")
        else:
            print(f"    - {action}")

    print(f"\n  风险指标:")
    for key, value in report.portfolio_metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return report.overall_level is not None


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("FinSage 日内数据模块测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = {}

    # 运行测试
    results['API连接'] = test_api_connection()
    results['小时级数据'] = test_hourly_data()
    results['分钟级数据'] = test_minute_data()
    results['实时报价'] = test_realtime_quote()
    results['VIX水平'] = test_vix_level()
    results['市场状态'] = test_market_status()
    results['风险监控集成'] = test_risk_monitor_integration()

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
