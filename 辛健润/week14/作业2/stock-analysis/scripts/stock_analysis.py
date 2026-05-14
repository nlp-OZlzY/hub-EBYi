#!/usr/bin/env python3
"""
股票分析 Skill - Claude Code 实现
可通过 /stock-analysis 命令调用
"""

# 解决 OpenMP 库冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_volatility(prices: pd.Series, window: int) -> pd.Series:
    """计算波动率"""
    returns = prices.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility


def get_stock_data(symbol: str, period: str = "3mo"):
    """获取股票数据并计算波动率"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)

        if df.empty:
            return {"error": f"无法获取股票 {symbol} 的数据"}

        # 计算波动率
        df['Daily_Volatility'] = calculate_volatility(df['Close'], window=5)
        df['Weekly_Volatility'] = calculate_volatility(df['Close'], window=20)

        info = stock.info
        return {
            "data": df,
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "currency": info.get('currency', 'USD'),
            "current_price": df['Close'].iloc[-1],
            "price_change": df['Close'].iloc[-1] - df['Close'].iloc[0],
            "price_change_pct": ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100,
        }
    except Exception as e:
        return {"error": f"获取数据失败: {str(e)}"}


def plot_stock_analysis(stock_data, save_path="stock_analysis.png"):
    """绘制股票分析图表"""
    if "error" in stock_data:
        return stock_data["error"]

    df = stock_data["data"]
    symbol = stock_data["symbol"]
    name = stock_data["name"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{name} ({symbol}) - 股票分析', fontsize=16, fontweight='bold')

    # 1. 价格走势
    ax1.plot(df.index, df['Close'], color='#2E86AB', linewidth=2, label='收盘价')
    ax1.fill_between(df.index, df['Low'], df['High'], alpha=0.3, color='#A23B72', label='日内波动')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.set_title('价格走势', fontsize=14, pad=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 添加价格变化
    price_change = stock_data["price_change"]
    price_change_pct = stock_data["price_change_pct"]
    color = 'red' if price_change >= 0 else 'green'
    ax1.text(0.02, 0.95, f'变化: {price_change:+.2f} ({price_change_pct:+.2f}%)',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # 2. 成交量
    colors = ['red' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'green'
              for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_title('成交量', fontsize=14, pad=10)
    ax2.grid(True, alpha=0.3)

    # 3. 波动率
    ax3.plot(df.index, df['Daily_Volatility'], color='#F18F01', linewidth=2,
             label='日波动率 (5日)', marker='o', markersize=3)
    ax3.plot(df.index, df['Weekly_Volatility'], color='#6A994E', linewidth=2,
             label='周波动率 (20日)', marker='s', markersize=3)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_ylabel('波动率', fontsize=12)
    ax3.set_title('波动率分析', fontsize=14, pad=10)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 格式化日期
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def analyze_trading_signals(stock_data):
    """分析买卖信号"""
    if "error" in stock_data:
        return stock_data

    df = stock_data["data"].dropna()

    # 计算趋势
    recent_days = 10
    if len(df) >= recent_days:
        recent_prices = df['Close'].tail(recent_days)
        price_trend = "上涨" if recent_prices.iloc[-1] > recent_prices.iloc[0] else "下跌"
        trend_strength = abs((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
    else:
        price_trend = "数据不足"
        trend_strength = 0

    # 买卖信号
    df['Buy_Score'] = (1 / (df['Daily_Volatility'] + 1e-6)) * (1 / (df['Close'] + 1e-6))
    best_buy_idx = df['Buy_Score'].idxmax()
    best_buy_date = best_buy_idx
    best_buy_price = df.loc[best_buy_idx, 'Close']

    df['Sell_Score'] = df['Close'] * (1 + df['Daily_Volatility'])
    best_sell_idx = df['Sell_Score'].idxmax()
    best_sell_date = best_sell_idx
    best_sell_price = df.loc[best_sell_idx, 'Close']

    # 低/高波动期
    daily_vol_median = df['Daily_Volatility'].median()
    low_volatility_periods = df[df['Daily_Volatility'] < daily_vol_median * 0.8]
    high_volatility_periods = df[df['Daily_Volatility'] > df['Daily_Volatility'].quantile(0.75)]

    return {
        "symbol": stock_data["symbol"],
        "current_price": df['Close'].iloc[-1],
        "current_volatility": df['Daily_Volatility'].iloc[-1],
        "price_trend": price_trend,
        "trend_strength": trend_strength,
        "low_volatility_days": len(low_volatility_periods),
        "high_volatility_days": len(high_volatility_periods),
        "best_buy_date": best_buy_date.strftime('%Y-%m-%d'),
        "best_buy_price": best_buy_price,
        "best_sell_date": best_sell_date.strftime('%Y-%m-%d'),
        "best_sell_price": best_sell_price,
        "potential_profit": best_sell_price - best_buy_price,
        "potential_profit_pct": ((best_sell_price - best_buy_price) / best_buy_price) * 100
    }


def format_analysis_report(stock_data, signals, chart_path):
    """格式化分析报告"""
    report = f"""
{'='*60}
📊 {stock_data['name']} ({stock_data['symbol']}) 分析报告
{'='*60}

【当前状态】
💰 当前价格: {signals['current_price']:.2f} {stock_data['currency']}
📈 价格趋势: {signals['price_trend']} (强度: {signals['trend_strength']:.2f}%)
📊 当前波动率: {signals['current_volatility']:.4f}

【波动分析】
🟢 低波动天数: {signals['low_volatility_days']} 天 (适合建仓)
🔴 高波动天数: {signals['high_volatility_days']} 天 (风险较高)

【历史最佳时机】
🟢 最佳买入时间: {signals['best_buy_date']}
   └─ 买入价格: {signals['best_buy_price']:.2f}

🔴 最佳卖出时间: {signals['best_sell_date']}
   └─ 卖出价格: {signals['best_sell_price']:.2f}

💰 潜在收益: {signals['potential_profit']:+.2f} ({signals['potential_profit_pct']:+.2f}%)

📁 可视化图表已保存: {chart_path}
{'='*60}
"""
    return report


def main():
    """主函数：命令行接口"""
    if len(sys.argv) < 2:
        print("用法: stock_analysis.py <股票代码> [周期]")
        print("示例: stock_analysis.py AAPL 3mo")
        sys.exit(1)

    symbol = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "3mo"

    print(f"\n{'='*60}")
    print(f"📊 开始分析股票: {symbol}")
    print(f"{'='*60}\n")

    # 获取数据
    print("📈 正在获取股票数据...")
    stock_data = get_stock_data(symbol, period)

    if "error" in stock_data:
        print(f"❌ 错误: {stock_data['error']}")
        sys.exit(1)

    print(f"✅ 数据获取成功: {stock_data['name']}")
    print(f"   当前价格: {stock_data['current_price']:.2f} {stock_data['currency']}")
    print(f"   周期涨跌: {stock_data['price_change']:+.2f} ({stock_data['price_change_pct']:+.2f}%)\n")

    # 绘制图表
    print("🎨 正在生成可视化图表...")
    chart_path = plot_stock_analysis(stock_data, f"stock_analysis_{symbol}.png")
    print(f"✅ 图表已保存: {chart_path}\n")

    # 分析信号
    print("🔍 正在分析交易信号...")
    signals = analyze_trading_signals(stock_data)

    # 输出报告
    report = format_analysis_report(stock_data, signals, chart_path)
    print(report)

    print("\n⚠️  免责声明：本分析仅供参考，不构成投资建议。股市有风险，投资需谨慎。\n")


if __name__ == "__main__":
    main()
