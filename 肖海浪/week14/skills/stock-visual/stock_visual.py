"""
股票波动可视化与买卖建议Skill
功能：获取股票周K线和日K线数据，绘制综合波动图，给出买卖建议
"""

import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from typing import Optional

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

TOKEN = "zgaLG8unUPr"


def get_week_kline(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """获取周K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/week"
    params = {"token": TOKEN, "code": code, "type": 0}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        print(f"获取周K线失败: {e}")
        return {}


def get_day_kline(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """获取日K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/day"
    params = {"token": TOKEN, "code": code, "type": 0}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        print(f"获取日K线失败: {e}")
        return {}


def parse_kline_data(data: dict):
    """解析K线数据，返回日期、开盘、收盘、最高、最低、成交量"""
    if not data or "data" not in data:
        return None
    records = data["data"]
    dates = []
    opens = []
    closes = []
    highs = []
    lows = []
    volumes = []
    for r in records:
        dates.append(datetime.strptime(r["date"], "%Y-%m-%d"))
        opens.append(float(r.get("open", 0)))
        closes.append(float(r.get("close", 0)))
        highs.append(float(r.get("high", 0)))
        lows.append(float(r.get("low", 0)))
        volumes.append(float(r.get("volume", 0)))
    return {
        "dates": dates,
        "opens": np.array(opens),
        "closes": np.array(closes),
        "highs": np.array(highs),
        "lows": np.array(lows),
        "volumes": np.array(volumes),
    }


def calc_ma(data: np.ndarray, window: int = 5):
    """计算移动平均线"""
    ma = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        ma[i] = np.mean(data[i - window + 1: i + 1])
    return ma


def calc_volatility(closes: np.ndarray, window: int = 5):
    """计算滚动波动率（标准差）"""
    vol = np.full_like(closes, np.nan)
    for i in range(window - 1, len(closes)):
        vol[i] = np.std(closes[i - window + 1: i + 1])
    return vol


def detect_signals(ma_short: np.ndarray, ma_long: np.ndarray):
    """检测金叉死叉信号"""
    signals = []
    for i in range(1, len(ma_short)):
        if np.isnan(ma_short[i]) or np.isnan(ma_long[i]):
            continue
        if np.isnan(ma_short[i - 1]) or np.isnan(ma_long[i - 1]):
            continue
        # 金叉：短均线从下方穿越长均线
        if ma_short[i - 1] < ma_long[i - 1] and ma_short[i] > ma_long[i]:
            signals.append((i, "golden"))
        # 死叉：短均线从上方穿越长均线
        elif ma_short[i - 1] > ma_long[i - 1] and ma_short[i] < ma_long[i]:
            signals.append((i, "death"))
    return signals


def analyze_trend(closes: np.ndarray, ma_short: np.ndarray, ma_long: np.ndarray):
    """分析当前趋势"""
    if len(closes) < 2:
        return "数据不足"
    last_close = closes[-1]
    last_ma_short = ma_short[-1] if not np.isnan(ma_short[-1]) else 0
    last_ma_long = ma_long[-1] if not np.isnan(ma_long[-1]) else 0

    if last_close > last_ma_short > last_ma_long:
        return "上升趋势"
    elif last_close < last_ma_short < last_ma_long:
        return "下降趋势"
    else:
        return "震荡整理"


def generate_advice(week_data, day_data):
    """生成综合买卖建议"""
    week_closes = week_data["closes"]
    day_closes = day_data["closes"]

    # 周线分析
    week_ma5 = calc_ma(week_closes, 5)
    week_ma20 = calc_ma(week_closes, 20)
    week_signals = detect_signals(week_ma5, week_ma20)
    week_trend = analyze_trend(week_closes, week_ma5, week_ma20)
    week_vol = calc_volatility(week_closes, 5)

    # 日线分析
    day_ma5 = calc_ma(day_closes, 5)
    day_ma20 = calc_ma(day_closes, 20)
    day_signals = detect_signals(day_ma5, day_ma20)
    day_trend = analyze_trend(day_closes, day_ma5, day_ma20)
    day_vol = calc_volatility(day_closes, 5)

    # 当前状态
    last_week_vol = week_vol[-1] if not np.isnan(week_vol[-1]) else 0
    last_day_vol = day_vol[-1] if not np.isnan(day_vol[-1]) else 0
    avg_week_vol = np.nanmean(week_vol)
    avg_day_vol = np.nanmean(day_vol)

    advice = {
        "周线趋势": week_trend,
        "日线趋势": day_trend,
        "周线波动率": "高" if last_week_vol > avg_week_vol * 1.2 else ("低" if last_week_vol < avg_week_vol * 0.8 else "中"),
        "日线波动率": "高" if last_day_vol > avg_day_vol * 1.2 else ("低" if last_day_vol < avg_day_vol * 0.8 else "中"),
        "最近周线信号": None,
        "最近日线信号": None,
    }

    # 最近信号
    if week_signals:
        idx, sig = week_signals[-1]
        advice["最近周线信号"] = f"{'金叉(买入信号)' if sig == 'golden' else '死叉(卖出信号)'}，日期: {week_data['dates'][idx].strftime('%Y-%m-%d')}"
    if day_signals:
        idx, sig = day_signals[-1]
        advice["最近日线信号"] = f"{'金叉(买入信号)' if sig == 'golden' else '死叉(卖出信号)'}，日期: {day_data['dates'][idx].strftime('%Y-%m-%d')}"

    # 综合建议
    buy_score = 0
    if week_trend == "上升趋势":
        buy_score += 2
    elif week_trend == "下降趋势":
        buy_score -= 2

    if day_trend == "上升趋势":
        buy_score += 1
    elif day_trend == "下降趋势":
        buy_score -= 1

    # 最近信号加权
    if week_signals:
        _, last_sig = week_signals[-1]
        if last_sig == "golden":
            buy_score += 2
        else:
            buy_score -= 2
    if day_signals:
        _, last_sig = day_signals[-1]
        if last_sig == "golden":
            buy_score += 1
        else:
            buy_score -= 1

    if buy_score >= 3:
        advice["综合建议"] = "强烈建议买入"
        advice["操作窗口"] = "当前为较好的买入时机，建议在日线回调至支撑位时分批建仓"
    elif buy_score >= 1:
        advice["综合建议"] = "建议逢低买入"
        advice["操作窗口"] = "趋势偏多，可在日线回调时适量买入，注意设置止损"
    elif buy_score <= -3:
        advice["综合建议"] = "强烈建议卖出/观望"
        advice["操作窗口"] = "当前趋势偏空，建议减仓或观望，等待趋势反转信号"
    elif buy_score <= -1:
        advice["综合建议"] = "建议逢高卖出"
        advice["操作窗口"] = "趋势偏弱，可在日线反弹时减仓，不宜追涨"
    else:
        advice["综合建议"] = "建议观望"
        advice["操作窗口"] = "当前处于震荡区间，建议等待明确信号后再操作"

    return advice, week_signals, day_signals


def plot_stock_chart(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None, save_path: Optional[str] = None):
    """绘制股票周波动+日波动综合图"""
    print(f"正在获取 {code} 的K线数据...")

    # 获取数据
    week_raw = get_week_kline(code, start_date, end_date)
    day_raw = get_day_kline(code, start_date, end_date)

    if not week_raw or not day_raw:
        print("获取数据失败，请检查股票代码是否正确")
        return None

    week_data = parse_kline_data(week_raw)
    day_data = parse_kline_data(day_raw)

    if not week_data or not day_data:
        print("解析数据失败")
        return None

    print(f"周K线数据: {len(week_data['dates'])} 条")
    print(f"日K线数据: {len(day_data['dates'])} 条")

    # 计算指标
    week_ma5 = calc_ma(week_data["closes"], 5)
    week_ma20 = calc_ma(week_data["closes"], 20)
    week_vol = calc_volatility(week_data["closes"], 5)

    day_ma5 = calc_ma(day_data["closes"], 5)
    day_ma20 = calc_ma(day_data["closes"], 20)
    day_vol = calc_volatility(day_data["closes"], 5)

    # 生成建议
    advice, week_signals, day_signals = generate_advice(week_data, day_data)

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1.5, 1]})
    fig.suptitle(f"股票 {code} 周波动与日波动综合分析", fontsize=16, fontweight='bold')

    # ---- 子图1: 价格走势 + 均线 ----
    ax1 = axes[0]
    # 周K线收盘价
    ax1.plot(week_data["dates"], week_data["closes"], 'b-', linewidth=2, label='周K收盘价', alpha=0.8)
    ax1.plot(week_data["dates"], week_ma5, 'orange', linewidth=1.5, label='周MA5', linestyle='--')
    ax1.plot(week_data["dates"], week_ma20, 'red', linewidth=1.5, label='周MA20', linestyle='--')

    # 日K线收盘价（半透明叠加）
    ax1_twin = ax1.twinx()
    ax1_twin.plot(day_data["dates"], day_data["closes"], 'g-', linewidth=1, label='日K收盘价', alpha=0.5)
    ax1_twin.set_ylabel('日K价格', fontsize=10, color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')

    # 标注金叉死叉
    for idx, sig in week_signals:
        color = 'red' if sig == 'golden' else 'green'
        marker = '^' if sig == 'golden' else 'v'
        ax1.scatter(week_data["dates"][idx], week_data["closes"][idx],
                    color=color, marker=marker, s=150, zorder=5,
                    label=f'周线{"金叉" if sig == "golden" else "死叉"}' if idx == week_signals[-1][0] else "")

    ax1.set_ylabel('周K价格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势与均线分析', fontsize=12)

    # ---- 子图2: 波动率对比 ----
    ax2 = axes[1]
    ax2.plot(week_data["dates"], week_vol, 'b-', linewidth=2, label='周波动率')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(day_data["dates"], day_vol, 'g-', linewidth=1.5, label='日波动率', alpha=0.7)
    ax2_twin.set_ylabel('日波动率', fontsize=10, color='green')

    # 标记高波动区间
    avg_week_vol = np.nanmean(week_vol)
    ax2.axhline(y=avg_week_vol, color='blue', linestyle=':', alpha=0.5, label='周均值')
    ax2.fill_between(week_data["dates"], avg_week_vol * 1.2, week_vol,
                     where=week_vol > avg_week_vol * 1.2, alpha=0.3, color='red', label='高波动区')

    ax2.set_ylabel('周波动率', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('波动率对比分析', fontsize=12)

    # ---- 子图3: 成交量 ----
    ax3 = axes[2]
    colors = ['red' if week_data["closes"][i] >= week_data["opens"][i] else 'green'
              for i in range(len(week_data["dates"]))]
    ax3.bar(week_data["dates"], week_data["volumes"], color=colors, alpha=0.7, width=5)
    ax3.set_ylabel('成交量', fontsize=12)
    ax3.set_title('周成交量', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()

    plt.close()

    # 打印建议
    print("\n" + "=" * 50)
    print("    技术分析与买卖建议")
    print("=" * 50)
    for key, value in advice.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    return advice


if __name__ == "__main__":
    # 示例：分析贵州茅台（600519）
    code = "600519"
    print(f"正在分析股票: {code}")
    advice = plot_stock_chart(code, save_path="stock_analysis.png")
