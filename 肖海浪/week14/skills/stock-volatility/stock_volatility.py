"""
股票周日波动对比与买卖建议Skill
功能：获取股票周K线和日K线数据，将周波动和日波动绘制在同一图中，基于波动率给出买卖建议
复用autostock API，参考已有stock-visual skill
"""

import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

TOKEN = "zgaLG8unUPr"


# ============ 数据获取（复用autostock API） ============

def get_week_kline(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
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


def get_day_kline(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
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


# ============ 数据解析 ============

def parse_kline_data(data: dict) -> Optional[Dict]:
    """解析K线数据"""
    if not data or "data" not in data:
        return None
    records = data["data"]
    dates, opens, closes, highs, lows, volumes = [], [], [], [], [], []
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


# ============ 波动率计算 ============

def calc_absolute_volatility(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """计算绝对波动率：最高价 - 最低价"""
    return highs - lows


def calc_relative_volatility(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """计算相对波动率：(最高价 - 最低价) / 收盘价 * 100%"""
    return (highs - lows) / closes * 100


def calc_rolling_volatility(closes: np.ndarray, window: int = 5) -> np.ndarray:
    """计算滚动波动率：收盘价的滚动标准差"""
    vol = np.full_like(closes, np.nan)
    for i in range(window - 1, len(closes)):
        vol[i] = np.std(closes[i - window + 1: i + 1])
    return vol


def calc_ma(data: np.ndarray, window: int = 5) -> np.ndarray:
    """计算移动平均线"""
    ma = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        ma[i] = np.mean(data[i - window + 1: i + 1])
    return ma


# ============ 波动率分析与信号生成 ============

def analyze_volatility_signals(
    week_data: Dict, day_data: Dict,
    week_vol: np.ndarray, day_vol: np.ndarray
) -> Dict:
    """分析波动率并生成买卖信号"""

    # 当前波动率
    last_week_vol = week_vol[-1] if not np.isnan(week_vol[-1]) else 0
    last_day_vol = day_vol[-1] if not np.isnan(day_vol[-1]) else 0
    avg_week_vol = np.nanmean(week_vol)
    avg_day_vol = np.nanmean(day_vol)

    # 波动率状态
    week_vol_status = "高" if last_week_vol > avg_week_vol * 1.2 else ("低" if last_week_vol < avg_week_vol * 0.8 else "中")
    day_vol_status = "高" if last_day_vol > avg_day_vol * 1.2 else ("低" if last_day_vol < avg_day_vol * 0.8 else "中")

    # 波动率趋势（最近5期 vs 之前5期）
    def get_vol_trend(vol_arr: np.ndarray) -> str:
        valid = vol_arr[~np.isnan(vol_arr)]
        if len(valid) < 10:
            return "数据不足"
        recent = np.mean(valid[-5:])
        prev = np.mean(valid[-10:-5])
        if recent > prev * 1.15:
            return "扩大"
        elif recent < prev * 0.85:
            return "收敛"
        else:
            return "稳定"

    week_vol_trend = get_vol_trend(week_vol)
    day_vol_trend = get_vol_trend(day_vol)

    # 综合评分
    score = 0

    # 周线波动率状态
    if week_vol_status == "低" and week_vol_trend == "收敛":
        score += 2  # 低波动收敛 → 蓄势，可能买入
    elif week_vol_status == "高" and week_vol_trend == "扩大":
        score -= 1  # 高波动扩大 → 风险

    # 日线波动率状态
    if day_vol_status == "低" and day_vol_trend == "收敛":
        score += 1
    elif day_vol_status == "高" and day_vol_trend == "扩大":
        score -= 1

    # 价格趋势配合
    week_closes = week_data["closes"]
    if len(week_closes) >= 5:
        recent_avg = np.mean(week_closes[-5:])
        prev_avg = np.mean(week_closes[-10:-5]) if len(week_closes) >= 10 else recent_avg
        if recent_avg > prev_avg:
            score += 1  # 价格上升
        elif recent_avg < prev_avg:
            score -= 1  # 价格下降

    # 生成建议
    if score >= 3:
        advice = "强烈建议买入"
        window = "当前波动率收敛，价格温和上升，是较好的买入时机。建议在日线回调时分批建仓。"
    elif score >= 1:
        advice = "建议逢低买入"
        window = "波动率趋于收敛，可关注日线低波动回调的买入机会。"
    elif score <= -3:
        advice = "强烈建议卖出/观望"
        window = "当前波动率扩大且价格走弱，建议减仓观望，等待波动率收敛。"
    elif score <= -1:
        advice = "建议逢高卖出"
        window = "波动率偏高，可在日线反弹时减仓。"
    else:
        advice = "建议观望"
        window = "波动率无明显方向信号，建议等待明确趋势后再操作。"

    return {
        "周线波动率状态": week_vol_status,
        "日线波动率状态": day_vol_status,
        "周线波动率趋势": week_vol_trend,
        "日线波动率趋势": day_vol_trend,
        "周线当前波动率": f"{last_week_vol:.2f}",
        "日线当前波动率": f"{last_day_vol:.2f}",
        "综合建议": advice,
        "操作时间窗口": window,
    }


# ============ 绘图 ============

def plot_volatility_chart(
    code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[Dict]:
    """绘制股票周日波动率对比图并给出买卖建议"""

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

    # 计算波动率
    week_abs_vol = calc_absolute_volatility(week_data["highs"], week_data["lows"])
    day_abs_vol = calc_absolute_volatility(day_data["highs"], day_data["lows"])
    week_rel_vol = calc_relative_volatility(week_data["highs"], week_data["lows"], week_data["closes"])
    day_rel_vol = calc_relative_volatility(day_data["highs"], day_data["lows"], day_data["closes"])
    week_roll_vol = calc_rolling_volatility(week_data["closes"], 5)
    day_roll_vol = calc_rolling_volatility(day_data["closes"], 5)

    # 生成建议
    advice = analyze_volatility_signals(week_data, day_data, week_roll_vol, day_roll_vol)

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2.5, 2, 1.5]})
    fig.suptitle(f"股票 {code} 周日波动率对比分析", fontsize=16, fontweight='bold')

    # ---- 子图1: 相对波动率对比（双Y轴） ----
    ax1 = axes[0]
    ax1.plot(week_data["dates"], week_rel_vol, 'b-', linewidth=2, label='周相对波动率(%)', alpha=0.8)
    ax1.fill_between(week_data["dates"], 0, week_rel_vol, alpha=0.15, color='blue')

    avg_week_rvol = np.nanmean(week_rel_vol)
    ax1.axhline(y=avg_week_rvol, color='blue', linestyle=':', alpha=0.5, label=f'周均值({avg_week_rvol:.1f}%)')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(day_data["dates"], day_rel_vol, 'g-', linewidth=1, label='日相对波动率(%)', alpha=0.6)
    ax1_twin.fill_between(day_data["dates"], 0, day_rel_vol, alpha=0.1, color='green')

    avg_day_rvol = np.nanmean(day_rel_vol)
    ax1_twin.axhline(y=avg_day_rvol, color='green', linestyle=':', alpha=0.5, label=f'日均值({avg_day_rvol:.1f}%)')
    ax1_twin.set_ylabel('日相对波动率(%)', fontsize=10, color='green')

    ax1.set_ylabel('周相对波动率(%)', fontsize=12, color='blue')
    ax1.set_title('相对波动率对比（周 vs 日）', fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- 子图2: 滚动波动率 + 信号标注 ----
    ax2 = axes[1]
    ax2.plot(week_data["dates"], week_roll_vol, 'b-', linewidth=2, label='周滚动波动率')
    ax2.plot(day_data["dates"], day_roll_vol, 'g-', linewidth=1, label='日滚动波动率', alpha=0.7)

    # 标记高波动区间
    avg_week_roll = np.nanmean(week_roll_vol)
    ax2.axhline(y=avg_week_roll, color='blue', linestyle=':', alpha=0.5)
    ax2.fill_between(week_data["dates"], avg_week_roll * 1.2, week_roll_vol,
                     where=week_roll_vol > avg_week_roll * 1.2, alpha=0.3, color='red', label='高波动区')
    ax2.fill_between(week_data["dates"], week_roll_vol, avg_week_roll * 0.8,
                     where=week_roll_vol < avg_week_roll * 0.8, alpha=0.3, color='green', label='低波动区(蓄势)')

    ax2.set_ylabel('波动率', fontsize=12)
    ax2.set_title('滚动波动率与信号区间', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- 子图3: 成交量配合 ----
    ax3 = axes[2]
    colors = ['red' if week_data["closes"][i] >= week_data["opens"][i] else 'green'
              for i in range(len(week_data["dates"]))]
    ax3.bar(week_data["dates"], week_data["volumes"], color=colors, alpha=0.7, width=5)
    ax3.set_ylabel('成交量', fontsize=12)
    ax3.set_title('周成交量（配合波动率判断）', fontsize=12)
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
    print("    波动率分析与买卖建议")
    print("=" * 50)
    for key, value in advice.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    return advice


# ============ 主函数 ============

if __name__ == "__main__":
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    print(f"正在分析股票: {code}")
    advice = plot_volatility_chart(code, save_path="stock_volatility_analysis.png")
