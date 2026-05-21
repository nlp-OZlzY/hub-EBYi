"""
股票波动可视化与买卖时机分析
==================================
功能：
1. 获取日K线和周K线数据
2. 计算日波动率和周波动率
3. 绘制日/周波动综合图
4. 基于波动大小给出买卖建议
"""

import requests
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import argparse

# ============ API 配置 ============
TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1/stock/kline"

# ============ matplotlib 配置 ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fetch_kline_data(code: str, kline_type: str,
                     start_date: str = None,
                     end_date: str = None) -> Optional[List[Dict]]:
    """获取K线数据。"""
    url = f"{BASE_URL}/{kline_type}?token={TOKEN}"

    params = {
        "code": code,
        "type": 1  # 前复权
    }
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("code") == 200 and "data" in data and isinstance(data["data"], list):
            raw_data = data["data"]
            # API returns data as arrays: [date, open, close, high, low, volume]
            # Convert to dict format expected by the rest of the script
            converted = []
            for item in raw_data:
                if isinstance(item, list) and len(item) >= 5:
                    converted.append({
                        "date": item[0],
                        "open": item[1],
                        "close": item[2],
                        "high": item[3],
                        "low": item[4],
                        "volume": item[5] if len(item) > 5 else "0"
                    })
            # Filter by date range if specified
            if start_date:
                converted = [d for d in converted if d["date"] >= start_date]
            if end_date:
                converted = [d for d in converted if d["date"] <= end_date]
            return converted
        else:
            print(f"[错误] 获取{kline_type}K线失败: {data.get('message', '未知错误')}")
            return None
    except Exception as e:
        print(f"[异常] 请求失败: {e}")
        return None


def calc_volatility(kline_list: List[Dict]) -> List[float]:
    """计算每根K线的涨跌幅（%）。"""
    volatility = []
    for k in kline_list:
        open_price = float(k["open"])
        close_price = float(k["close"])
        change_pct = (close_price - open_price) / open_price * 100
        volatility.append(round(change_pct, 2))
    return volatility


def plot_volatility_chart(code: str, name: str,
                          daily_data: List[Dict],
                          weekly_data: List[Dict],
                          output_dir: str = "output") -> str:
    """将日波动和周波动绘制在同一张图上。"""
    daily_dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data]
    daily_vol = calc_volatility(daily_data)

    weekly_dates = [datetime.strptime(w["date"], "%Y-%m-%d") for w in weekly_data]
    weekly_vol = calc_volatility(weekly_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"{name} ({code}) 波动分析图", fontsize=18, fontweight="bold")

    # 上图：日波动
    daily_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in daily_vol]
    ax1.bar(daily_dates, daily_vol, color=daily_colors, width=0.8, alpha=0.85)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("日涨跌幅 (%)", fontsize=12)
    ax1.set_title("日波动 (每日涨跌幅)", fontsize=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.grid(axis="y", alpha=0.3)

    # 下图：周波动
    weekly_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in weekly_vol]
    ax2.bar(weekly_dates, weekly_vol, color=weekly_colors, width=3.5, alpha=0.85)
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("周涨跌幅 (%)", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.set_title("周波动 (每周涨跌幅)", fontsize=14)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{code}_volatility.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[完成] 图表已保存至: {img_path}")
    return img_path


def analyze_signals(daily_data: List[Dict],
                    weekly_data: List[Dict],
                    code: str = "") -> Dict:
    """基于波动大小给出买卖时机建议。"""
    daily_vol = calc_volatility(daily_data)
    weekly_vol = calc_volatility(weekly_data)

    lookback = min(20, len(daily_vol))
    recent_daily = daily_vol[-lookback:]
    vol_mean = np.mean(np.abs(recent_daily))
    vol_std = np.std(np.abs(recent_daily))

    week_lookback = min(10, len(weekly_vol))
    recent_weekly = weekly_vol[-week_lookback:]
    week_vol_mean = np.mean(np.abs(recent_weekly))

    daily_trend = np.mean([v for v in recent_daily[-5:]])
    weekly_trend = np.mean([v for v in recent_weekly[-3:]])

    prices = [float(d["close"]) for d in daily_data]
    current_price = prices[-1]
    high_20 = max(prices[-min(20, len(prices)):])
    low_20 = min(prices[-min(20, len(prices)):])
    price_position = (current_price - low_20) / (high_20 - low_20) * 100

    latest_daily_vol = abs(daily_vol[-1])
    latest_weekly_vol = abs(weekly_vol[-1])

    if vol_std > 0:
        daily_zscore = (latest_daily_vol - vol_mean) / vol_std
    else:
        daily_zscore = 0

    signals = []
    buy_score = 0
    sell_score = 0

    # 买入信号1：波动收缩 + 周趋势向上
    if daily_zscore < -0.8 and weekly_trend > 0:
        signals.append({
            "type": "买入",
            "强度": "强",
            "时间": daily_data[-1]["date"],
            "价格": current_price,
            "理由": (f"日波动收缩至 {latest_daily_vol:.2f}%（低于均值{vol_mean:.2f}%），"
                     f"周趋势向上（{weekly_trend:.2f}%），"
                     f"蓄力后可能向上突破。"
                     f"建议买入价格区间：{round(current_price*0.98,2)} - {round(current_price,2)}")
        })
        buy_score += 3

    # 买入信号2：波动连续收缩 + 价格中低位
    if len(daily_vol) >= 4:
        last_4_vol = [abs(v) for v in daily_vol[-4:-1]]
        if all(last_4_vol[i] > last_4_vol[i+1] for i in range(len(last_4_vol)-1)):
            if price_position < 60:
                signals.append({
                    "type": "买入",
                    "强度": "中",
                    "时间": daily_data[-1]["date"],
                    "价格": current_price,
                    "理由": (f"波动率连续3天收缩，价格处于20日"
                             f"{price_position:.0f}%分位，变盘概率增大。"
                             f"建议关注次日走势，若放量上涨可跟进。")
                })
                buy_score += 2

    # 买入信号3：周波动温和 + 日趋势转正
    if 1.0 < latest_weekly_vol < 3.0 and daily_trend > 0 and price_position < 50:
        signals.append({
            "type": "买入",
            "强度": "中",
            "时间": daily_data[-1]["date"],
            "价格": current_price,
            "理由": (f"周波动温和（{latest_weekly_vol:.2f}%），日趋势转正，"
                     f"价格处于中低位，回调可能结束。")
        })
        buy_score += 2

    # 卖出信号1：日波动异常放大 + 价格高位
    if daily_zscore > 1.5 and price_position > 70:
        signals.append({
            "type": "卖出",
            "强度": "强",
            "时间": daily_data[-1]["date"],
            "价格": current_price,
            "理由": (f"日波动异常放大至 {latest_daily_vol:.2f}%（远超均值{vol_mean:.2f}%），"
                     f"价格处于20日{price_position:.0f}%高位，"
                     f"可能是短期顶部信号。建议减仓或设置止损。")
        })
        sell_score += 3

    # 卖出信号2：周波动放大 + 日趋势转负
    if latest_weekly_vol > week_vol_mean * 1.5 and daily_trend < 0:
        signals.append({
            "type": "卖出",
            "强度": "中",
            "时间": daily_data[-1]["date"],
            "价格": current_price,
            "理由": (f"周波动放大至{latest_weekly_vol:.2f}%（高于均值{week_vol_mean:.2f}%），"
                     f"且日趋势转负（{daily_trend:.2f}%），短期风险上升。")
        })
        sell_score += 2

    # 卖出信号3：连续大涨 + 波动急剧放大
    if len(daily_vol) >= 3:
        last_3 = daily_vol[-3:]
        if all(v > 1.0 for v in last_3) and latest_daily_vol > vol_mean * 2:
            signals.append({
                "type": "卖出",
                "强度": "强",
                "时间": daily_data[-1]["date"],
                "价格": current_price,
                "理由": (f"连续3日大涨且今日波动急剧放大({latest_daily_vol:.2f}%)，"
                         f"短期过热，建议分批止盈。")
            })
            sell_score += 3

    # 波动大小定性评估
    if vol_mean < 0.8:
        daily_vol_level = "偏低（日波动小于0.8%，股价相对平稳）"
    elif vol_mean < 1.5:
        daily_vol_level = "适中（日波动0.8%-1.5%，属于正常范围）"
    elif vol_mean < 2.5:
        daily_vol_level = "偏高（日波动1.5%-2.5%，波动较为明显）"
    else:
        daily_vol_level = "很高（日波动超过2.5%，股价剧烈波动）"

    if week_vol_mean < 2.0:
        weekly_vol_level = "偏低（周波动小于2%，中期趋势平稳）"
    elif week_vol_mean < 4.0:
        weekly_vol_level = "适中（周波动2%-4%，属于正常范围）"
    elif week_vol_mean < 7.0:
        weekly_vol_level = "偏高（周波动4%-7%，中期波动较大）"
    else:
        weekly_vol_level = "很高（周波动超过7%，中期剧烈波动）"

    if buy_score > sell_score:
        suggestion = "整体偏向【买入/持有】"
    elif sell_score > buy_score:
        suggestion = "整体偏向【卖出/减仓】"
    else:
        suggestion = "整体偏向【观望】"

    summary = {
        "股票代码": code,
        "当前价格": current_price,
        "近期日均波动": f"{vol_mean:.2f}%",
        "日波动水平": daily_vol_level,
        "近期周均波动": f"{week_vol_mean:.2f}%",
        "周波动水平": weekly_vol_level,
        "今日波动": f"{daily_vol[-1]:.2f}%",
        "本周波动": f"{weekly_vol[-1]:.2f}%",
        "日趋势方向": "向上" if daily_trend > 0 else "向下",
        "周趋势方向": "向上" if weekly_trend > 0 else "向下",
        "价格20日分位": f"{price_position:.0f}%",
        "买入信号强度": buy_score,
        "卖出信号强度": sell_score,
        "综合建议": suggestion
    }

    return {"signals": signals, "summary": summary}


def print_analysis(signals: List[Dict], summary: Dict):
    """打印分析结果到控制台。"""
    print("\n" + "=" * 60)
    print("  *** 股票波动分析报告 ***")
    print("=" * 60)

    print(f"\n  【基本信息】")
    for key in ["股票代码", "当前价格", "近期日均波动", "日波动水平",
                 "近期周均波动", "周波动水平",
                 "今日波动", "本周波动", "日趋势方向", "周趋势方向",
                 "价格20日分位"]:
        print(f"  - {key}: {summary.get(key, 'N/A')}")

    print(f"\n  【信号强度】")
    print(f"  - 买入信号: {summary['买入信号强度']} 分")
    print(f"  - 卖出信号: {summary['卖出信号强度']} 分")

    print(f"\n  【综合建议】{summary['综合建议']}")

    if signals:
        print(f"\n  【具体信号】")
        for i, sig in enumerate(signals, 1):
            tag = "[买入]" if sig["type"] == "买入" else "[卖出]"
            print(f"\n  {tag} 信号{i} 强度={sig['强度']}")
            print(f"     日期: {sig['时间']}  价格: {sig['价格']}")
            print(f"     理由: {sig['理由']}")
    else:
        print(f"\n  【具体信号】暂无明确交易信号，建议继续观察。")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="股票波动可视化与买卖时机分析"
    )
    parser.add_argument("--code", type=str, required=True,
                        help="股票代码，如 000001")
    parser.add_argument("--name", type=str, default="",
                        help="股票名称，如 平安银行")
    parser.add_argument("--days", type=int, default=90,
                        help="获取最近N天的日K线数据（默认90天）")

    args = parser.parse_args()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"\n正在查询 {args.name}({args.code}) 的数据...")
    print(f"时间范围: {start_date} ~ {end_date}")

    print("\n[1/4] 获取日K线数据...")
    daily_data = fetch_kline_data(args.code, "day", start_date, end_date)
    if not daily_data:
        print("无法获取日K线数据，请检查股票代码是否正确。")
        sys.exit(1)
    print(f"      获取到 {len(daily_data)} 条日K线记录")

    print("\n[2/4] 获取周K线数据...")
    weekly_data = fetch_kline_data(args.code, "week", start_date, end_date)
    if not weekly_data:
        print("无法获取周K线数据。")
        sys.exit(1)
    print(f"      获取到 {len(weekly_data)} 条周K线记录")

    print("\n[3/4] 生成波动可视化图表...")
    name = args.name if args.name else args.code
    plot_volatility_chart(args.code, name, daily_data, weekly_data)

    print("\n[4/4] 分析买卖时机...")
    result = analyze_signals(daily_data, weekly_data, code=args.code)
    print_analysis(result["signals"], result["summary"])

    print(f"分析完毕！图片保存在 output/ 目录下。")


if __name__ == "__main__":
    main()
