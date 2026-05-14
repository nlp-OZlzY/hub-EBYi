
"""
股票技术分析 Skill
name: 股票信息可视化，股票信息和股价查询、生成买卖建议
description: 通过调用第三方接口 https://api.autostock.cn 获取股票、指数、板块、K线等数据，绘制股票信息可视化图并生成买卖建议
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, Dict

# ==================== 配置 ====================
TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn"


# ==================== API 调用函数 ====================
def get_day_line(code: str, startDate: Optional[str] = None, endDate: Optional[str] = None, type: int = 1) -> Dict:
    """获取日K线数据"""
    url = f"{BASE_URL}/v1/stock/kline/day"
    params = {"token": TOKEN, "code": code, "type": type}
    if startDate:
        params["startDate"] = startDate
    if endDate:
        params["endDate"] = endDate
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"get_day_line 异常: {e}")
        return {}

def get_week_line(code: str, startDate: Optional[str] = None, endDate: Optional[str] = None, type: int = 1) -> Dict:
    """获取周K线数据"""
    url = f"{BASE_URL}/v1/stock/kline/week"
    params = {"token": TOKEN, "code": code, "type": type}
    if startDate:
        params["startDate"] = startDate
    if endDate:
        params["endDate"] = endDate
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"get_week_line 异常: {e}")
        return {}


def get_stock_info(code: str) -> Dict:
    """获取股票基础信息"""
    url = f"{BASE_URL}/v1/stock"
    params = {"token": TOKEN, "code": code}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"get_stock_info 异常: {e}")
        return {}


# ==================== 数据处理 ====================
def to_dataframe(data_dict: Dict) -> pd.DataFrame:
    """将API返回的JSON转为DataFrame，支持列表格式和字典格式，自动处理列数不一致"""
    if not data_dict:
        print("警告: API返回空数据")
        return pd.DataFrame()

    if "code" in data_dict and data_dict["code"] != 200:
        print(f"API错误: {data_dict.get('code')} - {data_dict.get('message', '未知')}")
        return pd.DataFrame()

    data = data_dict.get("data")
    if not isinstance(data, list) or len(data) == 0:
        print("警告: API返回的data字段为空或不是列表")
        return pd.DataFrame()

    first = data[0]
    if isinstance(first, dict):
        # 字典格式，直接转换
        df = pd.DataFrame(data)
        date_col = None
        for col in ["date", "trade_date", "day", "dt"]:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            print(f"错误: 字典格式中无日期列，实际列: {df.columns.tolist()}")
            return pd.DataFrame()
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})
    elif isinstance(first, list):
        # 列表格式，获取所有行的最大列数，确保列数统一
        num_cols = max(len(row) for row in data)
        # 根据常见的列数设置列名（优先支持6列或7列）
        if num_cols == 6:
            columns = ["date", "open", "high", "low", "close", "volume"]
        elif num_cols == 7:
            columns = ["date", "open", "high", "low", "close", "volume", "amount"]
        else:
            columns = [f"col_{i}" for i in range(num_cols)]
            print(f"注意: 使用通用列名，列数={num_cols}")
        # 截取或填充每行至相同列数（防止某行少列）
        aligned_data = []
        for row in data:
            if len(row) < num_cols:
                row = list(row) + [None] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:num_cols]
            aligned_data.append(row)
        df = pd.DataFrame(aligned_data, columns=columns)
        # 确保日期列命名为'date'
        if "date" not in df.columns and len(columns) > 0:
            df = df.rename(columns={columns[0]: "date"})
    else:
        print(f"错误: 未知数据类型 {type(first)}")
        return pd.DataFrame()

    # 转换日期列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
    else:
        print("错误: 处理后仍未找到日期列")
        return pd.DataFrame()

    # 确保数值列为浮点数
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def calc_ma(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window=window).mean()


def simple_trend_analysis(day_df: pd.DataFrame, week_df: pd.DataFrame) -> Dict:
    """多周期技术分析"""
    if len(week_df) < 10 or len(day_df) < 20:
        return {"trend": "sideways", "support": False, "resistance": False, "signal": "none", "volume_trend": "normal"}

    # 周线趋势
    week_ma20 = calc_ma(week_df, 20)
    last_week_close = week_df["close"].iloc[-1]
    last_week_ma20 = week_ma20.iloc[-1]
    if last_week_close > last_week_ma20 and week_df["close"].iloc[-5:].mean() > week_ma20.iloc[-5:].mean():
        trend = "up"
    elif last_week_close < last_week_ma20:
        trend = "down"
    else:
        trend = "sideways"

    # 支撑/压力
    support = abs(last_week_close - last_week_ma20) / last_week_ma20 <= 0.02 if last_week_ma20 != 0 else False
    recent_high = week_df["high"].iloc[-10:-1].max()
    resistance = last_week_close >= recent_high * 0.98 if recent_high else False

    # 日线信号
    last = day_df.iloc[-1]
    prev = day_df.iloc[-2]
    signal = "none"
    if prev["close"] < prev["open"] and last["close"] > last["open"] and last["close"] > prev["open"]:
        signal = "bullish_engulf"
    elif prev["close"] > prev["open"] and last["close"] < last["open"] and last["close"] < prev["open"]:
        signal = "bearish_engulf"
    else:
        body = abs(last["close"] - last["open"])
        lower_shadow = min(last["open"], last["close"]) - last["low"]
        if lower_shadow > 2 * body and last["close"] > last["open"]:
            signal = "hammer"

    # 成交量
    vol_avg5 = day_df["volume"].tail(5).mean()
    vol_avg10 = day_df["volume"].tail(10).mean()
    volume_trend = "volume_up" if vol_avg5 > 1.3 * vol_avg10 else (
        "volume_down" if vol_avg5 < 0.7 * vol_avg10 else "normal")

    return {"trend": trend, "support": support, "resistance": resistance, "signal": signal,
            "volume_trend": volume_trend}


def generate_advice(analysis: Dict, latest_price: float) -> Dict:
    trend = analysis["trend"]
    support = analysis["support"]
    resistance = analysis["resistance"]
    signal = analysis["signal"]
    volume = analysis["volume_trend"]

    action = "hold"
    reason = "无明显交易信号。"
    stop_loss = None

    if trend == "up" and support and signal in ["bullish_engulf", "hammer"]:
        action = "buy"
        reason = f"周线上升趋势且回调至支撑位，日线出现{signal}形态，成交量{volume}，建议买入。"
        stop_loss = latest_price * 0.95
    elif trend == "down" and signal in ["bearish_engulf"]:
        action = "sell"
        reason = f"周线下降趋势，日线出现{signal}形态，建议卖出或减仓。"
        stop_loss = latest_price * 1.05
    elif trend == "up" and resistance:
        action = "hold"
        reason = "周线接近压力位，建议等待突破确认。"
    elif trend == "sideways" and signal in ["bullish_engulf", "hammer"] and volume == "volume_up":
        action = "buy"
        reason = f"震荡市中，日线出现放量{signal}形态，可小仓位波段操作。"
        stop_loss = latest_price * 0.97
    else:
        action = "hold"
        reason = "当前无明确买卖信号，建议观望。"

    return {"action": action, "price": round(latest_price, 2), "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "reason": reason}


# ==================== 绘图 ====================
def plot_kline(day_df: pd.DataFrame, week_df: pd.DataFrame, code: str, stock_name: str, advice: Dict,
               save_path: str = None):
    if day_df.empty:
        print("无数据，无法绘图")
        return
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(day_df["date"], day_df["close"], color="blue", linewidth=1, label="日收盘价")
    for w, col, ls in [(5, "orange", "--"), (10, "green", "--"), (20, "red", "--")]:
        if len(day_df) >= w:
            ma = calc_ma(day_df, w)
            ax1.plot(day_df["date"], ma, color=col, linestyle=ls, linewidth=1, label=f"MA{w}")
    if not week_df.empty:
        ax1.scatter(week_df["date"], week_df["close"], color="magenta", s=20, label="周收盘价", zorder=5)
    ax1.set_ylabel("价格 (元)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(day_df["date"], day_df["volume"], color="gray", alpha=0.4, width=0.8, label="成交量")
    ax2.set_ylabel("成交量", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    title = f"{stock_name}({code}) 技术分析图\n最新:{advice['price']} 建议:{advice['action'].upper()}"
    plt.title(title)
    if advice["action"] == "buy":
        last_date = day_df["date"].iloc[-1]
        last_price = day_df["close"].iloc[-1]
        ax1.annotate("★ 买点", xy=(last_date, last_price), xytext=(last_date, last_price * 0.95),
                     arrowprops=dict(arrowstyle="->", color="green"), color="green")
    elif advice["action"] == "sell":
        last_date = day_df["date"].iloc[-1]
        last_price = day_df["close"].iloc[-1]
        ax1.annotate("★ 卖点", xy=(last_date, last_price), xytext=(last_date, last_price * 1.05),
                     arrowprops=dict(arrowstyle="->", color="red"), color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close()


# ==================== 主函数 ====================
def analyze_stock(code: str, start_date: str = None, end_date: str = None):
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")

    # 获取数据（并打印调试信息）
    day_json = get_day_line(code, start_date, end_date, type=1)
    print("日K API返回码/状态:", day_json.get("code") if isinstance(day_json, dict) else "非字典")
    week_json = get_week_line(code, start_date, end_date, type=1)

    day_df = to_dataframe(day_json)
    week_df = to_dataframe(week_json)

    if day_df.empty:
        print(f"无法获取 {code} 的有效日K数据，请检查网络和Token。")
        return None

    info = get_stock_info(code)
    stock_name = info.get("name", code) if info else code

    analysis = simple_trend_analysis(day_df, week_df)
    latest_price = day_df["close"].iloc[-1]
    advice = generate_advice(analysis, latest_price)

    chart_file = f"{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_kline(day_df, week_df, code, stock_name, advice, save_path=chart_file)

    print(f"股票：{stock_name}({code})")
    print(f"最新日期：{day_df['date'].iloc[-1].strftime('%Y-%m-%d')}，收盘价：{latest_price}")
    print(f"趋势判断：{analysis['trend']} | 周线支撑：{analysis['support']} | 日线信号：{analysis['signal']}")
    print(f"操作建议：{advice['action'].upper()}")
    print(f"理由：{advice['reason']}")
    if advice['stop_loss']:
        print(f"参考止损：{advice['stop_loss']}")
    print(f"图表已保存：{chart_file}")
    return advice


# ==================== 测试 ====================
if __name__ == "__main__":
    analyze_stock("002491")
