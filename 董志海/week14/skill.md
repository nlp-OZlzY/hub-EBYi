---
name: 股票可视化分析
description: 对股票进行可视化分析，绘制周波动和日波动图表，基于波动率给出买入卖出时机建议。
---

# 功能概述

## 1. 数据获取
- `get_week_volatility` - 获取股票周波动数据
- `get_day_volatility` - 获取股票日波动数据
- `get_stock_wave_analysis` - 获取波动分析结果和买卖建议

## 2. 可视化功能
- 在一张图中同时展示周波动和日波动
- 支持标注关键买卖点
- 显示波动率统计信息

## 3. 买卖建议算法
基于波动率计算：
- **波动率 > 15%**：高波动，显示超买/超卖信号
- **波动率 8%-15%**：正常波动，适合区间操作
- **波动率 < 8%**：低波动，显示趋势信号

买入信号：
- 日波动率突然放大且周波动率处于低位
- 连续下跌后波动率开始缩小

卖出信号：
- 日波动率快速放大且周波动率处于高位
- 连续上涨后波动率异常扩大

# 调用方法

```python
import requests
from typing import Annotated, Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import traceback

TOKEN = "zgaLG8unUPr"

@app.get("/get_week_volatility", operation_id="get_week_volatility")
async def get_week_volatility(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间"] = None,
        endDate: Annotated[Optional[str], "结束时间"] = None
) -> Dict:
    """获取周波动数据"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN

    try:
        payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": 1}
        response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        data = response.json()

        if data.get("code") == 0 and data.get("data"):
            items = data["data"]
            volatilities = []
            for item in items:
                high = float(item.get("high", 0))
                low = float(item.get("low", 0))
                close = float(item.get("close", 0))
                if close > 0:
                    volatility = ((high - low) / close) * 100
                    volatilities.append({
                        "date": item.get("date", ""),
                        "volatility": round(volatility, 2),
                        "close": close,
                        "high": high,
                        "low": low
                    })
            return {"code": 0, "data": volatilities}
        return {"code": -1, "msg": "无数据"}
    except Exception:
        print(traceback.format_exc())
        return {"code": -1, "msg": str(traceback.format_exc())}


@app.get("/get_day_volatility", operation_id="get_day_volatility")
async def get_day_volatility(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间"] = None,
        endDate: Annotated[Optional[str], "结束时间"] = None
) -> Dict:
    """获取日波动数据"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN

    try:
        payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": 1}
        response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        data = response.json()

        if data.get("code") == 0 and data.get("data"):
            items = data["data"]
            volatilities = []
            for item in items:
                high = float(item.get("high", 0))
                low = float(item.get("low", 0))
                close = float(item.get("close", 0))
                if close > 0:
                    volatility = ((high - low) / close) * 100
                    volatilities.append({
                        "date": item.get("date", ""),
                        "volatility": round(volatility, 2),
                        "close": close,
                        "high": high,
                        "low": low
                    })
            return {"code": 0, "data": volatilities}
        return {"code": -1, "msg": "无数据"}
    except Exception:
        print(traceback.format_exc())
        return {"code": -1, "msg": str(traceback.format_exc())}


@app.get("/get_stock_wave_analysis", operation_id="get_stock_wave_analysis")
async def get_stock_wave_analysis(
        code: Annotated[str, "股票代码"],
        weeks: Annotated[int, "周线数据数量"] = 52
) -> Dict:
    """获取波动分析结果和买卖建议"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=weeks * 7)).strftime("%Y-%m-%d")

        # 获取周线数据
        payload = {"code": code, "startDate": start_date, "endDate": end_date, "type": 1}
        week_response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        week_data = week_response.json()

        # 获取日线数据
        day_url = "https://api.autostock.cn/v1/stock/kline/day"
        day_payload = {"code": code, "startDate": start_date, "endDate": end_date, "type": 1}
        day_response = requests.request("GET", day_url, headers={}, params=day_payload, timeout=10)
        day_data = day_response.json()

        week_volatilities = []
        day_volatilities = []

        # 解析周线波动率
        if week_data.get("code") == 0 and week_data.get("data"):
            for item in week_data["data"]:
                high = float(item.get("high", 0))
                low = float(item.get("low", 0))
                close = float(item.get("close", 0))
                if close > 0:
                    volatility = ((high - low) / close) * 100
                    week_volatilities.append({
                        "date": item.get("date", ""),
                        "volatility": round(volatility, 2),
                        "close": close
                    })

        # 解析日线波动率
        if day_data.get("code") == 0 and day_data.get("data"):
            for item in day_data["data"]:
                high = float(item.get("high", 0))
                low = float(item.get("low", 0))
                close = float(item.get("close", 0))
                if close > 0:
                    volatility = ((high - low) / close) * 100
                    day_volatilities.append({
                        "date": item.get("date", ""),
                        "volatility": round(volatility, 2),
                        "close": close
                    })

        # 计算统计数据
        week_avg_vol = np.mean([v["volatility"] for v in week_volatilities]) if week_volatilities else 0
        day_avg_vol = np.mean([v["volatility"] for v in day_volatilities]) if day_volatilities else 0

        # 生成买卖建议
        signals = generate_signals(week_volatilities, day_volatilities, week_avg_vol, day_avg_vol)

        return {
            "code": 0,
            "data": {
                "week_volatility": week_volatilities[-20:] if len(week_volatilities) > 20 else week_volatilities,
                "day_volatility": day_volatilities[-60:] if len(day_volatilities) > 60 else day_volatilities,
                "stats": {
                    "week_avg_volatility": round(week_avg_vol, 2),
                    "day_avg_volatility": round(day_avg_vol, 2),
                    "week_max_volatility": round(max([v["volatility"] for v in week_volatilities]), 2) if week_volatilities else 0,
                    "week_min_volatility": round(min([v["volatility"] for v in week_volatilities]), 2) if week_volatilities else 0
                },
                "signals": signals
            }
        }
    except Exception:
        print(traceback.format_exc())
        return {"code": -1, "msg": str(traceback.format_exc())}


def generate_signals(week_data: List[Dict], day_data: List[Dict], week_avg: float, day_avg: float) -> List[Dict]:
    """生成买卖信号"""
    signals = []

    if len(day_data) < 5:
        return signals

    # 最近5日波动率变化
    recent_day_vols = [d["volatility"] for d in day_data[-5:]]
    day_vol_change = recent_day_vols[-1] - recent_day_vols[0] if len(recent_day_vols) >= 2 else 0

    # 最近5周波动率变化
    recent_week_vols = [w["volatility"] for w in week_data[-5:]] if len(week_data) >= 5 else [w["volatility"] for w in week_data]
    week_vol_change = recent_week_vols[-1] - recent_week_vols[0] if len(recent_week_vols) >= 2 else 0

    # 买入信号1：日波动率突然放大（>30%），但周波动率处于低位
    if recent_day_vols[-1] > 3 and recent_day_vols[-1] > day_avg * 1.5 and (not week_data or week_data[-1]["volatility"] < week_avg):
        signals.append({
            "type": "BUY",
            "reason": "日波动率异常放大但周线支撑良好，可能为启动信号",
            "date": day_data[-1]["date"],
            "strength": min(round(recent_day_vols[-1] / 10, 1), 5)
        })

    # 买入信号2：连续下跌后波动率开始缩小
    if len(recent_day_vols) >= 3 and recent_day_vols[0] > recent_day_vols[-1] and recent_day_vols[-1] < day_avg * 0.8:
        signals.append({
            "type": "BUY",
            "reason": "连续调整后波动率收缩，可能见底",
            "date": day_data[-1]["date"],
            "strength": min(round((day_avg - recent_day_vols[-1]) / day_avg * 3, 1), 5)
        })

    # 卖出信号1：日波动率快速放大且周波动率处于高位
    if recent_day_vols[-1] > 3 and recent_day_vols[-1] > day_avg * 2 and week_data and week_data[-1]["volatility"] > week_avg:
        signals.append({
            "type": "SELL",
            "reason": "日周波动率同时放大，警惕阶段性顶部",
            "date": day_data[-1]["date"],
            "strength": min(round(recent_day_vols[-1] / 10, 1), 5)
        })

    # 卖出信号2：连续上涨后波动率异常扩大
    if len(recent_day_vols) >= 3 and recent_day_vols[0] < recent_day_vols[-1] and recent_day_vols[-1] > day_avg * 1.5:
        signals.append({
            "type": "SELL",
            "reason": "连续上涨后波动率异常放大，注意锁定利润",
            "date": day_data[-1]["date"],
            "strength": min(round((recent_day_vols[-1] - day_avg) / day_avg * 3, 1), 5)
        })

    # 震荡信号：波动率持续在均值附近
    if 0.9 < recent_day_vols[-1] / day_avg < 1.1 and week_data and 0.9 < week_data[-1]["volatility"] / week_avg < 1.1:
        signals.append({
            "type": "WATCH",
            "reason": "波动率处于均值附近，等待方向明确",
            "date": day_data[-1]["date"],
            "strength": 0
        })

    return signals


@app.get("/plot_stock_volatility", operation_id="plot_stock_volatility")
async def plot_stock_volatility(
        code: Annotated[str, "股票代码"],
        weeks: Annotated[int, "周线数据数量"] = 52
) -> Dict:
    """绘制股票波动率图表"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=weeks * 7)).strftime("%Y-%m-%d")

        # 获取日线数据
        payload = {"code": code, "startDate": start_date, "endDate": end_date, "type": 1}
        response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        data = response.json()

        if data.get("code") != 0 or not data.get("data"):
            return {"code": -1, "msg": "获取数据失败"}

        volatilities = []
        dates = []
        for item in data["data"]:
            high = float(item.get("high", 0))
            low = float(item.get("low", 0))
            close = float(item.get("close", 0))
            if close > 0:
                volatility = ((high - low) / close) * 100
                volatilities.append(round(volatility, 2))
                dates.append(item.get("date", ""))

        # 计算均线
        if len(volatilities) >= 5:
            ma5 = np.convolve(volatilities, np.ones(5)/5, mode='valid').tolist()
        else:
            ma5 = volatilities

        if len(volatilities) >= 10:
            ma10 = np.convolve(volatilities, np.ones(10)/10, mode='valid').tolist()
        else:
            ma10 = volatilities

        avg_vol = np.mean(volatilities)

        # 绘制图表
        plt.figure(figsize=(14, 8))

        # 波动率曲线
        plt.subplot(2, 1, 1)
        plt.plot(dates[-60:], volatilities[-60:], 'b-', alpha=0.7, label='日波动率')
        if len(ma5) > 0:
            plt.plot(dates[-len(ma5):], ma5, 'r-', linewidth=2, label='5日均线')
        if len(ma10) > 0:
            plt.plot(dates[-len(ma10):], ma10, 'g-', linewidth=2, label='10日均线')
        plt.axhline(y=avg_vol, color='orange', linestyle='--', label=f'均值({avg_vol:.2f}%)')
        plt.fill_between(dates[-60:], 0, volatilities[-60:], alpha=0.3)
        plt.ylabel('波动率 (%)')
        plt.title(f'股票 {code} 日波动率分析')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 买卖信号标注
        signals_data = []
        for i in range(5, len(volatilities)):
            if volatilities[i] > avg_vol * 1.5 and volatilities[i-1] <= avg_vol * 1.5:
                plt.axvline(x=dates[i], color='red', alpha=0.5, linestyle='--')
                signals_data.append({"type": "SELL", "date": dates[i], "vol": volatilities[i]})
            elif volatilities[i] > avg_vol * 0.5 and volatilities[i-1] <= avg_vol * 0.5 and i > 5:
                if volatilities[i-5:i].count(min(volatilities[i-5:i])) > 0:
                    plt.axvline(x=dates[i], color='green', alpha=0.5, linestyle='--')
                    signals_data.append({"type": "BUY", "date": dates[i], "vol": volatilities[i]})

        # 价格走势
        plt.subplot(2, 1, 2)
        prices = [item["close"] for item in data["data"][-60:]]
        plt.plot(dates[-60:], prices, 'b-', linewidth=1.5, label='收盘价')
        plt.ylabel('价格')
        plt.xlabel('日期')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 标注买卖点
        for sig in signals_data[-5:]:
            idx = dates.index(sig["date"]) if sig["date"] in dates else -1
            if idx >= 0:
                color = 'red' if sig["type"] == "SELL" else 'green'
                plt.annotate(sig["type"], xy=(dates[idx], prices[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, color=color, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'volatility_{code}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 生成分析报告
        analysis = {
            "avg_volatility": round(avg_vol, 2),
            "max_volatility": round(max(volatilities), 2),
            "min_volatility": round(min(volatilities), 2),
            "current_volatility": volatilities[-1] if volatilities else 0,
            "buy_signals": [s for s in signals_data if s["type"] == "BUY"],
            "sell_signals": [s for s in signals_data if s["type"] == "SELL"]
        }

        # 生成建议
        recommendation = ""
        if volatilities[-1] > avg_vol * 1.8:
            recommendation = "当前波动率处于高位，建议谨慎或适当减仓"
        elif volatilities[-1] < avg_vol * 0.6:
            recommendation = "当前波动率处于低位，可能存在反弹机会"
        else:
            recommendation = "当前波动率处于正常区间，建议观望或小仓位操作"

        return {
            "code": 0,
            "data": {
                "analysis": analysis,
                "recommendation": recommendation,
                "chart_path": f"volatility_{code}.png"
            }
        }
    except Exception:
        print(traceback.format_exc())
        return {"code": -1, "msg": str(traceback.format_exc())}
```

# 使用说明

## 1. 获取波动数据
- 调用 `get_week_volatility` 获取周波动数据
- 调用 `get_day_volatility` 获取日波动数据

## 2. 获取分析建议
- 调用 `get_stock_wave_analysis` 获取综合波动分析和买卖信号

## 3. 绘制可视化图表
- 调用 `plot_stock_volatility` 生成波动率图表

## 波动率判断标准

| 波动率范围 | 市场状态 | 操作建议 |
|------------|----------|----------|
| < 5% | 极度低波动 | 可能酝酿突破，关注方向 |
| 5% - 8% | 低波动 | 趋势可能延续 |
| 8% - 12% | 正常波动 | 适合区间操作 |
| 12% - 15% | 高波动 | 警惕转折，注意止损 |
| > 15% | 极度高波动 | 风险较大，谨慎操作 |

## 买卖信号说明

**买入信号**：
1. 日波动率突然放大超过均值50%，且周波动率处于低位
2. 连续下跌后波动率开始明显收缩

**卖出信号**：
1. 日波动率快速放大超过均值100%，且周波动率处于高位
2. 连续上涨后波动率异常扩大

**观望信号**：
- 波动率持续在均值附近窄幅波动，等待方向明确
