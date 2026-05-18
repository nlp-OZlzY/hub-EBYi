---
name: 股票技术分析与可视化
description: 基于autostock API获取股票K线数据，进行技术分析、周波动、日波动可视化，并给出买入卖出建议。
---

# 功能概述

本Skill提供股票技术分析和可视化功能，包括：
1. K线数据获取（日、周、月）
2. 周波动和日波动可视化
3. 技术指标计算
4. 买入卖出时机建议

# 接口调用

```python
TOKEN = "zgaLG8unUPr"

import requests
from typing import Annotated, Optional, Dict, List
import traceback

@app.get("/get_week_line", operation_id="get_stock_week_kline_v2")
async def get_stock_week_kline(
    code: Annotated[str, "股票代码"],
    startDate: Annotated[Optional[str], "开始时间"] = None,
    endDate: Annotated[Optional[str], "结束时间"] = None,
    type: Annotated[int, "0不复权,1前复权,2后复权"] = 1
) -> Dict:
    """周K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN
    try:
        payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
        response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_day_line", operation_id="get_stock_day_kline_v2")
async def get_stock_day_kline(
    code: Annotated[str, "股票代码"],
    startDate: Annotated[Optional[str], "开始时间"] = None,
    endDate: Annotated[Optional[str], "结束时间"] = None,
    type: Annotated[int, "0不复权,1前复权,2后复权"] = 1
) -> Dict:
    """日K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN
    try:
        payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
        response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_month_line", operation_id="get_stock_month_kline_v2")
async def get_stock_month_kline(
    code: Annotated[str, "股票代码"],
    startDate: Annotated[Optional[str], "开始时间"] = None,
    endDate: Annotated[Optional[str], "结束时间"] = None,
    type: Annotated[int, "0不复权,1前复权,2后复权"] = 1
) -> Dict:
    """月K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/month" + "?token=" + TOKEN
    try:
        payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
        response = requests.request("GET", url, headers={}, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_info", operation_id="get_stock_info_v2")
async def get_stock_info(code: Annotated[str, "股票代码"]) -> Dict:
    """股票基础信息"""
    url = "https://api.autostock.cn/v1/stock" + "?token=" + TOKEN + "&code=" + code
    try:
        response = requests.request("GET", url, headers={}, data={}, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}
```

# 可视化与建议模块

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np

class StockVisualizer:
    """股票技术分析与可视化"""

    def __init__(self):
        self.plt_style = 'seaborn-v0_8-darkgrid'

    def calculate_volatility(self, prices: List[float]) -> float:
        """计算波动率"""
        if len(prices) < 2:
            return 0.0
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        return float(np.std(returns) * 100)

    def calculate_ma(self, prices: List[float], period: int) -> List[float]:
        """计算移动平均线"""
        if len(prices) < period:
            return prices
        ma = []
        for i in range(len(prices)):
            if i < period - 1:
                ma.append(prices[i])
            else:
                ma.append(sum(prices[i-period+1:i+1]) / period)
        return ma

    def identify_support_resistance(self, prices: List[float], windows: int = 5) -> Tuple[List[float], List[float]]:
        """识别支撑位和压力位"""
        supports = []
        resistances = []
        for i in range(windows, len(prices) - windows):
            is_support = all(prices[i] <= prices[i+j] for j in range(-windows, windows+1) if j != 0)
            is_resistance = all(prices[i] >= prices[i+j] for j in range(-windows, windows+1) if j != 0)
            if is_support:
                supports.append(prices[i])
            if is_resistance:
                resistances.append(prices[i])
        return supports, resistances

    def generate_signal(self, prices: List[float], volumes: List[int], period: int = 5) -> Dict[str, any]:
        """生成买入卖出信号"""
        ma5 = self.calculate_ma(prices, 5)
        ma10 = self.calculate_ma(prices, 10)
        ma20 = self.calculate_ma(prices, 20)

        volatility = self.calculate_volatility(prices)
        current_price = prices[-1]
        ma5_current = ma5[-1] if len(ma5) > 0 else current_price
        ma10_current = ma10[-1] if len(ma10) > 0 else current_price
        ma20_current = ma20[-1] if len(ma20) > 0 else current_price

        signal = "持有"
        confidence = 0.5
        reason = ""

        # 买入信号条件
        buy_conditions = 0
        if ma5_current > ma10_current:
            buy_conditions += 1
        if current_price > ma5_current:
            buy_conditions += 1
        if volatility < 5:
            buy_conditions += 1

        # 卖出信号条件
        sell_conditions = 0
        if ma5_current < ma10_current:
            sell_conditions += 1
        if current_price < ma5_current:
            sell_conditions += 1
        if volatility > 8:
            sell_conditions += 1

        if buy_conditions >= 2 and sell_conditions < 2:
            signal = "买入"
            confidence = 0.6 + (buy_conditions * 0.1)
            reason = f"MA5({ma5_current:.2f})>MA10({ma10_current:.2f})且波动率较低({volatility:.2f}%)"
        elif sell_conditions >= 2 and buy_conditions < 2:
            signal = "卖出"
            confidence = 0.6 + (sell_conditions * 0.1)
            reason = f"MA5({ma5_current:.2f})<MA10({ma10_current:.2f})或波动率较高({volatility:.2f}%)"
        else:
            reason = f"市场震荡，MA5={ma5_current:.2f}, MA10={ma10_current:.2f}, 波动率={volatility:.2f}%"

        return {
            "signal": signal,
            "confidence": min(confidence, 0.95),
            "current_price": current_price,
            "volatility": volatility,
            "ma5": ma5_current,
            "ma10": ma10_current,
            "ma20": ma20_current,
            "reason": reason
        }

    def plot_stock_analysis(
        self,
        dates: List[str],
        day_prices: List[float],
        day_volumes: List[int],
        week_prices: List[float],
        week_volumes: List[int],
        stock_code: str,
        save_path: Optional[str] = None
    ) -> str:
        """绘制股票分析图（日波动+周波动）"""
        try:
            plt.style.use(self.plt_style)
        except:
            plt.style.use('default')

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'{stock_code} 股票技术分析', fontsize=16, fontweight='bold')

        # 转换日期格式
        x_dates = [datetime.strptime(d, '%Y-%m-%d') if '-' in str(d) else datetime.now() for d in dates]
        x_range = range(len(x_dates))

        # === 图1: 日K线价格走势 ===
        ax1 = axes[0]
        ax1.plot(x_range, day_prices, 'b-', linewidth=1.5, label='日线价格')
        ax1.plot(x_range, self.calculate_ma(day_prices, 5), 'g--', alpha=0.7, label='MA5')
        ax1.plot(x_range, self.calculate_ma(day_prices, 10), 'r--', alpha=0.7, label='MA10')
        ax1.plot(x_range, self.calculate_ma(day_prices, 20), 'm--', alpha=0.7, label='MA20')
        ax1.set_ylabel('价格 (元)')
        ax1.set_title('日线价格走势', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(x_range, day_prices, alpha=0.3)

        # === 图2: 日波动率 ===
        ax2 = axes[1]
        day_returns = [0] + [(day_prices[i]/day_prices[i-1]-1)*100 for i in range(1, len(day_prices))]
        colors = ['green' if r >= 0 else 'red' for r in day_returns]
        ax2.bar(x_range, day_returns, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('日涨跌幅 (%)')
        ax2.set_title('日波动', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # === 图3: 周波动率 ===
        ax3 = axes[2]
        week_returns = [0]
        for i in range(1, len(week_prices)):
            if week_volumes[i] > 0:
                week_returns.append((week_prices[i]/week_prices[i-1]-1)*100)
            else:
                week_returns.append(0)
        colors = ['green' if r >= 0 else 'red' for r in week_returns]
        ax3.bar(range(len(week_returns)), week_returns, color=colors, alpha=0.7, width=0.6)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('周涨跌幅 (%)')
        ax3.set_xlabel('时间')
        ax3.set_title('周波动', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 设置x轴日期标签
        step = max(1, len(dates) // 10)
        ax3.set_xticks(x_range[::step])
        ax3.set_xticklabels([dates[i] if i < len(dates) else '' for i in range(0, len(dates), step)], rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path

        img_path = f'temp_stock_{stock_code}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        return img_path

    def analyze_and_recommend(
        self,
        day_prices: List[float],
        day_volumes: List[int],
        week_prices: List[float],
        week_volumes: List[int]
    ) -> Dict:
        """综合分析并给出投资建议"""
        day_signal = self.generate_signal(day_prices, day_volumes)
        week_signal = self.generate_signal(week_prices, week_volumes)

        day_volatility = self.calculate_volatility(day_prices)
        week_volatility = self.calculate_volatility(week_prices)

        supports, resistances = self.identify_support_resistance(day_prices[-20:])

        recommendation = "持有"
        action = ""

        # 综合判断
        if day_signal["signal"] == "买入" and week_signal["signal"] == "买入":
            recommendation = "强烈买入"
            action = f"日线和周线同时出现买入信号，建议在{day_signal['current_price']:.2f}元附近建仓"
        elif day_signal["signal"] == "卖出" and week_signal["signal"] == "卖出":
            recommendation = "建议卖出"
            action = f"日线和周线同时出现卖出信号，建议在{day_signal['current_price']:.2f}元附近减仓"
        elif day_signal["signal"] == "买入":
            recommendation = "轻度买入"
            action = f"日线出现买入信号，但周线趋势不明朗，建议轻仓试探"
        elif day_signal["signal"] == "卖出":
            recommendation = "轻度卖出"
            action = f"日线出现卖出信号，但周线仍在上行，可考虑分批减仓"
        else:
            action = f"市场震荡，建议观望等待更明确信号"

        return {
            "recommendation": recommendation,
            "action": action,
            "day_analysis": day_signal,
            "week_analysis": week_signal,
            "day_volatility": round(day_volatility, 2),
            "week_volatility": round(week_volatility, 2),
            "support_range": f"{min(supports):.2f}-{max(supports):.2f}" if supports else "未识别到明显支撑",
            "resistance_range": f"{min(resistances):.2f}-{max(resistances):.2f}" if resistances else "未识别到明显压力",
            "current_price": day_signal["current_price"]
        }
```

# 使用示例

```python
# 1. 获取数据
day_data = await get_stock_day_kline("600519", startDate="2024-01-01", type=1)
week_data = await get_stock_week_kline("600519", startDate="2024-01-01", type=1)

# 2. 提取数据
dates = [d["date"] for d in day_data.get("data", [])]
day_prices = [float(d["close"]) for d in day_data.get("data", [])]
day_volumes = [int(d["volume"]) for d in day_data.get("data", [])]
week_prices = [float(d["close"]) for d in week_data.get("data", [])]
week_volumes = [int(d["volume"]) for d in week_data.get("data", [])]

# 3. 可视化
visualizer = StockVisualizer()
img_path = visualizer.plot_stock_analysis(dates, day_prices, day_volumes, week_prices, week_volumes, "600519")

# 4. 分析建议
result = visualizer.analyze_and_recommend(day_prices, day_volumes, week_prices, week_volumes)

# 输出示例:
# {
#     "recommendation": "强烈买入",
#     "action": "日线和周线同时出现买入信号，建议在1850.00元附近建仓",
#     "day_analysis": {"signal": "买入", "confidence": 0.8, ...},
#     "week_analysis": {"signal": "买入", "confidence": 0.7, ...},
#     "day_volatility": 2.35,
#     "week_volatility": 5.21,
#     "support_range": "1780.00-1820.00",
#     "resistance_range": "1900.00-1950.00",
#     "current_price": 1850.00
# }
```
