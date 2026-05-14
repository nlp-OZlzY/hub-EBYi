---
name: 股票波动可视化与买卖建议
description: 获取股票的周K线和日K线数据，绘制周波动和日波动在同一图中，并基于技术分析给出买入卖出的最佳时间建议。
---

# 股票波动可视化与买卖建议

## 功能说明

本Skill通过调用autostock API获取股票的周K线和日K线数据，实现以下功能：
1. **周波动可视化**：绘制股票周K线的收盘价走势和波动范围
2. **日波动可视化**：绘制股票日K线的收盘价走势和日内波动幅度
3. **综合分析**：将周波动和日波动绘制在同一图中，便于对比分析
4. **买卖建议**：基于技术指标（均线、波动率、趋势）给出买入/卖出的最佳时间建议

## 技术指标分析逻辑

### 1. 均线分析
- **短期均线（MA5）**：反映近期趋势
- **中期均线（MA20）**：反映中期趋势
- **金叉信号**：短期均线上穿中期均线 → 买入信号
- **死叉信号**：短期均线下穿中期均线 → 卖出信号

### 2. 波动率分析
- **波动率计算**：使用收盘价的标准差衡量波动幅度
- **高波动期**：波动率显著上升 → 可能出现趋势转折
- **低波动期**：波动率持续低位 → 可能即将突破

### 3. 趋势判断
- **上升趋势**：收盘价持续在均线上方，且均线向上
- **下降趋势**：收盘价持续在均线下方，且均线向下
- **震荡整理**：价格在均线附近反复波动

## 调用方法

使用autostock API获取K线数据：

```python
TOKEN = "zgaLG8unUPr"
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

def get_week_kline(code, start_date=None, end_date=None):
    """获取周K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/week"
    params = {"token": TOKEN, "code": code, "type": 0}
    if start_date: params["startDate"] = start_date
    if end_date: params["endDate"] = end_date
    response = requests.get(url, params=params, timeout=10)
    return response.json()

def get_day_kline(code, start_date=None, end_date=None):
    """获取日K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/day"
    params = {"token": TOKEN, "code": code, "type": 0}
    if start_date: params["startDate"] = start_date
    if end_date: params["endDate"] = end_date
    response = requests.get(url, params=params, timeout=10)
    return response.json()
```

## 输出结果

### 图表输出
- 上半部分：周K线收盘价 + MA5/MA20均线 + 日K线收盘价叠加
- 下半部分：周波动率 + 日波动率对比

### 买卖建议输出
- 当前趋势方向（上升/下降/震荡）
- 最近的金叉/死叉信号
- 波动率状态（高/中/低）
- 综合建议：买入/卖出/持有/观望
- 建议的操作时间窗口

## 使用场景
- 投资者想要直观了解股票的中期（周线）和短期（日线）走势
- 需要综合周线和日线数据做出交易决策
- 寻找买入或卖出的最佳时机