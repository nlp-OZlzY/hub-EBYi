---
name: 股票周日波动对比与买卖建议
description: 获取股票的周K线和日K线数据，将周波动和日波动绘制在同一图中，基于波动率大小对比给出买入卖出的最佳时间建议。
---

# 股票周日波动对比与买卖建议

## 功能说明

本Skill通过调用autostock API获取股票的周K线和日K线数据，实现以下功能：
1. **周波动计算**：基于周K线的最高价和最低价计算每周波动幅度
2. **日波动计算**：基于日K线的最高价和最低价计算每日波动幅度
3. **同图绘制**：将周波动率和日波动率绘制在同一图表中进行对比
4. **买卖建议**：基于波动率的大小变化趋势给出买入/卖出的最佳时间建议

## 波动率分析逻辑

### 1. 波动率计算
- **绝对波动率**：最高价 - 最低价
- **相对波动率**：(最高价 - 最低价) / 收盘价 × 100%
- **滚动波动率**：使用N日/周收盘价的标准差

### 2. 波动率状态判断
- **高波动期**：当前波动率 > 平均波动率 × 1.2 → 市场情绪激烈，可能出现趋势转折
- **低波动期**：当前波动率 < 平均波动率 × 0.8 → 市场平静，可能即将突破
- **正常波动**：处于平均值附近 → 市场相对稳定

### 3. 买卖信号生成
- **缩量回调 + 低波动** → 可能的买入时机（蓄势待发）
- **放量突破 + 高波动** → 趋势启动信号
- **高位放量 + 高波动** → 可能的卖出时机（见顶风险）
- **持续低波动** → 观望，等待方向选择

## 调用方法

复用autostock API获取K线数据：

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
- **上半部分**：周波动率和日波动率的折线对比图（双Y轴）
- **下半部分**：波动率变化柱状图 + 买卖信号标注

### 买卖建议输出
- 当前周/日波动率状态（高/中/低）
- 波动率变化趋势（扩大/收敛/稳定）
- 最近的波动率异常点
- 综合建议：买入/卖出/观望
- 建议的操作时间窗口

## 使用场景
- 投资者想要通过波动率变化判断市场情绪
- 寻找波动率收缩后的突破买入机会
- 识别高位放量高波动的卖出信号
- 对比周线和日线的波动节奏
