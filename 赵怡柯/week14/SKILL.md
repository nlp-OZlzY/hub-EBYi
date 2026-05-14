---
name: 股票波动可视化分析
description: 对股票的周波动、日波动绘制在同一图中，基于波动率大小给出买入卖出的最佳时间建议
---

# 功能定义

1. **日波动率计算**: `(日最高价 - 日最低价) / 收盘价 × 100%`
2. **周波动率计算**: `(周最高价 - 周最低价) / 周收盘价 × 100%`
3. **双波动率叠加图**: 将日波动率和周波动率绘制在同一图表中
4. **买卖最佳时间建议**: 基于波动率历史走势，给出具体买入/卖出日期

# 波动率等级与操作建议

| 波动率范围 | 市场状态 | 操作建议 |
|-----------|---------|---------|
| 日波动率 < 2% 且 周波动率 < 5% | 低波动筑底 | ⭐ 最佳买入时机 |
| 日波动率 2%-4% | 正常波动 | 持有/观望 |
| 日波动率 4%-6% | 温和放大 | 谨慎操作 |
| 日波动率 > 6% | 高波动预警 | 考虑卖出 |
| 日波动率 > 8% | 异常波动 | ⭐ 强烈建议卖出 |

# 买入信号

- 日波动率连续3天 < 2%
- 周波动率 < 5% 且日波动率 < 2%
- 波动率从高位快速回落（超卖信号）
- 日波动率创近30天新低

# 卖出信号

- 日波动率 > 6% 且呈上升趋势
- 日波动率 > 8%（异常波动）
- 波动率创新高但价格创新低（背离）
- 日波动率连续3天上升

# ===== 核心：最佳时间建议 =====

## 买入最佳时间判断

从历史数据中找到满足以下条件的日期序列：
1. 日波动率 < 2% 的连续交易日
2. 对应的周波动率 < 5%
3. 价格处于近30天低位（< 20日均线）

**建议格式**:
```json
{
  "best_buy_date": "2026-04-28",
  "best_buy_reason": "日波动率1.28%创30日新低，且周波动率3.2%<5%，价格处于支撑位",
  "buy_confidence": 85
}
```

## 卖出最佳时间判断

从历史数据中找到满足以下条件的日期序列：
1. 日波动率 > 6% 或 > 8%（异常）
2. 波动率呈上升趋势
3. 价格处于近30天高位（> 20日均线）

**建议格式**:
```json
{
  "best_sell_date": "2026-05-12",
  "best_sell_reason": "日波动率6.8%创30日新高，波动率上升趋势确认，建议减仓",
  "sell_confidence": 78
}
```

## 综合时间轴建议

返回一段时期内的所有买卖信号点：
```json
{
  "timeline": [
    {"date": "2026-04-20", "action": "观望", "volatility": 2.5, "reason": "波动率正常"},
    {"date": "2026-04-25", "action": "买入", "volatility": 1.8, "reason": "低波动筑底", "confidence": 85},
    {"date": "2026-05-08", "action": "持有", "volatility": 3.2, "reason": "正常波动"},
    {"date": "2026-05-12", "action": "卖出", "volatility": 6.8, "reason": "波动率异常", "confidence": 78}
  ]
}
```

---

# API接口

## GET /api/stock/volatility/{code}

获取日波动率和周波动率数据

**参数**:
- `code`: 股票代码 (必填)
- `days`: 分析天数，默认30

**返回**:
```json
{
  "code": "000001",
  "name": "平安银行",
  "daily_volatility": [
    {"date": "2026-05-14", "value": 1.28},
    {"date": "2026-05-13", "value": 1.56}
  ],
  "weekly_volatility": [
    {"week": "2026-05-14", "value": 4.21}
  ],
  "buy_signal": true,
  "sell_signal": false,
  "recommendation": "买入",
  "reason": "日波动率1.28%<2%，周波动率4.21%<5%，建议买入"
}
```

## GET /api/stock/volatility/chart/{code}

获取双波动率叠加图数据

**参数**:
- `code`: 股票代码
- `weeks`: 周数，默认12

**返回**:
```json
{
  "code": "000001",
  "data": {
    "dates": ["2026-05-14", "2026-05-13", ...],
    "daily_values": [1.28, 1.56, ...],
    "weekly_values": [4.21, 3.85, ...],
    "buy_line": 2.0,
    "sell_line": 6.0
  }
}
```

## GET /api/stock/advice/{code}

获取具体买卖建议

**返回**:
```json
{
  "action": "买入",
  "score": 25,
  "signals": ["低波动筑底", "周日共振"],
  "entry_price": 10.76,
  "stop_loss": 10.50,
  "take_profit": 12.00,
  "risk_level": "低"
}
```

## GET /api/stock/best-timing/{code}  ⭐ 预测最佳时间建议

基于历史波动率趋势外推预测未来买卖时间点

**返回**:
```json
{
  "code": "000001",
  "name": "平安银行",
  "current_price": 11.05,
  "current_volatility": 1.85,
  "trend": "下降",
  "trend_description": "波动率近期呈下降趋势，市场趋于稳定",
  "predictions": [
    {"day": 1, "date": "预测第1天", "predicted_volatility": 1.72},
    {"day": 2, "date": "预测第2天", "predicted_volatility": 1.58},
    {"day": 3, "date": "预测第3天", "predicted_volatility": 1.45},
    {"day": 4, "date": "预测第4天", "predicted_volatility": 1.31},
    {"day": 5, "date": "预测第5天", "predicted_volatility": 1.18}
  ],
  "best_buy_prediction": {
    "date": "预测第5天",
    "predicted_volatility": 1.18,
    "reason": "预测波动率1.18%较低，市场可能企稳",
    "confidence": 76
  },
  "best_sell_prediction": null,
  "current_action": "观望",
  "action_reason": "波动率趋势不明显，建议等待",
  "recommendation": "观望"
}
```