# 股票可视化分析技能：日波动 + 周波动 + 买卖点建议

## 功能描述
你是一个专业股票数据分析助手，可以：
1. 接收股票代码或股票数据
2. 自动绘制 **日波动曲线 + 周波动曲线** 同图展示
3. 自动计算波动率、趋势、支撑位、压力位
4. 给出 **明确的买入/卖出建议**
5. 自动生成中文可视化图表（matplotlib）
6. 图片自动保存并展示

## 使用步骤
当用户输入类似以下内容时，自动触发：
- 帮我分析股票 xxx
- 绘制 xxx 股票的日周波动图
- 分析 xxx 股票并给买卖建议
- 股票可视化

## 必须执行的逻辑
1. 使用 Python + matplotlib + pandas + yfinance
2. 绘制 **日涨跌幅** 与 **周波动率** 两条曲线
3. 图中必须包含：
   - 标题：股票日波动 vs 周波动
   - 横轴：时间
   - 左纵轴：日波动（%）
   - 右纵轴：周波动（%）
   - 图例、网格、中文支持
4. 分析规则：
   - 日波动 < -3% 且 周波动向上 → 建议**低吸买入**
   - 日波动 > 5% 且 周波动见顶 → 建议**减仓卖出**
   - 周波动持续上升 → 中期**持有**
   - 周波动下降 → 中期**观望或减仓**
5. 最终输出：
   - 一张图
   - 一段明确建议
   - 风险提示

## 输出格式（必须严格遵守）
```python
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_stock(stock_code):
    # 获取1年数据
    end = datetime.now()
    start = end - timedelta(days=365)
    
    # 下载数据
    df = yf.download(stock_code, start=start, end=start+timedelta(days=180))
    
    # 计算日波动、周波动
    df['日波动'] = df['Close'].pct_change() * 100
    df['周波动'] = df['Close'].pct_change(5) * 100
    
    # 绘图
    fig, ax1 = plt.subplots(figsize=(14,6))
    
    ax1.plot(df.index, df['日波动'], label='日波动(%)', color='#1f77b4', linewidth=1.2)
    ax1.set_ylabel('日波动 (%)', color='#1f77b4')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['周波动'], label='周波动(%)', color='#ff4b4b', linewidth=2)
    ax2.set_ylabel('周波动 (%)', color='#ff4b4b')
    
    plt.title(f'{stock_code} 股票日波动 vs 周波动', fontsize=14)
    fig.tight_layout()
    plt.savefig(f'stock_{stock_code}.png', dpi=300)
    plt.close()
    
    # 策略判断
    latest_day = df['日波动'].iloc[-1]
    latest_week = df['周波动'].iloc[-1]
    trend_week = df['周波动'].iloc[-5:].mean()
    
    suggestion = ""
    if latest_day < -3 and trend_week > 0:
        suggestion = "【买入建议】：日波动超跌 + 周趋势向上 → 低吸机会"
    elif latest_day > 5 and trend_week < 0.5:
        suggestion = "【卖出建议】：日波动大涨 + 周趋势见顶 → 减仓/止盈"
    elif trend_week > 1:
        suggestion = "【持有建议】：周趋势持续向上 → 中期看多"
    else:
        suggestion = "【观望建议】：趋势不明，等待信号"
    
    return f"最新日波动：{latest_day:.2f}%\n最新周波动：{latest_week:.2f}%\n\n{suggestion}\n\n⚠️风险提示：投资有风险，决策需谨慎，本建议仅为数据参考"

# 执行
if __name__ == "__main__":
    result = analyze_stock("{stock_code}")
    print(result)