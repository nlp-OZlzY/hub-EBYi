# 📊 股票分析 Skill

股票分析 Skill 现已正确安装到 Claude Code！

## ✅ 目录结构

```
~/.claude/skills/stock-analysis/
├── stock-analysis.md           # Skill 定义文件（必需）
├── README.md                   # 使用说明
└── scripts/
    └── stock_analysis.py       # Python 实现脚本
```

## 🚀 使用方法

### 方法1：在 Claude Code 对话中直接使用（推荐）

直接描述你的需求，Claude 会自动调用这个 Skill：

```
请分析苹果公司（AAPL）的股票
```

### 方法2：命令行直接运行

```bash
cd ~/.claude/skills/stock-analysis/scripts
python stock_analysis.py AAPL 3mo
```

## 📊 功能特性

- ✅ 实时股票数据获取（美股/A股/港股）
- ✅ 日波动率 + 周波动率分析
- ✅ 三合一可视化图表
- ✅ 最佳买入/卖出时机识别
- ✅ 详细分析报告

## 🎯 支持的股票

- 美股：AAPL, TSLA, MSFT, GOOGL
- A股：600000.SS, 000001.SZ
- 港股：0700.HK, 9988.HK

## 🎉 快速测试

```bash
cd ~/.claude/skills/stock-analysis/scripts
python stock_analysis.py AAPL
```

查看详细说明：`stock-analysis.md`

---
**⚠️ 免责声明：仅供学习参考，不构成投资建议**
