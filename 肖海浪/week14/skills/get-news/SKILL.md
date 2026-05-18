---
name: 新闻获取器
description: 获取最新新闻、抖音新闻、github新闻、头条新闻和运动新闻
---

# 主要内容

| 工具                   | API端点             |
| :--------------------- | :------------------ |
| `get_today_daily_news` | `/api/tx/bulletin`  |
| `get_douyin_hot_news`  | `/api/tx/douyinhot` |
| `get_github_hot_news`  | `/api/github`       |
| `get_toutiao_hot_news` | `/api/tx/topnews`   |
| `get_sports_news`      | `/api/tx/esports`   |

# 使用方法

## 1. get_today_daily_news
```bash
curl -s "https://whyta.cn/api/tx/bulletin?key=6d997a997fbf"
```

## 2. get_douyin_hot_news
```bash
curl -s "https://whyta.cn/api/tx/douyinhot?key=6d997a997fbf"
```

## 3. get_github_hot_news
```bash
curl -s "https://whyta.cn/api/github?key=6d997a997fbf"
```

## 4. get_toutiao_hot_news
```bash
curl -s "https://whyta.cn/api/tx/topnews?key=6d997a997fbf"
```

## 5. get_sports_news
```bash
curl -s "https://whyta.cn/api/tx/esports?key=6d997a997fbf"
```

