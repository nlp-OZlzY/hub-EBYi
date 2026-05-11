\## 1. 前后端分离



后端用 FastAPI 提供 REST API（chat、user、stock 等接口），前端用 Streamlit 构建页面展示，前后端通过 HTTP JSON 通信，互不依赖。



\## 2. 历史对话存储与输入



历史对话存在 SQLite 两张表里：chat\_session 存会话信息，chat\_message 存每条消息（role 区分 user/assistant）。



对话输入给大模型靠的是 `AdvancedSQLiteSession`，它用 session\_id 关联，自动从数据库读历史消息拼到 prompt 里，大模型就能"记起"之前说了什么。

