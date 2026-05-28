"""
Qwen API 客户端 - 使用阿里云DashScope OpenAI兼容接口
"""
import os
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("需要安装 openai 包: pip install openai")

class QwenClient:
    """Qwen API 客户端 - 使用阿里云DashScope"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "qwen3.6-plus",
                 region: str = "cn-beijing"):
        """
        初始化 Qwen 客户端
        
        Args:
            api_key: DashScope API Key，若不提供则从环境变量 DASHSCOPE_API_KEY 获取
            model: 模型名称，默认 qwen3.6-plus
            region: 区域，可选值: cn-beijing, us-virginia, singapore, eu-frankfurt
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        
        # 根据区域选择对应的 base_url
        region_urls = {
            "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "us-virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "eu-frankfurt": "https://dashscope-eu-central-1.aliyuncs.com/compatible-mode/v1"
        }
        
        self.base_url = region_urls.get(region, region_urls["cn-beijing"])
        
        if not self.api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY environment variable or provide api_key parameter.")
        
        # 创建 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """生成响应"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Qwen API error: {e}")
            return f"API调用失败: {str(e)}"
    
    def think(self, role: str, context: str) -> str:
        """让Agent思考"""
        messages = [
            {
                "role": "system",
                "content": f"""你是一个狼人杀游戏中的{role}角色。
规则：
- 狼人阵营：每天晚上刀死一个好人，目标是让狼人数 >= 好人数
- 好人阵营：白天投票出局狼人，目标是消灭所有狼人
- 预言家：每晚可以查验一个玩家身份
- 女巫：有一瓶解药和一瓶毒药
- 猎人：被投票出局时可以开枪带走一人

你的任务：根据当前局势，分析应该采取什么行动。
请输出你的思考过程，用中文。"""
            },
            {
                "role": "user",
                "content": f"当前局势：\n{context}\n\n请分析你应该采取什么行动，并说明理由。"
            }
        ]
        
        return self.generate_response(messages, max_tokens=200)
    
    def speak(self, role: str, context: str, is_werewolf: bool = False) -> str:
        """让Agent发言"""
        messages = [
            {
                "role": "system",
                "content": f"""你是一个狼人杀游戏中的{role}角色，正在发言阶段。

注意：
- 如果你是狼人，要隐藏身份，误导好人
- 如果你是好人，要提供有价值的信息帮助找出狼人
- 发言要符合角色逻辑，不要暴露太多信息

请用自然的口语化中文发言，不要太长。"""
            },
            {
                "role": "user",
                "content": f"当前局势：\n{context}\n\n请说出你的发言。"
            }
        ]
        
        return self.generate_response(messages, max_tokens=150)
    
    def vote(self, role: str, context: str, candidates: List[str]) -> str:
        """让Agent投票"""
        messages = [
            {
                "role": "system",
                "content": f"""你是一个狼人杀游戏中的{role}角色，正在投票阶段。

请分析局势，选择一个玩家投票出局。
输出格式：只输出玩家编号，不要解释。"""
            },
            {
                "role": "user",
                "content": f"当前局势：\n{context}\n\n可投票玩家：{', '.join(candidates)}\n\n请投票给："
            }
        ]
        
        result = self.generate_response(messages, max_tokens=20)
        
        # 提取数字
        import re
        match = re.search(r'\d+', result)
        if match:
            return match.group()
        return result.strip()

# 全局实例
qwen_client = None

def init_qwen_client(api_key: str, model: str = "qwen3.6-plus", region: str = "cn-beijing"):
    """初始化 Qwen 客户端"""
    global qwen_client
    qwen_client = QwenClient(api_key, model, region)
    return qwen_client

def get_qwen_client() -> QwenClient:
    """获取 Qwen 客户端实例"""
    if qwen_client is None:
        raise ValueError("Qwen client not initialized. Call init_qwen_client first.")
    return qwen_client
