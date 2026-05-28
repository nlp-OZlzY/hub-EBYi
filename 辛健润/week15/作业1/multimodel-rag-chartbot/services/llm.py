"""LLM Service - Qwen-VL."""
import os
import yaml
import openai

_config = None


def load_config():
    global _config
    if _config is None:
        cfg_path = "config.yaml"
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        with open(cfg_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


class LLMService:
    def __init__(self):
        cfg = load_config()
        llm_cfg = cfg["llm"]

        self.client = openai.OpenAI(
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"]
        )
        self.model = llm_cfg["model"]

    def chat(self, question: str, context: str) -> str:
        """单轮对话"""
        system_prompt = f"""基于资料回答提问。

相关资料:
{context}

回答要求:
- 回答要客观，有逻辑，基于给出的资料
- 如果资料中包含图片链接，需要保留图的原始链接，放在相关内容位置
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return resp.choices[0].message.content


_llm_service = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service