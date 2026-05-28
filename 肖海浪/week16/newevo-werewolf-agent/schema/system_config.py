"""系统配置模块

加载并管理系统配置（模型地址、API Key 等）。
环境变量优先于配置文件，加载后自动注入环境变量供 SDK 使用。
"""

import os
from pydantic import BaseModel, Field, model_validator


class SystemConfig(BaseModel):
    """系统配置模型

    包含 LLM 服务的基础配置，支持从 JSON 文件和环境变量加载。
    优先级：环境变量 > 配置文件 > 默认值

    加载后自动将 api_key 和 base_url 注入环境变量（OPENAI_API_KEY、OPENAI_BASE_URL），
    以便 OpenAI SDK 自动读取，无需在每次创建客户端时手动传入。
    """
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", description="模型服务地址")
    api_key: str = Field(default="", description="模型认证 API Key")
    default_model: str = Field(default="qwen-flash", description="默认模型名称")

    @model_validator(mode="after")
    def after_load_hook(self) -> "SystemConfig":
        """加载后的钩子：环境变量覆盖配置文件，并注入环境变量供 SDK 使用"""
        env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if env_key:
            self.api_key = env_key
        env_base_url = os.environ.get("OPENAI_BASE_URL")
        if env_base_url:
            self.base_url = env_base_url
        # 将 key 注入环境变量供 SDK 使用
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_BASE_URL"] = self.base_url
        return self


def load_system_config(file_path: str) -> SystemConfig:
    """从 JSON 文件加载系统配置

    Args:
        file_path: 配置文件路径

    Returns:
        SystemConfig 实例
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = f.read()

    return SystemConfig.model_validate_json(json_data)
