"""Prompt 存储模块

管理各角色的策略 prompt 文件，支持读写、版本保存、历史查看和回滚。
自演化系统的核心存储层：SelfReflector 生成的新 prompt 通过此模块持久化。
"""

import os
import glob
from datetime import datetime
from typing import List


class PromptStore:
    """角色 Prompt 文件存储管理器

    职责：
    - 读写 prompts/roles/ 目录下的角色策略 prompt 文件
    - 在 prompt_versions/ 目录下保存历史版本（带时间戳）
    - 支持版本列表查看和回滚操作

    版本管理流程：
    1. SelfReflector 生成新 prompt 前，先调用 save_version 备份当前版本
    2. 新 prompt 通过 write_prompt 覆盖写入
    3. 可通过 list_versions 查看演化历史，rollback 回滚到任意版本

    文件结构示例：
        prompt_versions/
          werewolf/
            v001_20260528_1430.md
            v002_20260528_1500.md
          seer/
            v001_20260528_1430.md
    """

    def __init__(self, prompts_dir: str = "prompts/roles", versions_dir: str = "prompt_versions"):
        """初始化存储管理器

        Args:
            prompts_dir: 当前 prompt 文件存放目录
            versions_dir: 历史版本存放目录
        """
        self.prompts_dir = prompts_dir
        self.versions_dir = versions_dir
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(versions_dir, exist_ok=True)

    def read_prompt(self, role: str) -> str:
        """读取指定角色的当前 prompt

        Args:
            role: 角色类型，如 "werewolf", "seer"

        Returns:
            prompt 文件内容，文件不存在时返回空字符串
        """
        filepath = os.path.join(self.prompts_dir, f"{role}.md")
        if not os.path.exists(filepath):
            return ""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def write_prompt(self, role: str, content: str) -> None:
        """写入角色 prompt（覆盖现有内容）

        Args:
            role: 角色类型
            content: 新的 prompt 内容
        """
        os.makedirs(self.prompts_dir, exist_ok=True)
        filepath = os.path.join(self.prompts_dir, f"{role}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def save_version(self, role: str, content: str) -> str:
        """保存当前 prompt 为历史版本

        版本文件命名格式：v001_20260528_1430.md（序号_日期_时间）

        Args:
            role: 角色类型
            content: 要保存的 prompt 内容

        Returns:
            版本标识字符串
        """
        role_dir = os.path.join(self.versions_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        existing = glob.glob(os.path.join(role_dir, "v*.md"))
        next_num = len(existing) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version = f"v{next_num:03d}_{timestamp}"
        filepath = os.path.join(role_dir, f"{version}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return version

    def list_versions(self, role: str) -> List[str]:
        """列出指定角色的所有历史版本

        Args:
            role: 角色类型

        Returns:
            版本标识列表（按时间排序）
        """
        role_dir = os.path.join(self.versions_dir, role)
        if not os.path.exists(role_dir):
            return []
        files = sorted(glob.glob(os.path.join(role_dir, "v*.md")))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def rollback(self, role: str, version: str) -> None:
        """回滚角色 prompt 到指定历史版本

        Args:
            role: 角色类型
            version: 版本标识，如 "v001_20260528_1430"

        Raises:
            FileNotFoundError: 指定版本不存在时抛出
        """
        role_dir = os.path.join(self.versions_dir, role)
        filepath = os.path.join(role_dir, f"{version}.md")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Version {version} not found for role {role}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        self.write_prompt(role, content)
