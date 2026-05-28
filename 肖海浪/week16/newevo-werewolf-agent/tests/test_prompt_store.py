import pytest
import os
import shutil
from prompt_store.store import PromptStore


@pytest.fixture
def store(tmp_path):
    prompts_dir = str(tmp_path / "prompts" / "roles")
    versions_dir = str(tmp_path / "prompt_versions")
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(versions_dir, exist_ok=True)
    return PromptStore(prompts_dir=prompts_dir, versions_dir=versions_dir)


@pytest.fixture
def store_with_file(tmp_path):
    prompts_dir = str(tmp_path / "prompts" / "roles")
    versions_dir = str(tmp_path / "prompt_versions")
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(versions_dir, exist_ok=True)
    with open(os.path.join(prompts_dir, "werewolf.md"), "w", encoding="utf-8") as f:
        f.write("# 狼人策略指令\n\n## 身份信息\n你是狼人。")
    return PromptStore(prompts_dir=prompts_dir, versions_dir=versions_dir)


def test_read_prompt(store_with_file):
    content = store_with_file.read_prompt("werewolf")
    assert "狼人策略指令" in content
    assert "你是狼人" in content


def test_read_prompt_not_found(store):
    content = store.read_prompt("nonexistent")
    assert content == ""


def test_write_prompt(store):
    store.write_prompt("seer", "# 预言家策略\n\n你是预言家。")
    content = store.read_prompt("seer")
    assert "预言家策略" in content


def test_save_version(store_with_file):
    version = store_with_file.save_version("werewolf", "# 新版狼人策略")
    assert version.startswith("v")
    versions = store_with_file.list_versions("werewolf")
    assert len(versions) == 1
    assert versions[0] == version


def test_list_versions(store_with_file):
    store_with_file.save_version("werewolf", "v1 content")
    store_with_file.save_version("werewolf", "v2 content")
    versions = store_with_file.list_versions("werewolf")
    assert len(versions) == 2


def test_rollback(store_with_file):
    store_with_file.save_version("werewolf", "version 1 content")
    v2 = store_with_file.save_version("werewolf", "version 2 content")
    versions = store_with_file.list_versions("werewolf")
    store_with_file.rollback("werewolf", versions[0])
    content = store_with_file.read_prompt("werewolf")
    assert "version 1 content" in content
