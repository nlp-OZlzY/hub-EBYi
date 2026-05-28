"""Shared test configuration — sys.path setup and heavy dependency mocking."""

import os
import sys
import types
from unittest.mock import MagicMock

# ── Ensure src/ is importable ──────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Use temp DB for tests ──────────────────────────────────────────
os.environ["DB_PATH"] = os.path.join(os.path.dirname(__file__), "_test.db")

# ── Mock heavy dependencies before src modules load ────────────────
_MOCK_MODULES = [
    "sentence_transformers",
    "kafka",
    "kafka.errors",
    "pymilvus",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        mock_mod = types.ModuleType(mod_name)
        mock_mod.__dict__["KafkaConsumer"] = MagicMock()
        mock_mod.__dict__["KafkaProducer"] = MagicMock()
        mock_mod.__dict__["SentenceTransformer"] = MagicMock()
        mock_mod.__dict__["MilvusClient"] = MagicMock()
        sys.modules[mod_name] = mock_mod
