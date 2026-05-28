"""Tests for offline_process_worker.encode_text_and_image."""

import numpy as np
from unittest.mock import MagicMock

from offline_process_worker import encode_text_and_image


def _make_model(return_dim: int):
    """Create a mock SentenceTransformer that returns vectors of given dimension."""
    model = MagicMock()
    model.encode.return_value = np.random.randn(return_dim).astype(np.float32)
    return model


def test_bge_text_vector_dim_512():
    bge = _make_model(512)
    clip = _make_model(1024)
    bge_vec, clip_text_vec, clip_img_vec = encode_text_and_image(
        "hello world", "/fake/path.md", bge, clip
    )
    assert len(bge_vec) == 512


def test_clip_text_vector_dim_1024():
    bge = _make_model(512)
    clip = _make_model(1024)
    _, clip_text_vec, _ = encode_text_and_image(
        "hello world", "/fake/path.md", bge, clip
    )
    assert len(clip_text_vec) == 1024


def test_no_image_gives_zero_clip_image_vector():
    bge = _make_model(512)
    clip = _make_model(1024)
    _, _, clip_img_vec = encode_text_and_image(
        "plain text no images", "/fake/path.md", bge, clip
    )
    assert len(clip_img_vec) == 1024
    assert all(v == 0.0 for v in clip_img_vec)


def test_encode_exception_returns_zero_vectors():
    bge = MagicMock()
    bge.encode.side_effect = RuntimeError("model broken")
    clip = MagicMock()
    clip.encode.side_effect = RuntimeError("model broken")
    bge_vec, clip_text_vec, clip_img_vec = encode_text_and_image(
        "text", "/fake/path.md", bge, clip
    )
    assert len(bge_vec) == 512
    assert all(v == 0.0 for v in bge_vec)
    assert len(clip_text_vec) == 1024
    assert all(v == 0.0 for v in clip_text_vec)
    assert len(clip_img_vec) == 1024
    assert all(v == 0.0 for v in clip_img_vec)
