"""Tests for offline_process_worker.split_text2chunks."""

from offline_process_worker import split_text2chunks


def test_normal_text_respects_chunk_size():
    lines = ["a" * 100, "b" * 100, "c" * 100]
    chunks = split_text2chunks(lines, chunk_size=256)
    for chunk in chunks:
        assert len(chunk) <= 256 + 50  # some tolerance for newlines


def test_image_lines_preserved():
    lines = ["some text", "![img](images/pic.png)", "more text"]
    chunks = split_text2chunks(lines, chunk_size=256)
    combined = "\n".join(chunks)
    assert "![img](images/pic.png)" in combined


def test_empty_lines_skipped():
    lines = ["hello", "", "world"]
    chunks = split_text2chunks(lines, chunk_size=256)
    combined = "\n".join(chunks)
    assert "\n\n" not in combined.strip()


def test_references_header_skipped():
    lines = ["text before", "# References", "text after"]
    chunks = split_text2chunks(lines, chunk_size=256)
    combined = "\n".join(chunks)
    assert "# References" not in combined


def test_reference_number_lines_skipped():
    lines = ["some text", "[1] Author et al.", "[23] Another paper"]
    chunks = split_text2chunks(lines, chunk_size=256)
    combined = "\n".join(chunks)
    assert "[1]" not in combined
    assert "[23]" not in combined


def test_long_single_line_becomes_own_chunk():
    long_line = "x" * 500
    chunks = split_text2chunks([long_line], chunk_size=256)
    assert len(chunks) == 1
    assert chunks[0] == long_line


def test_empty_input():
    assert split_text2chunks([]) == []


def test_only_blank_lines():
    assert split_text2chunks(["", "", ""]) == []
