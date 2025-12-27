from pathlib import Path

from decayrag import parse_document, chunk_nodes


def test_parse_document_txt(tmp_path: Path) -> None:
    txt = tmp_path / "doc.txt"
    txt.write_text("para1\n\npara2")
    nodes = parse_document(str(txt))
    assert len(nodes) == 2
    assert nodes[0]["text"] == "para1"
    assert nodes[0]["level"] == []


def test_parse_document_md(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nIntro text\n\n## Section\n\nMore text")
    nodes = parse_document(str(md))
    assert nodes[0]["level"] == ["Title"]
    assert nodes[1]["level"] == ["Title", "Section"]


def test_chunk_nodes_assigns_positions() -> None:
    nodes = [
        {"doc_id": "x", "level": [], "text": "one two three four five"},
        {"doc_id": "x", "level": [], "text": "six seven eight"},
    ]
    chunks = chunk_nodes(nodes, max_tokens=2, overlap=0)
    assert chunks
    assert [chunk["position"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["doc_id"] == "x" for chunk in chunks)
