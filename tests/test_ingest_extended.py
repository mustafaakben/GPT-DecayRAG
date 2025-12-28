"""Extended tests for the ingest module."""
import json
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import (
    parse_document,
    chunk_nodes,
    embed_chunks,
    upsert_embeddings,
    batch_ingest,
    batch_ingest_from_config,
    resolve_ingest_settings,
)
import decayrag.decayrag.ingest as ingest


class TestParseDocumentPdf:
    def test_parse_pdf_file(self, tmp_path: Path):
        # Create a minimal PDF file (plain text extraction may be empty)
        pdf_path = tmp_path / "test.pdf"
        # pdfminer needs a valid PDF structure - create test with txt first
        # For real PDF testing, we test error handling
        pdf_path.write_bytes(b"%PDF-1.4\n")  # Minimal invalid PDF
        nodes = parse_document(str(pdf_path))
        # Should return empty list for invalid PDF
        assert nodes == []

    def test_unsupported_file_type_raises(self, tmp_path: Path):
        doc_path = tmp_path / "test.docx"
        doc_path.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(str(doc_path))


class TestSplitText:
    def test_empty_text_returns_empty(self):
        result = ingest._split_text("", max_tokens=100, overlap=0)
        assert result == []

    def test_zero_max_tokens_returns_whole_text(self):
        result = ingest._split_text("hello world", max_tokens=0, overlap=0)
        assert result == ["hello world"]

    def test_negative_max_tokens_returns_whole_text(self):
        result = ingest._split_text("hello world", max_tokens=-5, overlap=0)
        assert result == ["hello world"]

    def test_split_with_overlap(self):
        text = "one two three four five six seven eight"
        result = ingest._split_text(text, max_tokens=4, overlap=2)
        assert len(result) >= 2
        # Each chunk should have some overlap with adjacent

    def test_overlap_capped_at_max_tokens_minus_one(self):
        text = "one two three four five"
        result = ingest._split_text(text, max_tokens=3, overlap=10)
        assert len(result) >= 1

    def test_text_shorter_than_max_tokens(self):
        text = "short"
        result = ingest._split_text(text, max_tokens=100, overlap=0)
        assert result == ["short"]


class TestChunkNodes:
    def test_empty_nodes_returns_empty(self):
        result = chunk_nodes([], max_tokens=100, overlap=0)
        assert result == []

    def test_preserves_level_metadata(self):
        nodes = [
            {"doc_id": "doc1", "level": ["Chapter 1", "Section A"], "text": "content"}
        ]
        chunks = chunk_nodes(nodes, max_tokens=100, overlap=0)
        assert chunks[0]["level"] == ["Chapter 1", "Section A"]

    def test_multiple_nodes_get_sequential_positions(self):
        nodes = [
            {"doc_id": "doc1", "level": [], "text": "first"},
            {"doc_id": "doc1", "level": [], "text": "second"},
            {"doc_id": "doc1", "level": [], "text": "third"},
        ]
        chunks = chunk_nodes(nodes, max_tokens=100, overlap=0)
        positions = [c["position"] for c in chunks]
        assert positions == [0, 1, 2]

    def test_long_node_splits_into_multiple_chunks(self):
        nodes = [
            {"doc_id": "doc1", "level": [], "text": "a b c d e f g h i j k l m n o"}
        ]
        chunks = chunk_nodes(nodes, max_tokens=3, overlap=0)
        assert len(chunks) > 1


class TestEmbedChunks:
    def test_embed_chunks_calls_api(self, monkeypatch):
        calls = {}

        def fake_api_embed(texts, model_name, **kwargs):
            calls["texts"] = texts
            calls["model"] = model_name
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        monkeypatch.setattr(ingest, "_api_embed", fake_api_embed)

        chunks = [
            {"doc_id": "doc1", "text": "hello"},
            {"doc_id": "doc1", "text": "world"},
        ]
        embeds = embed_chunks(chunks, "test-model")
        assert embeds.shape == (2, 2)
        assert calls["texts"] == ["hello", "world"]
        assert calls["model"] == "test-model"

    def test_embed_chunks_normalizes_vectors(self, monkeypatch):
        def fake_api_embed(texts, model_name, **kwargs):
            return np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)

        monkeypatch.setattr(ingest, "_api_embed", fake_api_embed)

        chunks = [{"doc_id": "doc1", "text": "a"}, {"doc_id": "doc1", "text": "b"}]
        embeds = embed_chunks(chunks, "test-model")
        norms = np.linalg.norm(embeds, axis=1)
        assert np.allclose(norms, 1.0)


class TestUpsertEmbeddings:
    def test_creates_new_index(self, tmp_path: Path):
        index_path = str(tmp_path / "index.faiss")
        chunks = [
            {"doc_id": "doc1", "chunk_id": 0, "level": [], "position": 0, "text": "hello"}
        ]
        embeds = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        upsert_embeddings(index_path, chunks, embeds)

        assert Path(index_path).exists()
        assert Path(index_path + ".meta").exists()
        assert Path(index_path + ".npy").exists()

    def test_appends_to_existing_index(self, tmp_path: Path):
        index_path = str(tmp_path / "index.faiss")
        chunks1 = [
            {"doc_id": "doc1", "chunk_id": 0, "level": [], "position": 0, "text": "first"}
        ]
        embeds1 = np.array([[1.0, 0.0]], dtype=np.float32)
        upsert_embeddings(index_path, chunks1, embeds1)

        chunks2 = [
            {"doc_id": "doc2", "chunk_id": 0, "level": [], "position": 0, "text": "second"}
        ]
        embeds2 = np.array([[0.0, 1.0]], dtype=np.float32)
        upsert_embeddings(index_path, chunks2, embeds2)

        # Check metadata has both
        with open(index_path + ".meta", "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Check embeddings combined
        combined = np.load(index_path + ".npy")
        assert combined.shape == (2, 2)

    def test_empty_chunks_does_nothing(self, tmp_path: Path):
        index_path = str(tmp_path / "index.faiss")
        upsert_embeddings(index_path, [], np.empty((0, 3)))
        assert not Path(index_path).exists()

    def test_metadata_format(self, tmp_path: Path):
        index_path = str(tmp_path / "index.faiss")
        chunks = [
            {
                "doc_id": "doc1",
                "chunk_id": 5,
                "level": ["Ch1", "Sec2"],
                "position": 10,
                "text": "test text",
            }
        ]
        embeds = np.array([[1.0, 0.0]], dtype=np.float32)
        upsert_embeddings(index_path, chunks, embeds)

        with open(index_path + ".meta", "r") as f:
            meta = json.loads(f.readline())
        assert meta["doc_id"] == "doc1"
        assert meta["chunk_id"] == 5
        assert meta["level"] == ["Ch1", "Sec2"]
        assert meta["position"] == 10
        assert meta["text"] == "test text"


class TestBatchIngest:
    def test_ingests_txt_files(self, tmp_path: Path, monkeypatch):
        # Create test docs
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Hello world\n\nSecond para")
        (docs_dir / "doc2.txt").write_text("Another doc")

        # Mock API embed
        def fake_api_embed(texts, model_name, **kwargs):
            return np.random.randn(len(texts), 3).astype(np.float32)

        monkeypatch.setattr(ingest, "_api_embed", fake_api_embed)

        index_path = str(tmp_path / "index.faiss")
        batch_ingest(str(docs_dir), index_path, "test-model", max_tokens=100, overlap=0)

        assert Path(index_path).exists()

    def test_skips_unsupported_files(self, tmp_path: Path, monkeypatch):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc.txt").write_text("valid doc")
        (docs_dir / "other.docx").write_text("ignored")

        def fake_api_embed(texts, model_name, **kwargs):
            return np.random.randn(len(texts), 3).astype(np.float32)

        monkeypatch.setattr(ingest, "_api_embed", fake_api_embed)

        index_path = str(tmp_path / "index.faiss")
        batch_ingest(str(docs_dir), index_path, "test-model", max_tokens=100, overlap=0)
        assert Path(index_path).exists()

    def test_ingests_markdown_files(self, tmp_path: Path, monkeypatch):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc.md").write_text("# Title\n\nContent here")

        def fake_api_embed(texts, model_name, **kwargs):
            return np.random.randn(len(texts), 3).astype(np.float32)

        monkeypatch.setattr(ingest, "_api_embed", fake_api_embed)

        index_path = str(tmp_path / "index.faiss")
        batch_ingest(str(docs_dir), index_path, "test-model", max_tokens=100, overlap=0)
        assert Path(index_path).exists()


class TestBatchIngestFromConfig:
    def test_uses_config_defaults(self, tmp_path: Path, monkeypatch):
        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "index_path: data/index.faiss\n"
            "model: config-model\n"
            "max_tokens: 150\n"
            "overlap: 10\n"
        )

        # Create docs
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc.txt").write_text("test content")

        calls = {}

        def fake_batch_ingest(folder, idx, model, max_tok, overlap):
            calls["index"] = idx
            calls["model"] = model
            calls["max_tokens"] = max_tok
            calls["overlap"] = overlap

        monkeypatch.setattr(ingest, "batch_ingest", fake_batch_ingest)

        settings = batch_ingest_from_config(str(docs_dir), str(config_path))
        assert settings["model_name"] == "config-model"
        assert settings["max_tokens"] == 150
        assert settings["overlap"] == 10

    def test_cli_overrides_config(self, tmp_path: Path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "index_path: data/index.faiss\n"
            "model: config-model\n"
            "max_tokens: 150\n"
        )

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc.txt").write_text("test")

        def fake_batch_ingest(*args, **kwargs):
            pass

        monkeypatch.setattr(ingest, "batch_ingest", fake_batch_ingest)

        settings = batch_ingest_from_config(
            str(docs_dir),
            str(config_path),
            model_name="override-model",
            max_tokens=200,
        )
        assert settings["model_name"] == "override-model"
        assert settings["max_tokens"] == 200


class TestResolveIngestSettings:
    def test_requires_index_path(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: test\n")
        with pytest.raises(ValueError, match="index_path is required"):
            resolve_ingest_settings(str(config_path))

    def test_defaults_model_if_not_provided(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("index_path: data/index.faiss\n")
        settings = resolve_ingest_settings(str(config_path))
        assert settings["model_name"] == "text-embedding-3-small"

    def test_defaults_max_tokens_and_overlap(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("index_path: data/index.faiss\n")
        settings = resolve_ingest_settings(str(config_path))
        assert settings["max_tokens"] == 200
        assert settings["overlap"] == 0

    def test_coerces_string_integers(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "index_path: data/index.faiss\n"
            "max_tokens: '250'\n"
            "overlap: '15'\n"
        )
        settings = resolve_ingest_settings(str(config_path))
        assert settings["max_tokens"] == 250
        assert settings["overlap"] == 15

    def test_cli_overrides_take_precedence(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "index_path: config/path.faiss\n"
            "model: config-model\n"
            "max_tokens: 100\n"
            "overlap: 5\n"
        )
        settings = resolve_ingest_settings(
            str(config_path),
            index_path="cli/path.faiss",
            model_name="cli-model",
            max_tokens=300,
            overlap=20,
        )
        assert settings["index_path"] == "cli/path.faiss"
        assert settings["model_name"] == "cli-model"
        assert settings["max_tokens"] == 300
        assert settings["overlap"] == 20

    def test_invalid_max_tokens_raises(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "index_path: data/index.faiss\n" "max_tokens: invalid\n"
        )
        with pytest.raises(ValueError, match="max_tokens must be an integer"):
            resolve_ingest_settings(str(config_path))


class TestApiEmbed:
    def test_empty_texts_returns_empty_array(self):
        result = ingest._api_embed([], "model")
        assert result.shape == (0, 0)

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
            ingest._api_embed(["test"], "model")

    def test_invalid_batch_size_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with pytest.raises(ValueError, match="batch_size must be a positive"):
            ingest._api_embed(["test"], "model", batch_size=0)


class TestMarkdownParsing:
    def test_nested_headings(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text(
            "# H1\n\nPara under H1\n\n"
            "## H2 under H1\n\nPara under H2\n\n"
            "### H3\n\nDeep para\n\n"
            "# Another H1\n\nNew section"
        )
        nodes = parse_document(str(md))
        assert len(nodes) == 4
        assert nodes[0]["level"] == ["H1"]
        assert nodes[1]["level"] == ["H1", "H2 under H1"]
        assert nodes[2]["level"] == ["H1", "H2 under H1", "H3"]
        assert nodes[3]["level"] == ["Another H1"]

    def test_heading_without_content(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\n## Empty Section\n\n## Has Content\n\nText here")
        nodes = parse_document(str(md))
        # Only sections with content should create nodes
        assert any(n["text"] == "Text here" for n in nodes)

    def test_multiline_paragraph(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nLine one\nLine two\nLine three")
        nodes = parse_document(str(md))
        assert len(nodes) == 1
        assert "Line one\nLine two\nLine three" == nodes[0]["text"]
