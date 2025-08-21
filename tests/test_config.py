import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import load_config
import decayrag.decayrag.retrieval as retrieval


def test_load_config(tmp_path):
    cfg = {"model": "test-model", "max_tokens": 123}
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    data = load_config(str(cfg_file))
    assert data["model"] == "test-model"
    data2 = retrieval.load_config(str(cfg_file))
    assert data2["max_tokens"] == 123
