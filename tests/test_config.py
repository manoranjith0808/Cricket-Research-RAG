import importlib
import sys
from pathlib import Path


def test_config_reads_dotenv_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("src.config", None)

    config = importlib.import_module("src.config")

    assert config.BASE_DIR == str(Path(__file__).resolve().parents[1])
    assert config.DEFAULT_GEMINI_MODEL == "3.1-flash"
    assert "3.1-flash-lite" in config.GEMINI_MODEL_OPTIONS
