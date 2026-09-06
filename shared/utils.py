"""Shared helpers reused across mechanistic interpretability topics."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TOPICS_DIR = ROOT_DIR / "topics"
PLAYGROUND_DIR = ROOT_DIR / "playground"


def topic_path(topic_name: str) -> Path:
    """Return the absolute path for a topic directory."""
    return TOPICS_DIR / topic_name
