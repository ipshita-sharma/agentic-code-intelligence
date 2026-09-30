"""Optional adapter for the installed MTEB AppsRetrieval task.

MTEB evolves independently of this prototype.  This module keeps the
integration isolated and checks the installed API at runtime instead of
silently producing custom, incompatible JSON.
"""

from pathlib import Path
from typing import Any


class RetrievalEncoder:
    """MTEB-compatible encoder facade for embedding models."""

    def __init__(self, semantic_index):
        self.semantic_index = semantic_index

    def encode(self, sentences, **kwargs):
        return self.semantic_index.encoder.encode(list(sentences))

    def get_model_name(self):
        return self.semantic_index.backend_name


def run_mteb(model: Any, output_folder: str | Path = "outputs/mteb") -> Any:
    """Run AppsRetrieval using the installed MTEB API, or explain what is missing."""
    try:
        import mteb
    except ImportError as exc:
        raise RuntimeError("MTEB is optional. Install a compatible version with `pip install 'mteb>=1.30'` before using --mteb.") from exc
    if not hasattr(mteb, "MTEB"):
        raise RuntimeError("Installed mteb package does not expose the current MTEB runner API")
    task = mteb.get_tasks(tasks=["AppsRetrieval"]) if hasattr(mteb, "get_tasks") else ["AppsRetrieval"]
    runner = mteb.MTEB(tasks=task)
    return runner.run(model, output_folder=str(output_folder))
