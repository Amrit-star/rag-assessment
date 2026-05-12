from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k: int = 3
    data_path: Path = field(default_factory=lambda: ROOT_DIR / "data" / "technical_context.json")
    benchmark_md_path: Path = field(default_factory=lambda: ROOT_DIR / "retrieval_benchmark.md")
    benchmark_json_path: Path = field(default_factory=lambda: ROOT_DIR / "benchmark_results.json")
    log_level: str = "INFO"


config = Config()
