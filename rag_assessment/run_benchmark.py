"""
Benchmark runner: compares Strategy A (raw vector search) vs Strategy B (AI-enhanced retrieval).

Outputs:
  - Rich table to console
  - retrieval_benchmark.md  (required by assessment)
  - benchmark_results.json  (machine-readable)
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config import config
from src.embeddings import SentenceTransformerEmbedder
from src.orchestrator import ContextAwareRetrievalEngine
from src.query_expansion import MockVertexGenerativeModel
from src.vector_store import FAISSVectorStore

logging.basicConfig(level=config.log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

console = Console()

INDEX_STORE = Path(__file__).parent / "index_store"
_INDEX_FILE = INDEX_STORE / "index.faiss"

BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What happens when the database goes down?",
    "How do you keep users logged in?",
    "How do you know when something breaks?",
    "What if an entire region fails?",
]


def _index_is_fresh() -> bool:
    if not _INDEX_FILE.exists():
        return False
    return _INDEX_FILE.stat().st_mtime >= config.data_path.stat().st_mtime


def build_engine() -> ContextAwareRetrievalEngine:
    embedder = SentenceTransformerEmbedder()
    expander = MockVertexGenerativeModel()

    if _index_is_fresh():
        console.print("[dim]Loading pre-built FAISS index from disk...[/dim]")
        store = FAISSVectorStore.load_index(INDEX_STORE, dim=config.embedding_dim)
        engine = ContextAwareRetrievalEngine(embedder, store, expander)
        engine._ingested = True
    else:
        console.print("[dim]First run: embedding corpus and saving index to disk...[/dim]")
        store = FAISSVectorStore(dim=config.embedding_dim)
        engine = ContextAwareRetrievalEngine(embedder, store, expander)
        engine.ingest_dataset(config.data_path)
        store.save_index(INDEX_STORE)

    return engine


def run_benchmark(engine: ContextAwareRetrievalEngine) -> list[dict]:
    results = []
    for query in BENCHMARK_QUERIES:
        a_results = engine.strategy_a_search(query, k=config.top_k)
        expanded_query, b_results = engine.strategy_b_search(query, k=config.top_k)

        avg_a = statistics.mean(r.score for r in a_results) if a_results else 0.0
        avg_b = statistics.mean(r.score for r in b_results) if b_results else 0.0

        results.append({
            "query": query,
            "expanded_query": expanded_query,
            "strategy_a": {
                "results": [{"rank": r.rank, "score": round(r.score, 4), "text": r.text} for r in a_results],
                "avg_score": round(avg_a, 4),
            },
            "strategy_b": {
                "results": [{"rank": r.rank, "score": round(r.score, 4), "text": r.text} for r in b_results],
                "avg_score": round(avg_b, 4),
            },
            "delta": round(avg_b - avg_a, 4),
        })
    return results


def print_rich_table(results: list[dict]) -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Context-Aware Retrieval Engine[/bold cyan]\n"
            "[dim]Strategy A (Raw) vs Strategy B (AI-Enhanced)[/dim]",
            border_style="cyan",
        )
    )

    for entry in results:
        console.print(f"\n[bold yellow]Query:[/bold yellow] {entry['query']}")
        console.print(
            f"[bold green]Expanded:[/bold green] [dim]{entry['expanded_query'][:120]}...[/dim]"
        )

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="center", width=6)
        table.add_column("Strategy A — Raw Search", ratio=2)
        table.add_column("Score A", justify="center", width=9)
        table.add_column("Strategy B — AI-Enhanced", ratio=2)
        table.add_column("Score B", justify="center", width=9)

        a_results = entry["strategy_a"]["results"]
        b_results = entry["strategy_b"]["results"]
        for i in range(config.top_k):
            a = a_results[i] if i < len(a_results) else {}
            b = b_results[i] if i < len(b_results) else {}
            a_text = (a.get("text", "")[:90] + "…") if a else "—"
            b_text = (b.get("text", "")[:90] + "…") if b else "—"
            a_score = f"[cyan]{a.get('score', '—'):.4f}[/cyan]" if a else "—"
            b_score = f"[green]{b.get('score', '—'):.4f}[/green]" if b else "—"
            table.add_row(str(i + 1), a_text, a_score, b_text, b_score)

        delta_color = "green" if entry["delta"] >= 0 else "red"
        table.add_section()
        table.add_row(
            "Avg",
            "",
            f"[cyan]{entry['strategy_a']['avg_score']:.4f}[/cyan]",
            "",
            f"[{delta_color}]{entry['strategy_b']['avg_score']:.4f} ({entry['delta']:+.4f})[/{delta_color}]",
        )
        console.print(table)

    # Summary
    overall_delta = statistics.mean(e["delta"] for e in results)
    wins = sum(1 for e in results if e["delta"] > 0)
    console.print()
    console.print(
        Panel(
            f"[bold]Strategy B outperforms Strategy A in [green]{wins}/{len(results)}[/green] queries\n"
            f"Overall avg score delta: [bold green]{overall_delta:+.4f}[/bold green][/bold]",
            title="Summary",
            border_style="green",
        )
    )


def write_markdown(results: list[dict]) -> None:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_avg_a = statistics.mean(e["strategy_a"]["avg_score"] for e in results)
    overall_avg_b = statistics.mean(e["strategy_b"]["avg_score"] for e in results)
    overall_delta = overall_avg_b - overall_avg_a
    wins = sum(1 for e in results if e["delta"] > 0)

    lines += [
        "# Context-Aware Retrieval Engine — Benchmark Report\n",
        f"> **Generated:** {now}  \n"
        f"> **Embedding Model:** `{config.embedding_model_name}` ({config.embedding_dim}d)  \n"
        f"> **Vector Index:** FAISS `IndexFlatIP` (Cosine Similarity on L2-normalised vectors)  \n"
        f"> **Dataset:** {config.data_path.name} ({len(results)} queries · 100 GCP architecture chunks)\n",
        "---\n",
        "## Summary\n",
        "| # | Query | Avg Score A | Avg Score B | Delta | Winner |",
        "|---|-------|:-----------:|:-----------:|:-----:|:------:|",
    ]
    for i, e in enumerate(results, 1):
        winner = "B ✓" if e["delta"] > 0 else "A ✓"
        lines.append(
            f"| {i} | {e['query']} | {e['strategy_a']['avg_score']:.4f} "
            f"| {e['strategy_b']['avg_score']:.4f} | {e['delta']:+.4f} | **{winner}** |"
        )

    lines += [
        "",
        f"**Strategy B outperformed Strategy A in {wins}/{len(results)} queries.**  ",
        f"**Overall average score improvement: {overall_delta:+.4f}**\n",
        "---\n",
    ]

    for i, e in enumerate(results, 1):
        lines += [
            f"## Query {i}: \"{e['query']}\"\n",
            "**Strategy B Expanded Query:**",
            f"> *{e['expanded_query']}*\n",
            "| Rank | Strategy A — Raw Search | Score A | Strategy B — AI-Enhanced | Score B |",
            "|:----:|------------------------|:-------:|--------------------------|:-------:|",
        ]
        a_list = e["strategy_a"]["results"]
        b_list = e["strategy_b"]["results"]
        for j in range(config.top_k):
            a = a_list[j] if j < len(a_list) else {}
            b = b_list[j] if j < len(b_list) else {}
            a_text = (a.get("text", "")[:100] + "…") if a else "—"
            b_text = (b.get("text", "")[:100] + "…") if b else "—"
            a_score = f"{a.get('score', 0):.4f}" if a else "—"
            b_score = f"{b.get('score', 0):.4f}" if b else "—"
            lines.append(f"| #{j+1} | {a_text} | {a_score} | {b_text} | {b_score} |")

        delta_label = f"+{e['delta']:.4f} ↑" if e["delta"] > 0 else f"{e['delta']:.4f}"
        lines += [
            f"\n**Avg Score A:** `{e['strategy_a']['avg_score']:.4f}` | "
            f"**Avg Score B:** `{e['strategy_b']['avg_score']:.4f}` | "
            f"**Delta:** `{delta_label}`\n",
            "---\n",
        ]

    config.benchmark_md_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[dim]Markdown written -> {config.benchmark_md_path}[/dim]")


def write_json(results: list[dict]) -> None:
    overall_avg_a = statistics.mean(e["strategy_a"]["avg_score"] for e in results)
    overall_avg_b = statistics.mean(e["strategy_b"]["avg_score"] for e in results)
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "embedding_model": config.embedding_model_name,
            "embedding_dim": config.embedding_dim,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
            "dataset": str(config.data_path),
            "top_k": config.top_k,
        },
        "results": results,
        "summary": {
            "overall_avg_score_a": round(overall_avg_a, 4),
            "overall_avg_score_b": round(overall_avg_b, 4),
            "overall_delta": round(overall_avg_b - overall_avg_a, 4),
            "strategy_b_wins": sum(1 for e in results if e["delta"] > 0),
            "total_queries": len(results),
        },
    }
    config.benchmark_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[dim]JSON written -> {config.benchmark_json_path}[/dim]")


if __name__ == "__main__":
    with console.status("[bold cyan]Initialising RAG engine and ingesting dataset…[/bold cyan]"):
        engine = build_engine()

    with console.status("[bold cyan]Running benchmark queries…[/bold cyan]"):
        results = run_benchmark(engine)

    print_rich_table(results)
    write_markdown(results)
    write_json(results)
