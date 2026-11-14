"""Run the Day 06 ingestion, indexing, retrieval, and evaluation pipeline."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .models import DocumentPage
from .reranking import rerank
from .store import PgVectorStore


def _load_pages(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not record.get("eligible_content"):
            continue
        metadata = {
            "section": record["section"],
            "eligible_content": record["eligible_content"],
            "character_count": record["character_count"],
        }
        pages.append(
            DocumentPage(
                document_id=record["document_id"],
                source=record["source"],
                page=record["page"],
                text=record["text"],
                metadata=metadata,
            )
        )
    return pages


def _load_questions(path: Path) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if len(questions) < 20:
        raise ValueError("benchmark must contain at least 20 questions")
    if sum(q.get("split") == "dev" for q in questions) < 15:
        raise ValueError("benchmark must contain at least 15 dev questions")
    if sum(q.get("split") == "test" for q in questions) < 5:
        raise ValueError("benchmark must contain at least 5 test questions")
    return questions


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Build strategies A/B/C, persist vectors, and save retrieval traces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=Path, default=Path("../data/paper_pages.jsonl"))
    parser.add_argument("--benchmark", type=Path, default=Path("../benchmark/questions.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--truncate-dim", type=int, choices=[128, 256, 512, 768], default=768)
    parser.add_argument("--fixed-size", type=int, default=700)
    parser.add_argument("--fixed-overlap", type=int, default=100)
    parser.add_argument("--paragraph-size", type=int, default=1000)
    parser.add_argument("--query-prompt", choices=["QuestionAnswering", "SearchQuery"], default="QuestionAnswering")
    parser.add_argument("--rerank", choices=["none", "lexical"], default="none")
    parser.add_argument("--lexical-weight", type=float, default=0.2)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    model_name = args.model or os.environ.get("EMBEDDING_MODEL", "google/embeddinggemma-2")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if args.candidate_k < 3:
        raise ValueError("candidate-k must be at least 3")

    pages = _load_pages(args.pages)
    questions = _load_questions(args.benchmark)
    if len(pages) < 30:
        raise ValueError(f"expected at least 30 eligible content pages, found {len(pages)}")

    strategy_chunks = {
        "A": [chunk for page in pages for chunk in fixed_size(page, args.fixed_size, args.fixed_overlap)],
        "B": [chunk for page in pages for chunk in by_paragraph(page, args.paragraph_size)],
    }
    strategy_chunks["C"] = strategy_chunks["B"]

    started = time.perf_counter()
    embedder = MyEmbedder(model_name, truncate_dim=args.truncate_dim)
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    strategy_stats: dict[str, dict] = {}
    all_traces: list[dict] = []
    query_vectors = {
        q["id"]: embedder.embed_query(q["question"], prompt_name=args.query_prompt)
        for q in questions
    }
    vectors_by_strategy: dict[str, list[list[float]]] = {}

    for strategy, chunks in strategy_chunks.items():
        ingest_started = time.perf_counter()
        if strategy == "C":
            vectors = vectors_by_strategy["B"]
        else:
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        vectors_by_strategy[strategy] = vectors
        store.upsert(strategy, chunks, vectors)
        strategy_stats[strategy] = {
            "chunk_count": len(chunks),
            "chunk_length_min": min((len(chunk.text) for chunk in chunks), default=0),
            "chunk_length_max": max((len(chunk.text) for chunk in chunks), default=0),
            "chunk_length_mean": round(mean(len(chunk.text) for chunk in chunks), 2) if chunks else 0,
            "ingest_seconds": round(time.perf_counter() - ingest_started, 3),
        }
        if strategy == "C":
            strategy_stats[strategy]["embedding_reused_from"] = "B"
        traces: list[dict] = []
        for question in questions:
            filters = question.get("filter") if strategy == "C" else None
            candidate_k = max(3, args.candidate_k if args.rerank == "lexical" else 3)
            hits = store.search(strategy, query_vectors[question["id"]], top_k=candidate_k, filters=filters)
            if args.rerank == "lexical":
                hits = rerank(question["question"], hits, args.lexical_weight)
            hits = hits[:3]
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            traces.append(trace)
            all_traces.append(trace)
        _write_jsonl(args.results / f"traces_{strategy}.jsonl", traces)

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "metrics.json").write_text(
        json.dumps(summarize(all_traces), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.results / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "dimensions": embedder.dimensions,
                "vector_store": "PostgreSQL + pgvector",
                "metric": "cosine similarity",
                "query_prompt": args.query_prompt,
                "rerank": args.rerank,
                "lexical_weight": args.lexical_weight if args.rerank == "lexical" else None,
                "candidate_k": args.candidate_k if args.rerank == "lexical" else 3,
                "strategies": {
                    "A": {"chunking": "fixed_size", "size": args.fixed_size, "overlap": args.fixed_overlap},
                    "B": {"chunking": "by_paragraph", "max_size": args.paragraph_size},
                    "C": {"chunking": "by_paragraph", "max_size": args.paragraph_size, "filter": "section when supplied"},
                },
                "question_count": len(questions),
                "eligible_page_count": len(pages),
                "total_seconds": round(time.perf_counter() - started, 3),
                "strategy_stats": strategy_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": str(args.results / "metrics.json"), "strategy_stats": strategy_stats}, indent=2))

if __name__ == "__main__":
    main()
