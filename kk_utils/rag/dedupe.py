"""Dry-run-first duplicate document cleanup for ChromaDB RAG collections."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable

from kk_utils.rag.config import get_rag_config
from kk_utils.rag.rag_engine import RAGEngine


def _resolve_persist_directory(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()

    import os

    config = get_rag_config()
    config_path = os.environ.get("RAG_CONFIG_PATH")
    if config_path:
        backend_root = Path(config_path).expanduser().resolve().parent.parent
        return backend_root / "data" / config.storage.persist_directory
    return Path.cwd() / "data" / config.storage.persist_directory


def _document_timestamp(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("uploaded_at") or metadata.get("added_at") or "")


def _group_documents(ids: Iterable[str], documents: Iterable[str], metadatas: Iterable[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk_id, content, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        doc_id = str(metadata.get("doc_id") or chunk_id)
        grouped[doc_id].append(
            {
                "id": chunk_id,
                "content": content or "",
                "metadata": metadata,
                "chunk_index": int(metadata.get("chunk_index", 0)),
            }
        )
    for chunks in grouped.values():
        chunks.sort(key=lambda chunk: (chunk["chunk_index"], chunk["id"]))
    return grouped


def build_cleanup_plan(collection) -> dict:
    """Inspect a collection and return duplicate document groups without writing."""
    data = collection.get(include=["documents", "metadatas"])
    documents = _group_documents(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    )

    by_fingerprint: dict[str, list[dict]] = defaultdict(list)
    for doc_id, chunks in documents.items():
        fingerprint = RAGEngine.compute_chunk_fingerprint(
            [chunk["content"] for chunk in chunks]
        )
        first_metadata = chunks[0]["metadata"] if chunks else {}
        by_fingerprint[fingerprint].append(
            {
                "doc_id": doc_id,
                "source_file": first_metadata.get("source_file"),
                "timestamp": _document_timestamp(first_metadata),
                "chunk_ids": [chunk["id"] for chunk in chunks],
                "chunks": chunks,
                "fingerprint": fingerprint,
            }
        )

    duplicate_groups = []
    for fingerprint, candidates in by_fingerprint.items():
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda item: (item["timestamp"], item["doc_id"]), reverse=True)
        duplicate_groups.append(
            {
                "fingerprint": fingerprint,
                "keep": candidates[0],
                "remove": candidates[1:],
            }
        )

    return {
        "total_documents": len(documents),
        "total_chunks": len(data.get("ids", [])),
        "duplicate_groups": duplicate_groups,
        "documents_to_remove": sum(len(group["remove"]) for group in duplicate_groups),
        "chunks_to_remove": sum(
            len(document["chunk_ids"])
            for group in duplicate_groups
            for document in group["remove"]
        ),
    }


def apply_cleanup_plan(collection, plan: dict) -> dict:
    """Apply a previously generated cleanup plan."""
    removed_ids = [
        chunk_id
        for group in plan["duplicate_groups"]
        for document in group["remove"]
        for chunk_id in document["chunk_ids"]
    ]
    if removed_ids:
        collection.delete(ids=removed_ids)

    updated_chunks = 0
    for group in plan["duplicate_groups"]:
        keeper = group["keep"]
        ids = []
        metadatas = []
        for chunk in keeper["chunks"]:
            metadata = dict(chunk["metadata"])
            metadata["chunk_fingerprint"] = keeper["fingerprint"]
            ids.append(chunk["id"])
            metadatas.append(metadata)
        if ids:
            collection.update(ids=ids, metadatas=metadatas)
            updated_chunks += len(ids)

    return {
        "documents_removed": plan["documents_to_remove"],
        "chunks_removed": len(removed_ids),
        "keeper_chunks_updated": updated_chunks,
    }


def _print_plan(collection_name: str, persist_directory: Path, plan: dict) -> None:
    print(f"Collection: {collection_name}")
    print(f"Persist directory: {persist_directory}")
    print(f"Documents: {plan['total_documents']}")
    print(f"Chunks: {plan['total_chunks']}")
    print(f"Duplicate groups: {len(plan['duplicate_groups'])}")
    for group in plan["duplicate_groups"]:
        keeper = group["keep"]
        removals = ", ".join(document["doc_id"] for document in group["remove"])
        print(
            f"- keep {keeper['doc_id']} ({keeper['source_file']}, {keeper['timestamp']}); "
            f"remove {removals}"
        )
    print(f"Documents to remove: {plan['documents_to_remove']}")
    print(f"Chunks to remove: {plan['chunks_to_remove']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", help="ChromaDB collection name")
    parser.add_argument("--persist-directory", help="Override the configured ChromaDB directory")
    parser.add_argument("--apply", action="store_true", help="Back up the store and apply deletions")
    args = parser.parse_args()

    persist_directory = _resolve_persist_directory(args.persist_directory)
    if not persist_directory.exists():
        parser.error(f"Persist directory does not exist: {persist_directory}")

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:
        parser.error(f"Collection {args.collection!r} was not found: {exc}")

    plan = build_cleanup_plan(collection)
    _print_plan(args.collection, persist_directory, plan)
    if not args.apply:
        print("Dry run only. Re-run with --apply while the backend is stopped.")
        return 0

    del collection
    del client
    backup_directory = persist_directory.with_name(
        f"{persist_directory.name}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.copytree(persist_directory, backup_directory)
    print(f"Backup created: {backup_directory}")

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_collection(args.collection)
    result = apply_cleanup_plan(collection, plan)
    print(json.dumps(result, indent=2))
    print(f"Remaining chunks: {collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
