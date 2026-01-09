"""Unified Chroma tool for ingesting and querying vector collections.

Usage:
  # Ingest Tebliğ chunks:
  python src/chroma_tool.py ingest \
    --json data/teblig/chunks/teblig_chunks_vector_ready.json \
    --persist-dir .chroma/teblig \
    --collection teblig_chunks

  # Single query:
  python src/chroma_tool.py query \
    --query "kredi kartı aidatı iadesi" \
    --persist-dir .chroma/teblig \
    --collections teblig_chunks \
    --top-n 10

  # Batch query from source JSONs:
  python src/chroma_tool.py query \
    --source-jsons data/banka_dokumanlari/chunks/file1.json,file2.json \
    --persist-dir .chroma/teblig \
    --collections teblig_chunks \
    --scope-match \
    --min-sim 0.5 \
    --save-dir logs/output

  # Auto-run batch (default when no args):
  python src/chroma_tool.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Set

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PERSIST_DIR = Path(".chroma/teblig")
DEFAULT_COLLECTION = "teblig_chunks"

DEFAULT_AUTO_BATCH = {
    "source_jsons": [
        Path("data/banka_dokumanlari/chunks/KREDI_KARTI_SOZLESMESI_HIBRIT_01.08.2025_GUNCEL_vector_ready.json"),
        Path("data/banka_dokumanlari/chunks/Bireysel_Bankacilik_Hizmetleri_Sozlesmesi_2025_vector_ready.json"),
        Path("data/banka_dokumanlari/chunks/ucret_tarifeleri_2025_vector_ready.json"),
    ],
    "save_dir": Path("logs/banka_vs_teblig"),
    "scope_match": True,
    "min_sim": 0.4,
    "overwrite": True,
    "persist_dir": DEFAULT_PERSIST_DIR,
    "collections": DEFAULT_COLLECTION,
    "top_n": 5,
}

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def pick_text(row: Dict[str, Any]) -> str:
    """Extract embedding text from chunk row."""
    return (
        row.get("text_for_embedding")
        or row.get("text")
        or row.get("text_full")
        or ""
    ).strip()


def to_scope_set(val: Any) -> Set[str]:
    """Convert scope value to set of strings."""
    if isinstance(val, list):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",")]
        return {p for p in parts if p}
    return set()


# ============================================================================
# INGEST FUNCTIONS
# ============================================================================

def load_chunks_for_ingest(json_path: Path, include_parents: bool = False) -> List[Dict[str, Any]]:
    """Load chunks from JSON for ingestion."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of chunk objects.")

    if include_parents:
        return data
    return [row for row in data if row.get("chunk_kind") == "child"]


def build_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build Chroma metadata dict from chunk row."""
    scope = row.get("scope")
    # Chroma metadata values must be scalar; flatten scope lists to a string.
    if isinstance(scope, list):
        scope = ",".join([str(s) for s in scope])
    elif scope is not None:
        scope = str(scope)

    return {
        "doc_id": row.get("doc_id"),
        "unit_id": row.get("unit_id"),
        "chunk_kind": row.get("chunk_kind"),
        "parent_chunk_id": row.get("parent_chunk_id"),
        "scope": scope,
        "bolum_no": row.get("bolum_no"),
        "madde_no": row.get("madde_no"),
        "fikra_no": row.get("fikra_no"),
    }


def ingest(
    json_path: Path,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    include_parents: bool = False,
    batch_size: int = 64,
) -> None:
    """Ingest chunks into Chroma collection."""
    rows = load_chunks_for_ingest(json_path, include_parents=include_parents)
    if not rows:
        raise ValueError("No chunks to ingest.")

    texts = [pick_text(r) for r in rows]
    ids = [r.get("chunk_id") for r in rows]
    metadatas = [build_metadata(r) for r in rows]

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"Collection {collection_name} already exists with {collection.count()} items.")
        # Optional: collection.delete(ids=ids) to overwrite
        
    # Batch ingest
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        
        collection.upsert(
            documents=batch_texts,
            ids=batch_ids,
            metadatas=batch_metas
        )
        print(f"Ingested batch {i // batch_size + 1}/{(len(rows) + batch_size - 1) // batch_size}")
    
    print(f"✓ Ingestion complete. Collection: {collection_name} ({collection.count()} items)")


def ingest_results(
    json_path: Path,
    persist_dir: Path,
    collection_name: str = "compliance_insights",
    model_name: str = DEFAULT_MODEL,
) -> None:
    """Ingest COMPLIANCE RESULTS into a separate collection for the chatbot."""
    if not json_path.exists():
        print(f"⚠️ Result file not found: {json_path}")
        return

    with json_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Ingesting {len(results)} compliance results into '{collection_name}'...")
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)
    model = SentenceTransformer(model_name)
    
    docs = []
    ids = []
    metas = []
    
    seen_ids = set()
    for res in results:
        chunk_id = res.get("chunk_id", "unknown")
        
        # Skip duplicates
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        # Create a rich text representation for the chatbot context
        # Format: STATUS | CHUNK_ID | REASON | TEBLIG_CITATION
        text_content = (
            f"Durum: {res.get('status')}\n"
            f"Doküman Parçası: {res.get('chunk_text', '')}\n"
            f"Analiz Nedeni: {res.get('reason', '')}\n"
            f"İlgili Tebliğ: {res.get('citation', '')}\n"
            f"Düzeltilmiş Metin: {res.get('corrected_text', '')}"
        )
        
        docs.append(text_content)
        ids.append(chunk_id)
        metas.append({
            "chunk_id": chunk_id,
            "status": res.get("status", "UNKNOWN"),
            "document_type": "Unknown", # Can be populated if needed
        })
        
    collection.upsert(
        documents=docs,
        ids=ids,
        metadatas=metas
    )
    
    print(f"✓ Result ingestion complete. Collection: {collection_name} ({collection.count()} items)")



# ============================================================================
# QUERY FUNCTIONS
# ============================================================================

def query_collection(
    client: chromadb.ClientAPI, # pyright: ignore[reportPrivateImportUsage]
    collection_name: str,
    query_embedding,
    top_n: int,
) -> Dict[str, Any]:
    """Query a Chroma collection."""
    col = client.get_collection(name=collection_name)
    return col.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    ) # pyright: ignore[reportReturnType]


def print_results(collection_name: str, result: Dict[str, Any], emit: Callable[[str], None]) -> None:
    """Print query results."""
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0] if "ids" in result else [None] * len(docs)

    emit("")
    emit(f"=== {collection_name} ===")
    if not docs:
        emit("No results.")
        return

    for rank, (doc, meta, dist, cid) in enumerate(zip(docs, metas, dists, ids), start=1):
        sim = 1 - dist  # cosine space → similarity approximation
        scope = meta.get("scope")
        if scope is None:
            scope = "-"
        emit(f"[{rank:02d}] sim={sim:.4f} | id={cid}")
        emit(f"     scope={scope} | madde={meta.get('madde_no')} | fikra={meta.get('fikra_no')} | parent={meta.get('parent_chunk_id')}")
        # Do not truncate; downstream LLM needs full chunk text for accurate reasoning.
        full_text = doc.replace("\n", " ")
        emit(f"     text={full_text}")


def filter_result_by_scope(
    result: Dict[str, Any],
    q_scope: Set[str],
    top_n: int,
    scope_match: bool,
    min_sim: Optional[float],
) -> Dict[str, Any]:
    """Filter results by scope match and similarity threshold."""
    if not result:
        return result

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0] if "ids" in result else [None] * len(docs)

    filtered_docs = []
    filtered_metas = []
    filtered_dists = []
    filtered_ids = []

    for doc, meta, dist, cid in zip(docs, metas, dists, ids):
        sim = 1 - dist
        if min_sim is not None and sim < min_sim:
            continue
        target_scope = to_scope_set(meta.get("scope"))
        if scope_match and q_scope and not (target_scope & q_scope):
            continue
        filtered_docs.append(doc)
        filtered_metas.append(meta)
        filtered_dists.append(dist)
        filtered_ids.append(cid)
        if len(filtered_docs) >= top_n:
            break

    return {
        "documents": [filtered_docs],
        "metadatas": [filtered_metas],
        "distances": [filtered_dists],
        "ids": [filtered_ids],
    }


def load_source_chunks(path: Path, include_parents: bool) -> List[Dict[str, Any]]:
    """Load source chunks for batch querying."""
    if not path.exists():
        raise FileNotFoundError(f"Source JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected list of chunk objects in source JSON.")
    if include_parents:
        return data
    return [row for row in data if row.get("chunk_kind") == "child"]


def query_single(
    query_text: str,
    collections: List[str],
    persist_dir: Path,
    model: SentenceTransformer,
    top_n: int,
    min_sim: Optional[float],
    save_path: Optional[Path],
    overwrite: bool,
) -> None:
    """Execute a single query."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    file_handle = None
    emit: Callable[[str], None]
    if save_path:
        mode = "w" if overwrite else "a"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        file_handle = save_path.open(mode, encoding="utf-8")

        def _emit(line: str = "") -> None:
            print(line)
            file_handle.write(line + "\n")

        emit = _emit
    else:
        emit = print

    q_embed = model.encode(query_text, normalize_embeddings=True)
    for name in collections:
        try:
            result = query_collection(client, name, q_embed, top_n)
            filtered = filter_result_by_scope(result, set(), top_n, False, min_sim)
            print_results(name, filtered, emit)
        except Exception as exc:
            emit("")
            emit(f"=== {name} ===")
            emit(f"Error querying collection: {exc}")
    
    if file_handle:
        file_handle.close()


def query_batch(
    source_list: List[Path],
    collections: List[str],
    persist_dir: Path,
    model: SentenceTransformer,
    top_n: int,
    scope_match: bool,
    min_sim: Optional[float],
    include_parents: bool,
    source_limit: Optional[int],
    save_path: Optional[Path],
    save_dir: Optional[Path],
    overwrite: bool,
) -> None:
    """Execute batch queries from source JSONs."""
    client = chromadb.PersistentClient(path=str(persist_dir))

    for src_path in source_list:
        if not src_path:
            continue

        # Decide output file
        file_handle = None
        emit: Callable[[str], None]
        if save_path:
            out_path = save_path
        elif save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            base = src_path.stem
            suffix = collections[0] if collections else "results"
            out_path = save_dir / f"{base}_vs_{suffix}.txt"
        else:
            out_path = None

        if out_path:
            mode = "w" if overwrite else "a"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            file_handle = out_path.open(mode, encoding="utf-8")

            def _emit(line: str = "") -> None:
                print(line)
                file_handle.write(line + "\n") # pyright: ignore[reportOptionalMemberAccess]

            emit = _emit
        else:
            emit = print

        source_rows = load_source_chunks(src_path, include_parents=include_parents)
        if source_limit is not None:
            source_rows = source_rows[:source_limit]

        print(f"\n{'='*60}")
        print(f"Processing: {src_path.name} ({len(source_rows)} chunks)")
        print(f"{'='*60}")

        for idx, row in enumerate(source_rows, start=1):
            q_text = pick_text(row)
            q_id = row.get("chunk_id")
            q_scope = row.get("scope")
            q_madde = row.get("madde_no")
            q_fikra = row.get("fikra_no")

            q_scope_set = to_scope_set(q_scope)
            q_embed = model.encode(q_text, normalize_embeddings=True)
            emit("")
            emit(f"##### Source {idx}/{len(source_rows)} | id={q_id} | scope={q_scope} | madde={q_madde} | fikra={q_fikra}")
            src_snippet = q_text[:240].replace("\n", " ")
            emit(f"text={src_snippet}{'...' if len(q_text) > 240 else ''}")

            for name in collections:
                try:
                    result = query_collection(client, name, q_embed, top_n)
                    filtered = filter_result_by_scope(result, q_scope_set, top_n, scope_match, min_sim)
                    print_results(name, filtered, emit)
                except Exception as exc:
                    emit("")
                    emit(f"=== {name} ===")
                    emit(f"Error querying collection: {exc}")

        if file_handle:
            file_handle.close()
            print(f"✓ Results saved to: {out_path}")


# ============================================================================
# MAIN CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Chroma tool for ingesting and querying vector collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ========== INGEST SUBCOMMAND ==========
    ingest_parser = subparsers.add_parser("ingest", help="Ingest chunks into Chroma collection")
    ingest_parser.add_argument(
        "--json",
        type=Path,
        default=Path("data/teblig/chunks/teblig_chunks_vector_ready.json"),
        help="Path to chunk JSON file.",
    )
    ingest_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory for Chroma persistence.",
    )
    ingest_parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Chroma collection name.",
    )
    ingest_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name.",
    )
    ingest_parser.add_argument(
        "--include-parents",
        action="store_true",
        help="Also ingest parent chunks (not recommended for ranking).",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation.",
    )

    # ========== QUERY SUBCOMMAND ==========
    query_parser = subparsers.add_parser("query", help="Query Chroma collections")
    query_parser.add_argument("--query", help="Tek seferlik sorgu metni (source-json yoksa gerekli).")
    query_parser.add_argument(
        "--source-json",
        type=Path,
        help="Tek kaynak chunk JSON'u (örn. banka_word vector_ready). Her child chunk tek tek sorgu olur.",
    )
    query_parser.add_argument(
        "--source-jsons",
        type=str,
        help="Virgülle ayrılmış çoklu kaynak JSON listesi; her biri sırayla işlenir.",
    )
    query_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Chroma persistence directory (aynı dizinde koleksiyonlar bulunmalı).",
    )
    query_parser.add_argument(
        "--collections",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Virgülle ayrılmış koleksiyon isimleri (ör: teblig_chunks,banka_pdf).",
    )
    query_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Query embedding için SentenceTransformer modeli (ingest ile aynı olmalı).",
    )
    query_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Her koleksiyon için döndürülecek sonuç sayısı (varsayılan 5).",
    )
    query_parser.add_argument(
        "--min-sim",
        type=float,
        default=None,
        help="Benzerlik eşiği (0-1). Örn: 0.5 verildiğinde bu değerin altındakiler atılır.",
    )
    query_parser.add_argument(
        "--scope-match",
        action="store_true",
        help="Kaynak chunk scope'u ile hedef sonuç scope'unu kesişim olacak şekilde filtrele.",
    )
    query_parser.add_argument(
        "--include-parents",
        action="store_true",
        help="source-json okurken parent chunk'ları da sorgula (varsayılan: sadece child).",
    )
    query_parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Kaynak chunk sayısını sınırla (test için).",
    )
    query_parser.add_argument(
        "--save-path",
        type=Path,
        help="Sonuçları bu dosyaya yaz (yoksa oluşturur).",
    )
    query_parser.add_argument(
        "--save-dir",
        type=Path,
        help="Birden fazla kaynak için otomatik çıktı dizini; dosya adı kaynak adı + '_vs_' + ilk koleksiyon + '.txt' olur.",
    )
    query_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="--save-path ile dosyayı baştan yaz (varsayılan: ekle).",
    )

    args = parser.parse_args()

    # ========== AUTO-RUN BATCH MODE ==========
    if not args.command or (args.command == "query" and len(sys.argv) == 2):
        print("=" * 60)
        print("[AUTO] Argüman verilmedi, varsayılan banka_vs_teblig batch çalışıyor.")
        print("=" * 60)
        print(f"Model: {DEFAULT_MODEL}")
        print(f"Persist Dir: {DEFAULT_AUTO_BATCH['persist_dir']}")
        print(f"Collections: {DEFAULT_AUTO_BATCH['collections']}")
        print(f"Top-N: {DEFAULT_AUTO_BATCH['top_n']}")
        print(f"Scope Match: {DEFAULT_AUTO_BATCH['scope_match']}")
        print(f"Min Similarity: {DEFAULT_AUTO_BATCH['min_sim']}")
        print(f"Save Dir: {DEFAULT_AUTO_BATCH['save_dir']}")
        print("=" * 60)

        model = SentenceTransformer(DEFAULT_MODEL)

        query_batch(
            source_list=DEFAULT_AUTO_BATCH["source_jsons"],
            collections=[DEFAULT_AUTO_BATCH["collections"]],
            persist_dir=DEFAULT_AUTO_BATCH["persist_dir"],
            model=model,
            top_n=DEFAULT_AUTO_BATCH["top_n"],
            scope_match=DEFAULT_AUTO_BATCH["scope_match"],
            min_sim=DEFAULT_AUTO_BATCH["min_sim"],
            include_parents=False,
            source_limit=None,
            save_path=None,
            save_dir=DEFAULT_AUTO_BATCH["save_dir"],
            overwrite=DEFAULT_AUTO_BATCH["overwrite"],
        )
        print("\n✓ Auto-batch completed!")
        return

    # ========== INGEST COMMAND ==========
    if args.command == "ingest":
        args.persist_dir.mkdir(parents=True, exist_ok=True)
        ingest(
            json_path=args.json,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            model_name=args.model,
            include_parents=args.include_parents,
            batch_size=args.batch_size,
        )
        return

    # ========== QUERY COMMAND ==========
    if args.command == "query":
        if args.min_sim is not None and not (0 <= args.min_sim <= 1):
            raise ValueError("--min-sim 0 ile 1 arasında olmalı.")

        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
        if not collections:
            raise ValueError("At least one collection name is required.")

        # Resolve multiple sources
        source_list: List[Optional[Path]] = []
        if args.source_json:
            source_list.append(args.source_json)
        if args.source_jsons:
            for item in args.source_jsons.split(","):
                if item := item.strip():
                    source_list.append(Path(item))

        if not source_list and not args.query:
            query_parser.print_help()
            print("\nEn az bir kaynak (--query veya --source-jsons/--source-json) vermen gerekiyor.")
            return

        print(f"Loading model: {args.model}")
        model = SentenceTransformer(args.model)

        # Single query mode
        if not source_list:
            query_single(
                query_text=args.query,
                collections=collections,
                persist_dir=args.persist_dir,
                model=model,
                top_n=args.top_n,
                min_sim=args.min_sim,
                save_path=args.save_path,
                overwrite=args.overwrite,
            )
        else:
            # Batch query mode
            query_batch(
                source_list=source_list, # pyright: ignore[reportArgumentType]
                collections=collections,
                persist_dir=args.persist_dir,
                model=model,
                top_n=args.top_n,
                scope_match=args.scope_match,
                min_sim=args.min_sim,
                include_parents=args.include_parents,
                source_limit=args.source_limit,
                save_path=args.save_path,
                save_dir=args.save_dir,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
