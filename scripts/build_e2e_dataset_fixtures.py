"""Build small dataset fixtures for e2e dataset tests.

This script downloads/loads each source dataset, samples a small deterministic
subset, and writes fixture files under ``tests/e2e/fixtures``. The fixture
layout intentionally mirrors the raw files each dataset loader expects where
that is practical, so future e2e tests can patch the fetch/download helpers to
point at these smaller local sources.

The script is intentionally not wired into tests or CI. Run it manually when
refreshing fixtures:

    poetry run python scripts/build_e2e_dataset_fixtures.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("tests/e2e/fixtures/datasets")
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_RANDOM_STATE = 42


def _sample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy().reset_index(drop=True)
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _sample_by_ids(
    df: pd.DataFrame,
    id_column: str,
    ids: set,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if id_column not in df.columns:
        return _sample(df, sample_size, random_state)
    filtered = df[df[id_column].isin(ids)]
    if filtered.empty:
        return _sample(df, sample_size, random_state)
    return _sample(filtered, sample_size, random_state)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _copy_text_sample(
    source_path: Path,
    output_path: Path,
    sample_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r") as source, output_path.open("w") as output:
        for idx, line in enumerate(source):
            if idx >= sample_size:
                break
            output.write(line)


def build_msmarco_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    from cheat_at_search import msmarco_data

    msmarco_data.download_msmarco()
    source_dir = msmarco_data.msmarco_path
    fixture_dir = output_dir / "msmarco"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    collection = pd.read_csv(
        source_dir / "collection.tsv",
        sep="\t",
        names=["doc_id", "description"],
    )
    collection_sample = _sample(collection, sample_size, random_state)
    collection_sample.to_csv(
        fixture_dir / "collection.tsv",
        sep="\t",
        header=False,
        index=False,
    )

    sampled_doc_ids = set(collection_sample["doc_id"].tolist())
    for variant in ["dev.small", "train"]:
        qrels_path = source_dir / f"qrels.{variant}.tsv"
        queries_path = source_dir / f"queries.{variant}.tsv"
        if not qrels_path.exists() or not queries_path.exists():
            continue
        qrels = pd.read_csv(
            qrels_path,
            sep="\t",
            names=["query_id", "unused", "doc_id", "grade"],
        )
        qrels = _sample_by_ids(qrels, "doc_id", sampled_doc_ids, sample_size, random_state)
        qrels.to_csv(
            fixture_dir / f"qrels.{variant}.tsv",
            sep="\t",
            header=False,
            index=False,
        )

        query_ids = set(qrels["query_id"].tolist())
        queries = pd.read_csv(queries_path, sep="\t", names=["query_id", "query"])
        queries = _sample_by_ids(queries, "query_id", query_ids, sample_size, random_state)
        queries.to_csv(
            fixture_dir / f"queries.{variant}.tsv",
            sep="\t",
            header=False,
            index=False,
        )


def build_minimarco_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    """Build a MiniMARCO fixture from the MSMARCO source files."""
    build_msmarco_fixture(output_dir / "_shared", sample_size, random_state)
    source_dir = output_dir / "_shared" / "msmarco"
    fixture_dir = output_dir / "minimarco"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    shutil.copytree(source_dir, fixture_dir)
    shutil.rmtree(output_dir / "_shared")


def build_esci_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    from cheat_at_search import esci_data

    source_dir = esci_data.fetch_esci() / "shopping_queries_dataset"
    fixture_dir = output_dir / "esci" / "shopping_queries_dataset"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    products = pd.read_parquet(source_dir / "shopping_queries_dataset_products.parquet")
    products_sample = _sample(products, sample_size, random_state)
    products_sample.to_parquet(
        fixture_dir / "shopping_queries_dataset_products.parquet",
        index=False,
    )

    product_ids = set(products_sample["product_id"].tolist())
    examples = pd.read_parquet(source_dir / "shopping_queries_dataset_examples.parquet")
    examples_sample = _sample_by_ids(
        examples,
        "product_id",
        product_ids,
        sample_size,
        random_state,
    )
    examples_sample.to_parquet(
        fixture_dir / "shopping_queries_dataset_examples.parquet",
        index=False,
    )


def build_wands_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    from cheat_at_search import wands_data

    source_dir = wands_data.fetch_wands() / "dataset"
    fixture_dir = output_dir / "wands" / "dataset"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(source_dir / "product.csv", sep="\t")
    products_sample = _sample(products, sample_size, random_state)
    products_sample.to_csv(fixture_dir / "product.csv", sep="\t", index=False)

    product_ids = set(products_sample["product_id"].tolist())
    labels = pd.read_csv(source_dir / "label.csv", sep="\t")
    labels_sample = _sample_by_ids(labels, "product_id", product_ids, sample_size, random_state)
    labels_sample.to_csv(fixture_dir / "label.csv", sep="\t", index=False)

    query_ids = set(labels_sample["query_id"].tolist())
    queries = pd.read_csv(source_dir / "query.csv", sep="\t")
    queries_sample = _sample_by_ids(queries, "query_id", query_ids, sample_size, random_state)
    queries_sample.to_csv(fixture_dir / "query.csv", sep="\t", index=False)

    enriched_dir = source_dir / "enriched"
    if enriched_dir.exists():
        fixture_enriched_dir = fixture_dir / "enriched"
        fixture_enriched_dir.mkdir(parents=True, exist_ok=True)

        enriched_products_path = enriched_dir / "enriched_products.csv.gz"
        if enriched_products_path.exists():
            enriched_products = pd.read_csv(enriched_products_path, compression="gzip")
            enriched_products = _sample_by_ids(
                enriched_products,
                "product_id",
                product_ids,
                sample_size,
                random_state,
            )
            enriched_products.to_csv(
                fixture_enriched_dir / "enriched_products.csv.gz",
                compression="gzip",
                index=False,
            )

        query_attributes_path = enriched_dir / "query_attributes.csv"
        if query_attributes_path.exists():
            query_attributes = pd.read_csv(query_attributes_path)
            query_attributes = _sample_by_ids(
                query_attributes,
                "query_id",
                query_ids,
                sample_size,
                random_state,
            )
            query_attributes.to_csv(fixture_enriched_dir / "query_attributes.csv", index=False)


def build_tmdb_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    from cheat_at_search import tmdb_data

    tmdb_data.fetch_tmdb()
    source_dir = tmdb_data.tmdb_path
    fixture_dir = output_dir / "tmdb"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    movies_path = source_dir / "tmdb.json"
    if not movies_path.exists():
        with tarfile.open(source_dir / "movies.tgz", "r:gz") as tar:
            tar.extractall(path=source_dir)
    movies = pd.read_json(movies_path, orient="index")
    movies_sample = _sample(movies, sample_size, random_state)
    movies_sample.to_json(fixture_dir / "tmdb.json", orient="index")

    judgments_path = source_dir / "ai_pow_search_judgments.txt"
    if not judgments_path.exists():
        with tarfile.open(source_dir / "judgments.tgz", "r:gz") as tar:
            tar.extractall(path=source_dir)
    sampled_movie_ids = {str(movie_id) for movie_id in movies_sample.index.tolist()}
    with judgments_path.open("r") as source, (fixture_dir / "ai_pow_search_judgments.txt").open("w") as output:
        written = 0
        for line in source:
            parts = line.split()
            if line.startswith("#") or len(parts) < 4 or parts[2] != "#":
                continue
            if parts[3] not in sampled_movie_ids:
                continue
            output.write(line)
            written += 1
            if written >= sample_size:
                break


def build_doug_blog_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    from cheat_at_search import doug_blog_data

    fixture_dir = output_dir / "doug_blog"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    posts = doug_blog_data._docs()
    posts_sample = _sample(posts, sample_size, random_state)
    with gzip.open(fixture_dir / "posts.json.gz", "wt") as handle:
        posts_sample.to_json(handle, orient="records", lines=True)


def build_bc_plus_fixture(output_dir: Path, sample_size: int, random_state: int) -> None:
    from cheat_at_search import bc_plus_data

    fixture_dir = output_dir / "browsecomp_plus"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    queries = bc_plus_data._load_queries()
    queries_sample = _sample(queries, sample_size, random_state)
    judgments = bc_plus_data._build_judgments(queries_sample)
    corpus = bc_plus_data._load_corpus()
    doc_ids = set(judgments["doc_id"].tolist()) if "doc_id" in judgments.columns else set()
    corpus_sample = _sample_by_ids(corpus, "doc_id", doc_ids, sample_size, random_state)

    queries_sample[["query", "query_id"]].drop_duplicates().to_parquet(
        fixture_dir / "queries.parquet",
        index=False,
    )
    judgments.to_parquet(fixture_dir / "judgments.parquet", index=False)
    corpus_sample.to_parquet(fixture_dir / "corpus.parquet", index=False)


BUILDERS = {
    "msmarco": build_msmarco_fixture,
    "minimarco": build_minimarco_fixture,
    "esci": build_esci_fixture,
    "wands": build_wands_fixture,
    "tmdb": build_tmdb_fixture,
    "doug_blog": build_doug_blog_fixture,
    "bc_plus": build_bc_plus_fixture,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sampled fixtures should be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum rows to sample from each source file.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Deterministic pandas sample seed.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(BUILDERS),
        default=sorted(BUILDERS),
        help="Datasets to build fixtures for.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "sample_size": args.sample_size,
            "random_state": args.random_state,
            "datasets": args.datasets,
        },
    )
    for dataset in args.datasets:
        print(f"Building {dataset} fixture...")
        BUILDERS[dataset](args.output_dir, args.sample_size, args.random_state)
    print(f"Fixtures written to {args.output_dir}")


if __name__ == "__main__":
    main()
