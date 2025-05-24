from cheat_at_search.strategy import (
    BM25Search,
    MiniLMSearch,
    EnrichedBM25Search,
    EnrichedJustRoomBM25Search,
)
from cheat_at_search.wands_data import products, queries
from cheat_at_search.eval import grade_results
import argparse


strategies = {
    "bm25": BM25Search,
    "minilm": MiniLMSearch,
    "bm25+enriched": EnrichedBM25Search,
    "bm25+enriched_room": EnrichedJustRoomBM25Search,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Search strategy comparison")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=strategies.keys(),
        default="bm25",
        help="Search strategy to use (default: bm25)",
    )
    args = parser.parse_args()
    args.strategy = strategies[args.strategy]
    return args


def main():
    args = parse_args()
    search_strategy = args.strategy(products)
    results = search_strategy.search_all(queries)
    graded = grade_results(
        results,
        max_grade=2,
        k=10,
    )
    idcg = graded['idcg'].iloc[0]
    dcgs = graded.groupby("query")["discounted_gain"].sum().sort_values(ascending=False)
    ndcgs = dcgs / idcg
    for query, ndcg in zip(ndcgs.index, ndcgs):
        print(f"{query}: {ndcg:.4f}")


if __name__ == "__main__":
    main()
