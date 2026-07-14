import pandas as pd

from cheat_at_search import doug_blog_data


QUESTION_ANSWER = [
    ("What is a search relevance judgment list?",
     "A judgment list records how relevant particular documents are to particular queries. Search teams use those labels to evaluate ranking changes systematically instead of deciding that results merely look good."),
    ("Why does Doug prefer the term “judgment list” over “ground truth” or “golden set”?",
     "Because relevance labels are not an objective and final truth. Explicit raters, crowdsourced raters, clicks, purchases, and other signals each provide a different and biased view of relevance, so teams should often use multiple evaluation perspectives."),
    ("What does Doug's “time well spent” search metric attempt to model?",
     "It models the reward and frustration accumulated while a user scans results. Relevant results add value, while irrelevant results waste the user's time and subtract from the score."),
    ("Why does Doug say NDCG is overrated rather than useless?",
     "NDCG is a useful measurement of ranked query-document relevance, but it represents only one narrow view of search quality. Its judgments may have incomplete coverage and embedded biases, and it does not capture concerns such as result diversity, interface quality, latency, or whether a change behaved as intended.")
]


def _judgments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query_id": int(query_id), "query": query, "answer": answer}
            for query_id, (query, answer) in enumerate(QUESTION_ANSWER, start=1)
        ],
        columns=["query_id", "query", "answer"],
    )


def __getattr__(name):
    """Load the answer dataset lazily."""
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        judgments = _judgments()
        globals()["judgments"] = judgments
        queries = judgments[["query", "query_id"]].drop_duplicates().reset_index(
            drop=True
        )
        globals()["queries"] = queries
        return globals()[name]
    if name == "corpus":
        corpus = doug_blog_data.corpus
        globals()["corpus"] = corpus
        return corpus
    raise AttributeError(f"Module {__name__} has no attribute {name}")
