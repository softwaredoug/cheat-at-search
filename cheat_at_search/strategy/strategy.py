import gc
import pandas as pd
import numpy as np
from searcharray import SearchArray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class SearchStrategy:
    def __init__(self, corpus, top_k=5, workers=1):
        self.corpus = corpus
        self.top_k = top_k
        self.workers = workers

    def search_all(self, queries, k=10, batch_size=100, show_progress=True):
        if callable(getattr(self, "search_batch", None)):
            return self._search_all_batched(
                queries, k=k, batch_size=batch_size, show_progress=show_progress
            )
        return self._search_all_single(
            queries, k=k, batch_size=batch_size, show_progress=show_progress
        )

    def _search_all_single(self, queries, k=10, batch_size=100, show_progress=True):
        all_top_ks = []
        all_scores = []
        all_queries = []
        all_query_ids = []
        all_ranks = []
        total_queries = len(queries)
        if total_queries == 0:
            return pd.DataFrame()

        search_array_cols = [
            col
            for col in self.corpus.columns
            if isinstance(self.corpus[col].array, SearchArray)
        ]
        corpus_no_searcharray = self.corpus.drop(
            columns=search_array_cols, errors="ignore"
        )

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            progress = tqdm(
                total=total_queries,
                desc="Searching",
                disable=not show_progress,
            )
            try:
                for start in range(0, total_queries, batch_size):
                    batch = queries.iloc[start : start + batch_size]
                    futures = {}
                    for _, query_row in batch.iterrows():
                        future = executor.submit(self.search, query_row["query"], k)
                        futures[future] = query_row

                    for future in as_completed(futures):
                        query_row = futures[future]
                        top_k, scores = future.result()
                        query_id = query_row["query_id"]
                        ranks = np.arange(len(top_k)) + 1
                        query = query_row["query"]
                        all_top_ks.extend(list(top_k))
                        all_scores.extend(list(scores))
                        all_queries.extend([query] * len(top_k))
                        all_query_ids.extend([query_id] * len(top_k))
                        all_ranks.extend(list(ranks))
                        progress.update(1)

                    gc.collect()
            finally:
                progress.close()
        results = corpus_no_searcharray.iloc[all_top_ks].copy()
        results["score"] = all_scores
        results["query"] = all_queries
        results["query_id"] = all_query_ids
        results["rank"] = all_ranks
        return results

    def _search_all_batched(self, queries, k=10, batch_size=100, show_progress=True):
        all_top_ks = []
        all_scores = []
        all_queries = []
        all_query_ids = []
        all_ranks = []
        total_queries = len(queries)
        if total_queries == 0:
            return pd.DataFrame()

        search_array_cols = [
            col
            for col in self.corpus.columns
            if isinstance(self.corpus[col].array, SearchArray)
        ]
        corpus_no_searcharray = self.corpus.drop(
            columns=search_array_cols, errors="ignore"
        )
        progress = tqdm(
            total=total_queries,
            desc="Searching",
            disable=not show_progress,
        )
        try:
            for start in range(0, total_queries, batch_size):
                batch = queries.iloc[start : start + batch_size]
                batch_queries = batch["query"].tolist()
                batch_top_k, batch_scores = self.search_batch(batch_queries, k)
                for (_, query_row), top_k, scores in zip(
                    batch.iterrows(), batch_top_k, batch_scores
                ):
                    query_id = query_row["query_id"]
                    ranks = np.arange(len(top_k)) + 1
                    query = query_row["query"]
                    all_top_ks.extend(list(top_k))
                    all_scores.extend(list(scores))
                    all_queries.extend([query] * len(top_k))
                    all_query_ids.extend([query_id] * len(top_k))
                    all_ranks.extend(list(ranks))
                    progress.update(1)
                gc.collect()
        finally:
            progress.close()
        results = corpus_no_searcharray.iloc[all_top_ks].copy()
        results["score"] = all_scores
        results["query"] = all_queries
        results["query_id"] = all_query_ids
        results["rank"] = all_ranks
        return results

    def search(self, query, k):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses should implement this method.")
