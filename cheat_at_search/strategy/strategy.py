import gc
from hashlib import md5
from pathlib import Path

import pandas as pd
import numpy as np
from searcharray import SearchArray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cheat_at_search.data_dir import ensure_data_subdir


class SearchStrategy:
    def __init__(self, corpus, top_k=5, workers=1):
        self.corpus = corpus
        self.top_k = top_k
        self.workers = workers

    # @property
    # def cache_key(self):
    #     subclasses overriding this property can enable caching of search results based on the returned key

    def search_all(
        self,
        queries,
        k=10,
        batch_size=100,
        show_progress=True,
        cache=True,
    ):
        if callable(getattr(self, "search_batch", None)):
            return self._search_all_batched(
                queries,
                k=k,
                batch_size=batch_size,
                show_progress=show_progress,
                cache=cache,
            )
        return self._search_all_single(
            queries,
            k=k,
            batch_size=batch_size,
            show_progress=show_progress,
            cache=cache,
        )

    def answer_all(
        self,
        queries,
        batch_size=100,
        show_progress=True,
        cache=True,
    ):
        all_results = []
        total_queries = len(queries)
        if total_queries == 0:
            return pd.DataFrame()

        cache_dir = self._cache_dir()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            progress = tqdm(
                total=total_queries,
                desc="Answering",
                disable=not show_progress,
            )
            try:
                for batch_index, start in enumerate(
                    range(0, total_queries, batch_size)
                ):
                    batch = queries.iloc[start : start + batch_size]
                    cache_path = self._batch_cache_path(cache_dir, batch_index, batch)
                    if cache_path is not None and cache and cache_path.exists():
                        all_results.append(pd.read_pickle(cache_path))
                        progress.update(len(batch))
                        continue
                    futures = {}
                    for _, query_row in batch.iterrows():
                        future = executor.submit(self.answer, query_row["query"])
                        futures[future] = query_row
                    batch_answers = []
                    batch_queries = []
                    batch_query_ids = []
                    for future in as_completed(futures):
                        query_row = futures[future]
                        answer = future.result()
                        batch_answers.append(answer)
                        batch_queries.append(query_row["query"])
                        batch_query_ids.append(query_row["query_id"])
                        progress.update(1)

                    batch_results = pd.DataFrame(
                        {
                            "query_id": batch_query_ids,
                            "query": batch_queries,
                            "answer": batch_answers,
                        }
                    )
                    if cache_path is not None:
                        batch_results.to_pickle(cache_path)
                    all_results.append(batch_results)
                    gc.collect()
            finally:
                progress.close()
        return pd.concat(all_results) if all_results else pd.DataFrame()

    def _cache_dir(self):
        cache_key = getattr(self, "cache_key", None)
        if not cache_key:
            return None
        base_dir = Path(ensure_data_subdir("search_cache"))
        cache_dir = base_dir / str(cache_key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _batch_cache_path(self, cache_dir, batch_index, batch):
        if cache_dir is None:
            return None
        batch_signature = "|".join(
            (
                batch["query_id"].astype(str)
                + ":"
                + batch["query"].astype(str)
            ).tolist()
        )
        batch_hash = md5(batch_signature.encode("utf-8")).hexdigest()
        return cache_dir / f"batch_{batch_index}_{batch_hash}.pkl"

    def _search_all_single(
        self,
        queries,
        k=10,
        batch_size=100,
        show_progress=True,
        cache=True,
    ):
        all_results = []
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
        cache_dir = self._cache_dir()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            progress = tqdm(
                total=total_queries,
                desc="Searching",
                disable=not show_progress,
            )
            try:
                for batch_index, start in enumerate(
                    range(0, total_queries, batch_size)
                ):
                    batch = queries.iloc[start : start + batch_size]
                    cache_path = self._batch_cache_path(cache_dir, batch_index, batch)
                    if cache_path is not None and cache and cache_path.exists():
                        all_results.append(pd.read_pickle(cache_path))
                        progress.update(len(batch))
                        continue
                    futures = {}
                    for _, query_row in batch.iterrows():
                        future = executor.submit(self.search, query_row["query"], k)
                        futures[future] = query_row
                    batch_results = SearchResultBatch()
                    for future in as_completed(futures):
                        query_row = futures[future]
                        top_k, scores = future.result()
                        if len(top_k) == 0:
                            top_k = [-1]
                            scores = [0]
                        query_id = query_row["query_id"]
                        ranks = np.arange(len(top_k)) + 1
                        query = query_row["query"]
                        batch_results.append(
                            list(top_k),
                            list(scores),
                            [query] * len(top_k),
                            [query_id] * len(top_k),
                            list(ranks),
                        )
                        progress.update(1)
                    batch_results = batch_results.to_df(corpus_no_searcharray)
                    if cache_path is not None:
                        batch_results.to_pickle(cache_path)
                    all_results.append(batch_results)
                    gc.collect()
            finally:
                progress.close()
        return pd.concat(all_results) if all_results else pd.DataFrame()

    def _search_all_batched(
        self,
        queries,
        k=10,
        batch_size=100,
        show_progress=True,
        cache=True,
    ):
        all_results = []
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
        cache_dir = self._cache_dir()
        progress = tqdm(
            total=total_queries,
            desc="Searching",
            disable=not show_progress,
        )
        try:
            for batch_index, start in enumerate(
                range(0, total_queries, batch_size)
            ):
                batch = queries.iloc[start : start + batch_size]
                cache_path = self._batch_cache_path(cache_dir, batch_index, batch)
                if cache_path is not None and cache and cache_path.exists():
                    all_results.append(pd.read_pickle(cache_path))
                    progress.update(len(batch))
                    continue
                batch_queries = batch["query"].tolist()
                batch_top_k, batch_scores = self.search_batch(batch_queries, k)
                batch_results = SearchResultBatch()
                for (_, query_row), top_k, scores in zip(
                    batch.iterrows(), batch_top_k, batch_scores
                ):
                    if len(top_k) == 0:
                        top_k = [-1]
                        scores = [0]
                    query_id = query_row["query_id"]
                    ranks = np.arange(len(top_k)) + 1
                    query = query_row["query"]
                    batch_results.append(
                        list(top_k),
                        list(scores),
                        [query] * len(top_k),
                        [query_id] * len(top_k),
                        list(ranks),
                    )
                    progress.update(1)
                results = batch_results.to_df(corpus_no_searcharray)
                if cache_path is not None:
                    results.to_pickle(cache_path)
                all_results.append(results)
                gc.collect()
        finally:
            progress.close()
        return pd.concat(all_results) if all_results else pd.DataFrame()

    def search(self, query, k):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses should implement this method.")

    # def search_batch(self, queries, k):
    #     raise NotImplementedError("Subclasses can implement this method for batch searching, or rely on the default single search implementation.")

    def answer(self, question):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses should implement this method.")


class SearchResultBatch:
    def __init__(self):
        self.top_ks = []
        self.scores = []
        self.queries = []
        self.query_ids = []
        self.ranks = []

    def append(self, top_ks, scores, queries, query_ids, ranks):
        self.top_ks.extend(top_ks)
        self.scores.extend(scores)
        self.queries.extend(queries)
        self.query_ids.extend(query_ids)
        self.ranks.extend(ranks)

    def to_df(self, corpus):
        rows = []
        for idx in self.top_ks:
            if idx == -1:
                row = {col: None for col in corpus.columns}
                if "doc_id" in row:
                    row["doc_id"] = -1
                rows.append(row)
            else:
                rows.append(corpus.iloc[idx].to_dict())
        results = pd.DataFrame(rows)
        results["score"] = self.scores
        results["query"] = self.queries
        results["query_id"] = self.query_ids
        results["rank"] = self.ranks
        return results
