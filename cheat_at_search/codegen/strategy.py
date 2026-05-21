from __future__ import annotations

from typing import Optional

import numpy as np
from pandas.api.types import is_integer_dtype, is_string_dtype

from cheat_at_search.codegen.validators import _resolve_logger
from cheat_at_search.strategy import SearchStrategy


def _coerce_doc_ids(doc_ids, doc_id_col):
    target_type = None
    if is_integer_dtype(doc_id_col.dtype):
        target_type = int
    elif is_string_dtype(doc_id_col.dtype):
        target_type = str
    else:
        for value in doc_id_col:
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            if isinstance(value, (int, np.integer)):
                target_type = int
                break
            if isinstance(value, str):
                target_type = str
                break
    if target_type is None:
        return doc_ids
    coerced = []
    for doc_id in doc_ids:
        if target_type is int:
            try:
                coerced.append(int(doc_id))
            except (TypeError, ValueError):
                coerced.append(doc_id)
        else:
            try:
                coerced.append(str(doc_id))
            except Exception:
                coerced.append(doc_id)
    return coerced


class CodeGenSearchStrategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        tool_fns,
        module_name: str,
        code: Optional[str] = None,
        workers=1,
        logger=None,
    ):
        super().__init__(corpus, workers=workers)
        self.index = corpus
        self.tool_fns = tool_fns
        self.module_name = module_name
        self.code = code
        self.logger = _resolve_logger(logger)

    def _call_rerank(self, rerank_fn, query, top_k):
        try:
            return rerank_fn(query, top_k, *self.tool_fns)
        except TypeError:
            return rerank_fn(query, *self.tool_fns)

    def search(self, query, k=10):
        if not self.code:
            raise ValueError(
                "CodeGenSearchStrategy requires code to be provided; "
                "module import is no longer used."
            )
        rerank_fn = self._rerank_fn_from_code(self.code, module_name=self.module_name)

        doc_ids = self._call_rerank(rerank_fn, query, k)[:k]
        if doc_ids and isinstance(doc_ids[0], (list, tuple)):
            doc_ids = [doc_id for doc_id, _ in doc_ids]
        doc_ids = _coerce_doc_ids(doc_ids, self.index["doc_id"])
        scores = np.arange(len(doc_ids), 0, -1)
        top_k_ilocs = []
        for doc_id in doc_ids:
            iloc = self.index.index[self.index["doc_id"] == doc_id].tolist()
            if len(iloc):
                top_k_ilocs.append(iloc[0])
            else:
                self.logger.info("Doc ID %s not found in corpus", doc_id)
                continue
        scores = scores[:k]
        return top_k_ilocs, scores

    @staticmethod
    def _rerank_fn_from_code(code: str, module_name: Optional[str] = None):
        exec_globals = {}
        exec(code, exec_globals)
        if module_name:
            candidate = exec_globals.get(module_name)
            if callable(candidate):
                return candidate
        raise ValueError(
            "No rerank function found in code. Define a callable named "
            f"'{module_name}'."
        )
