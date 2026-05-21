from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from cheat_at_search.logger import log_to_stdout
from cheat_at_search.codegen.models import (
    Doc,
    Edit,
    EditResult,
    EvalResult,
    EvalResults,
    QueryEvalResult,
)
from cheat_at_search.codegen.grep import make_run_path_grep_tool
from cheat_at_search.codegen.train_log import write_training_logs
from cheat_at_search.codegen.validators import make_guardrail_checker, make_length_validator
from cheat_at_search.search import run_strategy
from cheat_at_search.codegen.strategy import CodeGenSearchStrategy

# Implementation breadcrumb:
# search-experiments should import this module directly instead of carrying a local copy.
# Porting notes:
# - Rerank functions should be defined as rerank_*(query, *tool_fns).
# - Guardrail checker uses OpenAIAgent.loop with reasoning_level.
# - Use Reranker.tools() to expose tool functions.


def _resolve_logger(logger=None, logger_name: str = "code"):
    if logger is not None:
        return logger
    return log_to_stdout(logger_name=logger_name)


class Reranker:
    def __init__(
        self,
        code_dir: str,
        tool_fns: list[callable],
        corpus: pd.DataFrame,
        judgments: pd.DataFrame,
        training_queries: List[str],
        validation_queries: Optional[List[str]] = None,
        module_name: str = "rerank_esci",
        guardrail_fns: List | None = None,
        eval_margin: float = 0.003,
        logger=None,
        files: dict[str, str] | None = None,
    ):
        if not training_queries:
            raise ValueError("training_queries must be provided for Reranker")
        self.code_dir = code_dir
        self.tool_fns = tool_fns
        self.module_name = module_name
        self.guardrail_fns = guardrail_fns or []
        self.eval_margin = eval_margin
        self.logger = _resolve_logger(logger)
        self.files = files or {}
        self.filepath = os.path.join(code_dir, f"{module_name}.py")
        self.backup_path = os.path.join(code_dir, f"{module_name}_backup.py")
        self.training_eval_fn = self._make_eval_guardrail(
            corpus,
            judgments,
            tool_fns=tool_fns,
            module_name=module_name,
            logger=self.logger,
            queries=training_queries,
        )
        self.validation_eval_fn = None
        if validation_queries:
            self.validation_eval_fn = self._make_eval_guardrail(
                corpus,
                judgments,
                tool_fns=tool_fns,
                module_name=module_name,
                logger=self.logger,
                queries=validation_queries,
            )
        self.grep_tool = make_run_path_grep_tool(
            Path(code_dir),
            files=self.files,
            module_name=module_name,
            logger=self.logger,
        )

    def tools(self):
        def search(query: str, top_k: int = 10):
            return self.search(query, top_k=top_k)

        def evaluate(edit: Optional[Edit] = None):
            return self.evaluate(edit)

        def commit_patch(edit: Edit):
            return self.commit_patch(edit)

        def grep(
            pattern: str,
            file_glob: str = "**/*",
            max_matches: int = 50,
            max_file_size_kb: int = 512,
        ):
            return self.grep_tool(pattern, file_glob, max_matches, max_file_size_kb)

        search.__doc__ = self.search.__doc__
        evaluate.__doc__ = self.evaluate.__doc__
        commit_patch.__doc__ = self.commit_patch.__doc__
        grep.__doc__ = self.grep_tool.__doc__

        return search, evaluate, commit_patch, grep

    @staticmethod
    def _make_eval_guardrail(
        corpus,
        judgments,
        tool_fns: list[callable],
        module_name: str = "rerank_esci",
        queries: Optional[List[str]] = None,
        seed=1234,
        num_queries=100,
        workers=4,
        logger=None,
    ) -> callable:
        def eval_guardrail(code: str, results: bool = False):
            strategy = CodeGenSearchStrategy(
                corpus,
                tool_fns=tool_fns,
                module_name=module_name,
                code=code,
                workers=workers,
                logger=logger,
            )
            results_df = run_strategy(
                strategy,
                judgments,
                queries=queries,
                num_queries=num_queries,
                seed=seed,
                cache=False,
            )
            ndcgs = results_df.groupby("query")["ndcg"].mean()
            if results:
                return ndcgs, results_df
            return ndcgs
        return eval_guardrail


    def _call_rerank(self, rerank_fn, query: str, top_k: int):
        try:
            return rerank_fn(query, top_k, *self.tool_fns)
        except TypeError:
            return rerank_fn(query, *self.tool_fns)

    def _load_code(self) -> str:
        with open(self.filepath, "r") as handle:
            return handle.read()

    def current_code(self) -> str:
        """Return the current reranker code."""
        return self._load_code()

    def _write_code(self, code: str) -> None:
        with open(self.filepath, "w") as handle:
            handle.write(code)

    def _commit_code(self, code: str) -> str:
        with open(self.filepath, "r") as current:
            with open(self.backup_path, "w") as backup:
                self.logger.info(
                    "Creating backup of %s.py at %s", self.module_name, self.backup_path
                )
                backup.write(current.read())

        self.logger.info("Committing changes to %s.py", self.module_name)
        self._write_code(code)
        return code

    def _patch_code(self, edit: Edit, test_queries=None) -> Tuple[str, str]:
        test_queries = test_queries or [
            "red dress",
            "real housewives of orange county",
        ]
        self.logger.info("Patching code with edits")
        self.logger.info("Goal: %s", edit.intention)
        self.logger.info("Why: %s", edit.why)
        self.logger.info("Expected improved queries: %s", edit.queries_expected_to_improve)
        code = self._load_code()
        existing_code = code

        anchor_index = code.find(edit.anchor)
        if anchor_index == -1:
            raise ValueError(f"Anchor '{edit.anchor}' not found in code.")
        block_index = code.find(edit.block_until, anchor_index)
        if block_index == -1:
            raise ValueError(
                f"Block until '{edit.block_until}' not found after anchor in code."
            )

        for guardrail in self.guardrail_fns:
            error_message = guardrail(edit.text)
            if error_message is not None:
                raise ValueError(error_message)

        if edit.action == "insert_after":
            insertion_point = block_index + len(edit.block_until)
            code = (
                code[:insertion_point]
                + "\n"
                + edit.text
                + "\n"
                + code[insertion_point:]
            )
        elif edit.action == "replace":
            code = (
                code[:anchor_index]
                + edit.text
                + code[block_index + len(edit.block_until):]
            )
        elif edit.action == "delete":
            code = code[:anchor_index] + code[block_index + len(edit.block_until):]
        else:
            raise ValueError(f"Unknown action '{edit.action}'.")

        local_vars = {}
        exec(code, {}, local_vars)
        if self.module_name not in local_vars:
            self.logger.error("Edited code does not define module_name")
            raise ValueError("The edited code does not define module_name.")
        if not callable(local_vars[self.module_name]):
            self.logger.error("module_name is not callable.")
            raise ValueError("module_name is not callable.")

        rerank_fn = local_vars[self.module_name]
        for query in test_queries:
            try:
                results = self._call_rerank(rerank_fn, query, 10)
                if not isinstance(results, list):
                    raise ValueError(
                        f"'{self.module_name}' did not return a list for query '{query}'."
                    )
            except Exception as exc:
                self.logger.error(
                    "Error calling %s with query '%s': %s",
                    self.module_name,
                    query,
                    exc,
                )
                self.logger.error(code)
                raise ValueError(
                    f"Error calling {self.module_name} with query '{query}': {exc}"
                )

        return code, existing_code

    def search(self, query: str, top_k: int = 10):
        """Run the current reranker and return the ranked doc IDs."""
        code = self.current_code()
        rerank_fn = CodeGenSearchStrategy._rerank_fn_from_code(code, module_name=self.module_name)
        doc_ids = self._call_rerank(rerank_fn, query, top_k)
        if doc_ids and isinstance(doc_ids[0], (list, tuple)):
            doc_ids = [doc_id for doc_id, _ in doc_ids]
        return doc_ids[:top_k]

    def evaluate(self, edit: Optional[Edit] = None) -> EvalResult:
        """Evaluate the current or patched reranker on training queries.

        Pass edit=None to evaluate the current reranker without running guardrails.
        When edit is provided, guardrails run before evaluation.
        """
        if edit is None:
            self.logger.info("Evaluating current reranker")
        else:
            self.logger.info("Evaluating patch")
        existing_code = self._load_code()

        try:
            if self.training_eval_fn is None:
                return None
            if edit is None:
                ndcgs_before: pd.Series = self.training_eval_fn(existing_code)
                results_df = None
                try:
                    ndcgs_after, results_df = self.training_eval_fn(
                        existing_code,
                        results=True,
                    )
                except TypeError:
                    ndcgs_after = self.training_eval_fn(existing_code)
                code = existing_code
            else:
                code, existing_code = self._patch_code(edit)
                ndcgs_before: pd.Series = self.training_eval_fn(existing_code)
                results_df = None
                try:
                    ndcgs_after, results_df = self.training_eval_fn(code, results=True)
                except TypeError:
                    ndcgs_after = self.training_eval_fn(code)
            deltas: pd.Series = ndcgs_after - ndcgs_before
            delta_dict = deltas.to_dict()
            changed_queries = {}
            for query in delta_dict:
                if delta_dict[query] != 0.0:
                    changed_queries[query] = delta_dict[query]

            icon = "❌"
            if ndcgs_after.mean() >= (ndcgs_before.mean() + self.eval_margin):
                icon = "✅"
            if ndcgs_after.mean() >= ndcgs_before.mean():
                icon = "⚠️"

            self.logger.info(
                "%s Evaluated patch successfully. train NDCG before: %s, after: %s",
                icon,
                ndcgs_before.mean(),
                ndcgs_after.mean(),
            )
            self.logger.info("Changed queries NDCG deltas: %s", changed_queries)
            self.logger.info("Code:")
            self.logger.info(code)
            warning = None
            if edit is not None and ndcgs_after.mean() < (ndcgs_before.mean() + self.eval_margin):
                warning = (
                    "⚠️ Warning: NDCG did not improve by at least "
                    f"{self.eval_margin} on training set: before={ndcgs_before.mean()}, "
                    f"after={ndcgs_after.mean()}. It might be rejected if applied. "
                    "Hint: look at changed queries, modify your change to get the upside "
                    "of your change, and minimize the downside."
                )
            self.logger.warning(warning)

            training_path = None
            if results_df is not None:
                training_path = write_training_logs(
                    code,
                    delta_dict,
                    results_df,
                    self.code_dir,
                )

            return EvalResult(
                success=True,
                error_message=warning,
                ndcg_deltas=changed_queries,
                ndcg_before=ndcgs_before.mean(),
                ndcg_after=ndcgs_after.mean(),
                current_code=existing_code,
                training_path=training_path,
            )
        except Exception as exc:
            self.logger.info("Error evaluating patch: %s", exc)
            return EvalResult(
                success=False,
                error_message=str(exc),
                ndcg_deltas={},
                current_code=existing_code,
            )

    def commit_patch(self, edit: Edit) -> EditResult:
        """Apply a code patch if validation queries improve by the margin."""
        try:
            self.logger.info("Comitting patch with edits... pending validation")
            code, existing_code = self._patch_code(edit)
            edit_result = EditResult(
                success=True, error_message=None, current_code=existing_code
            )
            if self.validation_eval_fn is not None:
                ndcg_before = self.validation_eval_fn(existing_code).mean()
                ndcg_after = self.validation_eval_fn(code).mean()
                if ndcg_after < (ndcg_before + self.eval_margin):
                    self.logger.warning(
                        "❌ Rejecting Change: Validation NDCG must increase at least %s "
                        "after applying patch: before=%s, after=%s",
                        self.eval_margin,
                        ndcg_before,
                        ndcg_after,
                    )
                    raise ValueError(
                        "Rejecting change as overfit must increase NDCG by at least "
                        f"{self.eval_margin}: before={ndcg_before}, after={ndcg_after}"
                    )
                else:
                    self.logger.info(
                        "✅ Validation NDCG improved: before=%s, after=%s",
                        ndcg_before,
                        ndcg_after,
                    )

            code = self._commit_code(code)
            if code:
                edit_result.current_code = code
                return edit_result
        except Exception as exc:
            self.logger.info("Error applying patch: %s", exc)
            existing_code = self._load_code()
            return EditResult(
                success=False,
                error_message=str(exc),
                current_code=existing_code,
            )



def _make_eval_guardrail(
    corpus,
    judgments,
    tool_fns: list[callable],
    module_name: str = "rerank_esci",
    queries: Optional[List[str]] = None,
    seed=1234,
    num_queries=100,
    workers=4,
    logger=None,
) -> callable:
    def eval_guardrail(code: str, results: bool = False):
        strategy = CodeGenSearchStrategy(
            corpus,
            tool_fns=tool_fns,
            module_name=module_name,
            code=code,
            workers=workers,
            logger=logger,
        )
        results_df = run_strategy(
            strategy,
            judgments,
            queries=queries,
            num_queries=num_queries,
            seed=seed,
            cache=False,
        )
        ndcgs = results_df.groupby("query")["ndcg"].mean()
        if results:
            return ndcgs, results_df
        return ndcgs
    return eval_guardrail
