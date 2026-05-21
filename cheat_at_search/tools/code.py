from __future__ import annotations

import os
import re
import hashlib
import importlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import SearchStrategy

# Implementation breadcrumb:
# search-experiments should import this module directly instead of carrying a local copy.
# Porting notes:
# - Rerank functions should be defined as rerank_*(query, *tool_fns).
# - Guardrail checker uses OpenAIAgent.loop with reasoning_level.
# - make_patch_fn exposes function_name/tool_fns and logger injection.


def _resolve_logger(logger=None, logger_name: str = "code"):
    if logger is not None:
        return logger
    return log_to_stdout(logger_name=logger_name)


class Edit(BaseModel):
    """A single edit to apply to the reranker code."""

    anchor: str = Field(
        ...,
        description="The anchor text to identify where the patch should be applied.",
    )
    block_until: str = Field(
        ...,
        description=(
            "The end of the block of text which the patch should be applied. "
            "Do not leave blank."
        ),
    )
    action: Literal["insert_after", "replace", "delete"] = Field(
        ..., description="The action to perform: insert_after, replace, or delete."
    )
    text: str = Field(
        ...,
        description="The text to insert or replace with. Ignored for delete action.",
    )
    intention: str = Field(
        None, description="A brief description of the intention behind this edit."
    )
    why: str = Field(
        None, description="An optional explanation of why this edit is being made."
    )
    queries_expected_to_improve: List[str] = Field(
        None,
        description="A list of training queries expected to have their NDCG changed by this edit.",
    )


class EditResult(BaseModel):
    """The result of applying edits to the reranker code."""

    success: bool = Field(
        ...,
        description="Whether the edits were applied successfully and the reranker passed tests.",
    )
    error_message: Optional[str] = Field(
        None,
        description="An error message if the edits failed to apply or tests failed.",
    )
    current_code: str = Field(
        None, description="The current reranker code after this call."
    )


class EvalResult(BaseModel):
    success: bool = Field(
        ...,
        description="Whether the edits can be applied succesfully without code errors.",
    )
    error_message: Optional[str] = Field(
        None,
        description=(
            "An error or warning message if the patch failed to be applied, "
            "evaluation failed, or NDCG did not improve sufficiently."
        ),
    )
    ndcg_deltas: Optional[Dict[str, float]] = Field(
        None, description="The NDCG deltas for the training dataset."
    )
    ndcg_before: Optional[float] = Field(
        0.0, description="The NDCG before applying the edit."
    )
    ndcg_after: Optional[float] = Field(
        0.0, description="The NDCG after applying the edit."
    )
    current_code: Optional[str] = Field(
        None, description="The current reranker code after this call."
    )
    training_path: Optional[str] = Field(
        None,
        description="Relative path to training run logs under code_dir/training.",
    )


def make_length_validator(
    max_lines: int = 10, max_cols=120
) -> Callable[[str], Optional[str]]:
    guardrail_desc = (
        f"Edits longer than {max_lines} and wider than {max_cols} "
        "characters will be rejected."
    )

    def length_validation(code: str) -> Optional[str]:
        if code.count("\n") > max_lines:
            return f"Code exceeds maximum length of {max_lines} lines."

        for line in code.split("\n"):
            if len(line) > max_cols + 20:
                return f"Line exceeds maximum length of {max_cols} characters: {line}"
        return None

    length_validation.__doc__ = guardrail_desc
    return length_validation


class GuardrailResponse(BaseModel):
    """The response from the guardrail checker."""

    compliant: bool = Field(
        ..., description="Whether the code complies with the guardrails."
    )
    issues: Optional[List[str]] = Field(
        None, description="A list of issues found in the code, if any."
    )


def make_guardrail_checker(
    prompt: str,
    model: str = "openai/gpt-5-mini",
    reasoning: str = "medium",
    logger=None,
):
    agent = OpenAIAgent(
        tools=[],
        model=model,
        response_model=GuardrailResponse,
        reasoning_level=reasoning,
    )
    logger = _resolve_logger(logger)

    def code_guardrails(code: str) -> Optional[str]:
        """Edits where the code appears to be overfit to training queries will be rejected."""
        inputs = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Please evaluate the following code for compliance:\n```python\n{code}\n```",
            },
        ]
        resp = agent.loop(inputs=inputs)
        if resp is None:
            return "Guardrail check failed: no response from model."
        if not resp.compliant:
            issues = (
                "\n".join(resp.issues)
                if resp.issues
                else "No specific issues provided."
            )
            return f"Code does not comply with guardrails:\n{issues}"
        logger.debug("Guardrail check passed.")

    return code_guardrails


def _get_rerank_fn(module_name: str):
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    rerank_fn = None
    for attr in dir(mod):
        if attr.startswith("rerank_"):
            rerank_fn = getattr(mod, attr)
            break
    return rerank_fn


def _rerank_fn_from_code(code: str):
    exec_globals = {}
    exec(code, exec_globals)
    rerank_fn = None
    for name, obj in exec_globals.items():
        if name.startswith("rerank_"):
            rerank_fn = obj
            break
    return rerank_fn


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
        if self.code:
            rerank_fn = _rerank_fn_from_code(self.code)
        else:
            rerank_fn = _get_rerank_fn(self.module_name)

        doc_ids = self._call_rerank(rerank_fn, query, k)[:k]
        if doc_ids and isinstance(doc_ids[0], (list, tuple)):
            doc_ids = [doc_id for doc_id, _ in doc_ids]
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


def grade_to_emoji(grade):
    if grade == 3:
        return "🤩"
    if grade == 2:
        return "🙂"
    if grade == 1:
        return "😐"
    if grade == 0:
        return "😭"
    return ""


class Doc(BaseModel):
    """A document returned by the search system."""

    title: str = Field(..., description="The title of the document.")
    label: Literal["🤩", "🙂", "😐", "😭", ""] = Field(
        ..., description="The human judgment label for the document."
    )


class QueryEvalResult(BaseModel):
    query: str = Field(..., description="The user query being evaluated.")
    ndcg: float = Field(..., description="The NDCG score for the query.")
    relevant_doc: Doc = Field(
        ..., description="An example of a relevant document for the query."
    )


class EvalResults(BaseModel):
    """The result of evaluating the reranker on ground truth judgments."""

    query_ndcgs: List[QueryEvalResult] = Field(
        ..., description="The NDCG scores for each query."
    )
    mean_ndcg: float = Field(
        ..., description="The mean NDCG across all queries."
    )


def make_run_path_grep_tool(
    run_path: Path,
    files: dict[str, str] | None = None,
    module_name: str = "rerank_esci",
    logger=None,
) -> Callable[[str, str, int, int], dict]:
    base_path = Path(run_path).expanduser().resolve()
    logger = _resolve_logger(logger)
    files = files or {}
    managed_files = {
        f"{module_name}.py": "reranker source",
        "queries.csv": "training query summary",
    }
    combined_files = {**managed_files, **files}
    files_doc = "\n".join(
        f"- {filename} ({description})"
        for filename, description in combined_files.items()
    )
    if files_doc:
        files_doc = f"\n\nTypical files to inspect:\n{files_doc}"
    training_layout_doc = (
        "\n\nTraining run layout (when logging is enabled):\n"
        "- training/<timestamp>/\n"
        "  - reranker.py\n"
        "  - queries.csv\n"
        "  - <query_path>/results.csv"
    )

    def grep_run_path(
        pattern: str,
        file_glob: str = "**/*",
        max_matches: int = 50,
        max_file_size_kb: int = 512,
    ) -> dict:
        """Search previous codegen run files for a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            file_glob: Glob pattern under the run path to scan.
            max_matches: Maximum number of matches to return.
            max_file_size_kb: Skip files larger than this limit.
        """
        if not base_path.exists():
            return {"matches": [], "error": f"run path not found: {base_path}"}

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"matches": [], "error": f"invalid regex: {exc}"}
        logger.info(
            "!GREP Searching for pattern '%s' in files matching '%s' under %s...",
            pattern,
            file_glob,
            base_path,
        )

        matches = []
        skipped = []
        truncated = False
        for path in sorted(base_path.rglob(file_glob)):
            if len(matches) >= max_matches:
                truncated = True
                break
            if path.is_dir():
                continue
            try:
                size_kb = path.stat().st_size / 1024
            except OSError:
                skipped.append(str(path))
                continue
            if size_kb > max_file_size_kb:
                skipped.append(
                    f"{path} (size {size_kb:.1f}kb > {max_file_size_kb}kb)"
                )
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped.append(f"{path} (non-utf8)")
                continue
            except OSError:
                skipped.append(str(path))
                continue
            for line_num, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"file": str(path), "line": line_num, "text": line})
                    if len(matches) >= max_matches:
                        truncated = True
                        break
        return {"matches": matches, "truncated": truncated, "skipped": skipped}

    grep_run_path.__name__ = "grep_run_path"
    grep_run_path.__doc__ = f"""Search previous codegen run files for a regex pattern.{files_doc}{training_layout_doc}

    Args:
        pattern: Regex pattern to search for.
        file_glob: Glob pattern under the run path to scan.
        max_matches: Maximum number of matches to return.
        max_file_size_kb: Skip files larger than this limit.
    """
    return grep_run_path


class Reranker:
    def __init__(
        self,
        code_dir: str,
        tool_fns: list[callable],
        module_name: str = "rerank_esci",
        guardrail_fns: List | None = None,
        training_eval_fn: Optional[Callable] = None,
        validation_eval_fn: Optional[Callable] = None,
        eval_margin: float = 0.003,
        logger=None,
        files: dict[str, str] | None = None,
    ):
        self.code_dir = code_dir
        self.tool_fns = tool_fns
        self.module_name = module_name
        self.guardrail_fns = guardrail_fns or []
        self.training_eval_fn = training_eval_fn
        self.validation_eval_fn = validation_eval_fn
        self.eval_margin = eval_margin
        self.logger = _resolve_logger(logger)
        self.files = files or {}
        self.filepath = os.path.join(code_dir, f"{module_name}.py")
        self.backup_path = os.path.join(code_dir, f"{module_name}_backup.py")
        self.grep_tool = make_run_path_grep_tool(
            Path(code_dir),
            files=self.files,
            module_name=module_name,
            logger=self.logger,
        )

    @classmethod
    def build(
        cls,
        code_dir: str,
        tool_fns: list[callable],
        module_name: str = "rerank_esci",
        guardrail_fns: List | None = None,
        training_eval_fn: Optional[Callable] = None,
        validation_eval_fn: Optional[Callable] = None,
        eval_margin: float = 0.003,
        logger=None,
        files: dict[str, str] | None = None,
    ):
        reranker = cls(
            code_dir=code_dir,
            tool_fns=tool_fns,
            module_name=module_name,
            guardrail_fns=guardrail_fns,
            training_eval_fn=training_eval_fn,
            validation_eval_fn=validation_eval_fn,
            eval_margin=eval_margin,
            logger=logger,
            files=files,
        )

        def run_reranker(query: str, top_k: int = 10):
            return reranker.run_reranker(query, top_k=top_k)

        def try_out_patch(edit: Edit):
            return reranker.try_out_patch(edit)

        def apply_patch(edit: Edit):
            return reranker.apply_patch(edit)

        def grep(pattern: str, file_glob: str = "**/*", max_matches: int = 50, max_file_size_kb: int = 512):
            return reranker.grep_tool(pattern, file_glob, max_matches, max_file_size_kb)

        return run_reranker, try_out_patch, apply_patch, grep

    @staticmethod
    def make_eval_guardrail(
        corpus,
        judgments,
        tool_fns: list[callable],
        module_name: str = "rerank_esci",
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

    def _slugify_query(self, query: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "-", query.strip().lower()).strip("-")
        if not normalized:
            normalized = "query"
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
        return f"{normalized[:40]}-{digest}"

    def _write_training_logs(
        self,
        code: str,
        ndcg_deltas: dict[str, float],
        results_df: pd.DataFrame,
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        training_root = os.path.join(self.code_dir, "training", timestamp)
        os.makedirs(training_root, exist_ok=True)

        reranker_path = os.path.join(training_root, "reranker.py")
        with open(reranker_path, "w") as handle:
            handle.write(code)

        if "doc_id" not in results_df.columns and "product_id" in results_df.columns:
            results_df = results_df.copy()
            results_df["doc_id"] = results_df["product_id"]
        for col in ["query", "rank", "doc_id", "title", "description"]:
            if col not in results_df.columns:
                results_df[col] = ""

        query_paths = {}
        for query in ndcg_deltas.keys():
            query_slug = self._slugify_query(query)
            query_paths[query] = query_slug

            query_dir = os.path.join(training_root, query_slug)
            os.makedirs(query_dir, exist_ok=True)

            query_results = results_df[results_df["query"] == query]
            query_results = query_results[
                ["query", "rank", "doc_id", "title", "description"]
            ]
            results_path = os.path.join(query_dir, "results.csv")
            query_results.to_csv(results_path, index=False)

        queries_rows = []
        for query, delta in ndcg_deltas.items():
            queries_rows.append({
                "query": query,
                "ndcg_delta": delta,
                "query_path": query_paths[query],
            })
        queries_df = pd.DataFrame(
            queries_rows,
            columns=["query", "ndcg_delta", "query_path"],
        )
        queries_path = os.path.join(training_root, "queries.csv")
        queries_df.to_csv(queries_path, index=False)

        return os.path.relpath(training_root, self.code_dir)

    def run_reranker(self, query: str, top_k: int = 10):
        rerank_fn = _get_rerank_fn(self.module_name)
        doc_ids = self._call_rerank(rerank_fn, query, top_k)
        if doc_ids and isinstance(doc_ids[0], (list, tuple)):
            doc_ids = [doc_id for doc_id, _ in doc_ids]
        return doc_ids[:top_k]

    def try_out_patch(self, edit: Edit) -> EvalResult:
        self.logger.info("Evaluating patch")
        existing_code = self._load_code()

        try:
            if self.training_eval_fn is None:
                return None
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
            if ndcgs_after.mean() < (ndcgs_before.mean() + self.eval_margin):
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
                training_path = self._write_training_logs(code, delta_dict, results_df)

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

    def apply_patch(self, edit: Edit) -> EditResult:
        try:
            self.logger.info("Applying patch with edits")
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

    def revert_changes(self) -> str:
        if not os.path.exists(self.backup_path):
            with open(self.filepath, "r") as current:
                with open(self.backup_path, "w") as backup:
                    backup.write(current.read())
        with open(self.backup_path) as backup:
            with open(self.filepath, "w") as handle:
                self.logger.info("Reverted %s.py to backup.", self.module_name)
                code = backup.read()
                handle.write(code)
                self.logger.info("Reverted changes successfully.")
                return code
        return "Error reverting changes."


def make_patch_fn(
    search_fn,
    corpus,
    code_dir: str,
    tool_fns: list[callable] | None = None,
    module_name: str = "rerank_esci",
    function_name: str | None = None,
    guardrail_fns: List = None,
    training_eval_fn: Optional[Callable] = None,
    validation_eval_fn: Optional[Callable] = None,
    eval_margin=0.003,
    logger=None,
) -> Tuple[callable, Optional[callable], callable]:
    """Returns a function that applies patches to the reranker code."""
    tool_fns = tool_fns or [search_fn]
    if guardrail_fns is None:
        guardrail_fns = []

    if training_eval_fn is not None:
        training_eval_fn = lru_cache(maxsize=64)(training_eval_fn)
    guardrail_doc_strs = "\n".join([func.__doc__ for func in guardrail_fns])
    full_guardrail_doc_strs = guardrail_doc_strs
    if validation_eval_fn is not None:
        validation_eval_fn = lru_cache(maxsize=64)(validation_eval_fn)
        full_guardrail_doc_strs += (
            "\nEdits that reduce validation NDCG will be rejected as overfitting "
            f"(must improve by at least {eval_margin})."
        )
        full_guardrail_doc_strs = (
            "Your code will be rejected if it does not meet these guardrails:\n"
            + full_guardrail_doc_strs
        )
        guardrail_doc_strs += (
            "\nNo checks to validation NDCG are performed in try_out_patch."
        )

    if guardrail_doc_strs:
        guardrail_doc_strs = (
            "Your code will be rejected if it does not meet these guardrails:\n"
            + guardrail_doc_strs
        )

    reranker = Reranker(
        code_dir=code_dir,
        tool_fns=tool_fns,
        module_name=module_name,
        guardrail_fns=guardrail_fns,
        training_eval_fn=training_eval_fn,
        validation_eval_fn=validation_eval_fn,
        eval_margin=eval_margin,
        logger=logger,
    )

    try_out_patch = reranker.try_out_patch
    apply_patch = reranker.apply_patch
    revert_changes = reranker.revert_changes

    try_out_patch.__doc__ = f"""Evaluate the proposed code change to analyze its impact on training queries.
    (Results won't be saved, this is used to evaluate potential patches before applying them.)

    {guardrail_doc_strs}

    """

    apply_patch.__doc__ = f"""Save the proposed code change to {module_name}.py.

    {full_guardrail_doc_strs}

    """
    if training_eval_fn is None:
        return apply_patch, None, revert_changes
    return apply_patch, try_out_patch, revert_changes


def set_to_start_code(code_dir: str) -> str:
    """Reset the reranker code to the original version from backup."""
    module_name = "rerank_esci"
    filepath = os.path.join(code_dir, f"{module_name}.py")
    backup_path = os.path.join(code_dir, f"{module_name}_backup.py")

    start_code = ""
    with open("cheat_at_search/start_rerank_esci.py", "r") as f:
        start_code = f.read()

    with open(filepath, "w") as f:
        f.write(start_code)

    with open(backup_path, "w") as backup:
        backup.write(start_code)
    return start_code


def set_code_to(code_dir: str, code: str) -> str:
    """Set the reranker code to the provided code."""
    module_name = "rerank_esci"
    filepath = os.path.join(code_dir, f"{module_name}.py")

    with open(filepath, "w") as f:
        f.write(code)
    return code


def current_code(code_dir: str) -> str:
    """Get the current reranker code."""
    module_name = "rerank_esci"
    filepath = os.path.join(code_dir, f"{module_name}.py")

    with open(filepath, "r") as f:
        code = f.read()
        return code
