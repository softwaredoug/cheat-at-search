from typing import List, Dict, Optional, Literal, Callable, Tuple, Union
from pydantic import BaseModel, Field
import pandas as pd
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.tools.eval import grade_to_emoji
from functools import lru_cache
import os


logger = log_to_stdout(logger_name="cheat_at_search.code")


class Edit(BaseModel):
    """A single edit to apply to the reranker code."""
    anchor: str = Field(..., description="The anchor text to identify where the patch/edit should be applied. This will be replaced with the new text. Do not leave blank.")
    block_until: str = Field(..., description="The end of the block of text which the patch should be applied. Do not leave blank.")
    action: Literal['replace', 'delete'] = Field(..., description="The action to perform: insert_after, replace, or delete.")
    text: str = Field(..., description="The text to insert or replace with. Ignored for delete action.")
    doing_differently: Optional[str] = Field(None, description="What you are doing differently in this edit compared to your previous attempt.")
    what_have_you_learned: Optional[str] = Field(None, description="What you have learned from previous attempt that informed this edit.")
    evidence_this_will_work: Optional[str] = Field(None, description="Any evidence or reasoning that this edit will improve the reranker. Use examples from query behaviors you've seen")
    past_mistakes: Optional[str] = Field(None, description="Mea. Culpa. Past mistakes you made in previous edits that you are avoiding in this edit.")


class EditResult(BaseModel):
    """The result of applying edits to the reranker code."""
    success: bool = Field(..., description="Whether the edits were applied successfully and the reranker passed tests.")
    error_messages: Optional[str] = Field(None, description="An error message if the edits failed to apply or tests failed.")
    info_messages: Optional[str] = Field(None, description="Informational message about the application of the edit.")
    current_code: str = Field(None, description="The current reranker code after this call.")


class PerQueryDelta(BaseModel):
    query: str = Field(..., description="The query for which the NDCG delta is reported.")
    ndcg_before: float = Field(..., description="The NDCG before applying the edit for this query.")
    ndcg_after: float = Field(..., description="The NDCG after applying the edit for this query.")
    relevant_doc: Dict[str, str] = Field(..., description="An example of a relevant document for this query.")


class EvalResult(BaseModel):
    success: bool = Field(..., description="Whether the edits can be applied succesfully without code errors.")
    warning_messages: Optional[str] = Field(None, description="An error or warning message if the patch failed to be applied, evaluation failed, or NDCG did not improve sufficiently.")
    info_messages: Optional[str] = Field(None, description="Informational message about the evaluation.")
    ndcg_before: Optional[float] = Field(0.0, description="The NDCG before applying the edit.")
    ndcg_after: Optional[float] = Field(0.0, description="The NDCG after applying the edit.")
    query_deltas: List[PerQueryDelta] = Field([], description="The per-query NDCG deltas after applying the edit.")
    current_code: Optional[str] = Field(None, description="The current reranker code after this call.")


def make_length_validator(max_lines: int = 10, max_cols=120) -> Callable[[str], Optional[str]]:
    guardrail_desc = f"""Edits longer than {max_lines} and wider than {max_cols} characters will be rejected."""

    def length_validation(code: str) -> Optional[str]:
        if code.count('\n') > max_lines:
            return f"Code exceeds maximum length of {max_lines} lines."

        for line in code.split('\n'):
            if len(line) > max_cols + 20:
                return f"Line exceeds maximum length of {max_cols} characters: {line}"
        return None
    length_validation.__doc__ = guardrail_desc
    return length_validation


class GuardrailResponse(BaseModel):
    """The response from the guardrail checker."""
    compliant: bool = Field(..., description="Whether the code complies with the guardrails.")
    issues: Optional[List[str]] = Field(None, description="A list of issues found in the code, if any.")


def make_guardrail_checker(prompt: str, model: str = "openai/gpt-5-mini"):
    agent = OpenAIAgent(tools=[],
                        model=model,
                        system_prompt=prompt,
                        response_model=GuardrailResponse)

    def code_guardrails(code: str) -> Optional[str]:
        """Edits where the code appears to be overfit to training queries will be rejected."""
        response = agent.loop(user_prompt=f"Please evaluate the following code for compliance:\n```python\n{code}\n```")
        if not response.compliant:
            issues = "\n".join(response.issues) if response.issues else "No specific issues provided."
            return f"Code does not comply with guardrails:\n{issues}"
    return code_guardrails


def score_prediction(queries_expected_to_change: List[str],
                     query_delta_ndcgs: Dict[str, float],
                     improved=True) -> Tuple[List[str], List[str]]:
    """Compute the following:

        expected to change, and did change
        did not expect to change, and did change

        change is improvement if improved=True, else degradation
    """
    def check(delta_ndcg):
        if improved:
            return delta_ndcg > 0
        return delta_ndcg < 0

    changed_as_expected = []
    changed_unexpectedly = []
    for query in queries_expected_to_change:
        if query in query_delta_ndcgs and check(query_delta_ndcgs[query]):
            changed_as_expected.append(query)

    for query in query_delta_ndcgs:
        if query not in queries_expected_to_change and check(query_delta_ndcgs[query]):
            changed_unexpectedly.append(query)

    return changed_as_expected, changed_unexpectedly


def patch_code(filepath: str,
               module_name: str,
               edit: Edit,
               guardrail_fns: List = None) -> Tuple[str, str, dict]:
    with open(filepath, "r") as f:
        code = f.read()
        existing_code = code

        anchor_index = code.find(edit.anchor)
        if anchor_index == -1:
            raise ValueError(f"Anchor '{edit.anchor}' not found in code.")
        block_index = code.find(edit.block_until, anchor_index)
        if block_index == -1:
            raise ValueError(f"Block until '{edit.block_until}' not found after anchor in code.")

        # Validate code
        if guardrail_fns is None:
            guardrail_fns = []
        for guardrail in guardrail_fns:
            error_message = guardrail(edit.text)
            if error_message is not None:
                raise ValueError(error_message)

        if edit.action == 'replace':
            # Get indent at anchor
            indent_level = anchor_index - code[:anchor_index].rfind("\n") - 1
            # Indent the edit text
            edit_lines = edit.text.split('\n')
            is_indented = all([line.startswith(' ' * indent_level) or line.strip() == '' for line in edit_lines])
            if not is_indented:
                anchor_indent = ' ' * indent_level
                indented_edit_lines = [edit_lines[0]]
                for line in edit_lines[1:]:
                    indented_edit_lines.append(anchor_indent + line if line.strip() != '' else line)
                edit.text = '\n'.join(indented_edit_lines)
            code = code[:anchor_index] + edit.text + code[block_index + len(edit.block_until):]
        elif edit.action == 'delete':
            code = code[:anchor_index] + code[block_index + len(edit.block_until):]
        else:
            raise ValueError(f"Unknown action '{edit.action}'.")
    # Attempt to eval the code
    local_vars = {}
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        logger.error("Error evaluatiing patched code: " + str(e))
        logger.error(existing_code)
        logger.error("Anchor\n" + edit.anchor)
        logger.error("Block Until\n" + edit.block_until)
        logger.error("Action: " + edit.action)
        logger.error("Text:\n" + edit.text)
        raise ValueError(f"Error evaluating patched code: {e}")
    if module_name not in local_vars:
        logger.error("Edited code does not define module_name")
        raise ValueError("The edited code does not define module_name.")
    # Test that rerank_esci is callable
    if not callable(local_vars[module_name]):
        logger.error("module_name is not callable.")
        raise ValueError("module_name is not callable.")
    return code, existing_code, local_vars


def make_patch_fn(search_fn,
                  corpus,
                  judgments: pd.DataFrame,
                  code_dir: str,
                  guardrail_fns: List = None,
                  training_eval_fn: Optional[Callable] = None,
                  validation_eval_fn: Optional[Callable] = None,
                  num_queries_improved_required: int = 1,
                  perc_queries_improved_required: float = 0.65,
                  min_ndcg_delta=0.05,  # Only count changes above this threshold
                  eval_margin=0.003) -> Tuple[callable, callable, callable, callable]:
    """Returns a function that applies patches to the reranker code."""

    module_name = "rerank_esci"

    filepath = os.path.join(code_dir, f"{module_name}.py")
    backup_path = os.path.join(code_dir, f"{module_name}_backup.py")

    if guardrail_fns is None:
        guardrail_fns = []

    if training_eval_fn is not None:
        training_eval_fn = lru_cache(maxsize=64)(training_eval_fn)
    guardrail_doc_strs = "\n".join([
        func.__doc__ for func in guardrail_fns
    ])
    full_guardrail_doc_strs = guardrail_doc_strs
    if validation_eval_fn is not None:
        validation_eval_fn = lru_cache(maxsize=64)(validation_eval_fn)
        full_guardrail_doc_strs += f"\nEdits that reduce validation NDCG will be rejected as overfitting (must improve by at least {eval_margin})."
        full_guardrail_doc_strs = "Your code will be rejected if it does not meet these guardrails:\n" + full_guardrail_doc_strs
        guardrail_doc_strs += "\nNo checks to validation NDCG are performed in try_out_patch."

    if guardrail_doc_strs:
        guardrail_doc_strs = "Your code will be rejected if it does not meet these guardrails:\n" + guardrail_doc_strs

    def revert_changes() -> str:
        """Undo the last patch to rerank_esci.py by restoring from backup."""
        with open(backup_path) as backup:
            with open(filepath, "w") as f:
                logger.info(f"Reverted {module_name}.py to backup.")
                code = backup.read()
                f.write(code)
                logger.info("Reverted changes successfully.")
                return code
        return "Error reverting changes."

    def _patch_code(edit: Edit,
                    test_queries=["red dress", "real housewives of orange county"]) -> Tuple[str, str]:
        logger.info("Patching code with edits")
        logger.info(f"Doing Differently: {edit.doing_differently}")
        logger.info(f"What Have You Learned: {edit.what_have_you_learned}")
        logger.info(f"Evidence This Will Work: {edit.evidence_this_will_work}")
        logger.info(f"Past Mistakes: {edit.past_mistakes}")
        code, existing_code, local_vars = patch_code(filepath,
                                                     module_name,
                                                     edit,
                                                     guardrail_fns)
        for query in test_queries:
            try:
                results = local_vars[module_name](search_fn, query)[:10]
            except Exception as e:
                logger.error(f"Error calling {module_name} with query '{query}': {e}")
                logger.error(existing_code)
                logger.error("Anchor\n" + edit.anchor)
                logger.error("Block Until\n" + edit.block_until)
                logger.error("Action: " + edit.action)
                logger.error("Text:\n" + edit.text)
                raise ValueError(f"Error calling {module_name} with query '{query}': {e}")

            try:
                if not isinstance(results, list):
                    logger.error(f"'{module_name}' did not return a list for query '{query}'.")
                    raise ValueError(f"'{module_name}' did not return a list for query '{query}'.")
            except Exception as e:
                logger.error(f"Error collecting results with query '{query}': {e}")
                raise ValueError(f"Error calling 'rerank_esci' with query '{query}': {e}")
        return code, existing_code, local_vars

    def _commit_code(code: str) -> Optional[str]:
        with open(filepath, "r") as f:
            with open(backup_path, "w") as backup:
                logger.info(f"Creating backup of {module_name}.py at {backup_path}")
                backup.write(f.read())

        with open(filepath, "w") as f:
            logger.info(f"Committing changes to {module_name}.py")
            f.write(code)
            return code

    def _training_eval_fn(code: str, existing_code: str):
        """Evaluate the code on training data and compare to existing code."""
        ndcgs_before = training_eval_fn(existing_code)
        ndcgs_after = training_eval_fn(code)
        deltas: pd.Series = ndcgs_after - ndcgs_before

        logger.info("Code:")
        logger.info(code)
        # Check if in margin
        msgs = []
        if ndcgs_after.mean() < (ndcgs_before.mean() + eval_margin):
            msgs.append(f"""⚠️ Warning: NDCG did not improve by at least {eval_margin} on training set:
            before={ndcgs_before.mean()},
            after={ndcgs_after.mean()}.
        It might be rejected if applied.

        Hint: look at changed queries, modify your change to get the upside of your change, and minimize the downside.""")
        else:
            msgs.append("✅ NDCG improved sufficiently on training set.")

        all_queries_improved = deltas[deltas > min_ndcg_delta].index.tolist()
        all_queries_harmed = deltas[deltas < -min_ndcg_delta].index.tolist()

        # Track what percentage of queries were positive improvements
        perc_queries_changed_improved = 0.0
        total_changed = len(all_queries_improved) + len(all_queries_harmed)
        if total_changed > 0:
            perc_queries_changed_improved = len(all_queries_improved) / total_changed

            msgs.append(f"ℹ️ Percentage of changed queries that improved by at least {min_ndcg_delta}: {perc_queries_changed_improved:.2f}")
            msgs.append(f"ℹ️ Number of improved queries by at least {min_ndcg_delta}: {len(all_queries_improved)}")
        else:
            msgs.append(f"ℹ️ No queries were changed by at least {min_ndcg_delta} NDCG.")

        if perc_queries_changed_improved < perc_queries_improved_required:
            msg = f"⚠️ Low percentage of changed queries that improved: {perc_queries_changed_improved:.2f}. Please make a better change (at least {perc_queries_improved_required * 100}% of changed queries should improve)."
            msgs.append(msg)
        else:
            msgs.append(f"✅ Sufficient percentage of changed queries improved: {perc_queries_changed_improved:.2f}.")

        if len(all_queries_improved) < num_queries_improved_required:
            msgs.append(f"⚠️ Very few changed queries ({len(all_queries_improved)}) improved. Please try to make a change that improves at least {num_queries_improved_required} training queries.")
        else:
            msgs.append(f"✅ Sufficient number of changed queries improved: {len(all_queries_improved)}.")

        infos = []
        warnings = []
        for msg in msgs:
            if msg.startswith("⚠️"):
                logger.warning(msg)
                warnings.append(msg)
            else:
                logger.info(msg)
                infos.append(msg)

        warning = None
        if len(warnings) > 0:
            warning = "PROBLEMS! Here are a list of guardrail violations on the training set:\n"
            warning += "\n".join(warnings)

        info = None
        if len(infos) > 0:
            info = "Here are some informational messages from evaluating on the training set:\n"
            info += "\n".join(infos)

        return ndcgs_before, ndcgs_after, warning, info

    def _per_query_deltas(ndcgs_before: pd.Series,
                          ndcgs_after: pd.Series) -> List[PerQueryDelta]:
        deltas: pd.Series = ndcgs_after - ndcgs_before
        per_query_deltas = []
        max_grade = judgments['grade'].max()
        for query in deltas.index:
            relevant_docs = judgments[(judgments['query'] == query) & (judgments['grade'] == max_grade)]
            example_relevant_doc = {}
            if not relevant_docs.empty:
                relevant_doc_id = relevant_docs.sample(1).iloc[0]['product_id']
                example_relevant_doc = corpus[corpus['doc_id'] == relevant_doc_id].iloc[0].to_dict()
                example_relevant_doc = {
                    "product_name": example_relevant_doc.get("product_name", ""),
                    "product_description": example_relevant_doc.get("product_description", ""),
                }
            per_query_deltas.append(PerQueryDelta(
                query=query,
                ndcg_before=ndcgs_before[query],
                ndcg_after=ndcgs_after[query],
                relevant_doc=example_relevant_doc
            ))
        return per_query_deltas

    def try_out_patch_on_query(edit: Edit, query: str) -> Union[List[Dict], str]:
        """Try out the proposed code change on a single query without saving changes.

        Returns the top 10 documents returned by the patched reranker for the given query.

        If the query exists in the judgments, results will be labeled

        If error, returns error message string."""
        try:
            logger.info(f"Trying out patch on query '{query}'")
            query_judgments = judgments[judgments['query'] == query]
            code, existing_code, local_vars = _patch_code(edit, test_queries=[query])
            reranker_fn = local_vars[module_name]
            results = reranker_fn(search_fn, query)
            top_docs = []
            for doc_id in results:
                judgment_row = query_judgments[query_judgments['doc_id'] == doc_id]
                doc = corpus[corpus['doc_id'] == doc_id].iloc[0]
                doc_dict = {}
                if not doc.empty:
                    doc_dict = {
                        "doc_id": doc_id,
                        "product_name": doc['product_name'],
                        "product_description": doc['product_description'],
                        "grade": 0,
                        "label": grade_to_emoji(0),
                    }
                if not judgment_row.empty and not judgment_row.iloc[0].isnull().any():
                    doc_dict['grade'] = judgment_row.iloc[0]['grade'].item()
                    doc_dict['label'] = grade_to_emoji(doc_dict['grade'])
                top_docs.append(doc_dict)
            return top_docs
        except Exception as e:
            logger.warning(f"Error trying out patch on query '{query}': {e}")
            return str(e)

    def try_out_patch(edit: Edit) -> EvalResult:
        logger.info("Evaluating patch")

        with open(filepath, "r") as f:
            existing_code = f.read()
        info = None

        try:
            if training_eval_fn is None:
                return None
            code, existing_code, local_vars = _patch_code(edit)
            ndcgs_before, ndcgs_after, warning, info = _training_eval_fn(
                code,
                existing_code)
            per_query_deltas = _per_query_deltas(ndcgs_before, ndcgs_after)

            res = EvalResult(success=True,
                             warning_messages=warning,
                             info_messages=info,
                             ndcg_before=ndcgs_before.mean(),
                             ndcg_after=ndcgs_after.mean(),
                             per_query_deltas=per_query_deltas,
                             current_code=existing_code)
            return res
        except Exception as e:
            msg = f"Error evaluating code in patch (no eval was performed): {e}"
            logger.warning(msg)
            res = EvalResult(success=False, warning_messages=msg,
                             ndcg_deltas={}, existing_code=existing_code,
                             info_messages=info)
            return res

    try_out_patch.__doc__ = f"""Evaluate the proposed code change to analyze its impact on all training queries.
    (Results won't be saved, this is used to evaluate potential patches before applying them.)

    Limitation: only returns the NDCG for a query, not the full results. Use try_out_patch_on_query to see full results for specific queries.

    {guardrail_doc_strs}

    You'll be warned if guardrails of apply_patch would be violated and therefore you should revise your change
    (though success will still be True if code runs without error).

    """

    def apply_patch(edit: Edit) -> EditResult:
        info = None
        try:
            logger.info("Applying patch with edits")
            code, existing_code, local_vars = _patch_code(edit)
            # Compare NDCG before and after
            edit_result = EditResult(success=True, error_message=None, current_code=existing_code)
            # Check on training data, to see if expected queries changed
            ndcgs_before, ndcgs_after, warning, info = _training_eval_fn(
                code,
                existing_code)
            if warning is not None:
                raise ValueError(warning)

            if validation_eval_fn is not None:
                ndcg_before = validation_eval_fn(existing_code).mean()
                ndcg_after = validation_eval_fn(code).mean()
                if ndcg_after < (ndcg_before + eval_margin):
                    logger.warning(f"❌ Rejecting Change: Validation NDCG must increase at least {eval_margin} after applying patch: before={ndcg_before}, after={ndcg_after}")
                    raise ValueError(f"Rejecting change as overfit to training set must increase validation NDCG by at least {eval_margin}: before={ndcg_before}, after={ndcg_after}")
                else:
                    logger.info(f"✅ Validation NDCG improved: before={ndcg_before}, after={ndcg_after}")

            code = _commit_code(code)
            if code:
                edit_result.current_code = code
                edit_result.success = True
                edit_result.error_messages = None
                edit_result.info_messages = info
                return edit_result
        except Exception as e:
            logger.warning(f"Error applying patch: {e}")
            with open(filepath, "r") as f:
                existing_code = f.read()
            return EditResult(success=False,
                              info_messages=info,
                              error_messages=str(e),
                              current_code=existing_code)
    apply_patch.__doc__ = f"""Save the proposed code change to rerank_esci.py.

    {full_guardrail_doc_strs}
    Code will be rejected if it does not improves / harm training data as expected (within some tolerance).
    Code will be rejected if a the percentage of changed queries by this change are not improvements (at least 75% of changed queries must improve).

    """
    if training_eval_fn is None:
        return apply_patch, None, revert_changes
    return apply_patch, try_out_patch, revert_changes, try_out_patch_on_query


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
