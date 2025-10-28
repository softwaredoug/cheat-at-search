from typing import List, Dict, Optional, Literal, Callable, Tuple
from pydantic import BaseModel, Field
import pandas as pd
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.agent.openai_agent import OpenAIAgent
from functools import lru_cache


logger = log_to_stdout(logger_name="cheat_at_search.code")


class Edit(BaseModel):
    """A single edit to apply to the reranker code."""
    anchor: str = Field(..., description="The anchor text to identify where the patch should be applied.")
    block_until: str = Field(..., description="The end of the block of text which the patch should be applied. Do not leave blank.")
    action: Literal['insert_after', 'replace', 'delete'] = Field(..., description="The action to perform: insert_after, replace, or delete.")
    text: str = Field(..., description="The text to insert or replace with. Ignored for delete action.")


class EditResult(BaseModel):
    """The result of applying edits to the reranker code."""
    success: bool = Field(..., description="Whether the edits were applied successfully and the reranker passed tests.")
    error_message: Optional[str] = Field(None, description="An error message if the edits failed to apply or tests failed.")
    current_code: str = Field(None, description="The current reranker code after this call.")


class EvalResult(BaseModel):
    success: bool = Field(..., description="Whether the edits can be applied succesfully without code errors.")
    error_message: Optional[str] = Field(None, description="An error message if the edits failed to apply or tests failed.")
    ndcg_deltas: Optional[Dict[str, float]] = Field(None, description="The NDCG deltas for the training dataset.")
    ndcg_before: Optional[float] = Field(0.0, description="The NDCG before applying the edit.")
    ndcg_after: Optional[float] = Field(0.0, description="The NDCG after applying the edit.")
    current_code: Optional[str] = Field(None, description="The current reranker code after this call.")


def make_length_validator(max_lines: int = 10, max_cols=120) -> Callable[[str], Optional[str]]:

    def length_validation(code: str) -> Optional[str]:
        """Validate that the code does not exceed max_lines."""

        if code.count('\n') > max_lines:
            return f"Code exceeds maximum length of {max_lines} lines."

        for line in code.split('\n'):
            if len(line) > max_cols + 20:
                return f"Line exceeds maximum length of {max_cols} characters: {line}"
        return None
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
        response = agent.loop(user_prompt=f"Please evaluate the following code for compliance:\n```python\n{code}\n```")
        if not response.compliant:
            issues = "\n".join(response.issues) if response.issues else "No specific issues provided."
            return f"Code does not comply with guardrails:\n{issues}"
    return code_guardrails


def make_patch_fn(search_fn, corpus, module_name: str,
                  guardrail_fns: List = [],
                  training_eval_fn: Optional[Callable] = None,
                  validation_eval_fn: Optional[Callable] = None,
                  eval_margin=0.003) -> callable:
    """Returns a function that applies patches to the reranker code."""

    if training_eval_fn is not None:
        training_eval_fn = lru_cache(maxsize=64)(training_eval_fn)
    if validation_eval_fn is not None:
        validation_eval_fn = lru_cache(maxsize=64)(validation_eval_fn)

    def revert_changes() -> str:
        """Undo the last patch to rerank_esci.py by restoring from backup."""
        with open(f"{module_name}_backup.py", "r") as backup:
            with open(f"{module_name}.py", "w") as f:
                logger.info(f"Reverted {module_name}.py to backup.")
                code = backup.read()
                f.write(code)
                logger.info("Reverted changes successfully.")
                return code
        return "Error reverting changes."

    def _patch_code(edit: Edit,
                    test_queries=["red dress", "real housewives of orange county"]) -> Tuple[str, str]:
        filepath = f"{module_name}.py"
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
            for guardrail in guardrail_fns:
                error_message = guardrail(edit.text)
                if error_message is not None:
                    raise ValueError(error_message)

            if edit.action == 'insert_after':
                insertion_point = block_index + len(edit.block_until)
                code = code[:insertion_point] + '\n' + edit.text + '\n' + code[insertion_point:]
            elif edit.action == 'replace':
                code = code[:anchor_index] + edit.text + code[block_index + len(edit.block_until):]
            elif edit.action == 'delete':
                code = code[:anchor_index] + code[block_index + len(edit.block_until):]
            else:
                raise ValueError(f"Unknown action '{edit.action}'.")
        # Attempt to eval the code
        local_vars = {}
        exec(code, {}, local_vars)
        if module_name not in local_vars:
            logger.error("Edited code does not define module_name")
            raise ValueError("The edited code does not define module_name.")
        # Test that rerank_esci is callable
        if not callable(local_vars[module_name]):
            logger.error("module_name is not callable.")
            raise ValueError("module_name is not callable.")
        for query in test_queries:
            try:
                results = local_vars[module_name](search_fn, query)[:10]
            except Exception as e:
                logger.error(f"Error calling {module_name} with query '{query}': {e}")
                logger.error(code)
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
        filepath = f"{module_name}.py"
        backup_path = f"{module_name}_backup.py"
        with open(filepath, "r") as f:
            with open(backup_path, "w") as backup:
                logger.info(f"Creating backup of {module_name}.py at {backup_path}")
                backup.write(f.read())

        with open(filepath, "w") as f:
            f.write(code)
            return code

    def try_out_patch(edit: Edit) -> EvalResult:
        """Evaluate the proposed code change to analyze its impact on training queries.
        (Results won't be saved, this is used to evaluate potential patches before applying them.)

        Edits more than 10 lines will be rejected.
        Edits with lines longer than 120 characters will be rejected.
        Edits that fail to compile / eval will be rejected
        Edits where the code appears to be overfit to training queries will be rejected
        NO checks to validation NDCG are performed in this function.

        """
        logger.info("Evaluating patch")

        with open(f"{module_name}.py", "r") as f:
            existing_code = f.read()

        try:
            if training_eval_fn is None:
                return None
            code, existing_code, local_vars = _patch_code(edit)
            ndcgs_before: pd.Series = training_eval_fn(existing_code)
            ndcgs_after: pd.Series = training_eval_fn(code)
            deltas: pd.Series = ndcgs_after - ndcgs_before
            delta_dict = deltas.to_dict()
            changed_queries = {}
            for query in delta_dict:
                if delta_dict[query] != 0.0:
                    changed_queries[query] = delta_dict[query]

            icon = '✅' if ndcgs_after.mean() >= ndcgs_before.mean() else '❌'
            logger.info(f"{icon} Evaluated patch successfully. train NDCG before: {ndcgs_before.mean()}, after: {ndcgs_after.mean()}")
            logger.info(f"Changed queries NDCG deltas: {changed_queries}")
            logger.info("Code:")
            logger.info(code)
            return EvalResult(success=True, error_message=None, ndcg_deltas=changed_queries,
                              ndcg_before=ndcgs_before.mean(), ndcg_after=ndcgs_after.mean(),
                              current_code=existing_code)
        except Exception as e:
            logger.info(f"Error evaluating patch: {e}")
            return EvalResult(success=False, error_message=str(e), ndcg_deltas={}, existing_code=existing_code)

    def apply_patch(edit: Edit) -> EditResult:
        """Save the proposed code change to rerank_esci.py.

        Edits more than 10 lines will be rejected.
        Edits with lines longer than 120 characters will be rejected.
        Edits that fail to compile / eval will be rejected
        Edits where the code appears to be overfit to training queries will be rejected
        Edits that reduce validation NDCG will be rejected as overfitting

        """
        try:
            logger.info("Applying patch with edits")
            code, existing_code, local_vars = _patch_code(edit)
            # Compare NDCG before and after
            edit_result = EditResult(success=True, error_message=None, current_code=existing_code)
            if validation_eval_fn is not None:
                ndcg_before = validation_eval_fn(existing_code).mean()
                ndcg_after = validation_eval_fn(code).mean()
                if ndcg_after < (ndcg_before + eval_margin):
                    logger.warning(f"❌ Rejecting Change: Validation NDCG must increase at least {eval_margin} after applying patch: before={ndcg_before}, after={ndcg_after}")
                    raise ValueError(f"Rejecting change as overfit must increase NDCG by at least {eval_margin}: before={ndcg_before}, after={ndcg_after}")
                else:
                    logger.info(f"✅ Validation NDCG improved: before={ndcg_before}, after={ndcg_after}")

            code = _commit_code(code)
            if code:
                edit_result.current_code = code
                return edit_result
        except Exception as e:
            logger.info(f"Error applying patch: {e}")
            with open(f"{module_name}.py", "r") as f:
                existing_code = f.read()
            return EditResult(success=False, error_message=str(e), query_results={},
                              current_code=existing_code)
    return apply_patch, try_out_patch, revert_changes
