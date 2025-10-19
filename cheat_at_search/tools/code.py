from typing import List, Dict, Union, Optional, Literal, Callable
from pydantic import BaseModel, Field
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.agent.openai_agent import OpenAIAgent


logger = log_to_stdout(logger_name="cheat_at_search.code")


class Edit(BaseModel):
    """A single edit to apply to the reranker code."""
    anchor: str = Field(..., description="The anchor text to identify where the patch should be applied.")
    block_until: str = Field(..., description="The end of the block of text which the patch should be applied. Do not leave blank.")
    action: Literal['insert_after', 'replace', 'delete'] = Field(..., description="The action to perform: insert_after, replace, or delete.")
    text: str = Field(..., description="The text to insert or replace with. Ignored for delete action.")
    test_queries: List[str] = Field(..., description="A list of test queries to validate the reranker after applying edits.")


class EditResult(BaseModel):
    """The result of applying edits to the reranker code."""
    success: bool = Field(..., description="Whether the edits were applied successfully and the reranker passed tests.")
    error_message: Optional[str] = Field(None, description="An error message if the edits failed to apply or tests failed.")
    query_results: Dict[str, Union[List[Dict], str]] = Field(..., description="The results of running the reranker on the test queries after applying edits.")
    current_code: str = Field(None, description="The current reranker code after this call.")


def make_length_validator(max_lines: int = 10, max_cols=120) -> Callable[[str], Optional[str]]:

    def length_validation(code: str) -> Optional[str]:
        """Validate that the code does not exceed max_lines."""

        if code.count('\n') > max_lines:
            return f"Code exceeds maximum length of {max_lines} lines."

        for line in code.split('\n'):
            if len(line) > max_cols:
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
                  validation_fns: List = [],
                  eval_fn: Optional[Callable] = None) -> callable:
    """Returns a function that applies patches to the reranker code."""

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

    def apply_patch(edit: Edit) -> EditResult:
        """Apply an incremental edit to reranker code.

        Edits more than 10 lines will be rejected.
        Edits with lines longer than 120 characters will be rejected.
        Edits that fail to compile / eval will be rejected
        Edits where the code appears to be overfit to training queries will be rejected
        Edits that reduce validation NDCG will be rejected as overfitting

        """
        try:
            logger.info("Applying patch with edits:")
            existing_code = ""
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
                for validation_fn in validation_fns:
                    error_message = validation_fn(edit.text)
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
            # Call with test_queries
            edit_result = EditResult(success=True, error_message=None, query_results={})
            for query in edit.test_queries:
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
                    dict_results = []
                    for result in results:
                        product = corpus[corpus['product_id'] == result]
                        if len(product) == 0:
                            continue
                        product = product.iloc[0]
                        dict_results.append({
                            'id': product['product_id'],
                            'title': product['title'],
                            'description': product['description'],
                        })
                    edit_result.query_results[query] = dict_results
                except Exception as e:
                    logger.error(f"Error collecting results with query '{query}': {e}")
                    raise ValueError(f"Error calling 'rerank_esci' with query '{query}': {e}")
            # Compare NDCG before and after
            if eval_fn:
                ndcg_before = eval_fn(existing_code)
                ndcg_after = eval_fn(code)
                if ndcg_after < ndcg_before:
                    logger.warning(f"Rejecting Change: Validation NDCG decreased after applying patch: before={ndcg_before}, after={ndcg_after}")
                    raise ValueError(f"Rejecting change as overfit: Validation NDCG decreased on test set after applying patch: before={ndcg_before}, after={ndcg_after}")

            backup_path = f"{module_name}_backup.py"
            with open(filepath, "r") as f:
                with open(backup_path, "w") as backup:
                    logger.info(f"Creating backup of {module_name}.py at {backup_path}")
                    backup.write(f.read())

            with open(filepath, "w") as f:
                f.write(code)
                edit_result.current_code = code
                logger.info(f"Patched {filepath} successfully.")
                return edit_result
        except Exception as e:
            logger.info(f"Error applying patch: {e}")
            return EditResult(success=False, error_message=str(e), query_results={},
                              current_code=existing_code)
    return apply_patch, revert_changes
