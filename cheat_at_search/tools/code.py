from typing import List, Dict, Union, Optional, Literal
from pydantic import BaseModel, Field
from cheat_at_search.logger import log_to_stdout


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


def make_patch_fn(search_fn, corpus, module_name: str):
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

        Edits more than 10 lines will be rejected

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
                if edit.text.count('\n') > 11:
                    raise ValueError("Edit text exceeds 10 lines limit. Please keep it incremental.")

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
