from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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


class GuardrailResponse(BaseModel):
    """The response from the guardrail checker."""

    compliant: bool = Field(
        ..., description="Whether the code complies with the guardrails."
    )
    issues: Optional[List[str]] = Field(
        None, description="A list of issues found in the code, if any."
    )
