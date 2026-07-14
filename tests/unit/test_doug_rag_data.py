import pandas as pd

from cheat_at_search import doug_rag_data


def test_judgments_are_answer_rows():
    judgments = doug_rag_data._judgments()

    assert list(judgments.columns) == ["query_id", "query", "answer"]
    assert judgments["query_id"].is_unique
    assert (judgments["answer"] != "").all()
    assert "doc_id" not in judgments.columns
    assert "grade" not in judgments.columns


def test_judgments_are_data_frame():
    assert isinstance(doug_rag_data._judgments(), pd.DataFrame)
