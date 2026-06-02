from unittest.mock import patch

import pandas as pd

from cheat_at_search import doug_blog_data


@patch("cheat_at_search.doug_blog_data._docs")
def test_judgments_match_title_phrases(mock_docs):
    mock_docs.return_value = pd.DataFrame(
        [
            {"doc_id": 1, "title": "BM25 for search"},
            {"doc_id": 2, "title": "Learning to rank basics"},
            {"doc_id": 3, "title": "Other topic"},
        ]
    )

    judgments = doug_blog_data._judgments()

    assert set(judgments["doc_id"]) == {1, 2}
    assert set(judgments["query_id"]) == {"bm25", "learning_to_rank"}
    assert (judgments["grade"] == 1).all()
