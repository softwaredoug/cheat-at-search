import pytest
import importlib
import numpy as np


@pytest.mark.parametrize(
    "data_module",
    [
        "minimarco_data",
        "esci_data",
        "bc_plus_data",
        "wands_data",
        "tmdb_data",
        "doug_blog_data",
    ],
)
def test_common_imports(data_module):
    """Confirm no import error when importing these values from the dataset module."""
    import_dfs_to_expected_columns = {
        "queries": ["query", "query_id"],
        "judgments": ["query_id", "doc_id", "grade"],
        "corpus": [
            "doc_id",
            "title",
            "description",
            "title_snowball",
            "description_snowball",
        ],
    }
    try:
        module = importlib.import_module(f"cheat_at_search.{data_module}")
        for df_import, expected_columns in import_dfs_to_expected_columns.items():
            try:
                df = getattr(module, df_import)
                for col in expected_columns:
                    if col not in df.columns:
                        pytest.fail(
                            f"DataFrame {df_import} from cheat_at_search.{data_module} is missing expected column: {col}"
                        )
            except AttributeError as e:
                pytest.fail(
                    f"Accessing {df_import} from cheat_at_search.{data_module} raised AttributeError: {e}"
                )
    except ImportError as e:
        pytest.fail(f"Importing cheat_at_search.{data_module} raised ImportError: {e}")


def test_bc_plus_judgments_include_answer():
    module = importlib.import_module("cheat_at_search.bc_plus_data")
    judgments = getattr(module, "judgments")
    assert "answer" in judgments.columns


@pytest.mark.parametrize(
    "data_module",
    [
        "msmarco_data",
        "minimarco_data",
        "esci_data",
        "bc_plus_data",
        "wands_data",
        "tmdb_data",
        "doug_blog_data",
    ],
)
def test_lexical_indexes_score_arrays(data_module):
    module = importlib.import_module(f"cheat_at_search.{data_module}")
    corpus = getattr(module, "corpus")
    title_scores = corpus["title_snowball"].array.score("test")
    description_scores = corpus["description_snowball"].array.score("test")
    assert isinstance(title_scores, np.ndarray)
    assert isinstance(description_scores, np.ndarray)
    assert len(title_scores) == len(corpus)
    assert len(description_scores) == len(corpus)
