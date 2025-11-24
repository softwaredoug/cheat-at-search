import pytest
import importlib


@pytest.mark.parametrize("data_module", ["msmarco_data", "esci_data", "wands_data"])
def test_common_imports(data_module):
    """Confirm no import error when importing these values from the dataset module."""
    import_dfs_to_expected_columns = {
        "queries": ["query", "query_id"],
        "judgments": ["query_id", "doc_id", "grade"],
        "corpus": ["doc_id", "title", "description"],
    }
    try:
        module = importlib.import_module(f"cheat_at_search.{data_module}")
        for df_import, expected_columns in import_dfs_to_expected_columns.items():
            try:
                df = getattr(module, df_import)
                for col in expected_columns:
                    if col not in df.columns:
                        pytest.fail(f"DataFrame {df_import} from cheat_at_search.{data_module} is missing expected column: {col}")
            except AttributeError as e:
                pytest.fail(f"Accessing {df_import} from cheat_at_search.{data_module} raised AttributeError: {e}")
    except ImportError as e:
        pytest.fail(f"Importing cheat_at_search.{data_module} raised ImportError: {e}")
