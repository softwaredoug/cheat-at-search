import pytest
import importlib
import numpy as np
from pathlib import Path

from cheat_at_search.data_dir import mount


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "datasets"


@pytest.fixture(autouse=True)
def dataset_fixtures(tmp_path, monkeypatch):
    mount(use_gdrive=False, manual_path=str(tmp_path), load_keys=False)

    def _clear_cached_attrs(module):
        for attr in [
            "queries",
            "judgments",
            "corpus",
            "products",
            "labels",
            "labeled_queries",
            "labeled_query_products",
            "ideal_top_10",
        ]:
            module.__dict__.pop(attr, None)

    msmarco_data = importlib.import_module("cheat_at_search.msmarco_data")
    _clear_cached_attrs(msmarco_data)
    monkeypatch.setattr(msmarco_data, "msmarco_path", FIXTURE_ROOT / "msmarco")
    monkeypatch.setattr(msmarco_data, "download_msmarco", lambda: None)

    minimarco_data = importlib.import_module("cheat_at_search.minimarco_data")
    _clear_cached_attrs(minimarco_data)

    esci_data = importlib.import_module("cheat_at_search.esci_data")
    _clear_cached_attrs(esci_data)
    monkeypatch.setattr(esci_data, "fetch_esci", lambda: FIXTURE_ROOT / "esci")

    wands_data = importlib.import_module("cheat_at_search.wands_data")
    _clear_cached_attrs(wands_data)
    monkeypatch.setattr(wands_data, "fetch_wands", lambda: FIXTURE_ROOT / "wands")

    tmdb_data = importlib.import_module("cheat_at_search.tmdb_data")
    _clear_cached_attrs(tmdb_data)
    monkeypatch.setattr(tmdb_data, "tmdb_path", FIXTURE_ROOT / "tmdb")
    monkeypatch.setattr(tmdb_data, "fetch_tmdb", lambda: FIXTURE_ROOT / "tmdb")

    bc_plus_data = importlib.import_module("cheat_at_search.bc_plus_data")
    _clear_cached_attrs(bc_plus_data)
    monkeypatch.setattr(
        bc_plus_data,
        "ensure_data_subdir",
        lambda _subdir: FIXTURE_ROOT / "browsecomp_plus",
    )

    doug_blog_data = importlib.import_module("cheat_at_search.doug_blog_data")
    _clear_cached_attrs(doug_blog_data)

    class BlogResource:
        def joinpath(self, *_parts):
            return FIXTURE_ROOT / "doug_blog" / "posts.json.gz"

    monkeypatch.setattr(doug_blog_data.resources, "files", lambda _package: BlogResource())


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
