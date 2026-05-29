import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from cheat_at_search.data_dir import mount


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "datasets"


def _clear_module_cache(module):
    for attr in ["queries", "judgments", "corpus", "products", "labels"]:
        module.__dict__.pop(attr, None)


def _mount_temp_cache():
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return data_dir


@patch("cheat_at_search.tmdb_data.tmdb_path", FIXTURE_ROOT / "tmdb")
@patch("cheat_at_search.tmdb_data.fetch_tmdb")
@patch("cheat_at_search.wands_data.fetch_wands")
def test_bm25_search_fixtures(mock_fetch_wands, mock_fetch_tmdb):
    _mount_temp_cache()
    mock_fetch_wands.return_value = FIXTURE_ROOT / "wands"
    mock_fetch_tmdb.return_value = FIXTURE_ROOT / "tmdb"

    for module_name in ["wands_data", "tmdb_data"]:
        module = importlib.import_module(f"cheat_at_search.{module_name}")
        _clear_module_cache(module)
        from cheat_at_search.search import run_strategy
        from cheat_at_search.strategy import BM25Search

        corpus = getattr(module, "corpus")
        judgments = getattr(module, "judgments")
        strategy = BM25Search(corpus)
        graded_results = run_strategy(strategy, judgments, num_queries=5, seed=123)
        assert len(graded_results) > 0


@patch("cheat_at_search.wands_data.fetch_wands")
def test_vs_ideal_wands_fixture(mock_fetch_wands):
    _mount_temp_cache()
    mock_fetch_wands.return_value = FIXTURE_ROOT / "wands"

    module = importlib.import_module("cheat_at_search.wands_data")
    _clear_module_cache(module)
    from cheat_at_search.search import run_strategy, vs_ideal
    from cheat_at_search.strategy import BM25Search

    corpus = module.corpus
    judgments = module.judgments
    strategy = BM25Search(corpus)
    graded_results = run_strategy(strategy, judgments, num_queries=2, seed=123)

    comparison = vs_ideal(graded_results, judgments, corpus=corpus)
    assert len(comparison) > 0
    assert comparison["rank_actual"].max() <= 10
    assert comparison["rank_ideal"].max() <= 10


@patch("cheat_at_search.wands_data.fetch_wands")
def test_run_bm25_cached_fixture(mock_fetch_wands):
    cache_dir = _mount_temp_cache()
    mock_fetch_wands.return_value = FIXTURE_ROOT / "wands"

    module = importlib.import_module("cheat_at_search.wands_data")
    _clear_module_cache(module)
    from cheat_at_search.search import run_bm25, vs_ideal

    corpus = module.corpus
    judgments = module.judgments

    graded_bm25 = run_bm25(corpus, judgments)
    assert (Path(cache_dir) / "bm25_results" / "graded_bm25.pkl").exists()
    assert "mrr" in graded_bm25.columns

    comparison = vs_ideal(graded_bm25, judgments, corpus=corpus)
    assert len(comparison) > 0
    assert comparison["rank_actual"].max() <= 10
