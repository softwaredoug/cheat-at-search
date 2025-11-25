from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import DATA_PATH, sync_git_repo
from pathlib import Path
import tarfile
import pandas as pd

tmdb_path = Path(DATA_PATH) / "tmdb"

logger = log_to_stdout("tmdb_data")


def fetch_tmdb():
    # Download to fixtures
    print("Downloading TheMovieDB (TMDB) dataset")
    git_url = "https://github.com/ai-powered-search/tmdb.git"

    return sync_git_repo(tmdb_path, git_url)


def _judgments():
    fetch_tmdb()
    judgments_path = tmdb_path / "judgments.tgz"
    judgments_extracted_path = tmdb_path / "ai_pow_search_judgments.txt"
    if not judgments_extracted_path.exists():
        with tarfile.open(judgments_path, "r:gz") as tar:
            tar.extractall(path=tmdb_path)
    # Build a buffer, stripping anything with comments
    without_comments = []
    judgments = []
    with open(judgments_extracted_path, "r") as f:
        without_comments = [line for line in f if not line.startswith("#")]
        for line in without_comments:
            if line.strip() == "":
                continue
            sep = line.split()
            grade = int(sep[0])
            query_id = sep[1]
            query_id = int(query_id.replace("qid:", ""))
            assert sep[2] == '#'
            doc_id = int(sep[3])
            query = sep[4].strip()
            assert len(query) > 0
            judgments.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "grade": grade,
                "query": query
            })
    return pd.DataFrame(judgments)


def _corpus():
    fetch_tmdb()
    corpus_path = tmdb_path / "movies.tgz"
    corpus_extracted_path = tmdb_path / "tmdb.json"
    if not corpus_extracted_path.exists():
        with tarfile.open(corpus_path, "r:gz") as tar:
            tar.extractall(path=tmdb_path)
    movies = pd.read_json(corpus_extracted_path, orient="index")
    movies = movies[~movies['title'].isna()]
    movies['overview'] = movies['overview'].fillna("")
    movies['description'] = movies['overview']
    movies['doc_id'] = movies.index.astype(int)
    return movies


def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]
    if name == "judgments":
        ds = _judgments()
        globals()["judgments"] = ds
        return globals()[name]
    elif name == "corpus":
        ds = _corpus()
        globals()["corpus"] = ds
        return globals()[name]
    elif name == "queries":
        ds = _judgments()
        queries = ds[['query', 'query_id']].drop_duplicates().reset_index(drop=True)
        globals()["queries"] = queries
        return globals()["queries"]


if __name__ == "__main__":
    _judgments()
