from importlib import resources
from pathlib import Path

import pandas as pd

from cheat_at_search.data_dir import get_project_root
from cheat_at_search.logger import log_to_stdout


logger = log_to_stdout("doug_blog_data")


def _docs() -> pd.DataFrame:
    posts_resource = None
    try:
        posts_resource = resources.files("cheat_at_search").joinpath(
            "data/posts.json.gz"
        )
    except Exception:
        posts_resource = None

    if posts_resource is not None and posts_resource.is_file():
        with posts_resource.open("rb") as handle:
            posts = pd.read_json(handle, lines=True, compression="gzip")
    else:
        project_root = Path(get_project_root())
        posts_path = project_root / "cheat_at_search" / "data" / "posts.json.gz"
        if not posts_path.exists():
            raise FileNotFoundError(f"Expected posts.json.gz at {posts_path}")
        posts = pd.read_json(posts_path, lines=True, compression="gzip")
    posts = posts.rename(columns={"body": "description"})
    posts["doc_id"] = range(len(posts))
    if "title" not in posts.columns:
        posts["title"] = ""
    if "description" not in posts.columns:
        posts["description"] = ""
    return posts[["doc_id", "title", "description"]]


def _judgments() -> pd.DataFrame:
    return pd.DataFrame(columns=["query_id", "query", "doc_id", "grade"])


def __getattr__(name):
    """Load dataset lazily."""
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        ds = _judgments()
        globals()["judgments"] = ds
        queries = ds[["query", "query_id"]].drop_duplicates().reset_index(drop=True)
        globals()["queries"] = queries
        return globals()[name]
    if name == "corpus":
        ds = _docs()
        globals()["corpus"] = ds
        return globals()[name]
    raise AttributeError(f"Module {__name__} has no attribute {name}")


if __name__ == "__main__":
    _docs()
