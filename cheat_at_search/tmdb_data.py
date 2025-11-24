from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import ensure_data_subdir, download_file, sync_git_repo
from pathlib import Path


tmdb_path = Path(ensure_data_subdir("tmdb"))

logger = log_to_stdout("tmdb_data")


def fetch_tmdb():
    # Download to fixtures
    print("Downloading TheMovieDB (TMDB) dataset")
    git_url = "https://github.com/ai-powered-search/tmdb.git"

    return sync_git_repo(tmdb_path, git_url)


if __name__ == "__main__":
    path = fetch_tmdb()
    import pdb; pdb.set_trace()
