import os
import logging
import pathlib


logger = logging.getLogger(__name__)


def get_project_root():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(file_dir))


DATA_PATH = pathlib.Path(get_project_root()) / "data"
if os.environ.get("CHEAT_AT_SEARCH_DATA_PATH"):
    DATA_PATH = os.environ["CHEAT_AT_SEARCH_DATA_PATH"]
    logger.info(f"Using WANDS data path from environment variable: {DATA_PATH}")


def ensure_data_subdir(subdir: str):
    """
    Ensure a subdirectory exists within the data directory.

    Args:
        subdir: Name of the subdirectory to ensure exists
    """
    subdir_path = DATA_PATH / subdir
    if not subdir_path.exists():
        logger.info(f"Creating data subdirectory: {subdir_path}")
        subdir_path.mkdir(parents=True, exist_ok=True)
    return subdir_path
