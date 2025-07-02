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
    subdir_path = pathlib.Path(DATA_PATH) / subdir
    if not subdir_path.exists():
        logger.info(f"Creating data subdirectory: {subdir_path}")
        subdir_path.mkdir(parents=True, exist_ok=True)
    return subdir_path


def mount(use_gdrive=True, mount_path=None):
    """
    Mount the data directory to a specific path.

    Args:
        use_grive: If True, mount using grive; otherwise, use local path.
        mount_path: Optional path to mount the data directory.
    """
    if use_gdrive:
        # Assumes you're running this in Google Colab
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            DATA_PATH = '/content/drive/MyDrive/cheat-at-search-data/'
        except ImportError:
            logger.error("Google Colab drive module not found. Ensure you're running this in Google Colab.")
            raise
    else:
        path_directory = pathlib.Path('cheat-at-search-data/')
        if not path_directory.exists():
            logger.info(f"Creating data directory: {path_directory}")
            path_directory.mkdir(parents=True, exist_ok=True)
        DATA_PATH = path_directory

    logger.info(f"Mounting data directory at {DATA_PATH}")
