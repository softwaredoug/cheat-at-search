import os
import logging
import pathlib
import getpass


logger = logging.getLogger(__name__)


def get_project_root():
    # Loop backwards until "cheat-at-search"
    # is found in the directory structure
    file_dir = os.path.abspath(__file__)
    while not os.path.basename(file_dir) == "cheat_at_search":
        file_dir = os.path.dirname(file_dir)
        if not file_dir or file_dir == "/":
            raise ValueError("Project root directory 'cheat-at-search' not found in the path.")

    return os.path.dirname(file_dir)


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


def mount(use_gdrive=True, manual_path=None):
    """
    Mount the data directory to a specific path.

    Args:
        use_grive: If True, mount using grive; otherwise, use 'cheat-at-search-data/' directory.
    """
    global DATA_PATH
    if manual_path:
        if not pathlib.Path(manual_path).exists():
            logger.info(f"Creating manual data directory: {manual_path}")
            pathlib.Path(manual_path).mkdir(parents=True, exist_ok=True)
        DATA_PATH = pathlib.Path(manual_path)
    if use_gdrive:
        # Assumes you're running this in Google Colab
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            DATA_PATH = '/content/drive/MyDrive/cheat-at-search-data/'
            if not pathlib.Path(DATA_PATH).exists():
                logger.info(f"Creating Google Drive data directory: {DATA_PATH}")
                pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
        except ImportError:
            logger.error("Google Colab drive module not found. Ensure you're running this in Google Colab.")
            raise
    else:
        path_directory = pathlib.Path('cheat-at-search-data/')
        if not path_directory.exists():
            logger.info(f"Creating data directory: {path_directory}")
            path_directory.mkdir(parents=True, exist_ok=True)
        DATA_PATH = path_directory

    # Check for OpenAI key in data directory
    KEY_PATH = f"{DATA_PATH}/openai_key.txt"
    try:
        logger.info(f"Reading OpenAI API key from {KEY_PATH}")
        with open(KEY_PATH, "r") as f:
            openai_key = f.read().strip()
            globals()['openai_key'] = openai_key
    except FileNotFoundError:
        key = getpass.getpass("Enter your openai key: ")
        with open(os.path.join(KEY_PATH), 'w') as f:
            logger.info(f"Saving OpenAI API key to {KEY_PATH}")
            f.write(key)
            globals()['openai_key'] = key

    logger.info(f"Mounting data directory at {DATA_PATH}")


def __getattr__(name):
    """
    Allow access to DATA_PATH as a module attribute.
    """
    if name == "OPENAI_KEY":
        print("Accessing OpenAI API key.")
        if os.getenv("OPENAI_API_KEY"):
            openai_key = os.getenv("OPENAI_API_KEY")
            globals()['openai_key'] = openai_key
            return openai_key
        elif 'openai_key' in globals():
            print("OpenAI Key available from previous mount.")
            return globals()['openai_key']
        else:
            logger.warning("OpenAI key not set. Please mount the data directory first.")
            return ''
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
