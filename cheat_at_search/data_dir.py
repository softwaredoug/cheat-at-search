import os
import pathlib
import getpass
import json
from cheat_at_search.logger import log_to_stdout
import subprocess
from pathlib import Path
import requests


logger = log_to_stdout(logger_name="data_dir")


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


def download_file(url, dest_path):
    filename = url.split('/')[-1]
    local_filename = Path(dest_path) / filename

    if Path(local_filename).exists():
        print(f"File {local_filename} already exists, skipping download.")
        return local_filename

    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        print(f"Downloading {url} to {local_filename}")
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {local_filename}")
    return local_filename


def sync_git_repo(data_dir: str, repo_url: str):
    # Convert to absolute path
    data_path = Path(data_dir).absolute()

    # Create directories if they don't exist
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the directory already exists
    if data_path.exists():
        logger.info(f"Directory {data_path} already exists. Checking for updates...")
        try:
            subprocess.run(
                ["git", "-C", str(data_path), "fetch", "origin"],
                check=True,
                capture_output=True,
                text=True
            )
            subprocess.run(
                ["git", "-C", str(data_path), "reset", "--hard", "origin/main"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Updated {repo_url} dataset at {data_path}")
            return data_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to update {repo_url} dataset: {e.stderr}. Will delete and re-clone.")
            # If fetch or reset fails, delete and re-clone
            subprocess.run(["rm", "-rf", str(data_path)], check=True)

    logger.info(f"Cloning dataset from {repo_url} to {data_path}")

    try:
        # Clone the repository
        logger.info(f"Cloning {repo_url} into {data_path}")
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(data_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully cloned {repo_url} dataset to {data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone {repo_url} dataset: {e.stderr}")
        raise

    return data_path


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


def mount_key(keyname):
    keyname = keyname.lower().strip()
    KEY_PATH = f"{DATA_PATH}/keys.json"
    key_json = {}
    try:
        logger.info(f"Reading {keyname} API key from {KEY_PATH}")
        with open(KEY_PATH, "r") as f:
            key_json = json.load(f)
            openai_key = key_json[keyname]
            globals()[keyname] = openai_key
            logger.info(f"{keyname} key loaded successfully.")
            return openai_key
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print("You're going to be prompted for your API key. This will be stored in a local file")
        print("If you'd prefer to set it as an environment variable, set it as:")
        print(f"    export {keyname.upper()}=your_api_key_here")
        key = getpass.getpass(f"Enter your {keyname}: ")
        key_json[keyname] = key
        with open(os.path.join(KEY_PATH), 'w') as f:
            json.dump(key_json, f)
            logger.info(f"Saving {keyname} key to {KEY_PATH}")
        globals()[keyname] = key
        return key


def mount(use_gdrive=True, manual_path=None, load_keys=True):
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
    elif use_gdrive:
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

    logger.info(f"Mounting data directory at {DATA_PATH}")


def key_for_provider(provider: str) -> str:
    keyname = provider.lower().strip()
    keyname_env = f"{keyname}_API_KEY".upper()
    logger.info(f"Looking for {keyname} in environment variables or globals...")
    if os.getenv(keyname_env):
        openai_key = os.getenv(keyname_env)
        globals()[keyname] = openai_key
        return openai_key
    elif keyname in globals():
        logger.info(f"{keyname} available from previous mount.")
        return globals()[keyname]
    else:
        return mount_key(keyname_env)


def __getattr__(name):
    """
    Allow access to DATA_PATH as a module attribute.
    """
    if name == "DATA_PATH":
        return DATA_PATH

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
