import os
import importlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
from platformdirs import user_cache_dir

from cheat_at_search import data_dir


def test_get_project_root():
    result = data_dir.get_project_root()
    assert isinstance(result, str)
    assert "cheat-at-search" in result or "cheat_at_search" in result


@patch('cheat_at_search.data_dir.requests.get')
@patch('cheat_at_search.data_dir.Path.exists')
def test_download_file_skips_if_exists(mock_exists, mock_requests_get):
    mock_exists.return_value = True
    downloaded = data_dir.download_file("http://example.com/file.txt", "/tmp")
    assert downloaded == Path("/tmp") / "file.txt"
    mock_requests_get.assert_not_called()


@patch('cheat_at_search.data_dir.requests.get')
@patch('builtins.open', new_callable=mock_open)
def test_download_file_downloads_when_missing(mock_file_open, mock_requests_get):
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_requests_get.return_value.__enter__.return_value = mock_response

    with patch('cheat_at_search.data_dir.Path.exists', return_value=False):
        data_dir.download_file("http://example.com/file.txt", "/tmp")

    mock_requests_get.assert_called_once()
    mock_file_open.assert_called_once()


@patch('cheat_at_search.data_dir.subprocess.run')
@patch('cheat_at_search.data_dir.Path.exists')
def test_sync_git_repo_updates_existing(mock_exists, mock_subprocess_run):
    mock_exists.return_value = True
    result = data_dir.sync_git_repo("/tmp/test_repo", "https://github.com/test/repo.git")
    assert result == Path("/tmp/test_repo").absolute()
    calls = [call for call in mock_subprocess_run.call_args_list]
    assert len(calls) >= 2


@patch('cheat_at_search.data_dir.subprocess.run')
@patch('cheat_at_search.data_dir.Path.exists')
def test_sync_git_repo_clones_new(mock_exists, mock_subprocess_run):
    mock_exists.return_value = False
    result = data_dir.sync_git_repo("/tmp/test_repo", "https://github.com/test/repo.git")
    assert result == Path("/tmp/test_repo").absolute()
    clone_calls = [call for call in mock_subprocess_run.call_args_list
                   if 'clone' in str(call)]
    assert len(clone_calls) >= 1


@patch('cheat_at_search.data_dir.subprocess.run')
@patch('cheat_at_search.data_dir.Path.exists')
def test_sync_git_repo_reclones_on_fetch_failure(mock_exists, mock_subprocess_run):
    mock_exists.return_value = True
    from subprocess import CalledProcessError
    mock_subprocess_run.side_effect = [
        CalledProcessError(1, "git fetch"),
        MagicMock(),
        MagicMock(),
    ]
    result = data_dir.sync_git_repo("/tmp/test_repo", "https://github.com/test/repo.git")
    assert result == Path("/tmp/test_repo").absolute()


@patch('cheat_at_search.data_dir.Path.mkdir')
@patch('cheat_at_search.data_dir.Path.exists')
def test_ensure_data_subdir_creates_if_missing(mock_exists, mock_mkdir):
    mock_exists.return_value = False
    with patch.object(data_dir, 'DATA_PATH', '/tmp/data'):
        result = data_dir.ensure_data_subdir("test_subdir")
        assert str(result) == "/tmp/data/test_subdir"
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch('cheat_at_search.data_dir.Path.exists')
def test_ensure_data_subdir_returns_existing(mock_exists):
    mock_exists.return_value = True
    with patch.object(data_dir, 'DATA_PATH', '/tmp/data'):
        result = data_dir.ensure_data_subdir("test_subdir")
        assert str(result) == "/tmp/data/test_subdir"


@patch('builtins.open', new_callable=mock_open, read_data='{"openai": "test-key-123"}')
@patch('cheat_at_search.data_dir.Path')
def test_mount_key_reads_from_file(mock_path, mock_file):
    with patch.object(data_dir, 'DATA_PATH', '/tmp/data'):
        result = data_dir.mount_key("openai")
        assert result == "test-key-123"


@patch('builtins.open', new_callable=mock_open)
@patch('cheat_at_search.data_dir.getpass.getpass')
@patch('cheat_at_search.data_dir.json.dump')
def test_mount_key_prompts_when_missing(mock_json_dump, mock_getpass, mock_file):
    mock_getpass.return_value = "prompted-key"
    with patch.object(data_dir, 'DATA_PATH', '/tmp/data'):
        with patch('cheat_at_search.data_dir.Path.exists', return_value=False):
            result = data_dir.mount_key("openai")
            assert result == "prompted-key"
            mock_getpass.assert_called_once()


@patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True)
def test_key_for_provider_from_env():
    result = data_dir.key_for_provider("openai")
    assert result == "env-key"


def test_data_path_accessible():
    assert hasattr(data_dir, 'DATA_PATH')
    assert data_dir.DATA_PATH is not None


@pytest.fixture
def restore_data_path():
    original_data_path = data_dir.DATA_PATH
    yield
    data_dir.DATA_PATH = original_data_path


def test_default_data_path_is_shared_user_cache_directory(monkeypatch, restore_data_path):
    monkeypatch.delenv("CHEAT_AT_SEARCH_DATA_PATH", raising=False)
    importlib.reload(data_dir)
    expected = Path(user_cache_dir("cheat-at-search"))

    assert Path(data_dir.DATA_PATH) == expected


def test_mount_manual_path_sets_data_path_and_subdirectories(tmp_path, restore_data_path):
    manual_path = tmp_path / "mounted-data"

    data_dir.mount(manual_path=str(manual_path), load_keys=False)

    assert data_dir.DATA_PATH == manual_path
    assert manual_path.is_dir()
    assert data_dir.ensure_data_subdir("msmarco") == manual_path / "msmarco"


def test_mount_local_uses_cache_dir(tmp_path, monkeypatch, restore_data_path):
    monkeypatch.chdir(tmp_path)

    data_dir.mount(use_gdrive=False, load_keys=False)

    expected = Path(user_cache_dir("cheat-at-search"))
    assert data_dir.DATA_PATH == expected
    assert (tmp_path / expected).is_dir()
    assert data_dir.ensure_data_subdir("msmarco") == expected / "msmarco"


def test_mount_google_drive_uses_legacy_path(restore_data_path):
    google = MagicMock()
    colab = MagicMock()
    drive = MagicMock()
    colab.drive = drive
    google.colab = colab

    with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
        with patch.object(data_dir.pathlib.Path, "exists", return_value=True):
            data_dir.mount(use_gdrive=True, load_keys=False)

    drive.mount.assert_called_once_with("/content/drive")
    assert data_dir.DATA_PATH == "/content/drive/MyDrive/cheat-at-search-data/"


def test_environment_data_path_overrides_default(tmp_path, restore_data_path):
    with patch.dict(os.environ, {"CHEAT_AT_SEARCH_DATA_PATH": str(tmp_path)}):
        reloaded_data_dir = importlib.reload(data_dir)
        assert reloaded_data_dir.DATA_PATH == str(tmp_path)

    importlib.reload(data_dir)


def test_mount_overrides_environment_data_path(tmp_path, restore_data_path):
    manual_path = tmp_path / "manual-data"

    with patch.dict(os.environ, {"CHEAT_AT_SEARCH_DATA_PATH": str(tmp_path / "env-data")}):
        importlib.reload(data_dir)
        data_dir.mount(manual_path=str(manual_path), load_keys=False)

        assert data_dir.DATA_PATH == manual_path

    importlib.reload(data_dir)
