import logging
from unittest.mock import patch, MagicMock
from cheat_at_search.logger import log_at, log_to_stdout


@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_at_configures_package_logger(mock_stream_handler):
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    with patch.object(logging.getLogger("cheat_at_search"), 'hasHandlers', return_value=False):
        with patch.object(logging.getLogger("cheat_at_search"), 'handlers', []):
            with patch.object(logging.getLogger("cheat_at_search"), 'setLevel') as mock_set_level:
                log_at(logging.INFO)
                mock_set_level.assert_called_with(logging.INFO)


@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_at_with_string_level(mock_stream_handler):
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    with patch.object(logging.getLogger("cheat_at_search"), 'hasHandlers', return_value=False):
        with patch.object(logging.getLogger("cheat_at_search"), 'handlers', []):
            with patch.object(logging.getLogger("cheat_at_search"), 'setLevel') as mock_set_level:
                log_at("DEBUG")
                mock_set_level.assert_called_with("DEBUG")


def test_log_at_skips_if_has_handlers():
    with patch.object(logging.getLogger("cheat_at_search"), 'hasHandlers', return_value=True):
        with patch.object(logging.getLogger("cheat_at_search"), 'addHandler') as mock_add_handler:
            log_at(logging.INFO)
            mock_add_handler.assert_not_called()


@patch('cheat_at_search.logger.logging.getLogger')
@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_to_stdout_creates_logger(mock_stream_handler, mock_get_logger):
    mock_logger = MagicMock()
    mock_logger.handlers = []
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    result = log_to_stdout("test_module", "INFO")

    mock_get_logger.assert_called_with("cheat_at_search.test_module")
    mock_logger.setLevel.assert_called_with(logging.INFO)
    assert result == mock_logger


@patch('cheat_at_search.logger.logging.getLogger')
@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_to_stdout_with_none_name(mock_stream_handler, mock_get_logger):
    mock_logger = MagicMock()
    mock_logger.handlers = []
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    log_to_stdout(None, "ERROR")

    mock_get_logger.assert_called_with(None)


@patch('cheat_at_search.logger.logging.getLogger')
@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_to_stdout_removes_existing_handlers(mock_stream_handler, mock_get_logger):
    mock_logger = MagicMock()
    old_handler = MagicMock()
    mock_logger.handlers = [old_handler]
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    log_to_stdout("test", "INFO")

    mock_logger.removeHandler.assert_called_with(old_handler)


@patch('cheat_at_search.logger.logging.getLogger')
@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_to_stdout_prefixes_non_package_name(mock_stream_handler, mock_get_logger):
    mock_logger = MagicMock()
    mock_logger.handlers = []
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    log_to_stdout("some_module", "INFO")

    mock_get_logger.assert_called_with("cheat_at_search.some_module")


@patch('cheat_at_search.logger.logging.getLogger')
@patch('cheat_at_search.logger.logging.StreamHandler')
def test_log_to_stdout_uses_custom_format(mock_stream_handler, mock_get_logger):
    mock_logger = MagicMock()
    mock_logger.handlers = []
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    custom_format = "%(name)s - %(message)s"
    log_to_stdout("test", "INFO", custom_format)

    call_args = mock_handler.setFormatter.call_args
    assert call_args is not None
    formatter = call_args[0][0]
    assert formatter._fmt == custom_format
