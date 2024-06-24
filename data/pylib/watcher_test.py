import pytest
import pytest_mock

from data.pylib import watcher


def test_report_error(mocker: pytest_mock.MockerFixture) -> None:
    mock_send_error_message = mocker.patch("data.pylib.watcher._send_error_message")

    @watcher.report_error
    def test_func() -> None:
        raise Exception("test error")  # pylint: disable=broad-exception-raised

    with pytest.raises(Exception):
        test_func()

    filename = mock_send_error_message.call_args[0][0]
    error_message = mock_send_error_message.call_args[0][1]
    assert filename.endswith("watcher_test.py")
    assert error_message.startswith("Traceback (most recent call last):\n")
    assert error_message.endswith("Exception: test error\n")
