import subprocess

import pytest

from data.pylib import command_utils


def test_run():
    assert command_utils.run("echo hello")


def test_ignore_error():
    assert command_utils.run("exit 1", ignore_error=True) is False


def test_stderr():
    with pytest.raises(subprocess.CalledProcessError):
        command_utils.run("exit 1")


def test_return_stdout():
    assert command_utils.run("echo hello && echo && echo world", return_stdout=True) == "hello\n\nworld"


def test_detect_shell_flag():
    assert command_utils.run("echo hello") == command_utils.run(["echo", "hello"], shell=True)
