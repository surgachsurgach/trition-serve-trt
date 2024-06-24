import subprocess
from typing import List, Optional, Union

from loguru import logger


def run(
    command: Union[str, List[str]],
    shell: Optional[bool] = None,
    ignore_error: bool = False,
    return_stdout: bool = False,
    return_stderr: bool = False,
    echo_command: bool = False,
) -> Union[str, bool]:
    """Execute a shell command.

    Args:
        command (List[str]): Command to execute.
        shell (bool, optional): Whether to execute the command in a shell. Defaults to True.
        ignore_error (bool, optional): When False, raises subprocess.CalledProcessError if the command fails.
            When True, instead of raising an exception, returns False. Defaults to False.
        return_stdout (bool, optional): When True, returns the stdout of the command. Defaults to False.
        return_stderr (bool, optional): When True, returns the stderr of the command. Defaults to False.
        echo_command (bool, optional): When True, prints the command to stdout. Defaults to False.
    Returns:
        Union[str, bool]: The output of the command or False if the command failed.
    """

    if shell is None:
        shell = isinstance(command, str)

    if return_stderr and ignore_error:
        raise ValueError("return_stderr and ignore_error cannot be True at the same time.")

    stdout = subprocess.PIPE if return_stdout else None
    stderr = None
    if ignore_error:
        stderr = subprocess.DEVNULL
    elif return_stderr:
        stderr = subprocess.PIPE

    if echo_command:
        logger.info("Running command: {}", command)

    result = subprocess.run(
        command,
        shell=shell,
        check=not ignore_error,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )

    if return_stdout or return_stderr:
        result_str = ""
        if return_stdout:
            result_str = result.stdout.decode("utf-8").strip()
        if return_stderr and result.stderr:
            result_str += "\n" + result.stderr.decode("utf-8").strip()
        return result_str

    return result.returncode == 0
