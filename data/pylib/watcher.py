import functools
import traceback
from typing import Any, Callable, Generator

from data.pylib import slack as slack_utils

_MAX_SLACK_MESSAGE_LENGTH = 3000


def _chunk_str_to_list(input_string: str, chunk_size: int) -> Generator[str, None, None]:
    for i in range(0, len(input_string), chunk_size):
        yield input_string[i : i + chunk_size]


def _get_error_code_block_attachment(error: str) -> dict:
    return {"attachments": [{"text": f"```{error}```"}]}


def _get_initial_message(filename: str, error: str) -> dict:
    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":crying-octopus: *Spark Job failed.*",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*File*: `{filename}` \n *Log*:",
                },
            },
        ],
        **_get_error_code_block_attachment(error),
    }


def _send_error_message(filename: str, error: str) -> None:
    error_chunks = list(_chunk_str_to_list(error, _MAX_SLACK_MESSAGE_LENGTH))
    initial_message = _get_initial_message(filename, error_chunks[0])
    slack_utils.send_alert(initial_message)

    for chunk in error_chunks[1:]:
        followup_message = _get_error_code_block_attachment(chunk)
        slack_utils.send_alert(followup_message)


def report_error(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            stack = traceback.extract_stack()

            # The second-to-last item in the stack represents the caller's frame
            caller_frame = stack[-2]

            # Extract the file name from the caller's frame
            caller_filename = caller_frame.filename
            error = traceback.format_exc()
            _send_error_message(caller_filename, error)
            raise e

    return wrapper
