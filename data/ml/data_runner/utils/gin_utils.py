import os
import sys
import zipfile

import gin

_PYZIP_FILE = "src.zip"


def _get_pyzip_dirname(zip_name: str = _PYZIP_FILE):
    for path in sys.path:
        if path.endswith(zip_name):
            return os.path.dirname(path)


def _unzip_src():
    pyzip_dirname = _get_pyzip_dirname()
    if not pyzip_dirname:
        return

    with zipfile.ZipFile(os.path.join(pyzip_dirname, _PYZIP_FILE), "r") as zip_ref:
        zip_ref.extractall(pyzip_dirname)

    return pyzip_dirname


def resolve_gin_path():
    """Must be called before gin.parse_config_file()."""

    unzipped_dirname = _unzip_src()
    assert unzipped_dirname, "Failed to unzip src.zip"
    # Add the unzipped directory to the search path.
    gin.add_config_file_search_path(unzipped_dirname)
    # Remove the default file reader to avoid conflict with absolute path. (e.g. /path/to/file)
    gin.config._FILE_READERS.pop()  # pylint:disable=protected-access
