import os

import gin


def resolve_gin_path():
    """Must be called before gin.parse_config_file()."""
    gin.add_config_file_search_path(os.getenv("GIN_SEARCH_PATH", "/opt/ml/code"))
    gin.config._FILE_READERS.pop()  # pylint:disable=protected-access
