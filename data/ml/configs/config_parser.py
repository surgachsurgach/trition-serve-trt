from typing import List, Optional, Union

import gin
from loguru import logger

_CONFIG_SEPARATOR = "|"


def gin_parse_config(gin_files: List[str], gin_params: Optional[Union[List[str], str]]):
    if gin_params and isinstance(gin_params, list) and len(gin_params) == 1:
        gin_params = gin_params[0]
        gin_params = [f"{param}" for param in gin_params.split(_CONFIG_SEPARATOR)]

    logger.debug(f"gin files: {gin_files}")
    logger.debug(f"gin params: {gin_params}")

    gin.parse_config_files_and_bindings(gin_files, gin_params)
    logger.info(gin.config_str())
