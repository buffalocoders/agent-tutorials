# **********************************************************
#                   日志
# **********************************************************
import logging
import structlog
from functools import lru_cache
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    stream=sys.stdout,
)


@lru_cache
def get_logger(name: str):
    """
    获取 logger
    :param name:
    :return:
    """
    return structlog.get_logger(name=name)
