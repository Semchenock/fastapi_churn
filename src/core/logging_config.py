import logging
import os

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging() -> None:
    """Настроить корневой логгер сервиса (идемпотентно)."""
    global _configured
    if _configured:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
    )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Вернуть именованный логгер, гарантируя что логирование настроено."""
    configure_logging()
    return logging.getLogger(name)
