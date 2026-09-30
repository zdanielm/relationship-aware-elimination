import logging


def setup_logger(name="network_pruner", level=logging.INFO, logfile=None):
    """
    Configures a global logger for the project.

    Parameters:
    -----------
    - name: str
        Logger name (default is "my_project")
    - level: int
        Logging level (default is logging.INFO)
    - logfile: str, optional
        Optional file path to log to

    Returns:
    --------
    - logger: logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def log_or_print(msg, level="info", logger_name="network_pruner"):
    """
    Logs a message using the configured logger, or prints it if no handlers are configured.

    Parameters:
    -----------
    - msg: str
        Message to log or print
    - level: str, optional
        Logging level (default is "info")
    - logger_name: str, optional
        Logger name (default is "network_pruner")
    """
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        getattr(logger, level)(msg)
    else:
        print(msg)
