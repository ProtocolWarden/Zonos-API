from logger.logger_helper import CustomLogger, get_logger

# Default logger to be used by other modules
logger: CustomLogger = get_logger(logger_name="initializer")


# Setter function to initialize the logger with a dynamic name
def set_logger_name(logger_name=None):
    global logger  # Ensure the logger is set globally
    logger = get_logger(logger_name=logger_name)
