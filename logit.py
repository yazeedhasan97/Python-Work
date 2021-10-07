"""Am aiming to color the outputs for different levels of the logging library without making any use of
a third-party library"""
import logging
from datetime import datetime

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': WHITE,
    'CRITICAL': MAGENTA,
    'ERROR': RED
}
FORMAT = '$BOLD%(levelname)s-%(name)s: %(pathname)s (%(module)s) (%(funcName)s:%(lineno)d) [p%(process)s]' \
         '\n[%(asctime)s]: %(message)s$RESET'
COLOR_FORMAT = formatter_message(FORMAT, True)


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname
            record.levelname = levelname_color

        return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
def getLogger(name='', level=logging.DEBUG, filename=None):
    ColoredFormatter(COLOR_FORMAT)

    if not logging.getLogger(name).hasHandlers():
        logger = logging.getLogger(name)
    else:
        raise ValueError("This name already has a logger attached to it. please try another name.")

    logger.setLevel(level)

    if filename:
        handler = logging.FileHandler(filename=filename, mode='w')
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)

    handler.setFormatter(ColoredFormatter(COLOR_FORMAT))
    logger.addHandler(handler)
    return logger


# Usage Example
if __name__ == '__main__':
    # logger = getLogger('app', logging.DEBUG, f'RunApp_{datetime.now():%Y%m%d_%H%M%S}.log')
    logger = getLogger()
    logger.critical("WTF")
