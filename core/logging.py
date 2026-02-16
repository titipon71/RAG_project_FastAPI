import logging
from http import HTTPStatus

# ============================================================
#                  CUSTOM COLOR LOGGING (FIXED)
# ============================================================

class CustomColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_RED = "\033[1;31m"
    WHITE = "\033[37m"

    timestamp_format = "%d/%m/%Y | %H:%M "

    def format(self, record):
        asctime = self.formatTime(record, self.timestamp_format)
        time_str = f"{self.WHITE}{asctime}{self.RESET}"

        if record.levelno == logging.INFO:
            level_color = self.BOLD_GREEN
        elif record.levelno == logging.WARNING:
            level_color = self.YELLOW
        elif record.levelno == logging.ERROR:
            level_color = self.BOLD_RED
        else:
            level_color = self.WHITE

        level_str = f"{level_color}{record.levelname}:{self.RESET}"

        if "uvicorn.access" in record.name and isinstance(record.args, tuple) and len(record.args) == 5:
            client_addr = record.args[0]
            method = record.args[1]
            path = record.args[2]
            protocol = record.args[3]
            status_code = record.args[4]

            client_addr_fmt = f"{self.MAGENTA}{client_addr}{self.RESET}"

            if status_code < 200:
                status_color = self.CYAN
            elif status_code < 300:
                status_color = self.GREEN
            elif status_code < 400:
                status_color = self.YELLOW
            elif status_code < 500:
                status_color = self.RED
            else:
                status_color = self.BOLD_RED

            try:
                phrase = HTTPStatus(status_code).phrase
                status_text = f" {phrase}"
            except ValueError:
                status_text = " Unknown"

            status_fmt = f"{status_color}{status_code}{status_text}{self.RESET}"
            message = f'{client_addr_fmt} - "{method} {path} HTTP/{protocol}" {status_fmt}'
        else:
            message = record.getMessage()

        return f"{time_str} {level_str} {message}"

custom_formatter = CustomColorFormatter()
logger = logging.getLogger("uvicorn.error")

access_logger = logging.getLogger("uvicorn.access")
if access_logger.handlers:
    access_logger.handlers[0].setFormatter(custom_formatter)

uvicorn_logger = logging.getLogger("uvicorn")
if uvicorn_logger.handlers:
    uvicorn_logger.handlers[0].setFormatter(custom_formatter)